import asyncio
import email.utils
import logging
import random
import secrets
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import httpx

from fake_platform import AmbiguousWrite, RateLimited, ReportingSuspended

log = logging.getLogger("platform")

DEFAULT_ACCOUNTS_DB = Path(__file__).parent / "data" / "accounts.db"
LIST_KEYS = ("data", "items", "results", "posts", "twoots", "notifications")
MAX_RETRIES = 4
MAX_CONCURRENT = 8
MAX_RATE_WAIT = 300.0  # seconds a request will wait out a rate limit; longer and it fails at once instead
JITTER = 1.0  # seconds, only ever added to a server-given wait
# Requests that never reached the server: safe to repeat even when they create something.
NOT_SENT = (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)


def retry_after(resp: httpx.Response, now: datetime) -> Optional[float]:
    """Seconds the server asked us to wait, from delay-seconds or an HTTP-date (RFC 9110 10.2.3);
    None when the header is missing or malformed."""
    value = (resp.headers.get("retry-after") or "").strip()
    if value.isdigit():
        return float(value)
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError):
        return None
    if when is None:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return max(0.0, (when - now).total_seconds())


def _scope(path: str) -> str:
    """Rate limits are shared by every request in a scope. Their real scope is unknown until launch, so all
    requests share one (the cautious choice), except reports: a reporting ban mustn't stall the bots."""
    return "report" if path.rstrip("/").endswith("/report") else "default"
NOTIFICATION_KINDS = {"comment": "reply", "mention": "reply", "reply": "reply", "like": "like",
                      "repost": "repost", "follow": "follow"}


def _unwrap(payload: Any) -> Any:
    """Responses wrap their content as {"data": ...}; lists sometimes sit under another key."""
    while isinstance(payload, dict):
        key = next((k for k in LIST_KEYS if k in payload), None)
        if key is None:
            return payload
        payload = payload[key]
    return payload


def _name(value: Any) -> Optional[str]:
    if isinstance(value, dict):
        value = value.get("username") or value.get("handle") or value.get("name")
    return str(value).lstrip("@") if value is not None else None


def _id(value: Any) -> Optional[int]:
    if isinstance(value, dict):
        value = value.get("id")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def normalise_post(raw: dict) -> dict:
    """An API twoot in the shape the rest of the code uses, keeping every original field."""
    return {
        **raw,
        "id": _id(raw.get("id")),
        "author": _name(raw.get("author") or raw.get("user") or raw.get("username")),
        "content": raw.get("content") or raw.get("text") or "",
        "parent_id": _id(raw.get("parent_id") or raw.get("parent")),
        "created_at": raw.get("created_at") or raw.get("createdAt") or raw.get("timestamp"),
    }


def normalise_notification(raw: dict) -> dict:
    kind = str(raw.get("kind") or raw.get("type") or raw.get("event") or "").lower()
    twoot = raw.get("twoot") or raw.get("post") or {}
    post_id = _id(raw.get("twoot_id") or raw.get("post_id") or raw.get("reply_id") or twoot)
    return {
        **raw,
        "id": _id(raw.get("id")),
        "kind": NOTIFICATION_KINDS.get(kind, kind),
        "actor": _name(raw.get("actor") or raw.get("from_user") or raw.get("user")
                       or (twoot.get("author") if isinstance(twoot, dict) else None)),
        "post_id": post_id,
    }


class AccountStore:
    """Passwords and session tokens for our platform accounts. Lives in data/, never committed."""

    def __init__(self, path=DEFAULT_ACCOUNTS_DB) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.execute("CREATE TABLE IF NOT EXISTS accounts (username TEXT PRIMARY KEY, password TEXT NOT NULL, "
                        "token TEXT, token_type TEXT)")

    def password(self, username: str) -> str:
        row = self.db.execute("SELECT password FROM accounts WHERE username = ?", (username,)).fetchone()
        if row:
            return row[0]
        password = secrets.token_urlsafe(18)
        with self.db:
            self.db.execute("INSERT INTO accounts (username, password) VALUES (?, ?)", (username, password))
        return password

    def token(self, username: str) -> Optional[tuple]:
        row = self.db.execute("SELECT token, token_type FROM accounts WHERE username = ?", (username,)).fetchone()
        return row if row and row[0] else None

    def save_token(self, username: str, token: Optional[str], token_type: Optional[str]) -> None:
        with self.db:
            self.db.execute("UPDATE accounts SET token = ?, token_type = ? WHERE username = ?",
                            (token, token_type, username))


def _token_from(resp: httpx.Response) -> tuple:
    """(token, type) from a login/register response: a bearer token in the body, or a session cookie."""
    try:
        body = resp.json()
    except ValueError:
        body = {}
    data = body.get("data") if isinstance(body.get("data"), dict) else body
    token = data.get("access_token") or data.get("token") if isinstance(data, dict) else None
    if token:
        return token, data.get("token_type") or "Bearer"
    for name in ("session", "access_token", "token"):
        if name in resp.cookies:
            return resp.cookies[name], f"Cookie:{name}"
    for name, value in resp.cookies.items():
        return value, f"Cookie:{name}"
    raise RuntimeError(f"login gave no token or session cookie (HTTP {resp.status_code})")


class RealPlatform:
    """Legit Social through its HTTP API, mirroring FakePlatform's methods.

    Endpoints follow the organisers' twooter SDK (1.1.0, built for 2025). Field names in responses
    are read leniently because the 2026 shapes are only confirmed once the platform opens.
    """

    def __init__(self, base_url: str, bot_key: Optional[str] = None, accounts: Optional[AccountStore] = None,
                 transport: Optional[httpx.AsyncBaseTransport] = None, max_concurrent: int = MAX_CONCURRENT) -> None:
        self.client = httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=30, transport=transport)
        self.bot_key = bot_key
        self.accounts = accounts or AccountStore()
        self.max_concurrent = max_concurrent
        self._sem: Optional[asyncio.Semaphore] = None  # created inside the running loop (Python 3.9)
        self.blocked_until = {}  # rate-limit scope -> monotonic time before which nothing in it is sent
        self.clock = time.monotonic  # replaced by a fake clock in tests
        self.sleep = asyncio.sleep

    async def aclose(self) -> None:
        await self.client.aclose()

    async def ensure_account(self, username: str, display_name: str) -> None:
        """Log in, registering the account with the team's bot key the first time."""
        password = self.accounts.password(username)
        resp = await self._send("POST", "/auth/login", json={"username": username, "password": password})
        if resp.status_code in (401, 403, 404) and self.bot_key:
            resp = await self._send("POST", "/auth/register-bot", json={
                "username": username, "display_name": display_name, "password": password,
                "competition_bot_key": self.bot_key,
            })
            if resp.status_code == 409:
                resp = await self._send("POST", "/auth/login", json={"username": username, "password": password})
        resp.raise_for_status()
        self.accounts.save_token(username, *_token_from(resp))
        log.info("logged in as %s", username)

    async def update_profile(self, username: str, display_name: str, bio: str) -> None:
        (await self._call("POST", "/users/me", username, json={"display_name": display_name, "bio": bio})).raise_for_status()

    async def post(self, username: str, content: str, parent_id: Optional[int] = None) -> dict:
        resp = await self._call("POST", "/twoots/", username, json={"content": content, "parent_id": parent_id},
                                idempotent=False)
        resp.raise_for_status()
        return normalise_post(_unwrap(resp.json()))

    async def reply(self, username: str, parent_id: int, content: str) -> dict:
        return await self.post(username, content, parent_id=parent_id)

    async def like(self, username: str, post_id: int) -> bool:
        return await self._toggle("POST", f"/twoots/{post_id}/like", username)

    async def repost(self, username: str, post_id: int) -> bool:
        return await self._toggle("POST", f"/twoots/{post_id}/repost", username)

    async def follow(self, username: str, target: str) -> bool:
        return await self._toggle("POST", f"/users/@{target}/follow", username)

    async def unfollow(self, username: str, target: str) -> bool:
        return await self._toggle("DELETE", f"/users/@{target}/follow", username)

    async def get_post(self, post_id: int) -> dict:
        resp = await self._send("GET", f"/twoots/{post_id}/")
        if resp.status_code == 404:
            raise KeyError(f"post {post_id} does not exist")
        resp.raise_for_status()
        return normalise_post(_unwrap(resp.json()))

    async def feed(self, key: str, username: Optional[str] = None, at: Optional[str] = None) -> list:
        """A named feed: latest, trending, home or explore (the last two need an account)."""
        params = {"at": at} if at else None
        resp = await (self._call("GET", f"/feeds/{key}", username, params=params) if username
                      else self._send("GET", f"/feeds/{key}", params=params))
        resp.raise_for_status()
        return [normalise_post(p) for p in _unwrap(resp.json()) or [] if isinstance(p, dict)]

    async def feed_latest(self, n: int = 10, before: Optional[int] = None) -> list:
        """Newest first; `before` (a post id) pages back, using that post's time as the feed's `at`."""
        at = (await self.get_post(before))["created_at"] if before is not None else None
        posts = await self.feed("latest", at=at)
        if before is not None:
            posts = [p for p in posts if p["id"] is not None and p["id"] < before]
        return sorted(posts, key=lambda p: p["id"] or 0, reverse=True)[:n]

    async def get_notifications(self, username: str) -> list:
        """Unread notifications. They stay unread until ack_notifications, which callers run once they have
        stored them, so a failure in between loses nothing."""
        resp = await self._call("GET", "/notifications/unread", username)
        resp.raise_for_status()
        return [normalise_notification(n) for n in _unwrap(resp.json()) or [] if isinstance(n, dict)]

    async def ack_notifications(self, username: str, notes: list) -> int:
        """Mark notifications read, each on its own: one that fails stays unread and comes back next time."""
        done = 0
        for n in notes:
            if n["id"] is None:
                continue
            try:
                resp = await self._call("POST", f"/notifications/{n['id']}/mark-read", username, json={})
                resp.raise_for_status()
                done += 1
            except Exception as e:
                log.warning("couldn't mark %s's notification %s read: %s", username, n["id"], str(e) or type(e).__name__)
        return done

    async def report(self, username: str, post_id: int, reason: str = "inauthentic") -> None:
        """File a report; the moderator rules later, so the outcome isn't known here."""
        resp = await self._call("POST", f"/twoots/{post_id}/report", username, json={"reason": reason},
                                idempotent=False)
        if resp.status_code in (403, 429) and any(w in resp.text.lower() for w in ("suspend", "block", "banned")):
            raise ReportingSuspended(resp.text[:200])
        resp.raise_for_status()

    async def trending_tags(self) -> list:
        resp = await self._send("GET", "/tags/trending")
        resp.raise_for_status()
        return _unwrap(resp.json()) or []

    async def search(self, query: str) -> list:
        resp = await self._send("GET", "/search", params={"query": query})
        resp.raise_for_status()
        return [normalise_post(p) for p in _unwrap(resp.json()) or [] if isinstance(p, dict)]

    async def _toggle(self, method: str, path: str, username: str) -> bool:
        resp = await self._call(method, path, username, json={} if method == "POST" else None)
        if resp.status_code == 409:
            return False  # already liked / reposted / followed
        resp.raise_for_status()
        return True

    def _auth_headers(self, username: str) -> dict:
        token = self.accounts.token(username)
        if token is None:
            raise RuntimeError(f"{username} isn't logged in; call ensure_account first")
        value, kind = token
        if kind and kind.startswith("Cookie:"):
            return {"Cookie": f"{kind.split(':', 1)[1]}={value}"}
        return {"Authorization": f"Bearer {value}"}

    async def _call(self, method: str, path: str, username: str, idempotent: bool = True,
                    **kwargs) -> httpx.Response:
        """An authenticated request; logs in again once if the session has expired (a 401 created nothing)."""
        resp = await self._send(method, path, idempotent, headers=self._auth_headers(username), **kwargs)
        if resp.status_code == 401:
            await self.ensure_account(username, username)
            resp = await self._send(method, path, idempotent, headers=self._auth_headers(username), **kwargs)
        return resp

    async def _send(self, method: str, path: str, idempotent: bool = True, **kwargs) -> httpx.Response:
        """One request, retried on rate limits, server errors and dropped connections.

        A rate limit is waited out in full (never retried early), for every request in its scope; one longer than
        MAX_RATE_WAIT raises RateLimited instead. A request that isn't idempotent (it creates a post or a report)
        is only repeated when it can't have reached the server; otherwise it raises AmbiguousWrite.
        """
        if self._sem is None:
            self._sem = asyncio.Semaphore(self.max_concurrent)
        scope = _scope(path)
        resp, error = None, None
        for attempt in range(MAX_RETRIES):
            wait = self.blocked_until.get(scope, 0.0) - self.clock()
            if wait > MAX_RATE_WAIT:
                raise RateLimited(wait)
            if wait > 0:
                await self.sleep(wait + random.uniform(0, JITTER))
            delay = 2.0 ** attempt
            try:
                async with self._sem:
                    resp, error = await self.client.request(method, path, **kwargs), None
            except NOT_SENT as e:
                error = e
            except httpx.TransportError as e:
                if not idempotent:
                    raise AmbiguousWrite(f"{method} {path}: {e or type(e).__name__}") from e
                error = e
            else:
                if resp.status_code == 429:
                    server_wait = retry_after(resp, datetime.now(timezone.utc))
                    wait = delay if server_wait is None else server_wait
                    self.blocked_until[scope] = max(self.blocked_until.get(scope, 0.0), self.clock() + wait)
                    if wait > MAX_RATE_WAIT:
                        raise RateLimited(wait)
                    delay = 0.0  # the scope's block above is the wait
                elif resp.status_code >= 500 and not idempotent:
                    raise AmbiguousWrite(f"{method} {path}: HTTP {resp.status_code}")
                elif resp.status_code < 500:
                    return resp
            log.warning("%s %s failed (%s), attempt %d", method, path,
                        error or f"HTTP {resp.status_code}", attempt + 1)
            if attempt < MAX_RETRIES - 1 and delay:
                await self.sleep(delay)
        if error is not None:
            raise error
        return resp


class MockLegitSocial:
    """An in-memory stand-in for the HTTP API, shaped like the twooter SDK expects, for offline tests."""

    def __init__(self, bot_key: str) -> None:
        self.bot_key = bot_key
        self.users = {}  # username -> password
        self.tokens = {}  # token -> username
        self.posts = {}
        self.notifications = {}  # username -> list
        self.reports = []
        self.calls = 0
        self.rate_limit_next = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        import json
        self.calls += 1
        if self.rate_limit_next:
            self.rate_limit_next -= 1
            return httpx.Response(429, headers={"retry-after": "0"})
        path = request.url.path.removeprefix("/api")
        body = json.loads(request.content or b"{}") if request.method == "POST" else {}
        user = self.tokens.get(request.headers.get("authorization", "").removeprefix("Bearer "))
        if path == "/auth/login":
            if self.users.get(body["username"]) != body["password"]:
                return httpx.Response(404 if body["username"] not in self.users else 401)
            return self._login(body["username"])
        if path == "/auth/register-bot":
            if body.get("competition_bot_key") != self.bot_key:
                return httpx.Response(403)
            if body["username"] in self.users:
                return httpx.Response(409)
            self.users[body["username"]] = body["password"]
            return self._login(body["username"])
        if path == "/feeds/latest":
            at = request.url.params.get("at")
            posts = [p for p in self.posts.values() if at is None or p["created_at"] <= at]
            return httpx.Response(200, json={"data": sorted(posts, key=lambda p: p["id"], reverse=True)[:20]})
        if path.startswith("/twoots/") and request.method == "GET":
            post = self.posts.get(int(path.split("/")[2]))
            return httpx.Response(200, json={"data": post}) if post else httpx.Response(404)
        if user is None:
            return httpx.Response(401)
        if path == "/twoots/":
            pid = len(self.posts) + 1
            post = {"id": pid, "content": body["content"], "parent_id": body.get("parent_id"),
                    "author": {"id": 100 + pid, "username": user},
                    "created_at": datetime.now(timezone.utc).isoformat(timespec="microseconds")}
            self.posts[pid] = post
            parent = self.posts.get(body.get("parent_id"))
            if parent:
                self.notifications.setdefault(parent["author"]["username"], []).append(
                    {"id": len(self.reports) + pid, "type": "comment", "actor": {"username": user},
                     "twoot": post, "read": False})
            return httpx.Response(201, json={"data": post})
        if path.endswith("/like") or path.endswith("/repost") or path.endswith("/follow"):
            return httpx.Response(200, json={"data": True})
        if path.endswith("/report"):
            self.reports.append((user, int(path.split("/")[2]), body["reason"]))
            return httpx.Response(200, json={"data": True})
        if path == "/notifications/unread":
            return httpx.Response(200, json={"data": [n for n in self.notifications.get(user, []) if not n["read"]]})
        if path.startswith("/notifications/") and path.endswith("/mark-read"):
            for n in self.notifications.get(user, []):
                if n["id"] == int(path.split("/")[2]):
                    n["read"] = True
            return httpx.Response(200, json={"data": True})
        return httpx.Response(404)

    def _login(self, username: str) -> httpx.Response:
        token = secrets.token_hex(8)
        self.tokens[token] = username
        return httpx.Response(200, json={"data": {"access_token": token, "token_type": "Bearer"}})


async def _demo() -> None:
    mock = MockLegitSocial(bot_key="team-key")
    p = RealPlatform("https://legit.test/api", bot_key="team-key", accounts=AccountStore(":memory:"),
                     transport=httpx.MockTransport(mock.handler))
    await p.ensure_account("joe_tama", "Joe Tama")
    await p.ensure_account("npc_1", "Npc One")
    assert set(mock.users) == {"joe_tama", "npc_1"}
    p.accounts.save_token("joe_tama", "expired", "Bearer")  # forces a silent re-login

    mine = await p.post("joe_tama", "milk went up again")
    assert mine["author"] == "joe_tama" and mine["id"] == 1
    reply = await p.reply("npc_1", mine["id"], "same at our shop")
    assert reply["parent_id"] == mine["id"]
    notes = await p.get_notifications("joe_tama")
    assert [(n["kind"], n["actor"], n["post_id"]) for n in notes] == [("reply", "npc_1", reply["id"])]
    assert await p.ack_notifications("joe_tama", notes) == 1 and await p.get_notifications("joe_tama") == []

    mock.rate_limit_next = 2
    for i in range(3):
        await p.post("npc_1", f"post {i}")
    latest = await p.feed_latest(n=3)
    assert [x["id"] for x in latest] == [5, 4, 3]
    assert [x["id"] for x in await p.feed_latest(n=10, before=3)] == [2, 1]
    assert (await p.get_post(2))["content"] == "same at our shop"
    try:
        await p.get_post(99)
        raise AssertionError("missing post should raise KeyError")
    except KeyError:
        pass
    assert await p.like("joe_tama", 2) and await p.follow("joe_tama", "npc_1")
    await p.report("joe_tama", 5)
    assert mock.reports == [("joe_tama", 5, "inauthentic")]
    await p.aclose()
    print(f"real platform ok ({mock.calls} HTTP calls)")


async def _retry_demo() -> None:
    """Rate limits are honoured in full and shared; writes that may have landed are never sent twice."""
    class Clock:
        def __init__(self) -> None:
            self.now, self.sleeps = 0.0, []

        def __call__(self) -> float:
            return self.now

        async def sleep(self, seconds: float) -> None:
            self.sleeps.append(seconds)
            self.now += seconds
            await asyncio.sleep(0)

    def platform(handler) -> tuple:
        clock = Clock()
        p = RealPlatform("https://legit.test/api", accounts=AccountStore(":memory:"),
                         transport=httpx.MockTransport(handler))
        p.clock, p.sleep = clock, clock.sleep
        p.accounts.password("joe")
        p.accounts.save_token("joe", "t", "Bearer")
        return p, clock

    def scripted(*answers):
        """A handler replying with each answer in turn (an exception is raised), noting when each request came."""
        seen = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(clock_of[0]())
            answer = answers[min(len(seen), len(answers)) - 1]
            if isinstance(answer, Exception):
                raise answer
            return answer
        return handler, seen

    ok = httpx.Response(200, json={"data": {"id": 1, "author": "joe", "content": "x"}})
    future = email.utils.format_datetime(datetime.now(timezone.utc).replace(microsecond=0) + timedelta(seconds=90),
                                         usegmt=True)
    past = "Wed, 21 Oct 2015 07:28:00 GMT"
    cases = [  # (Retry-After, shortest allowed wait before the retry, longest: jitter adds up to JITTER)
        ("120", 120, 121), (future, 85, 92), (past, 0, 0), (None, 1, 2), ("soon", 1, 2),
    ]
    for header, low, high in cases:
        clock_of = [None]
        handler, seen = scripted(httpx.Response(429, headers={"retry-after": header} if header else {}), ok)
        p, clock = platform(handler)
        clock_of[0] = clock
        await p.get_post(1)
        assert len(seen) == 2 and low <= seen[1] <= high, (header, seen)

    clock_of = [None]  # a long limit fails at once, for every request in its scope, but not for reports
    handler, seen = scripted(httpx.Response(429, headers={"retry-after": "600"}), ok)
    p, clock = platform(handler)
    clock_of[0] = clock
    for _ in range(2):
        try:
            await p.get_post(1)
            raise AssertionError("expected RateLimited")
        except RateLimited:
            pass
    assert len(seen) == 1 and clock.sleeps == []
    await p.report("joe", 5)  # the reports scope isn't blocked
    clock.now = 601
    await p.get_post(1)
    assert len(seen) == 3

    clock_of = [None]  # a concurrent caller waits for the scope's block instead of hitting the server early
    handler, seen = scripted(httpx.Response(429, headers={"retry-after": "120"}), ok, ok)
    p, clock = platform(handler)
    clock_of[0] = clock
    first = asyncio.create_task(p.get_post(1))
    await asyncio.sleep(0)
    await asyncio.gather(first, p.get_post(2))
    assert all(t >= 120 for t in seen[1:]), seen

    clock_of = [None]  # cancelling a caller that is waiting out a block sends nothing
    handler, seen = scripted(httpx.Response(429, headers={"retry-after": "120"}), ok)
    p, clock = platform(handler)
    clock_of[0] = clock
    gate = asyncio.Event()

    async def stuck(seconds: float) -> None:
        await gate.wait()
    p.sleep = stuck
    task = asyncio.create_task(p.get_post(1))
    await asyncio.sleep(0.01)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    assert len(seen) == 1

    clock_of = [None]  # no pointless sleep after the last attempt
    handler, seen = scripted(httpx.Response(503))
    p, clock = platform(handler)
    clock_of[0] = clock
    try:
        await p.get_post(1)
    except httpx.HTTPStatusError:
        pass
    assert len(seen) == MAX_RETRIES and clock.sleeps == [1.0, 2.0, 4.0], clock.sleeps

    # Writes: committed, then the answer is lost -> AmbiguousWrite and exactly one post on the server.
    for failure in (httpx.ReadTimeout("response lost"), httpx.RemoteProtocolError("connection dropped"), "500"):
        mock = MockLegitSocial(bot_key="k")

        def committed_then_fails(request: httpx.Request, mock=mock, failure=failure) -> httpx.Response:
            resp = mock.handler(request)
            if request.url.path.endswith("/twoots/"):
                if failure == "500":
                    return httpx.Response(500)
                raise failure
            return resp
        p = RealPlatform("https://legit.test/api", bot_key="k", accounts=AccountStore(":memory:"),
                         transport=httpx.MockTransport(committed_then_fails))
        p.sleep = Clock().sleep
        await p.ensure_account("joe", "Joe")
        try:
            await p.post("joe", "hello")
            raise AssertionError("expected AmbiguousWrite")
        except AmbiguousWrite:
            pass
        assert len(mock.posts) == 1, (failure, len(mock.posts))

    mock, attempts = MockLegitSocial(bot_key="k"), []  # a connection never made is safe to retry

    def refused_once(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/twoots/") and not attempts:
            attempts.append(1)
            raise httpx.ConnectError("connection refused")
        return mock.handler(request)
    p = RealPlatform("https://legit.test/api", bot_key="k", accounts=AccountStore(":memory:"),
                     transport=httpx.MockTransport(refused_once))
    p.sleep = Clock().sleep
    await p.ensure_account("joe", "Joe")
    assert (await p.post("joe", "hello"))["id"] == 1 and len(mock.posts) == 1
    print("retries ok")


if __name__ == "__main__":
    asyncio.run(_demo())
    asyncio.run(_retry_demo())
