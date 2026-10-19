import asyncio
import email.utils
import logging
import random
import secrets
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
import base64
import itertools
import json
from typing import Any, Optional

import httpx

from ctn import paths
from ctn.infra.fake_platform import AmbiguousWrite, Incomplete, RateLimited, ReportingSuspended, ReportQueueFull

log = logging.getLogger("platform")

DEFAULT_ACCOUNTS_DB = paths.DATA / "accounts.db"
PAGE = 100  # items asked for per page; the server may send fewer
MAX_BODY = 500  # characters the platform accepts in a post
RELIST_AT = 20  # unknown authors in one page above which the whole account list is fetched again
SKIP_PAGES = 1000  # pages of nothing but reposts a feed call reads through before it fails (never returns empty)
NEIGHBOURS = 20  # older post ids tried when paging back from a post that is gone
MAX_RETRIES = 4
MAX_CONCURRENT = 8
MAX_RATE_WAIT = 300.0  # seconds a request will wait out a rate limit; longer and it fails at once instead
JITTER = 1.0  # seconds, only ever added to a wait (a server-given one, or a feed request's turn)
EXHAUSTED_PAUSE = 60.0  # seconds a scope stays blocked after every attempt was rate-limited (even "Retry-After: 0")
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
def iso(value: Optional[str]) -> Optional[str]:
    """The API's UTC times end in "Z", which Python 3.9 can't parse; everything we store uses "+00:00"."""
    return value[:-1] + "+00:00" if isinstance(value, str) and value.endswith("Z") else value


def normalise_post(raw: dict, names: dict, badges: Optional[dict] = None) -> dict:
    """An API post in the shape the rest of the code uses (id, author username, content, parent_id), keeping
    every original field. `names` maps account ids to usernames, `badges` to whether they're verified."""
    return {
        **raw,
        "id": raw.get("post_id"),
        "author": names.get(raw.get("author_id")),
        "author_verified": bool((badges or {}).get(raw.get("author_id"))),
        "content": raw.get("body") or "",
        "parent_id": raw.get("reply_to_post_id"),
        "created_at": iso(raw.get("created_at")),
    }


def cursor_for(post_id: int, created_at: str) -> str:
    """The page cursor that continues after this post in a newest-first list, in the format the server issues:
    unpadded base64url of {"t": created_at, "i": post_id}."""
    raw = json.dumps({"t": created_at, "i": post_id}, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


class AccountStore:
    """Passwords, session tokens and account ids for our platform accounts. Lives in data/, never committed.
    A password can't be changed once an account is registered, so losing this file loses the accounts."""

    def __init__(self, path=DEFAULT_ACCOUNTS_DB) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.execute("CREATE TABLE IF NOT EXISTS accounts (username TEXT PRIMARY KEY, password TEXT NOT NULL, "
                        "token TEXT, token_type TEXT)")
        if "account_id" not in {r[1] for r in self.db.execute("PRAGMA table_info(accounts)")}:
            self.db.execute("ALTER TABLE accounts ADD COLUMN account_id INTEGER")

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

    def save_token(self, username: str, token: Optional[str], token_type: Optional[str],
                   account_id: Optional[int] = None) -> None:
        with self.db:
            self.db.execute("UPDATE accounts SET token = ?, token_type = ?, account_id = COALESCE(?, account_id) "
                            "WHERE username = ?", (token, token_type, account_id, username))

    def ids(self) -> dict:
        """username -> account id, for the accounts that have logged in at least once."""
        return dict(self.db.execute("SELECT username, account_id FROM accounts WHERE account_id IS NOT NULL"))


# The three feeds allow a team 10 reads a minute between them (SDK 0.5.0): every feed request waits its turn.
FEED_GAP = 6.5


class RealPlatform:
    """Legit Real (2026) through its HTTP API, mirroring FakePlatform's methods.

    Endpoints and fields were read from the platform's public web client in October 2026 (notes in
    data/platform2026/API_NOTES.md), before any documentation: check them against the docs once published.
    Posts name their author by account id, so usernames are looked up and cached. There is no notifications
    endpoint: a reply to one of our posts, seen in any page we read, becomes a notification for its author.
    """

    def __init__(self, base_url: str, team_key: Optional[str] = None, accounts: Optional[AccountStore] = None,
                 transport: Optional[httpx.AsyncBaseTransport] = None, max_concurrent: int = MAX_CONCURRENT,
                 feed_gap: float = FEED_GAP) -> None:
        self.client = httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=30, transport=transport)
        self.team_key = team_key
        self.accounts = accounts or AccountStore()
        self.max_concurrent = max_concurrent
        self._sem: Optional[asyncio.Semaphore] = None  # created inside the running loop (Python 3.9)
        self.blocked_until = {}  # rate-limit scope -> monotonic time before which nothing in it is sent
        self.clock = time.monotonic  # replaced by a fake clock in tests
        self.feed_gap = feed_gap
        self.feed_next = 0.0  # clock time before which no feed request may be sent
        self._feed_lock: Optional[asyncio.Lock] = None
        self.sleep = asyncio.sleep
        self.ids = self.accounts.ids()  # username -> account id
        self.names = {i: u for u, i in self.ids.items()}  # account id -> username
        self.badges = {}  # account id -> verified
        self.created = {}  # post id -> created_at exactly as the server wrote it, for page cursors
        self.roots = {}  # post id -> the post its thread starts from
        self.ours = {}  # post id -> which of our accounts wrote it
        self.inbox = {}  # username -> {reply id: notification}
        self.noted = set()  # replies made into a notification already, however often they turn up again
        self.after = {}  # oldest post id a recent-feed call returned -> the server's cursor to carry on from it
        self.searching = {}  # username -> the search cursor a lookup stopped at

    async def aclose(self) -> None:
        await self.client.aclose()

    def adopt(self, post_id: int, username: str) -> None:
        """One of our posts from before a restart: replies to it are notifications again."""
        self.ours[post_id] = username

    async def ensure_account(self, username: str, display_name: str, register: bool = True) -> None:
        """Log in, registering the account onto the team with its team code the first time (never, with
        register=False: a failed login then raises)."""
        password = self.accounts.password(username)
        login = {"username": username, "password": password}
        resp = await self._send("POST", "/auth/login", json=login)
        if resp.status_code in (400, 401, 403, 404) and self.team_key and register:
            try:
                resp = await self._send("POST", "/auth/register", idempotent=False, json={
                    **login, "display_name": display_name or None, "team_key": self.team_key})
            except AmbiguousWrite:
                resp = None  # it may exist now: logging in tells
            if resp is None or resp.status_code == 409:
                resp = await self._send("POST", "/auth/login", json=login)
        resp.raise_for_status()
        body = resp.json()
        account = body.get("account") or {}
        if not body.get("token"):
            raise RuntimeError(f"login gave no token (HTTP {resp.status_code})")
        self._know(account)
        self.accounts.save_token(username, body["token"], "Bearer", account.get("account_id"))
        log.info("logged in as %s", username)

    async def post(self, username: str, content: str, parent_id: Optional[int] = None) -> dict:
        body = {"kind": "post", "body": content, "visibility": "public"}
        if parent_id is not None:
            root = self.roots.get(parent_id)
            if root is None:
                root = (await self.get_post(parent_id))["root_post_id"] or parent_id
            body.update(kind="reply", reply_to_post_id=parent_id, root_post_id=root)
        resp = await self._call("POST", "/posts", username, json=body, idempotent=False)
        resp.raise_for_status()
        post = (await self._posts([resp.json()]))[0]
        self.ours[post["id"]] = username
        return post

    async def reply(self, username: str, parent_id: int, content: str) -> dict:
        return await self.post(username, content, parent_id=parent_id)

    async def like(self, username: str, post_id: int) -> bool:
        """True if this is a new like; False if it was liked already."""
        return await self._toggle("POST", f"/posts/{post_id}/like", username)

    async def repost(self, username: str, post_id: int) -> bool:
        """A repost is a post of its own pointing at the original."""
        resp = await self._call("POST", "/posts", username, idempotent=False,
                                json={"kind": "repost", "body": None, "visibility": "public",
                                      "repost_of_post_id": post_id})
        if resp.status_code == 409:
            return False
        if resp.status_code == 404:
            raise KeyError(f"post {post_id} does not exist")
        resp.raise_for_status()
        return True

    async def follow(self, username: str, target: str) -> bool:
        return await self._toggle("POST", f"/accounts/{await self.account_id(target)}/follow", username)

    async def unfollow(self, username: str, target: str) -> bool:
        return await self._toggle("DELETE", f"/accounts/{await self.account_id(target)}/follow", username)

    async def get_post(self, post_id: int) -> dict:
        resp = await self._send("GET", f"/posts/{post_id}")
        if resp.status_code == 404:  # also what a hidden post answers
            raise KeyError(f"post {post_id} does not exist")
        resp.raise_for_status()
        return (await self._posts([resp.json()]))[0]

    async def feed_latest(self, n: int = 10, before: Optional[int] = None) -> list:
        """Newest first, without reposts; `before` (a post id) pages back from that post."""
        params = {"sort": "recent", "limit": min(max(n, 1), PAGE)}
        found, cursor = [], None
        if before is not None:
            params["cursor"], found = await self._cursor_before(before)
            found = [p for p in found if p.get("kind") != "repost"]
            if found:  # the nearest post below a missing one is a page of its own, delivered before reading on
                return found
        for _ in range(SKIP_PAGES):  # a page of nothing but reposts isn't the end: read on until one isn't
            resp = await self._send("GET", "/posts", params=params)
            resp.raise_for_status()
            page = resp.json()
            items = page.get("items") or []
            fresh = [p for p in await self._posts(items) if p.get("kind") != "repost"]
            found += fresh
            cursor = page.get("next_cursor")
            if fresh or not cursor:
                break
            if cursor == params.get("cursor"):
                raise RuntimeError("the feed's cursor didn't move")
            params.update(cursor=cursor, limit=PAGE)
        else:
            # Only reposts so far. The last one read is a post id like any other: the caller can keep it as
            # where its history stands, and a call from it carries on with the server's own cursor.
            below = items[-1]["post_id"]
            self.after[below] = cursor
            raise Incomplete(f"{SKIP_PAGES} pages of nothing but reposts below post {before}", below=below)
        if before is not None:
            found = [p for p in found if p["id"] < before]
        if found and cursor:
            self.after[min(p["id"] for p in found)] = cursor
        return sorted((p for p in found if p["author"] is not None), key=lambda p: p["id"], reverse=True)[:n]

    async def feed_locked(self) -> bool:
        """Whether the platform has locked its feed: only citizens and moderators can act then."""
        resp = await self._send("GET", "/feed/lock")
        resp.raise_for_status()
        return bool(resp.json().get("locked"))

    async def get_notifications(self, username: str) -> list:
        """Replies to this account's posts seen since they were last acknowledged. They stay until
        ack_notifications, which callers run once they have stored them."""
        return list(self.inbox.get(username, {}).values())

    async def ack_notifications(self, username: str, notes: list) -> int:
        box = self.inbox.get(username, {})
        return sum(box.pop(n["id"], None) is not None for n in notes)

    async def report(self, username: str, post_id: int, detail: str, reason: str = "inauthentic") -> None:
        """File a report, `detail` saying why after the platform's fixed opening (it refuses a blank one); the
        moderator rules later, so the outcome isn't known here."""
        resp = await self._call("POST", f"/posts/{post_id}/report", username,
                                json={"reason": reason, "detail": detail}, idempotent=False)
        if resp.status_code == 403 and "awaiting a moderator" in resp.text.lower():
            raise ReportQueueFull(resp.text[:200])
        if resp.status_code in (403, 429) and any(w in resp.text.lower() for w in ("suspend", "block", "banned")):
            raise ReportingSuspended(resp.text[:200])
        resp.raise_for_status()

    async def report_account(self, username: str, account: str, detail: str, reason: str = "inauthentic") -> None:
        """Report an account (by username) as inauthentic, `detail` after the platform's opening for accounts."""
        resp = await self._call("POST", f"/accounts/{await self.account_id(account)}/report", username,
                                json={"reason": reason, "detail": detail}, idempotent=False)
        if resp.status_code == 403 and "awaiting a moderator" in resp.text.lower():
            raise ReportQueueFull(resp.text[:200])
        if resp.status_code in (403, 429) and any(w in resp.text.lower() for w in ("suspend", "block", "banned")):
            raise ReportingSuspended(resp.text[:200])
        resp.raise_for_status()

    async def verified_now(self, username: str) -> bool:
        """Whether this account has the platform's verified badge, asked afresh (a badge can come later)."""
        resp = await self._send("GET", f"/accounts/{await self.account_id(username)}")
        resp.raise_for_status()
        self._know(resp.json())
        return bool(self.badges.get(self.ids[username]))

    async def account_id(self, username: str) -> int:
        """The id behind a username: from the cache, else searching page by page until it turns up or the
        search ends (a search matches parts of names, so the exact one can be on any page)."""
        cursor = self.searching.pop(username, None)
        for _ in range(SKIP_PAGES):
            if username in self.ids:
                return self.ids[username]
            params = {"q": username, "limit": PAGE, **({"cursor": cursor} if cursor else {})}
            resp = await self._send("GET", "/accounts", params=params)
            resp.raise_for_status()
            page = resp.json()
            for account in page.get("items") or []:
                self._know(account)
            cursor = page.get("next_cursor")
            if not cursor:
                break
        else:
            if username not in self.ids:
                self.searching[username] = cursor
                raise Incomplete(f"searched {SKIP_PAGES} pages for {username} without reaching the end")
        if username not in self.ids:
            raise KeyError(f"no account {username}")
        return self.ids[username]

    def _know(self, account: dict) -> None:
        if account.get("account_id") is not None and account.get("username"):
            self.names[account["account_id"]] = account["username"]
            self.ids[account["username"]] = account["account_id"]
            self.badges[account["account_id"]] = bool(account.get("verified"))

    async def _list_accounts(self) -> None:
        cursor = None
        for _ in range(500):
            resp = await self._send("GET", "/accounts", params={"limit": PAGE, **({"cursor": cursor} if cursor else {})})
            resp.raise_for_status()
            page = resp.json()
            for account in page.get("items") or []:
                self._know(account)
            cursor = page.get("next_cursor")
            if not cursor:
                return

    async def _resolve(self, ids: set) -> None:
        """Learn the usernames behind these account ids: all at once from the account list if there are many,
        else one by one. An account that no longer exists stays unknown."""
        missing = {i for i in ids if i is not None and i not in self.names}
        if len(missing) > RELIST_AT:
            await self._list_accounts()
            missing = {i for i in missing if i not in self.names}
        for account_id in missing:
            resp = await self._send("GET", f"/accounts/{account_id}")
            if resp.status_code == 404:
                continue
            resp.raise_for_status()
            self._know(resp.json())

    async def _posts(self, items: list) -> list:
        """Normalise API posts, remembering what paging and replying need, and turning replies to our posts
        into notifications."""
        items = [i for i in items if isinstance(i, dict) and i.get("post_id") is not None]
        await self._resolve({i.get("author_id") for i in items})
        posts = []
        for raw in items:
            post = normalise_post(raw, self.names, self.badges)
            self.created[post["id"]] = raw.get("created_at")
            self.roots[post["id"]] = raw.get("root_post_id") or post["id"]
            self._notice(post["id"], post["author"], post["parent_id"])
            posts.append(post)
        return posts

    def recall(self, replies: list) -> None:
        """Replies the collector stored, (id, author, parent_id): after a restart, ones that never reached a bot
        become notifications again (a bot skips any it has answered)."""
        for post_id, author, parent_id in replies:
            self._notice(post_id, author, parent_id)

    def _notice(self, post_id: int, author: Optional[str], parent_id: Optional[int]) -> None:
        owner = self.ours.get(parent_id)
        if owner is not None and author not in (None, owner) and post_id not in self.noted:
            self.noted.add(post_id)
            self.inbox.setdefault(owner, {})[post_id] = {
                "id": post_id, "kind": "reply", "actor": author, "post_id": post_id}

    async def _cursor_before(self, post_id: int) -> tuple:
        """(cursor, posts) to read on from just below this post: the server's cursor from the page that ended at it,
        else one written in the server's format from its creation time (as after a restart). If the post is gone
        (hidden or deleted), from the nearest older one that isn't, which then comes back too since the cursor
        starts after it. The format is undocumented, so a server-issued cursor is always preferred."""
        if post_id in self.after:
            return self.after[post_id], []
        if self.created.get(post_id):
            return cursor_for(post_id, self.created[post_id]), []
        for candidate in range(post_id, max(0, post_id - NEIGHBOURS), -1):
            try:  # fetched even when seen before: having seen a post doesn't mean the caller has it
                post = await self.get_post(candidate)
            except KeyError:
                continue
            return cursor_for(candidate, self.created[candidate]), [] if candidate == post_id else [post]
        raise KeyError(f"can't page back from post {post_id}: it and the {NEIGHBOURS - 1} before it are gone")

    async def _toggle(self, method: str, path: str, username: str) -> bool:
        """True when this call made the like or follow (HTTP 201), False when it was there already."""
        resp = await self._call(method, path, username)
        if resp.status_code == 404 and method == "POST":
            raise KeyError(path)
        if resp.status_code in (404, 409):
            return False
        resp.raise_for_status()
        return resp.status_code == 201 if method == "POST" else True

    def _auth_headers(self, username: str) -> dict:
        token = self.accounts.token(username)
        if token is None:
            raise RuntimeError(f"{username} isn't logged in; call ensure_account first")
        return {"Authorization": f"Bearer {token[0]}"}

    async def _dispatch(self, scope: str, method: str, path: str, **kwargs) -> httpx.Response:
        """Send once the scope's rate-limit block allows it. Checked again after getting a connection slot and
        after every wait, since another request may hit a limit (or extend one) meanwhile; a blocked request
        gives its slot back while it waits. Feed requests, retries included, also leave one at a time, `feed_gap`
        after the last one actually went."""
        feed = bool(self.feed_gap) and method == "GET" and path == "/posts"
        if not feed:
            return await self._dispatch_when(scope, method, path, False, **kwargs)
        if self._feed_lock is None:
            self._feed_lock = asyncio.Lock()
        async with self._feed_lock:
            return await self._dispatch_when(scope, method, path, True, **kwargs)

    async def _dispatch_when(self, scope: str, method: str, path: str, feed: bool, **kwargs) -> httpx.Response:
        while True:
            wait = self.blocked_until.get(scope, 0.0) - self.clock()
            if wait > MAX_RATE_WAIT:
                raise RateLimited(wait)
            if feed:
                wait = max(wait, self.feed_next - self.clock())
            if wait > 0:
                await self.sleep(wait + random.uniform(0, JITTER))
                continue
            async with self._sem:
                now = self.clock()
                if self.blocked_until.get(scope, 0.0) <= now and (not feed or self.feed_next <= now):
                    if feed:
                        self.feed_next = now + self.feed_gap
                    return await self.client.request(method, path, **kwargs)

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
            delay = 2.0 ** attempt
            try:
                resp, error = await self._dispatch(scope, method, path, **kwargs), None
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
        if resp is not None and resp.status_code == 429:
            # Still limited after every attempt: a typed outcome the caller can wait out, never a refusal.
            until = max(self.blocked_until.get(scope, 0.0), self.clock() + EXHAUSTED_PAUSE)
            self.blocked_until[scope] = until
            raise RateLimited(until - self.clock())
        return resp


class MockLegitSocial:
    """An in-memory stand-in for the 2026 HTTP API, shaped as the web client expects, for offline tests.

    Posts are kept as {"id", "author": {"id", "username"}, "content", "parent_id", "created_at" (+00:00), ...}
    and sent in the API's shape (post_id, author_id, body, ..., created_at ending in "Z")."""

    def __init__(self, team_key: str) -> None:
        self.team_key = team_key
        self.users = {}  # username -> password
        self.accounts = {}  # account id -> account
        self.tokens = {}  # token -> username
        self.posts = {}
        self.hidden = set()  # post ids that answer 404, as moderated posts do
        self.locked = False  # the feed lock: only reads go through
        self.likes, self.follows = set(), set()
        self.reports = []  # (username, post id, reason)
        self.calls = 0
        self.rate_limit_next = 0

    def add_user(self, username: str, display_name: Optional[str] = None) -> str:
        """An account made outside the API (a citizen); returns a session token for it."""
        self.users.setdefault(username, secrets.token_hex(4))
        if username not in self._ids():
            account_id = len(self.accounts) + 1
            self.accounts[account_id] = {"account_id": account_id, "username": username,
                                         "display_name": display_name, "avatar_url": None, "verified": False,
                                         "created_at": _now().replace("+00:00", "Z")}
        token = secrets.token_hex(8)
        self.tokens[token] = username
        return token

    def handler(self, request: httpx.Request) -> httpx.Response:
        import re
        self.calls += 1
        if self.rate_limit_next:
            self.rate_limit_next -= 1
            return httpx.Response(429, headers={"retry-after": "0"})
        path, method, query = request.url.path.removeprefix("/api").rstrip("/"), request.method, request.url.params
        body = json.loads(request.content) if request.content else {}
        user = self.tokens.get(request.headers.get("authorization", "").removeprefix("Bearer "))
        error = lambda status, text: httpx.Response(status, json={"error": text})
        if path == "/auth/login" and method == "POST":
            if self.users.get(body.get("username")) != body.get("password"):
                return error(401, "invalid username or password")
            return self._session(body["username"], 200)
        if path == "/auth/register" and method == "POST":
            if body.get("team_key") != self.team_key:
                return error(403, "unknown team key")
            if body.get("username") in self.users:
                return error(409, "username taken")
            self.add_user(body["username"], body.get("display_name"))
            self.users[body["username"]] = body["password"]
            return self._session(body["username"], 201)
        if path == "/feed/lock" and method == "GET":
            return httpx.Response(200, json={"locked": self.locked, "reason": None, "updated_at": None})
        if self.locked and method != "GET" and not path.startswith("/auth/"):
            return error(403, "the feed is locked")
        if path == "/posts" and method == "GET":
            if query.get("sort") not in ("recent", "trending", "home"):
                return error(400, "sort must be recent, trending or home")
            posts = sorted((p for i, p in self.posts.items() if i not in self.hidden),
                           key=lambda p: (p["created_at"], p["id"]), reverse=True)
            if query.get("cursor"):
                at = json.loads(base64.urlsafe_b64decode(query["cursor"] + "=" * (-len(query["cursor"]) % 4)))
                posts = [p for p in posts if (p["created_at"], p["id"]) < (iso(at["t"]), at["i"])]
            limit = int(query.get("limit", 20))
            page = [self._wire(p) for p in posts[:limit]]
            more = len(posts) > limit
            return httpx.Response(200, json={"items": page, "next_cursor": cursor_for(
                page[-1]["post_id"], page[-1]["created_at"]) if more else None})
        if path == "/accounts" and method == "GET":
            found = [a for i, a in sorted(self.accounts.items()) if query.get("q", "") in a["username"]]
            if query.get("cursor"):
                after = json.loads(base64.urlsafe_b64decode(query["cursor"] + "=" * (-len(query["cursor"]) % 4)))["i"]
                found = [a for a in found if a["account_id"] > after]
            limit = int(query.get("limit", 20))
            more = len(found) > limit
            next_cursor = base64.urlsafe_b64encode(json.dumps({"i": found[limit - 1]["account_id"]}).encode()
                                                   ).decode().rstrip("=") if more else None
            return httpx.Response(200, json={"items": found[:limit], "next_cursor": next_cursor})
        match = re.fullmatch(r"/(posts|accounts)/(\d+)", path)
        if match and method == "GET":
            kind, key = match.group(1), int(match.group(2))
            if kind == "accounts":
                return httpx.Response(200, json=self.accounts[key]) if key in self.accounts else error(404, "not found")
            if key not in self.posts or key in self.hidden:
                return error(404, "not found")
            return httpx.Response(200, json=self._wire(self.posts[key]))
        if user is None:
            return error(401, "sign in first")
        if path == "/posts" and method == "POST":
            return self._create(user, body, error)
        match = re.fullmatch(r"/(posts|accounts)/(\d+)/(like|follow|report)", path)
        if match:
            kind, key, verb = match.group(1), int(match.group(2)), match.group(3)
            if key not in (self.accounts if kind == "accounts" else self.posts):
                return error(404, "not found")
            if verb == "report":
                if body.get("reason") not in ("hate_speech", "illegal_content", "inauthentic", "other"):
                    return error(400, "unknown reason")
                opening = "I think that this account is conducting " if kind == "accounts" else \
                    "I think that this content is "
                if not str(body.get("detail") or "").startswith(opening):
                    return error(400, f"detail must begin '{opening}inauthentic (bot) activity because...'")
                self.reports.append((user, key, body["reason"]))
                return httpx.Response(201, json={})
            done = self.likes if verb == "like" else self.follows
            if method == "DELETE":
                done.discard((user, key))
                return httpx.Response(204)
            if (user, key) in done:
                return httpx.Response(200, json={})
            done.add((user, key))
            return httpx.Response(201, json={})
        return error(404, "not found")

    def _create(self, user: str, body: dict, error) -> httpx.Response:
        kind, text = body.get("kind"), body.get("body")
        if kind in ("post", "reply") and not 1 <= len(text or "") <= MAX_BODY:
            return error(400, "body must be 1-500 characters")
        parent = body.get("reply_to_post_id") if kind == "reply" else None
        original = body.get("repost_of_post_id") if kind == "repost" else None
        if parent is not None and parent not in self.posts or original is not None and original not in self.posts:
            return error(404, "not found")
        if original is not None and any(p["author"]["username"] == user and p.get("repost_of") == original
                                        for p in self.posts.values()):
            return error(409, "already reposted")
        pid = len(self.posts) + 1
        post = {"id": pid, "kind": kind, "content": text or "", "parent_id": parent, "root_id": body.get("root_post_id"),
                "repost_of": original, "author": {"id": self._ids()[user], "username": user}, "created_at": _now()}
        self.posts[pid] = post
        return httpx.Response(201, json=self._wire(post))

    def _wire(self, post: dict) -> dict:
        return {"post_id": post["id"], "author_id": post["author"]["id"], "kind": post.get("kind", "post"),
                "body": None if post.get("kind") == "repost" else post["content"],
                "reply_to_post_id": post["parent_id"], "root_post_id": post.get("root_id"),
                "repost_of_post_id": post.get("repost_of"), "visibility": "public", "embed": None,
                "created_at": post["created_at"].replace("+00:00", "Z"), "edited_at": None}

    def _ids(self) -> dict:
        return {a["username"]: i for i, a in self.accounts.items()}

    def _session(self, username: str, status: int) -> httpx.Response:
        token = secrets.token_hex(8)
        self.tokens[token] = username
        expires = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat().replace("+00:00", "Z")
        return httpx.Response(status, json={"token": token, "expires_at": expires,
                                            "account": self.accounts[self._ids()[username]]})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


async def _demo() -> None:
    mock = MockLegitSocial(team_key="K7Q2WXRM4TPV9HZD")
    transport = httpx.MockTransport(mock.handler)
    p = RealPlatform("https://legit.test/api", feed_gap=0, team_key=mock.team_key, accounts=AccountStore(":memory:"),
                     transport=transport)
    await p.ensure_account("joe_tama", "Joe Tama")
    await p.ensure_account("npc_1", "Npc One")
    assert set(mock.users) == {"joe_tama", "npc_1"}
    p.accounts.save_token("joe_tama", "expired", "Bearer")  # forces a silent re-login

    mine = await p.post("joe_tama", "milk went up again")
    assert mine["author"] == "joe_tama" and mine["id"] == 1 and mine["created_at"].endswith("+00:00")
    reply = await p.reply("npc_1", mine["id"], "same at our shop")
    deeper = await p.reply("joe_tama", reply["id"], "and bread")
    assert reply["parent_id"] == mine["id"] and mock.posts[reply["id"]]["root_id"] == mine["id"]
    assert mock.posts[deeper["id"]]["root_id"] == mine["id"], "a reply to a reply names the thread's first post"
    notes = await p.get_notifications("joe_tama")
    assert [(n["kind"], n["actor"], n["post_id"]) for n in notes] == [("reply", "npc_1", reply["id"])]
    assert await p.ack_notifications("joe_tama", notes) == 1 and await p.get_notifications("joe_tama") == []

    stranger = mock.add_user("stranger")  # replies through the API without us: we only see it in the feed
    mock.handler(httpx.Request("POST", "https://legit.test/api/posts", headers={"authorization": f"Bearer {stranger}"},
                               json={"kind": "reply", "body": "who asked", "visibility": "public",
                                     "reply_to_post_id": mine["id"], "root_post_id": mine["id"]}))
    await p.feed_latest(n=5)
    assert [n["actor"] for n in await p.get_notifications("joe_tama")] == ["stranger"], "authors are looked up"
    restarted = RealPlatform("https://legit.test/api", feed_gap=0,
                             team_key=mock.team_key, accounts=p.accounts, transport=transport)
    restarted.recall([(4, "stranger", mine["id"])])
    assert restarted.inbox == {}, "a reply to a post not known as ours is nobody's notification"
    restarted.adopt(mine["id"], "joe_tama")
    restarted.recall([(4, "stranger", mine["id"]), (4, "stranger", mine["id"])])
    assert list(restarted.inbox["joe_tama"]) == [4], "a stored reply comes back once"
    await restarted.feed_latest(n=5)
    assert {n["actor"] for n in await restarted.get_notifications("joe_tama")} == {"stranger", "npc_1"}, \
        "after a restart replies come back; the bot's store knows which it has had"

    mock.rate_limit_next = 2
    made = [(await p.post("npc_1", f"post {i}"))["id"] for i in range(3)]
    assert await p.repost("joe_tama", made[0]) and not await p.repost("joe_tama", made[0])
    newest = await p.feed_latest(n=3)  # the newest is the repost: a page can come back short
    assert [x["id"] for x in newest] == [made[2], made[1]], "reposts aren't posts to read"
    older = [x["id"] for x in await p.feed_latest(n=10, before=made[1])]
    assert older == [5, 4, 3, 2, 1] and made[1] in p.after, "paged on with the server's cursor"
    assert made[1] not in restarted.after
    assert [x["id"] for x in await restarted.feed_latest(n=10, before=made[1])] == older, "or one written like it"
    mock.hidden.add(made[1])
    blind = RealPlatform("https://legit.test/api", feed_gap=0,
                         team_key=mock.team_key, accounts=p.accounts, transport=transport)
    nearest = [x["id"] for x in await blind.feed_latest(n=10, before=made[1])]
    assert nearest == older[:1], "below a hidden post, its nearest neighbour comes first, as a page of its own"
    assert [x["id"] for x in await blind.feed_latest(n=10, before=nearest[0])] == older[1:]
    mock.hidden.discard(made[1])
    for who, target in (("npc_1", made[1]), ("joe_tama", made[2]), ("npc_1", made[2])):
        await p.repost(who, target)
    assert [x["id"] for x in await p.feed_latest(n=2)] == made[::-1][:2], "a page of reposts is read through"

    assert (await p.get_post(reply["id"]))["content"] == "same at our shop"
    assert not (await p.get_post(reply["id"]))["author_verified"]
    mock.accounts[mock._ids()["npc_1"]]["verified"] = True
    badged = RealPlatform("https://legit.test/api", feed_gap=0,
                          team_key=mock.team_key, accounts=AccountStore(":memory:"),
                          transport=transport)
    assert (await badged.get_post(reply["id"]))["author_verified"], "the badge comes with the author"
    await badged.aclose()
    mock.accounts[mock._ids()["npc_1"]]["verified"] = False
    mock.hidden.add(reply["id"])
    for missing in (99, reply["id"]):
        try:
            await p.get_post(missing)
            raise AssertionError(f"post {missing} should raise KeyError")
        except KeyError:
            pass
    assert not await p.feed_locked()
    mock.locked = True
    assert await p.feed_locked()
    try:
        await p.like("joe_tama", 2)
        raise AssertionError("a like went through a locked feed")
    except httpx.HTTPStatusError as e:
        assert e.response.status_code == 403
    mock.locked = False
    assert await p.like("joe_tama", 2) and not await p.like("joe_tama", 2)
    assert await p.follow("joe_tama", "npc_1") and not await p.follow("joe_tama", "npc_1")
    assert await p.unfollow("joe_tama", "npc_1")
    try:
        await p.follow("joe_tama", "nobody_here")
        raise AssertionError("following an unknown account should raise KeyError")
    except KeyError:
        pass
    try:
        await p.report("joe_tama", 5, "")
        raise AssertionError("a report without a detail is refused, as the platform does")
    except httpx.HTTPStatusError as e:
        assert e.response.status_code == 400
    await p.report("joe_tama", 5, "I think that this content is inauthentic (bot) activity because it posts nonstop.")
    assert mock.reports == [("joe_tama", 5, "inauthentic")]

    stranger_team = RealPlatform("https://legit.test/api", feed_gap=0,
                                 team_key="WRONGWRONGWRONG2", accounts=AccountStore(":memory:"),
                                 transport=transport)
    try:
        await stranger_team.ensure_account("intruder", "Intruder")
        raise AssertionError("a wrong team key registered an account")
    except httpx.HTTPStatusError as e:
        assert e.response.status_code == 403

    for i in range(120):  # a search matches parts of names: the exact one may be on a later page
        mock.add_user(f"target_{i}")
    mock.add_user("target")
    finder = RealPlatform("https://legit.test/api", feed_gap=0,
                          team_key=mock.team_key, accounts=AccountStore(":memory:"),
                          transport=transport)
    assert await finder.account_id("target") == mock._ids()["target"]
    try:
        await finder.account_id("targe")
        raise AssertionError("a near name was taken for the one asked")
    except KeyError:
        pass
    await finder.aclose()

    lost = []  # the registration lands but its answer is lost: logging in finds the account

    def lose_register(request: httpx.Request) -> httpx.Response:
        resp = mock.handler(request)
        if request.url.path.endswith("/auth/register") and not lost:
            lost.append(1)
            raise httpx.ReadTimeout("answer lost")
        return resp
    q = RealPlatform("https://legit.test/api", feed_gap=0, team_key=mock.team_key, accounts=AccountStore(":memory:"),
                     transport=httpx.MockTransport(lose_register))
    await q.ensure_account("kai_lind", "Kai")
    assert "kai_lind" in mock.users and q.accounts.token("kai_lind")
    for client in (p, restarted, blind, stranger_team, q):
        await client.aclose()
    print(f"real platform ok ({mock.calls} HTTP calls)")


async def _repost_gap_demo() -> None:
    """A long run of reposts isn't the end of history: a gap behind it is read to its end, and a reply buried
    under it still reaches its post's author after a restart."""
    from ctn.infra.collector import Collector, NpcStore

    for page_size, run in ((1, 5), (50, 300)):
        mock = MockLegitSocial(team_key="k")
        transport = httpx.MockTransport(mock.handler)
        p = RealPlatform("https://legit.test/api", feed_gap=0,
                         team_key="k", accounts=AccountStore(":memory:"), transport=transport)
        for who in ("joe", "npc"):
            await p.ensure_account(who, who)
        first = await p.post("joe", "the oldest post")
        buried = await p.reply("npc", first["id"], "a reply under the reposts")
        for _ in range(run):
            pid = len(mock.posts) + 1
            mock.posts[pid] = {"id": pid, "kind": "repost", "content": "", "parent_id": None, "root_id": None,
                               "repost_of": first["id"], "author": {"id": mock._ids()["npc"], "username": "npc"},
                               "created_at": _now()}
        newest = await p.post("npc", "the newest post")
        restarted = RealPlatform("https://legit.test/api", feed_gap=0,
                                 team_key="k", accounts=p.accounts, transport=transport)
        restarted.adopt(first["id"], "joe")
        store = NpcStore(":memory:")
        collector = Collector(restarted, store, page_size=page_size, min_interval=0)
        await collector.poll()
        assert store.gaps(), (page_size, "the first look should leave history to fill")
        await collector.fill_gaps()
        assert store.count() == 3 and store.gaps() == [], (page_size, store.count(), store.gaps())
        assert {r[0] for r in store.posts_after(0)} == {first["id"], buried["id"], newest["id"]}
        assert list(restarted.inbox.get("joe", {})) == [buried["id"]], "the buried reply is a notification"
        for client in (p, restarted):
            await client.aclose()
    global SKIP_PAGES, PAGE
    saved = SKIP_PAGES, PAGE
    SKIP_PAGES, PAGE = 2, 1  # a budget too small for the run: each call stops and the next carries on
    try:
        mock = MockLegitSocial(team_key="k")
        transport = httpx.MockTransport(mock.handler)
        p = RealPlatform("https://legit.test/api", feed_gap=0,
                         team_key="k", accounts=AccountStore(":memory:"), transport=transport)
        await p.ensure_account("npc", "npc")
        first = await p.post("npc", "the oldest post")
        for _ in range(7):
            pid = len(mock.posts) + 1
            mock.posts[pid] = {"id": pid, "kind": "repost", "content": "", "parent_id": None, "root_id": None,
                               "repost_of": first["id"], "author": {"id": mock._ids()["npc"], "username": "npc"},
                               "created_at": _now()}
        newest = await p.post("npc", "the newest post")
        store = NpcStore(":memory:")
        collector = Collector(p, store, page_size=1, min_interval=0)
        await collector.poll()
        rounds = 0
        while store.gaps():
            before = store.gaps()[0][2]
            await collector.fill_gaps()  # a read that stops short ends the round, its gap kept and moved down
            rounds += 1
            assert rounds < 10, "it never got past the reposts"
            assert not store.gaps() or store.gaps()[0][2] < before, "the gap kept where it got to"
            p = RealPlatform("https://legit.test/api", feed_gap=0,
                             team_key="k", accounts=p.accounts, transport=transport)
            collector = Collector(p, store, page_size=1, min_interval=0)  # and a restart carries on from there
        assert rounds >= 3 and {r[0] for r in store.posts_after(0)} == {first["id"], newest["id"]}, rounds

        # A gap whose top post is hidden: its visible neighbour (a post or a reply, fetched before by something
        # else or not), then more reposts than one round reads.
        for kind, seen in itertools.product(("post", "reply"), (False, True)):
            mock = MockLegitSocial(team_key="k")
            transport = httpx.MockTransport(mock.handler)
            p = RealPlatform("https://legit.test/api", feed_gap=0, team_key="k", accounts=AccountStore(":memory:"),
                             transport=transport)
            await p.ensure_account("npc", "npc")
            kept = [(await p.post("npc", "the oldest post"))["id"]]
            for _ in range(7):
                pid = len(mock.posts) + 1
                mock.posts[pid] = {"id": pid, "kind": "repost", "content": "", "parent_id": None, "root_id": None,
                                   "repost_of": kept[0], "author": {"id": mock._ids()["npc"], "username": "npc"},
                                   "created_at": _now()}
            kept.append((await p.post("npc", "just below a hidden post", kept[0] if kind == "reply" else None))["id"])
            hidden = (await p.post("npc", "hidden"))["id"]
            mock.hidden.add(hidden)
            store = NpcStore(":memory:")
            store.add_gap(None, hidden)
            p = RealPlatform("https://legit.test/api", feed_gap=0,
                             team_key="k", accounts=p.accounts, transport=transport)
            if seen:
                await p.get_post(kept[1])  # e.g. a health check read it: the platform has seen it, the store hasn't
            for _ in range(10):  # a restart before every round, unless the cache is what's being tested
                await Collector(p, store, page_size=1, min_interval=0).fill_gaps()
                if not store.gaps():
                    break
                if not seen:
                    p = RealPlatform("https://legit.test/api", feed_gap=0,
                                     team_key="k", accounts=p.accounts, transport=transport)
            assert store.gaps() == [] and {r[0] for r in store.posts_after(0)} == set(kept), (kind, seen)

        for i in range(5):
            mock.add_user(f"seeker_{i}")
        mock.add_user("seeker")
        finder = RealPlatform("https://legit.test/api", feed_gap=0, team_key="k", accounts=AccountStore(":memory:"),
                              transport=transport)
        lookups = 0
        while True:
            try:
                found = await finder.account_id("seeker")
                break
            except Incomplete:
                lookups += 1  # not "no such account": unfinished, and the next try carries on
            assert lookups < 10
        assert found == mock._ids()["seeker"] and lookups >= 2, lookups
        for client in (p, finder):
            await client.aclose()
    finally:
        SKIP_PAGES, PAGE = saved
    print("repost gaps ok")


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
        p = RealPlatform("https://legit.test/api", feed_gap=0, accounts=AccountStore(":memory:"),
                         transport=httpx.MockTransport(handler))
        p.clock, p.sleep = clock, clock.sleep
        p.accounts.password("joe")
        p.accounts.save_token("joe", "t", "Bearer")
        p.names[7] = "joe"  # no account lookups in these scripts
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

    ok = httpx.Response(200, json={"post_id": 1, "author_id": 7, "kind": "post", "body": "x",
                                   "created_at": "2026-10-19T06:00:00.000000Z"})
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
    # the reports scope isn't blocked
    await p.report("joe", 5, "I think that this content is inauthentic (bot) activity because of a test.")
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

    # A caller queued for a connection slot when another request hits a limit must not slip through it.
    gate, parked, sent = asyncio.Event(), asyncio.Event(), []

    async def held(request: httpx.Request) -> httpx.Response:
        sent.append((request.url.path, clock.now))
        if request.url.path.endswith("/posts/1") and len(sent) == 1:
            await gate.wait()
            return httpx.Response(429, headers={"retry-after": "120"})
        return ok

    clock = Clock()
    p = RealPlatform("https://legit.test/api", feed_gap=0,
                     accounts=AccountStore(":memory:"), transport=httpx.MockTransport(held),
                     max_concurrent=1)
    p.names[7] = "joe"

    async def parked_sleep(seconds: float) -> None:  # time stands still until the test moves it
        await parked.wait()
    p.clock, p.sleep = clock, parked_sleep
    first = asyncio.create_task(p.get_post(1))
    await asyncio.sleep(0.01)
    second = asyncio.create_task(p.get_post(2))
    await asyncio.sleep(0.01)
    gate.set()
    await asyncio.sleep(0.05)
    assert sent == [("/api/posts/1", 0.0)], f"a request went out inside the block: {sent}"
    clock.now = 121.0
    parked.set()
    await asyncio.gather(first, second)
    assert all(t >= 120 for _, t in sent[1:]) and len(sent) == 3, sent

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
        mock = MockLegitSocial(team_key="k")

        def committed_then_fails(request: httpx.Request, mock=mock, failure=failure) -> httpx.Response:
            resp = mock.handler(request)
            if request.url.path.endswith("/posts") and request.method == "POST":
                if failure == "500":
                    return httpx.Response(500)
                raise failure
            return resp
        p = RealPlatform("https://legit.test/api", feed_gap=0, team_key="k", accounts=AccountStore(":memory:"),
                         transport=httpx.MockTransport(committed_then_fails))
        p.sleep = Clock().sleep
        await p.ensure_account("joe", "Joe")
        try:
            await p.post("joe", "hello")
            raise AssertionError("expected AmbiguousWrite")
        except AmbiguousWrite:
            pass
        assert len(mock.posts) == 1, (failure, len(mock.posts))

    mock, attempts = MockLegitSocial(team_key="k"), []  # a connection never made is safe to retry

    def refused_once(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/posts") and request.method == "POST" and not attempts:
            attempts.append(1)
            raise httpx.ConnectError("connection refused")
        return mock.handler(request)
    p = RealPlatform("https://legit.test/api", feed_gap=0, team_key="k", accounts=AccountStore(":memory:"),
                     transport=httpx.MockTransport(refused_once))
    p.sleep = Clock().sleep
    await p.ensure_account("joe", "Joe")
    assert (await p.post("joe", "hello"))["id"] == 1 and len(mock.posts) == 1
    print("retries ok")


async def _feed_pace_demo() -> None:
    """Every feed request waits its turn, however many callers want one at once and however a call pages: at most
    10 reach the server in any minute."""
    sent = []

    class Clock:
        now = 0.0

        def __call__(self) -> float:
            return self.now

        async def sleep(self, seconds: float) -> None:  # each sleeper wakes at its own time, not after the others
            until = self.now + seconds
            await asyncio.sleep(0)
            self.now = max(self.now, until)

    def item(post_id: int, kind: str = "post") -> dict:
        return {"post_id": post_id, "author_id": 7, "kind": kind,
                "body": None if kind == "repost" else f"post {post_id}",
                "reply_to_post_id": None, "root_post_id": None, "repost_of_post_id": 1 if kind == "repost" else None,
                "created_at": "2026-10-19T06:00:00.000000Z"}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/posts") and request.method == "GET":
            sent.append(clock.now)
            if request.url.params.get("cursor") == "reposts":  # a page of nothing but reposts, then the rest
                return httpx.Response(200, json={"items": [item(50, "repost")], "next_cursor": "after-reposts"})
            return httpx.Response(200, json={"items": [item(100 - len(sent))], "next_cursor": "reposts"})
        return httpx.Response(404, json={"error": "not found"})

    clock = Clock()
    p = RealPlatform("https://legit.test/api", accounts=AccountStore(":memory:"),
                     transport=httpx.MockTransport(handler))
    p.clock, p.sleep = clock, clock.sleep
    p.names[7] = "citizen"
    await asyncio.gather(*(p.feed_latest(n=5) for _ in range(6)))  # callers all at once
    p.after[42] = "reposts"
    for _ in range(4):
        await p.feed_latest(n=5, before=42)  # each reads a page of reposts, then one more
    assert len(sent) == 14, sent
    assert all(b - a >= FEED_GAP for a, b in zip(sent, sent[1:])), sent
    assert max(sum(t <= s < t + 60 for s in sent) for t in sent) <= 10
    await p.aclose()

    def spaced(times: list) -> bool:
        return (all(b - a >= FEED_GAP for a, b in zip(times, times[1:]))
                and max(sum(t <= u < t + 60 for u in times) for t in times) <= 10)

    def fresh(handler, **kwargs) -> RealPlatform:
        q = RealPlatform("https://legit.test/api", accounts=AccountStore(":memory:"),
                         transport=httpx.MockTransport(handler), **kwargs)
        q.clock, q.sleep = clock, clock.sleep
        q.names[7] = "citizen"
        return q

    page = {"items": [item(9)], "next_cursor": None}
    sent.clear()
    failed = set()

    def flaky(request: httpx.Request) -> httpx.Response:  # every read fails once, then works
        if request.url.path.endswith("/posts"):
            sent.append(clock.now)
            if len(sent) not in failed and len(sent) % 2:
                failed.add(len(sent))
                return httpx.Response(503, json={"error": "busy"})
            return httpx.Response(200, json=page)
        return httpx.Response(404, json={"error": "not found"})
    q = fresh(flaky)
    for _ in range(6):
        await q.feed_latest(n=5)
    assert len(sent) == 12 and spaced(sent), sent  # the retries wait their turn too
    await q.aclose()

    sent.clear()
    hold = asyncio.Event()

    async def congested(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/feed/lock"):
            await hold.wait()  # keeps the only connection slot busy
            return httpx.Response(200, json={"locked": False})
        sent.append(clock.now)
        return httpx.Response(200, json=page)
    q = fresh(congested, max_concurrent=1)
    lock = asyncio.ensure_future(q.feed_locked())
    await asyncio.sleep(0)
    reads = asyncio.gather(*(q.feed_latest(n=5) for _ in range(3)))
    for _ in range(20):
        await asyncio.sleep(0)
    clock.now += 30  # the slot stays busy long past every reader's turn
    hold.set()
    await lock
    await reads
    assert len(sent) == 3 and spaced(sent), sent  # not all at once when the slot frees
    await q.aclose()

    sent.clear()

    def limited(request: httpx.Request) -> httpx.Response:
        sent.append(clock.now)
        if len(sent) == 1:
            return httpx.Response(429, headers={"retry-after": "10"}, json={"error": "slow down"})
        return httpx.Response(200, json=page)
    q = fresh(limited)
    await asyncio.gather(*(q.feed_latest(n=5) for _ in range(4)))
    assert len(sent) == 5 and sent[1] >= sent[0] + 10 and spaced(sent[1:]), sent  # out of the wait, one by one

    waiting = asyncio.ensure_future(q.feed_latest(n=5))  # next turn is 6.5 s off
    await asyncio.sleep(0)
    waiting.cancel()
    try:
        await waiting
    except asyncio.CancelledError:
        pass
    assert await q.feed_latest(n=5) and not q._feed_lock.locked(), "a cancelled read gives its turn back"
    await q.aclose()
    print("feed pace ok")


if __name__ == "__main__":
    asyncio.run(_demo())
    asyncio.run(_feed_pace_demo())
    asyncio.run(_repost_gap_demo())
    asyncio.run(_retry_demo())
