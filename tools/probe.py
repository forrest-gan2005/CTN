"""Check the live platform against what RealPlatform assumes, reading only, and keep what it answered.

    python3 -m tools.probe                  anonymous reads: feeds, a post, its replies, accounts, tags, the lock
    python3 -m tools.probe --login NAME     also logs in as one of our accounts (registering it with
                                            $CTN_TEAM_KEY if it's new) and reads /auth/me, its mentions, home

Nothing is ever posted, liked, followed or reported. Samples and a report go to data/platform2026/probe-<time>/.
"""
import argparse
import asyncio
import base64
import json
import os
from datetime import datetime, timezone

from ctn import paths
from ctn.infra.llm import load_env
from ctn.infra.real_platform import AccountStore, RealPlatform, cursor_for, iso, normalise_post
from run import API_URL

POST_FIELDS = ("post_id", "author_id", "kind", "body", "reply_to_post_id", "root_post_id", "repost_of_post_id",
               "created_at")
ACCOUNT_FIELDS = ("account_id", "username", "verified", "created_at")


class Probe:
    def __init__(self, platform: RealPlatform, out) -> None:
        self.p, self.out, self.lines, self.problems = platform, out, [], []

    def note(self, line: str) -> None:
        self.lines.append(line)
        print(line)

    def problem(self, line: str) -> None:
        self.problems.append(line)
        self.note(f"  PROBLEM: {line}")

    async def get(self, name: str, path: str, params=None, headers=None):
        resp = await self.p._send("GET", path, params=params, headers=headers)
        (self.out / f"{name}.json").write_text(json.dumps(
            {"status": resp.status_code, "headers": dict(resp.headers), "body": _body(resp)}, indent=1))
        limits = {k: v for k, v in resp.headers.items() if "ratelimit" in k.lower() or k.lower() == "retry-after"}
        self.note(f"GET {path} {params or ''} -> {resp.status_code}" + (f" limits {limits}" if limits else ""))
        if resp.status_code >= 400:
            self.problem(f"{path} answered {resp.status_code}: {resp.text[:200]}")
            return None
        return _body(resp)

    def fields(self, what: str, item, wanted) -> None:
        if not isinstance(item, dict):
            self.problem(f"{what} is {type(item).__name__}, not an object")
            return
        missing = [f for f in wanted if f not in item]
        if missing:
            self.problem(f"{what} lacks {missing}; has {sorted(item)}")
        extra = sorted(set(item) - set(wanted))
        if extra:
            self.note(f"  {what} also has {extra}")

    def page(self, what: str, body) -> list:
        if not isinstance(body, dict) or "items" not in body or "next_cursor" not in body:
            self.problem(f"{what} isn't an items/next_cursor page: {str(body)[:200]}")
            return []
        return body["items"]

    async def anonymous(self) -> None:
        await self.get("healthz", "/healthz")
        lock = await self.get("feed_lock", "/feed/lock")
        if lock is not None and "locked" not in lock:
            self.problem(f"/feed/lock has no 'locked': {lock}")
        recent_body = await self.get("posts_recent", "/posts", {"sort": "recent", "limit": 5})
        recent = self.page("recent feed", recent_body)
        for item in recent[:1]:
            self.fields("a recent post", item, POST_FIELDS)
            try:
                datetime.fromisoformat(iso(item["created_at"]))
            except (TypeError, ValueError, KeyError):
                self.problem(f"created_at {item.get('created_at')!r} doesn't parse")
            normalised = normalise_post(item, {item.get("author_id"): "someone"})
            if normalised["id"] is None or normalised["author"] is None:
                self.problem(f"normalise_post lost the id or author: {normalised}")
        if recent_body and recent_body.get("next_cursor") and recent:
            await self.cursors(recent_body["next_cursor"], recent[-1])
        trending = self.page("trending feed", await self.get("posts_trending", "/posts", {"sort": "trending", "limit": 5}))
        sample = next((i for i in recent + trending if i.get("kind") != "repost"), None)
        if sample is not None:
            pid = sample["post_id"]
            self.fields("a single post", await self.get("post", f"/posts/{pid}"), POST_FIELDS)
            self.page("replies", await self.get("replies", f"/posts/{pid}/replies", {"limit": 5}))
            await self.get("counters", f"/posts/{pid}/counters")
            account = await self.get("account", f"/accounts/{sample['author_id']}")
            self.fields("an account", account, ACCOUNT_FIELDS)
        accounts = self.page("account list", await self.get("accounts", "/accounts", {"limit": 5}))
        for item in accounts[:1]:
            self.fields("a listed account", item, ACCOUNT_FIELDS)
        await self.get("tags_popular", "/tags/popular")
        await self.get("embeds_hosts", "/embeds/hosts")
        missing = await self.p._send("GET", "/posts/999999999")
        self.note(f"GET /posts/999999999 -> {missing.status_code} (we treat 404 as gone or hidden)")
        if missing.status_code != 404:
            self.problem(f"a missing post answers {missing.status_code}, not 404")

    async def cursors(self, server: str, last: dict) -> None:
        """Our own cursor for the page after `last` must give what the server's cursor gives."""
        try:
            decoded = json.loads(base64.urlsafe_b64decode(server + "=" * (-len(server) % 4)))
        except ValueError:
            self.problem(f"the feed cursor isn't base64 JSON: {server!r}; paging back after a restart won't work")
            return
        self.note(f"  feed cursor decodes to {decoded}")
        ours = cursor_for(last["post_id"], last["created_at"])
        theirs = self.page("next page", await self.get("posts_recent_next", "/posts",
                                                      {"sort": "recent", "limit": 5, "cursor": server}))
        mine = self.page("next page (our cursor)", await self.get("posts_recent_ours", "/posts",
                                                                 {"sort": "recent", "limit": 5, "cursor": ours}))
        if [i["post_id"] for i in theirs] != [i["post_id"] for i in mine]:
            self.problem("a cursor written like the server's doesn't page the same: paging back after a restart "
                         "would skip or repeat posts")
        else:
            self.note("  our cursor pages exactly like the server's")

    async def logged_in(self, username: str) -> None:
        await self.p.ensure_account(username, username)
        headers = self.p._auth_headers(username)
        me = await self.get("auth_me", "/auth/me", headers=headers)
        if isinstance(me, dict):
            self.fields("/auth/me", me, ("account", "role"))
            self.fields("/auth/me account", me.get("account"), ACCOUNT_FIELDS)
            self.note(f"  role {me.get('role')!r}; team shown: {'team_id' in (me.get('account') or {})}")
        account_id = self.p.ids.get(username)
        if account_id is not None:
            self.page("mentions", await self.get("mentions", f"/accounts/{account_id}/mentions", {"limit": 5},
                                                 headers=headers))
        home = self.page("home feed", await self.get("posts_home", "/posts", {"sort": "home", "limit": 5},
                                                     headers=headers))
        if any("team_id" in i or "role" in i for i in home):
            self.note("  NOTE: posts carry team_id/role for a logged-in reader")


def _body(resp):
    try:
        return resp.json()
    except ValueError:
        return resp.text[:2000]


async def main(login: str) -> None:
    load_env()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = paths.DATA / "platform2026" / f"probe-{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    platform = RealPlatform(os.environ.get("CTN_API_URL") or API_URL, team_key=os.environ.get("CTN_TEAM_KEY"),
                            accounts=AccountStore())
    probe = Probe(platform, out)
    try:
        await probe.anonymous()
        if login:
            if not platform.team_key and platform.accounts.token(login) is None:
                probe.problem("set CTN_TEAM_KEY in .env to register the account")
            else:
                await probe.logged_in(login)
    finally:
        await platform.aclose()
    report = "\n".join(probe.lines + ["", f"{len(probe.problems)} problem(s)"] + probe.problems)
    (out / "report.txt").write_text(report + "\n")
    print(f"\n{len(probe.problems)} problem(s); samples and report in {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--login", metavar="USERNAME", default="",
                        help="one of our accounts to log in as (registered with $CTN_TEAM_KEY if new)")
    asyncio.run(main(parser.parse_args().login))
