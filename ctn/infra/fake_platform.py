import asyncio
import contextvars
import itertools
from datetime import datetime, timedelta, timezone
from typing import Optional

REPORT_SUSPENSION = timedelta(hours=6)
# How soon a write in this task goes out of the team's write lane, lower first and in order of arrival within a
# level: an ANCHOR post, then a reply to a CITIZEN (the writes that can change a mind, best read while the citizen is
# still about), then the team's LIFT of the anchor in its rush, then the rest.
ANCHOR, CITIZEN, LIFT, NORMAL = 0, 1, 2, 3
WRITE_PRIORITY = contextvars.ContextVar("write_priority", default=NORMAL)
# A writer's marker that a platform which `marks_sends` sets {"left": True} on just before a write's request goes
# out: a write cancelled with it still False never left this machine. A dict, so a task the platform starts in a
# copied context still marks the writer's.
WRITE_LEFT = contextvars.ContextVar("write_left", default=None)


class RateLimited(Exception):
    """The platform asked us to wait longer than a request is allowed to."""

    def __init__(self, wait: float) -> None:
        super().__init__(f"rate-limited for another {wait:.0f}s")
        self.wait = wait


class AmbiguousWrite(Exception):
    """A request that creates something may or may not have taken effect, so it is never resent blindly."""


class Incomplete(RuntimeError):
    """A read stopped at its page budget before it could tell; calling again carries on where it stopped.
    `below` is the oldest post id it read past, when that is progress worth keeping."""

    def __init__(self, message: str, below: Optional[int] = None) -> None:
        super().__init__(message)
        self.below = below


class ReportingSuspended(Exception):
    """The team reported a genuine citizen's post and can't report anything for six hours."""


class AccountGone(Exception):
    """The account behind a post no longer exists on the platform."""


class ReportQueueFull(Exception):
    """The team has as many reports awaiting a moderator's ruling as the platform holds (20); more can be filed
    once some are ruled on."""


ACCOUNT_OPENING = "I think that this account is conducting "  # the platform's fixed opening for an account report


class FakePlatform:
    def __init__(self, competitors=()) -> None:
        self.posts = {}          # post_id -> post dict
        self.following = {}      # username -> set of usernames they follow
        self.notifications = {}  # username -> list of notification dicts
        self.competitors = set(competitors)  # the moderator's ground truth: accounts run by teams
        self.reports = {}        # post_id -> usernames that reported it
        self.trending = []       # post ids in Trending, top first, as a test sets them
        self.hidden = set()      # posts the moderator confirmed as inauthentic
        self.account_reports = []  # (reporter, account reported)
        self.reporting_blocked_until: Optional[datetime] = None
        self._ids = itertools.count(1)
        self._note_ids = itertools.count(1)

    async def post(self, username: str, content: str, parent_id: Optional[int] = None) -> dict:
        if parent_id is not None:
            parent = self._get(parent_id)
        post = {
            "id": next(self._ids),
            "author": username,
            "content": content,
            "parent_id": parent_id,
            "likes": set(),
            "reposts": set(),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self.posts[post["id"]] = post
        if parent_id is not None:
            self._notify(parent["author"], "reply", username, post["id"])
        return post

    async def reply(self, username: str, parent_id: int, content: str) -> dict:
        return await self.post(username, content, parent_id=parent_id)

    async def like(self, username: str, post_id: int) -> bool:
        """Idempotent: returns False if already liked, never toggles."""
        post = self._get(post_id)
        if username in post["likes"]:
            return False
        post["likes"].add(username)
        self._notify(post["author"], "like", username, post_id)
        return True

    async def unlike(self, username: str, post_id: int) -> bool:
        post = self._get(post_id)
        if username not in post["likes"]:
            return False
        post["likes"].remove(username)
        return True

    async def repost(self, username: str, post_id: int) -> bool:
        post = self._get(post_id)
        if username in post["reposts"]:
            return False
        post["reposts"].add(username)
        self._notify(post["author"], "repost", username, post_id)
        return True

    async def set_private(self, username: str, private: bool = True) -> None:
        self.private = getattr(self, "private", set())
        (self.private.add if private else self.private.discard)(username)

    async def delete_post(self, username: str, post_id: int) -> bool:
        if post_id not in self.posts:
            return False
        if self.posts[post_id]["author"] != username:
            raise PermissionError(f"post {post_id} isn't {username}'s")
        del self.posts[post_id]
        return True

    async def follow(self, username: str, target: str) -> bool:
        if username == target:
            return False
        followed = self.following.setdefault(username, set())
        if target in followed:
            return False
        followed.add(target)
        self._notify(target, "follow", username, None)
        return True

    async def unfollow(self, username: str, target: str) -> bool:
        followed = self.following.get(username, set())
        if target not in followed:
            return False
        followed.remove(target)
        return True

    async def report_account(self, username: str, account: str, detail: str) -> None:
        """Report an account as inauthentic; the platform rules later (suspends it if it has a hidden post and
        100 actions, otherwise the report lapses)."""
        if not detail.startswith(ACCOUNT_OPENING):
            raise ValueError(f"an account report's detail must begin {ACCOUNT_OPENING!r}: {detail!r}")
        now = datetime.now(timezone.utc)
        if self.reporting_blocked_until is not None and now < self.reporting_blocked_until:
            raise ReportingSuspended(f"reporting blocked until {self.reporting_blocked_until.isoformat()}")
        self.account_reports.append((username, account))

    async def report(self, username: str, post_id: int, detail: str) -> bool:
        """Report a post as inauthentic, saying why as the platform requires. This moderator rules at once; True if
        upheld."""
        if not detail.startswith("I think that this content is "):
            raise ValueError(f"a report's detail must begin 'I think that this content is': {detail!r}")
        now = datetime.now(timezone.utc)
        if self.reporting_blocked_until is not None and now < self.reporting_blocked_until:
            raise ReportingSuspended(f"reporting blocked until {self.reporting_blocked_until.isoformat()}")
        post = self._get(post_id)
        self.reports.setdefault(post_id, []).append(username)
        if post["author"] in self.competitors:
            self.hidden.add(post_id)
            return True
        self.reporting_blocked_until = now + REPORT_SUSPENSION
        return False

    async def get_post(self, post_id: int) -> dict:
        return {**self._get(post_id), "hidden": post_id in self.hidden}

    async def feed_latest(self, n: int = 10, before: Optional[int] = None) -> list:
        """Newest first; pass `before` (a post id) to page further back in time."""
        posts = [p for p in self.posts.values() if before is None or p["id"] < before]
        return sorted(posts, key=lambda p: p["id"], reverse=True)[:n]

    async def feed_trending(self, n: int = 50) -> list:
        return list(self.trending[:n])

    async def feed_home(self, username: str, n: int = 10) -> list:
        followed = self.following.get(username, set())
        followed_posts = [p for p in self.posts.values() if p["author"] in followed]
        return sorted(followed_posts, key=lambda p: p["id"], reverse=True)[:n]

    async def replies(self, post_id: int) -> list:
        self._get(post_id)
        return [p for p in self.posts.values() if p["parent_id"] == post_id]

    async def search(self, query: str, n: int = 10) -> list:
        q = query.lower()
        hits = [p for p in self.posts.values() if q in p["content"].lower()]
        return sorted(hits, key=lambda p: p["id"], reverse=True)[:n]

    async def get_notifications(self, username: str) -> list:
        """Unread notifications; they stay unread until acknowledged."""
        return list(self.notifications.get(username, []))

    async def ack_notifications(self, username: str, notes: list) -> int:
        done = {n["id"] for n in notes}
        self.notifications[username] = [n for n in self.notifications.get(username, []) if n["id"] not in done]
        return len(done)

    def _get(self, post_id: int) -> dict:
        if post_id not in self.posts:
            raise KeyError(f"post {post_id} does not exist")
        return self.posts[post_id]

    def _notify(self, to: str, kind: str, actor: str, post_id: Optional[int]) -> None:
        if to == actor:
            return
        self.notifications.setdefault(to, []).append({
            "id": next(self._note_ids),
            "kind": kind,
            "actor": actor,
            "post_id": post_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
        })


async def _demo() -> None:
    p = FakePlatform()
    a = await p.post("alice", "Clinics on the outer islands are already stretched.")
    await p.like("bob", a["id"])
    assert (await p.like("bob", a["id"])) is False
    await p.reply("bob", a["id"], "My aunt waited 3 days for a nurse in 2021.")
    await p.follow("carol", "alice")
    await p.repost("carol", a["id"])

    assert len(a["likes"]) == 1
    assert [x["author"] for x in await p.feed_home("carol")] == ["alice"]
    notes = await p.get_notifications("alice")
    assert {n["kind"] for n in notes} == {"like", "reply", "follow", "repost"}
    assert await p.get_notifications("alice") == notes, "reading alone doesn't clear"
    assert await p.ack_notifications("alice", notes) == 4 and await p.get_notifications("alice") == []
    assert len(await p.search("NURSE")) == 1
    try:
        await p.like("bob", 999)
    except KeyError:
        pass

    p.competitors = {"bob"}
    reply = [x for x in p.posts.values() if x["author"] == "bob"][0]
    why = "I think that this content is inauthentic (bot) activity because it posts every 20 seconds."
    try:
        await p.report("alice", reply["id"], "")
        raise AssertionError("a report without a reason should be refused")
    except ValueError:
        pass
    assert await p.report("alice", reply["id"], why) and reply["id"] in p.hidden
    assert not await p.report("alice", a["id"], why)
    try:
        await p.report("alice", reply["id"], why)
        raise AssertionError("reporting should be suspended after a wrong report")
    except ReportingSuspended:
        pass
    print("ok:", await p.feed_latest())


if __name__ == "__main__":
    asyncio.run(_demo())
