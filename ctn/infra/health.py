import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

log = logging.getLogger("health")

QUIET_AT = 2  # posts suspected or confirmed taken down before an account stops writing; three ban it
SUSPENDED_AT = 3
SUSPECTED = ("hidden", "missing")
# How the 2026 API marks a moderator-confirmed takedown is unknown until launch; fill this in then, e.g.
# {"moderation_status": {"upheld"}}. Until then nothing counts as confirmed, so the policy runs on suspicion.
CONFIRMED_MARKERS = {}
# Older posts are looked at less often but never dropped, since moderation can come days later.
RECHECK = ((timedelta(days=1), timedelta(hours=2)), (timedelta(days=3), timedelta(hours=6)),
           (None, timedelta(hours=24)))


def outcome(post: dict) -> str:
    """What the platform shows for one of our posts: confirmed, hidden or visible. Field names are read
    leniently; scoring separates a temporary hide (five reports, awaiting a moderator) from a confirmed one."""
    if any(str(post.get(field) or "").lower() in values for field, values in CONFIRMED_MARKERS.items()):
        return "confirmed"
    if any(post.get(k) for k in ("hidden", "is_hidden", "removed", "deleted", "is_deleted")):
        return "hidden"
    if str(post.get("visibility") or post.get("status") or "").lower() in ("hidden", "removed", "deleted"):
        return "hidden"
    return "visible"


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


class Health:
    """Watches our own posts for takedowns.

    Every report on one of our posts is upheld (the accounts are a team's), so a post that turns hidden or
    disappears probably counts towards the three that suspend an account. Without a confirmation signal from
    the API that is only a suspicion, recorded as such; the policy is deliberately cautious anyway: an account
    with QUIET_AT suspected or confirmed takedowns stops posting and replying and only likes, reposts and
    follows, which can't be reported. A post that comes back is marked restored and stops counting.
    """

    def __init__(self, platform, state, interval: float = 600.0, since: str = "") -> None:
        self.platform = platform
        self.state = state
        self.interval = interval
        self.since = since  # watch our posts from this time on (UTC ISO); "" means all of them
        self.checked = {}  # post_id -> when it was last fetched successfully

    def counts(self) -> dict:
        """username -> posts suspected or confirmed taken down."""
        return {u: c["suspected"] + c["confirmed"] for u, c in self.state.moderation_counts().items()}

    def quiet(self) -> set:
        return {u for u, n in self.counts().items() if n >= QUIET_AT}

    async def check(self) -> int:
        """Fetch our posts that are due; returns how many changed status. A failed fetch is retried next round."""
        now = datetime.now(timezone.utc)
        changed = 0
        for username, post_id, published in self.state.made_posts(self.since):
            last = self.checked.get(post_id)
            age = now - datetime.fromisoformat(published)
            if last is not None and now - last < next(e for limit, e in RECHECK if limit is None or age < limit):
                continue
            try:
                status = outcome(await self.platform.get_post(post_id))
            except KeyError:
                status = "missing"
            except Exception as e:
                log.warning("couldn't check %s's post #%s: %s", username, post_id, str(e) or type(e).__name__)
                continue
            self.checked[post_id] = now
            previous = self.state.moderation_status(post_id)
            if status == "visible":
                if previous in SUSPECTED and self.state.set_moderation(post_id, username, "restored", _iso(now)):
                    changed += 1
                    log.info("%s: post #%s is visible again", username, post_id)
                continue
            if self.state.set_moderation(post_id, username, status, _iso(now)):
                changed += 1
                self._report(username, post_id, status)
        return changed

    def _report(self, username: str, post_id: int, status: str) -> None:
        c = self.state.moderation_counts().get(username, {"suspected": 0, "confirmed": 0})
        total = c["suspected"] + c["confirmed"]
        log.warning("%s: post #%s %s (%d suspected, %d confirmed)%s", username, post_id, status, c["suspected"],
                    c["confirmed"], "; going quiet" if total == QUIET_AT else "")
        if c["confirmed"] >= SUSPENDED_AT:
            log.error("%s has %d confirmed takedowns and is suspended: activate a reserve with "
                      "`python3 -m ctn.text.personas activate <username>`", username, c["confirmed"])
        elif total >= SUSPENDED_AT:
            log.error("%s has %d posts suspected taken down and may be suspended; check the account", username, total)

    async def run(self, until: Optional[float] = None, on_change=None) -> None:
        """Check every `interval`, then hand the quiet accounts to `on_change` from what is stored,
        whether or not this round found anything, so a round that failed halfway can't leave bots out of date."""
        loop = asyncio.get_running_loop()
        while until is None or loop.time() < until:
            try:
                await self.check()
            except Exception:
                log.exception("health check failed")
            if on_change is not None:
                on_change(self.quiet())
            await asyncio.sleep(self.interval if until is None else max(0.0, min(self.interval, until - loop.time())))


async def _demo() -> None:
    from ctn.infra.state import StateStore
    from ctn.red.bot import Action

    class Platform:
        """Answers get_post from a table: a dict is the post, an exception is raised."""

        def __init__(self) -> None:
            self.answers = {}

        async def get_post(self, post_id: int) -> dict:
            answer = self.answers[post_id]
            if isinstance(answer, Exception):
                raise answer
            return answer

    def publish(state, username: str, post_id: int, age: timedelta) -> None:
        ts = _iso(datetime.now(timezone.utc) - age)
        state.record(username, ts, Action("post", content=f"post {post_id}", made_id=post_id))

    # R1: two hidden posts, then a fetch that fails; the failure neither hides them from the bots nor counts as
    # a check, and later rounds with nothing new still hand over the quiet set.
    p, state = Platform(), StateStore(":memory:")
    for pid in (1, 2, 3, 4):
        publish(state, "demo", pid, timedelta(minutes=30))
    p.answers = {1: {"hidden": True}, 2: {"hidden": True}, 3: RuntimeError("HTTP 503"), 4: {"hidden": False}}
    health, seen = Health(p, state, interval=0.01), []
    loop = asyncio.get_running_loop()
    await health.run(loop.time() + 0.05, on_change=seen.append)
    assert seen and all(s == {"demo"} for s in seen) and len(seen) >= 2, seen
    assert 3 not in health.checked and 4 in health.checked
    p.answers[3] = {"visible": True}
    assert await health.check() == 0 and 3 in health.checked

    # R5: temporary hide, restored, 404, explicit confirmation and transport failure all end differently.
    p, state = Platform(), StateStore(":memory:")
    for pid in (10, 11, 12, 13, 14):
        publish(state, "ana", pid, timedelta(hours=1))
    p.answers = {10: {"visibility": "hidden"}, 11: {"hidden": True}, 12: KeyError(12),
                 13: {"moderation_status": "upheld"}, 14: ConnectionError()}
    CONFIRMED_MARKERS["moderation_status"] = {"upheld"}
    health = Health(p, state)
    try:
        assert await health.check() == 4
        p.answers[11] = {"hidden": False}  # the moderator cleared it
        health.checked.clear()
        assert await health.check() == 1
    finally:
        CONFIRMED_MARKERS.clear()
    assert [state.moderation_status(i) for i in (10, 11, 12, 13, 14)] == ["hidden", "restored", "missing",
                                                                          "confirmed", None]
    assert state.moderation_counts() == {"ana": {"suspected": 2, "confirmed": 1}} and health.quiet() == {"ana"}

    # R4: a four-day-old post is still watched, on the slowest schedule.
    p, state = Platform(), StateStore(":memory:")
    publish(state, "old", 20, timedelta(days=4))
    p.answers = {20: {"hidden": False}}
    health = Health(p, state)
    assert await health.check() == 0 and 20 in health.checked
    p.answers[20] = {"hidden": True}
    assert await health.check() == 0, "rechecked within a day"
    health.checked[20] -= timedelta(hours=25)
    assert await health.check() == 1 and state.moderation_status(20) == "hidden"
    assert outcome({"visibility": "Hidden"}) == "hidden" and outcome({"visibility": "public"}) == "visible"
    print("health ok")


if __name__ == "__main__":
    asyncio.run(_demo())
