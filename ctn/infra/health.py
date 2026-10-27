import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from ctn.infra.fake_platform import ANCHOR, LIFT, WRITE_PRIORITY, RateLimited

log = logging.getLogger("health")

# Posts suspected or confirmed taken down before an account stops writing; three ban it, and one is enough for the
# next report on a new post to suspend an account with 100 actions. One until 31 October, to save registrations
# (200 a team) for the x4 days; two from 1 November.
QUIET_AT = 1
SUSPENDED_AT = 3
SUSPECTED = ("hidden", "missing")
# How the 2026 API marks a moderator-confirmed takedown is unknown until launch; fill this in then, e.g.
# {"moderation_status": {"upheld"}}. Until then nothing counts as confirmed, so the policy runs on suspicion.
CONFIRMED_MARKERS = {}
# Older posts are looked at less often but never dropped, since moderation can come days later.
RECHECK = ((timedelta(days=1), timedelta(hours=2)), (timedelta(days=3), timedelta(hours=6)),
           (None, timedelta(hours=24)))
# Every reply the moderator hid to 23 October was hidden within 12 hours of posting (posts, up to 63 hours on),
# and replies are nine in ten of what we publish, so older replies are looked at less and less often.
REPLY_RECHECK = ((timedelta(hours=12), timedelta(hours=3)), (timedelta(days=1), timedelta(hours=6)),
                 (timedelta(days=3), timedelta(hours=24)), (None, timedelta(days=7)))
# Due posts are fetched newest first, replies past a day last, one every HEALTH_PACE and at most HEALTH_BATCH a
# round: fetching all of them at once (~430 after a restart) drew ~35 rate limits a minute for 12 minutes against
# what looks like a limit of 30 a minute, and a rate limit stalls every bot.
HEALTH_PACE = 3.0
HEALTH_BATCH = 150
# An anchor's time on Trending is over by 3.4 hours (median 2.2, 50 anchors to 24 October), while the reports that
# hid anchors came later: of 61 hidden to 24 October, 57 were hidden 3+ hours after posting, many in waves such as
# the one just after midnight, when each team's daily report allowance starts again. A report already in is still
# ruled, so an anchor is deleted WITHDRAW_AFTER posting (if the platform still shows it), older ones too, which
# leaves later reports nothing to land on. Points already scored stay. Two hours from 25 October (Forrest): an anchor
# squad account was suspended for three hidden anchors, three of its six in a day hidden before the 3-hour delete.
# Off from 27 October evening (Forrest: the later likes are worth having; watch whether suspensions jump): rival
# anchors stay up and stay in Trending 7+ hours, 12% of the citizen likes on them came 2-4 hours after posting, and
# their anchor accounts post about hourly without being suspended.
WITHDRAW = False
WITHDRAW_AFTER = timedelta(hours=2)
WITHDRAW_EVERY = 300.0
REPLIES_KEY = "withdraw_replies"  # meta: whether our replies survive a deleted post ("survive" | "vanish")
WITHDRAW_BATCH = 5  # a look and a delete each; the looks share Health's pace, the deletes the write lane
# Teammates' replies lifting an anchor go once the anchor has (WITHDRAW_LIFTS), as soon as replies are known to outlive
# it: on 24 October six lift replies, 5-6.5 hours old, were hidden after their anchors had been deleted at 3 hours,
# and two of their accounts were suspended within the hour. With the anchor gone they lift nothing. A loop of their
# own, so a delete waiting in the write lane never holds up an anchor's, at LIFT priority (behind citizen replies,
# ahead of ordinary writes), and only those under LIFTS_MAX_AGE: every reply hidden to 24 October went within 12 hours.
WITHDRAW_LIFTS = True
LIFTS_MAX_AGE = timedelta(hours=24)


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


# bot.SUSPENDED: the meta key prefix of an account the platform refused as suspended. Set here too: on 26 October the
# anchor account was suspended while every draft it wrote was dropped, so no write of its was refused, nothing marked
# it, and the anchor slots waited on it until set by hand.
SUSPENDED = "suspended:"


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


class Pacer:
    """One request every `gap` seconds across everyone sharing it: Health's and Withdraw's post reads together."""

    def __init__(self, gap: float) -> None:
        self.gap = gap
        self.last = None
        self._lock = None  # made inside the running loop (Python 3.9)

    async def wait(self) -> None:
        if not self.gap:
            return
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            loop = asyncio.get_running_loop()
            if self.last is not None and self.last + self.gap > loop.time():
                await asyncio.sleep(self.last + self.gap - loop.time())
            self.last = loop.time()


class Health:
    """Watches our own posts for takedowns.

    Every report on one of our posts is upheld (the accounts are a team's), so a post that turns hidden or
    disappears probably counts towards the three that suspend an account. Without a confirmation signal from
    the API that is only a suspicion, recorded as such; the policy is deliberately cautious anyway: an account
    with QUIET_AT suspected or confirmed takedowns stops posting and replying and only likes, reposts and
    follows, which can't be reported. A post that comes back is marked restored and stops counting.
    """

    def __init__(self, platform, state, interval: float = 600.0, since: str = "", pace: Optional[float] = None) -> None:
        self.platform = platform
        self.state = state
        self.interval = interval
        self.since = since  # watch our posts from this time on (UTC ISO); "" means all of them
        self.pace = HEALTH_PACE if pace is None else pace
        self.reads = Pacer(self.pace)
        self.checked = {}  # post_id -> when it was last fetched successfully
        self.behind = False  # the last round left due posts for the next one

    def counts(self) -> dict:
        """username -> posts suspected or confirmed taken down."""
        return {u: c["suspected"] + c["confirmed"] for u, c in self.state.moderation_counts().items()}

    def quiet(self) -> set:
        return {u for u, n in self.counts().items() if n >= QUIET_AT}

    async def check(self) -> int:
        """Fetch our posts that are due, newest first and replies past a day last, one every `pace` and at most
        HEALTH_BATCH; returns how many changed status. A failed fetch is retried next round; a rate limit that
        outlasts the platform's own waiting ends the round."""
        now = datetime.now(timezone.utc)
        due = []
        for username, post_id, published, kind in self.state.watched_posts(self.since):
            last = self.checked.get(post_id)
            age = now - datetime.fromisoformat(published)
            schedule = REPLY_RECHECK if kind == "reply" else RECHECK
            if last is None or now - last >= next(e for limit, e in schedule if limit is None or age < limit):
                due.append((kind == "reply" and age >= timedelta(days=1), age, post_id, username, kind))
        due.sort()
        self.behind = len(due) > HEALTH_BATCH
        changed = 0
        for i, (_, _, post_id, username, kind) in enumerate(due[:HEALTH_BATCH]):
            await self.reads.wait()
            try:
                status = outcome(await self.platform.get_post(post_id))
            except KeyError:
                status = "missing"
            except RateLimited:
                log.warning("rate limited checking %s's post #%s; %d due posts wait for the next round", username,
                            post_id, len(due) - i)
                self.behind = False  # wait the full interval out
                break
            except Exception as e:
                log.warning("couldn't check %s's post #%s: %s", username, post_id, str(e) or type(e).__name__)
                continue
            if status != "visible" and self.state.withdrawn_status(post_id) is not None:
                continue  # we deleted it ourselves while this round was under way
            if (status != "visible" and kind == "reply" and self.state.get_meta(REPLIES_KEY) != "survive"
                    and self.state.parent_withdrawn(post_id)):
                continue  # may have gone with the post we deleted: Withdraw finds out which
            now = datetime.now(timezone.utc)
            self.checked[post_id] = now
            previous = self.state.moderation_status(post_id)
            if status == "visible":
                if previous in SUSPECTED and self.state.set_moderation(post_id, username, "restored", _iso(now)):
                    changed += 1
                    log.info("%s: post #%s is visible again", username, post_id)
                continue
            changed += self.record(username, post_id, status)
        return changed

    def record(self, username: str, post_id: int, status: str) -> bool:
        """A takedown seen (by this check or by Withdraw's look); True if it is news."""
        now = datetime.now(timezone.utc)
        self.checked[post_id] = now
        if not self.state.set_moderation(post_id, username, status, _iso(now)):
            return False
        self._report(username, post_id, status)
        return True

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
            wait = self.pace if self.behind else self.interval  # a round that left posts due goes on after a pause
            await asyncio.sleep(wait if until is None else max(0.0, min(wait, until - loop.time())))


class Withdraw:
    """Deletes our anchor posts once their time on Trending is over (WITHDRAW_AFTER), so the reports that come in
    hours later find nothing. Each is looked at first: one the platform no longer shows was taken down, and is
    recorded as Health would. A delete is recorded as being sent before it goes, so Health never takes our own
    delete for a takedown; one whose outcome is unknown is settled by the next look (gone after we sent a delete
    means deleted). Only accounts logged in now delete, and nothing is done while the feed is locked. Until it is
    known whether a reply outlives its deleted post, a teammate's reply seen visible just before a delete is looked
    at again a round later: if it vanished, Health would take such replies for takedowns, so withdrawing stops
    for good (kept in meta across restarts)."""

    def __init__(self, platform, state, health: Health, online, locked=lambda: False,
                 interval: float = WITHDRAW_EVERY, batch: int = WITHDRAW_BATCH, lifts: bool = False) -> None:
        self.platform = platform
        self.state = state
        self.health = health
        self.online = online  # () -> usernames logged in now
        self.locked = locked
        self.interval = interval
        self.batch = batch
        self.lifts = lifts  # this loop deletes lift replies under deleted anchors instead of anchors
        self.refused = set()  # accounts the platform refused a delete as suspended or blocked: left alone this run
        self.skipped = set()  # posts it refused a delete for any other reason: left alone this run
        found = state.get_meta(REPLIES_KEY)  # "survive" | "vanish", once looked at
        self.replies_survive = None if found is None else found == "survive"
        self.stopped = found == "vanish"
        self.probe = None  # (anchor id, username, reply id): a reply seen visible just before its anchor's delete

    async def withdraw(self) -> int:
        """One round: up to `batch` due anchors, newest first; returns how many were deleted. Until it is known
        whether replies outlive their post, an anchor with replies of ours under it is deleted only as the one
        whose reply is looked at again, one at a time."""
        if self.stopped or self.locked():
            return 0
        if self.lifts and (not WITHDRAW_LIFTS or self.state.get_meta(REPLIES_KEY) != "survive"):
            return 0
        if self.probe is not None and self.state.withdrawn_status(self.probe[0]) == "deleted":
            await self._look_under()
            if self.stopped:
                return 0
        now = datetime.now(timezone.utc)
        online = set(self.online())
        due = ([(u, p, False) for u, p in self.state.lifts_due(_iso(now - LIFTS_MAX_AGE))] if self.lifts
               else [(u, p, True) for u, p in self.state.anchors_due(_iso(now - WITHDRAW_AFTER))])
        due = [(u, p, a) for u, p, a in due if u in online and u not in self.refused and p not in self.skipped]
        deleted = 0
        for username, post_id, is_anchor in due[:self.batch]:
            kind = "anchor" if is_anchor else "lift reply"
            if username in self.refused or self.stopped or self.locked():
                continue
            sent = self.state.withdrawn_status(post_id) == "deleting"
            await self.health.reads.wait()
            try:
                status = outcome(await self.platform.get_post(post_id))
            except KeyError:
                status = "missing"
            except RateLimited:
                log.warning("rate limited looking at %s's %s #%s; the rest wait", username, kind, post_id)
                break
            except Exception as e:
                log.warning("couldn't look at %s's %s #%s: %s", username, kind, post_id, str(e) or type(e).__name__)
                continue
            if status != "visible":
                if sent and status == "missing":
                    self.state.set_withdrawn(post_id, username, "deleted", _iso(datetime.now(timezone.utc)))
                    deleted += 1
                    log.info("%s's %s #%s is gone after our delete", username, kind, post_id)
                    continue
                self.state.set_withdrawn(post_id, username, None, "")  # taken down before our delete landed
                self.health.record(username, post_id, status)
                continue
            if is_anchor and self.replies_survive is None and self._untaken(post_id):
                if self.probe is not None:
                    continue  # one such delete at a time until the question is settled
                self.probe = await self._probe_for(post_id)
                if self.probe is None:
                    continue
            self.state.set_withdrawn(post_id, username, "deleting", _iso(datetime.now(timezone.utc)))
            # Ahead of every queued write: at NORMAL a delete waited 15+ minutes in the write lane on 24 October,
            # and anchors were hidden and their accounts suspended 3.3 to 4.3 hours after posting while it waited.
            priority = WRITE_PRIORITY.set(ANCHOR if is_anchor else LIFT)
            try:
                await self.platform.delete_post(username, post_id)
            except RateLimited:
                log.warning("rate limited deleting %s's %s #%s; tried again next round", username, kind, post_id)
                self._drop_probe(post_id)
                break
            except Exception as e:
                response = getattr(e, "response", None)
                code = getattr(response, "status_code", None)
                if code is not None and 400 <= code < 500:  # refused: the post is watched again
                    body = (getattr(response, "text", "") or "").lower()
                    text = body[:200]
                    if code == 401 or any(w in body for w in ("suspend", "block", "banned")):
                        self.refused.add(username)
                    # as a refused write does in bot.step(); a team's reporting stand-down says "suspended" too
                    if code == 403 and "suspend" in body and "reporting" not in body:
                        self.state.set_meta(SUSPENDED + username, _iso(datetime.now(timezone.utc)))
                    else:
                        self.skipped.add(post_id)
                    self.state.set_withdrawn(post_id, username, None, "")
                    log.warning("couldn't delete %s's %s #%s: HTTP %s %s", username, kind, post_id, code, text)
                else:  # lost on the way: the next look settles it
                    log.warning("couldn't delete %s's %s #%s: %s", username, kind, post_id, str(e) or type(e).__name__)
                self._drop_probe(post_id)
                continue
            finally:
                WRITE_PRIORITY.reset(priority)
            self.state.set_withdrawn(post_id, username, "deleted", _iso(datetime.now(timezone.utc)))
            deleted += 1
            log.info("deleted %s's %s #%s, %s", username, kind, post_id,
                     "past its time on Trending" if is_anchor else "its anchor deleted")
        return deleted

    def _drop_probe(self, post_id: int) -> None:
        if self.probe is not None and self.probe[0] == post_id:
            self.probe = None

    def _untaken(self, post_id: int) -> list:
        """Our replies straight to `post_id` with no takedown on record."""
        return [(u, r) for u, r in self.state.our_replies_to(post_id)
                if self.state.moderation_status(r) in (None, "restored")]

    async def _probe_for(self, post_id: int) -> Optional[tuple]:
        """One of our replies to `post_id` that the platform shows now, to look at again after the delete."""
        replies = self._untaken(post_id)
        if not replies:
            return None
        username, reply_id = replies[0]
        await self.health.reads.wait()
        try:
            visible = outcome(await self.platform.get_post(reply_id)) == "visible"
        except Exception:
            return None
        return (post_id, username, reply_id) if visible else None

    async def _look_under(self) -> None:
        """A round after the delete, look at the reply again: gone means replies vanish with their post."""
        post_id, username, reply_id = self.probe
        await self.health.reads.wait()
        try:
            gone = outcome(await self.platform.get_post(reply_id)) != "visible"
        except KeyError:
            gone = True
        except Exception as e:
            log.warning("couldn't look at %s's reply #%s under deleted #%s: %s", username, reply_id, post_id,
                        str(e) or type(e).__name__)
            return
        if gone:
            self.stopped = True
            self.state.set_meta(REPLIES_KEY, "vanish")
            for u, r in self.state.replies_to_withdrawn():  # gone with their posts, not taken down
                if self.state.moderation_status(r) in (None, "restored"):
                    self.state.set_withdrawn(r, u, "deleted", _iso(datetime.now(timezone.utc)))
            log.error("our replies vanish with a deleted post (%s's #%s under #%s): withdrawing stops; if that was "
                      "a takedown instead, clear meta %s to try again", username, reply_id, post_id, REPLIES_KEY)
            return
        self.probe, self.replies_survive = None, True
        self.state.set_meta(REPLIES_KEY, "survive")
        log.info("%s's reply #%s outlives the deleted #%s", username, reply_id, post_id)

    async def run(self, until: Optional[float] = None) -> None:
        loop = asyncio.get_running_loop()
        while until is None or loop.time() < until:
            try:
                await self.withdraw()
            except Exception:
                log.exception("withdrawing anchors failed")
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
    health, seen = Health(p, state, interval=0.01, pace=0), []
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
    health = Health(p, state, pace=0)
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
    health = Health(p, state, pace=0)
    assert await health.check() == 0 and 20 in health.checked
    p.answers[20] = {"hidden": True}
    assert await health.check() == 0, "rechecked within a day"
    health.checked[20] -= timedelta(hours=25)
    assert await health.check() == 1 and state.moderation_status(20) == "hidden"
    assert outcome({"visibility": "Hidden"}) == "hidden" and outcome({"visibility": "public"}) == "visible"

    # R6: newest first, at most HEALTH_BATCH a round, a rate limit ends the round, and a reply past a day waits a day.
    class Counting(Platform):
        def __init__(self) -> None:
            super().__init__()
            self.asked = []

        async def get_post(self, post_id: int) -> dict:
            self.asked.append(post_id)
            return await super().get_post(post_id)

    p, state = Counting(), StateStore(":memory:")
    for pid in range(100, 100 + HEALTH_BATCH + 5):
        publish(state, "many", pid, timedelta(minutes=500 - (pid - 100)))
        p.answers[pid] = {"hidden": False}
    health = Health(p, state, pace=0)
    await health.check()
    assert len(p.asked) == HEALTH_BATCH and p.asked[0] == 100 + HEALTH_BATCH + 4 and health.behind, p.asked[:3]
    await health.check()
    assert len(p.asked) == HEALTH_BATCH + 5 and not health.behind
    p, state = Counting(), StateStore(":memory:")
    for pid in (30, 31, 32):
        publish(state, "limited", pid, timedelta(hours=pid - 29))
    p.answers = {30: {"hidden": False}, 31: RateLimited(60), 32: {"hidden": True}}
    health = Health(p, state, pace=0)
    assert await health.check() == 0 and p.asked == [30, 31] and 32 not in health.checked and not health.behind
    p, state = Platform(), StateStore(":memory:")
    state.record("old", _iso(datetime.now(timezone.utc) - timedelta(days=2)),
                 Action("reply", content="a reply", made_id=40))
    publish(state, "old", 41, timedelta(days=2))
    p.answers = {40: {"hidden": False}, 41: {"hidden": False}}
    health = Health(p, state, pace=0)
    await health.check()
    p.answers = {40: {"hidden": True}, 41: {"hidden": True}}
    for pid in (40, 41):
        health.checked[pid] -= timedelta(hours=7)
    assert await health.check() == 1 and state.moderation_status(41) == "hidden" and state.moderation_status(40) is None
    p, state = Counting(), StateStore(":memory:")
    state.record("old", _iso(datetime.now(timezone.utc) - timedelta(days=2)),
                 Action("reply", content="a reply", made_id=50))
    publish(state, "old", 51, timedelta(days=2, hours=12))
    p.answers = {50: {"hidden": False}, 51: {"hidden": False}}
    await Health(p, state, pace=0).check()
    assert p.asked == [51, 50], "a post goes before a reply past a day, however old"

    # R7: anchors past WITHDRAW_AFTER are deleted if still shown; young ones, other posts and offline accounts' are
    # left; one already taken down is recorded as Health would; a delete whose outcome is lost is settled by the
    # next look; Health never counts our own delete as a takedown, even mid-round.
    class Deleting(Platform):
        def __init__(self) -> None:
            super().__init__()
            self.deleted, self.fail, self.refuse = [], set(), {}

        async def delete_post(self, username: str, post_id: int) -> bool:
            if post_id in self.refuse:
                raise Refused(self.refuse[post_id])
            if post_id in self.fail:
                self.fail.discard(post_id)
                self.answers[post_id] = KeyError(post_id)  # it went through, but the answer was lost
                raise ConnectionError()
            self.deleted.append(post_id)
            self.answers[post_id] = KeyError(post_id)
            return True

    class Refused(Exception):
        def __init__(self, text: str) -> None:
            super().__init__(f"HTTP 403 {text}")
            self.response = type("R", (), {"status_code": 403, "text": text})()

    def anchor(state, username: str, post_id: int, age: timedelta, kind: str = "anchor") -> None:
        ts = _iso(datetime.now(timezone.utc) - age)
        state.record(username, ts, Action("post", content=f"post {post_id}", topic_kind=kind, made_id=post_id))

    p, state = Deleting(), StateStore(":memory:")
    for pid, user, age, kind in ((60, "ann", 4, "anchor"), (61, "ann", 2, "anchor"), (62, "ann", 5, "issue"),
                                 (63, "off", 5, "anchor"), (64, "ann", 6, "anchor"), (65, "ann", 7, "anchor")):
        anchor(state, user, pid, timedelta(hours=age), kind)
        p.answers[pid] = {"hidden": False}
    p.answers[64] = KeyError(64)  # taken down before we got to it
    p.fail = {65}
    health = Health(p, state, pace=0)
    w = Withdraw(p, state, health, online=lambda: {"ann"})
    assert await w.withdraw() == 1 and p.deleted == [60]
    assert state.withdrawn_status(60) == "deleted" and state.withdrawn_status(65) == "deleting"
    assert state.withdrawn_status(64) is None and state.moderation_status(64) == "missing" and health.quiet() == {"ann"}
    assert state.withdrawn_status(61) is None and state.withdrawn_status(63) is None
    await health.check()
    assert state.moderation_status(60) is None and state.moderation_status(65) is None, "our deletes aren't takedowns"
    assert await w.withdraw() == 1 and state.withdrawn_status(65) == "deleted" and p.deleted == [60]
    assert await w.withdraw() == 0

    # A refusal about the post leaves only that post; one about the account (suspended) leaves the account; and
    # nothing at all happens while the feed is locked.
    for pid, age in ((66, 4), (67, 5), (68, 6)):
        anchor(state, "ann", pid, timedelta(hours=age))
        p.answers[pid] = {"hidden": False}
    p.refuse = {66: "post awaiting a moderator"}
    locked = [True]
    w.locked = lambda: locked[0]
    assert await w.withdraw() == 0 and p.deleted == [60]
    locked[0] = False
    assert await w.withdraw() == 2 and w.skipped == {66} and not w.refused and p.deleted == [60, 67, 68]
    assert state.withdrawn_status(66) is None and 66 in [r[1] for r in state.watched_posts("")]
    anchor(state, "bob", 69, timedelta(hours=4))
    anchor(state, "bob", 71, timedelta(hours=5))
    p.answers.update({69: {"hidden": False}, 71: {"hidden": False}})
    p.refuse = {69: "account suspended", 71: "account suspended"}
    w.online = lambda: {"ann", "bob"}
    assert await w.withdraw() == 0 and w.refused == {"bob"} and state.withdrawn_status(71) is None
    assert state.get_meta(SUSPENDED + "bob") and state.get_meta(SUSPENDED + "ann") is None, "a suspension is marked"
    # read in full, however long the body; a reporting stand-down isn't an account's suspension
    for who, pid, why in (("cy", 72, "x" * 300 + " account suspended"), ("di", 73, "reporting is suspended until 10:00")):
        anchor(state, who, pid, timedelta(hours=4))
        p.answers[pid] = {"hidden": False}
        p.refuse[pid] = why
    w.online = lambda: {"cy", "di"}
    await w.withdraw()
    assert state.get_meta(SUSPENDED + "cy") and state.get_meta(SUSPENDED + "di") is None

    # After the first delete, a teammate's reply to it is looked at: if it vanished too, withdrawing stops and
    # the reply isn't counted as a takedown; if it is still there, no more looks.
    class Cascading(Deleting):
        async def delete_post(self, username: str, post_id: int) -> bool:
            await super().delete_post(username, post_id)
            if self.vanish:
                self.answers[81] = KeyError(81)
            return True

    for vanishes in (True, False):
        p, state = Cascading(), StateStore(":memory:")
        p.vanish = vanishes
        anchor(state, "ann", 80, timedelta(hours=4))
        anchor(state, "ann", 82, timedelta(hours=5))
        anchor(state, "ann", 84, timedelta(hours=6))
        for reply, ago in ((83, 3.5), (81, 3)):  # 83 was taken down before: never the one looked at
            state.record("cat", _iso(datetime.now(timezone.utc) - timedelta(hours=ago)),
                         Action("reply", post_id=80, content="lift", made_id=reply))
        state.set_moderation(83, "cat", "hidden", "t")
        p.answers = {80: {"hidden": False}, 81: {"hidden": False}, 82: {"hidden": False}, 84: {"hidden": False}}
        health = Health(p, state, pace=0)
        state.record("dan", _iso(datetime.now(timezone.utc) - timedelta(hours=3)),
                     Action("reply", post_id=82, content="lift", made_id=85))
        p.answers[85] = {"hidden": False}
        w = Withdraw(p, state, health, online=lambda: {"ann"})
        # 84 has no reply of ours and goes; 82 waits while 80's reply is the one looked at.
        assert await w.withdraw() == 2 and w.probe == (80, "cat", 81) and not w.stopped and p.deleted == [80, 84]
        if vanishes:
            p.answers[85] = KeyError(85)  # as if 82 had gone too and taken its reply with it
            state.set_withdrawn(82, "ann", "deleted", "t")
        await health.check()
        assert state.moderation_status(81) is None and state.moderation_status(85) is None, "not yet known"
        assert await w.withdraw() == (0 if vanishes else 1) and w.stopped == vanishes
        assert w.replies_survive is (None if vanishes else True) and (w.probe is None) != vanishes
        assert state.withdrawn_status(83) is None and state.moderation_status(83) == "hidden"
        assert state.withdrawn_status(81) == ("deleted" if vanishes else None)
        assert state.withdrawn_status(85) == ("deleted" if vanishes else None)
        health.checked.clear()
        await health.check()
        assert state.moderation_status(81) is None and state.moderation_status(85) is None
        again = Withdraw(p, state, health, online=lambda: {"ann"})
        assert again.stopped == vanishes and again.replies_survive is (not vanishes), "kept in meta"

    # Once replies are known to outlive their post, a lift reply goes after its anchor, at the write lane's normal
    # priority; not before that is known, nor while its anchor is still up.
    p, state = Deleting(), StateStore(":memory:")
    anchor(state, "ann", 90, timedelta(hours=4))
    anchor(state, "ann", 92, timedelta(hours=1))
    for reply, parent in ((91, 90), (93, 92)):
        state.record("cat", _iso(datetime.now(timezone.utc) - timedelta(minutes=50)),
                     Action("reply", post_id=parent, content="lift", made_id=reply))
    p.answers = {i: {"hidden": False} for i in (90, 91, 92, 93)}
    priorities = []
    deletes = p.delete_post

    async def noting(username: str, post_id: int) -> bool:
        priorities.append((post_id, WRITE_PRIORITY.get()))
        return await deletes(username, post_id)

    p.delete_post = noting
    state.set_meta(REPLIES_KEY, "survive")
    health = Health(p, state, pace=0)
    w = Withdraw(p, state, health, online=lambda: {"ann", "cat"})
    lifts = Withdraw(p, state, health, online=lambda: {"ann", "cat"}, lifts=True)
    assert await lifts.withdraw() == 0, "no lift reply is due while its anchor is up"
    assert await w.withdraw() == 1 and p.deleted == [90], "the anchor loop deletes anchors only"
    assert await lifts.withdraw() == 1 and p.deleted == [90, 91], "then the lift loop its lift reply"
    assert priorities == [(90, ANCHOR), (91, LIFT)] and state.withdrawn_status(93) is None
    state.set_meta(REPLIES_KEY, "")
    anchor(state, "ann", 94, timedelta(hours=5))
    state.record("cat", _iso(datetime.now(timezone.utc) - timedelta(hours=4)),
                 Action("reply", post_id=94, content="lift", made_id=95))
    p.answers.update({94: {"hidden": False}, 95: {"hidden": False}})
    await w.withdraw()
    assert await lifts.withdraw() == 0, "nothing until replies are known to outlive their post"
    state.set_meta(REPLIES_KEY, "survive")
    state.record("cat", _iso(datetime.now(timezone.utc) - LIFTS_MAX_AGE - timedelta(hours=1)),
                 Action("reply", post_id=94, content="old lift", made_id=96))
    p.answers[96] = {"hidden": False}
    assert await lifts.withdraw() == 1 and 95 in p.deleted and 96 not in p.deleted, "past LIFTS_MAX_AGE: left"

    class Racing(Deleting):
        """Our delete lands while Health's round is under way."""

        async def get_post(self, post_id: int) -> dict:
            state.set_withdrawn(post_id, "ann", "deleted", "t")
            raise KeyError(post_id)

    p, state = Racing(), StateStore(":memory:")
    anchor(state, "ann", 70, timedelta(hours=1))
    health = Health(p, state, pace=0)
    assert await health.check() == 0 and state.moderation_status(70) is None and 70 not in health.checked
    pacer, began = Pacer(0.05), loop.time()
    await asyncio.gather(pacer.wait(), pacer.wait(), pacer.wait())
    assert loop.time() - began >= 0.1, "one read every gap, however many wait"
    print("health ok")


if __name__ == "__main__":
    asyncio.run(_demo())
