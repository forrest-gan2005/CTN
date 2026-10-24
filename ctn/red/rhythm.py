import math
import random
from datetime import datetime, timedelta, timezone
from typing import Optional, Sequence
from zoneinfo import ZoneInfo

TIMEZONE = ZoneInfo("Australia/Sydney")

# Relative posting activity per Sydney hour (0-23), used until the collector has enough citizen posts.
DEFAULT_CURVE = (1.0, 0.5, 0.3, 0.2, 0.2, 0.4, 1.2, 2.5, 3.5, 3.5, 3.5, 4.0,
                 5.0, 4.5, 4.0, 4.0, 4.5, 5.0, 5.5, 6.5, 7.0, 6.5, 5.0, 2.5)
MIN_CURVE_POSTS = 500
QUIET_FLOOR = 0.1  # the quietest hour still gets this share of an average hour's activity
SESSION_GAP_MEDIAN = 45.0  # seconds between actions within one sitting
# Sittings a day, drawn once per account. Raised on the opening night, when the feed went from about two posts a
# minute to ten as other teams came online, and citizens post round the clock; doubled three hours in, when the
# whole team wrote 29 posts and replies an hour against the platform's 180 and the leaders were far ahead; doubled
# again when the team took to lifting anchor posts, which takes forty likes and twenty-five replies within minutes.
SESSIONS_PER_DAY = (48.0, 72.0)
FIRST_SITTING = (60.0, 600.0)  # seconds after coming online, or a restart, before a first look round
# Low-key until LOW_KEY_UNTIL: every account sits down LOW_KEY_RATE as often till then. On from 02:35 to 11:30 on 24
# October (Forrest): it cost about half the hourly score and suspensions went on regardless (22 that day, after one
# from 09:38 to 23:42 the day before at full pace), so it ended there; the machinery stays for later.
LOW_KEY_UNTIL = datetime(2026, 10, 24, 11, 30, tzinfo=TIMEZONE)
LOW_KEY_RATE = 0.7
# Nights stay near silent (Forrest, 24 October): from midnight to 09:00 that day 22 accounts were suspended, while the
# day before at full pace lost one; citizens hardly post then, and a team busy at 3 a.m. stands out. Between
# NIGHT_HOURS (Sydney) every account sits down NIGHT_RATE as often as the curve alone says; bot.py posts no anchors
# then, so no rush wakes anyone either.
NIGHT_HOURS = (23, 7)
NIGHT_RATE = 0.3


def at_night(t: datetime, hours: tuple = NIGHT_HOURS) -> bool:
    start, end = hours
    h = t.astimezone(TIMEZONE).hour
    return h >= start or h < end if start > end else start <= h < end


def normalise(weights: Sequence[float]) -> list:
    total = float(sum(weights))
    return [w / total for w in weights]


def npc_curve(post_times, min_posts: int = MIN_CURVE_POSTS) -> list:
    """Hourly share of citizen posting in Sydney time, from UTC ISO timestamps: posts in each hour of the day over
    how many times that hour was watched, so a short or uneven record isn't read as quiet hours. An hour never
    watched yet (the platform was reset at 23:08 on 18 October, so the first evening had no posts to count) gets
    the average. The default curve until there are enough posts."""
    stamps = sorted(datetime.fromisoformat(ts).astimezone(TIMEZONE) for ts in post_times if ts)
    if len(stamps) < min_posts:
        return normalise(DEFAULT_CURVE)
    counts, watched = [0] * 24, [0] * 24
    for stamp in stamps:
        counts[stamp.hour] += 1
    hour = stamps[0].replace(minute=0, second=0, microsecond=0)
    while hour <= stamps[-1]:
        watched[hour.hour] += 1
        hour += timedelta(hours=1)
    seen = [counts[h] / watched[h] for h in range(24) if watched[h]]
    mean = sum(seen) / len(seen)
    rate = [counts[h] / watched[h] if watched[h] else mean for h in range(24)]
    # Neighbouring hours smooth out noise; the floor keeps quiet hours possible, just rare.
    return normalise([rate[h - 1] + 2 * rate[h] + rate[(h + 1) % 24] + QUIET_FLOOR * mean for h in range(24)])


class UniformPacer:
    """Fixed random gaps, for quick offline runs."""

    def __init__(self, min_gap: float = 2.0, max_gap: float = 8.0) -> None:
        self.min_gap = min_gap
        self.max_gap = max_gap

    def next_wait(self, now: datetime) -> float:
        return random.uniform(self.min_gap, self.max_gap)


class Rhythm:
    """One account's day: sittings of a few actions, starting more often when citizens are online.

    Sittings start as a Poisson process whose hourly rate follows the citizen activity curve,
    shifted by a personal offset, so every account keeps its own stable habits across restarts.
    """

    def __init__(self, username: str, curve: Sequence[float], sessions_per_day: Optional[float] = None) -> None:
        personal = random.Random(username)
        self.shift = personal.uniform(-1.5, 1.5)  # hours; early birds and night owls
        self.sessions_per_day = sessions_per_day or personal.uniform(*SESSIONS_PER_DAY)
        self.actions_per_session = personal.uniform(2, 4)
        self.curve = normalise(curve)
        self.left = 0  # actions still to take in the current sitting
        self.started = False

    def rate(self, t: datetime) -> float:
        """Expected sittings per hour at time t."""
        local = t.astimezone(TIMEZONE) - timedelta(hours=self.shift)
        h = local.hour + local.minute / 60 - 0.5  # each hour's weight sits at the middle of that hour
        i = math.floor(h)
        frac = h - i
        weight = self.curve[i % 24] * (1 - frac) + self.curve[(i + 1) % 24] * frac
        return (self.sessions_per_day * weight * (LOW_KEY_RATE if t < LOW_KEY_UNTIL else 1.0)
                * (NIGHT_RATE if at_night(t - timedelta(hours=self.shift)) else 1.0))  # nights by its own habits

    def next_wait(self, now: datetime) -> float:
        """Seconds until this account's next action."""
        if not self.started:  # just online, newly signed up or after a restart: a look round soon, as people do
            self.started = True
            self.left = self._sitting() - 1
            if at_night(now - timedelta(hours=self.shift)):  # but not the whole team at 3 a.m. after a restart
                return self._until_next_session(now)
            return random.uniform(*FIRST_SITTING)
        if self.left > 0:
            self.left -= 1
            return min(300.0, random.lognormvariate(math.log(SESSION_GAP_MEDIAN), 0.6))
        self.left = self._sitting() - 1
        return self._until_next_session(now)

    def _sitting(self) -> int:
        """How many actions the next sitting takes: geometric, averaging actions_per_session."""
        n = 1
        while random.random() > 1 / self.actions_per_session:
            n += 1
        return n

    def _until_next_session(self, now: datetime) -> float:
        # Thinning: draw candidates at the peak rate, keep each with probability rate(t) / peak.
        peak = self.sessions_per_day * max(self.curve)
        t = 0.0
        while True:
            t += random.expovariate(peak) * 3600
            if random.random() * peak < self.rate(now + timedelta(seconds=t)):
                return t


if __name__ == "__main__":
    random.seed(1)
    curve = normalise(DEFAULT_CURVE)
    assert npc_curve([]) == curve
    stamps = [f"2026-10-1{d}T09:{m:02d}:00+00:00" for d in range(5) for m in range(0, 60, 2)] * 4
    sydney_peak = npc_curve(stamps, min_posts=100)
    assert max(range(24), key=sydney_peak.__getitem__) == 20  # 09:00 UTC is 20:00 in Sydney (AEDT)
    # Steady posting watched for 19 hours, as on the day after the reset: the five hours never watched are no
    # quieter than the rest, and the curve comes out flat.
    first_day = [f"2026-10-18T{h:02d}:{m:02d}:00+00:00" for h in range(12, 24) for m in range(0, 60, 2)] + \
                [f"2026-10-19T{h:02d}:{m:02d}:00+00:00" for h in range(0, 7) for m in range(0, 60, 2)]
    flat = npc_curve(first_day)
    assert len(first_day) >= MIN_CURVE_POSTS and max(flat) / min(flat) < 1.05, flat
    # Twice as many posts in the hour watched only once as in hours watched twice: per watch, the same.
    uneven = [f"2026-10-18T{h:02d}:00:00+00:00" for h in range(0, 24) for _ in range(20)] + \
             [f"2026-10-19T{h:02d}:00:00+00:00" for h in range(0, 12) for _ in range(20)]
    assert max(npc_curve(uneven)) / min(npc_curve(uneven)) < 1.05

    r = Rhythm("joe_tama", curve)
    assert Rhythm("joe_tama", curve).shift == r.shift
    before, after = LOW_KEY_UNTIL - timedelta(days=1), LOW_KEY_UNTIL + timedelta(days=1)  # same hour of day
    assert abs(r.rate(before) - LOW_KEY_RATE * r.rate(after)) < 1e-9, "low-key: fewer sittings till the x4 days"
    noon = datetime(2026, 10, 27, 12, tzinfo=TIMEZONE)
    assert at_night(noon.replace(hour=23)) and at_night(noon.replace(hour=3)) and not at_night(noon.replace(hour=7))
    assert not at_night(noon) and at_night(noon, (12, 13)) and not at_night(noon.replace(hour=13), (12, 13))
    night, same = noon.replace(hour=2), Rhythm("joe_tama", [1.0] * 24)
    assert abs(same.rate(night) - NIGHT_RATE * same.rate(noon)) < 1e-9, "nights: NIGHT_RATE as many sittings"
    waits = [Rhythm(f"n{i}", curve).next_wait(noon.replace(hour=3)) for i in range(200)]
    assert sum(w <= FIRST_SITTING[1] for w in waits) < 40, "a restart at night doesn't wake the team"
    assert SESSIONS_PER_DAY[0] <= r.sessions_per_day <= SESSIONS_PER_DAY[1]
    fresh = Rhythm("joe_tama", curve)  # comes online: the first action within minutes, not hours
    first = fresh.next_wait(datetime(2026, 10, 19, 7, tzinfo=timezone.utc))
    assert FIRST_SITTING[0] <= first <= FIRST_SITTING[1] and fresh.started
    now = datetime(2026, 10, 19, 6, tzinfo=timezone.utc)
    end = now + timedelta(days=400)
    hours = [0] * 24
    actions = 0
    while now < end:
        now += timedelta(seconds=r.next_wait(now))
        hours[(now.astimezone(TIMEZONE) - timedelta(hours=r.shift)).hour] += 1
        actions += 1
    per_day = actions / 400
    expected = r.sessions_per_day * r.actions_per_session
    share = normalise(hours)
    tvd = sum(abs(a - b) for a, b in zip(share, curve)) / 2
    print(f"{per_day:.1f} actions/day (expected ~{expected:.1f}), shift {r.shift:+.1f}h, "
          f"distance from curve {tvd:.3f}")
    assert abs(per_day - expected) / expected < 0.15
    assert tvd < 0.07  # sampling noise is ~0.04; a flat schedule scores ~0.27
    print("rhythm ok")
