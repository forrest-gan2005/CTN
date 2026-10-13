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
SESSION_GAP_MEDIAN = 45.0  # seconds between actions within one sitting


def normalise(weights: Sequence[float]) -> list:
    total = float(sum(weights))
    return [w / total for w in weights]


def npc_curve(post_times, min_posts: int = MIN_CURVE_POSTS) -> list:
    """Hourly share of citizen posting in Sydney time, from UTC ISO timestamps."""
    counts = [0] * 24
    for ts in post_times:
        if ts:
            counts[datetime.fromisoformat(ts).astimezone(TIMEZONE).hour] += 1
    if sum(counts) < min_posts:
        return normalise(DEFAULT_CURVE)
    # Neighbouring hours smooth out noise; +1 keeps quiet hours possible, just rare.
    return normalise([counts[h - 1] + 2 * counts[h] + counts[(h + 1) % 24] + 1 for h in range(24)])


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
        self.sessions_per_day = sessions_per_day or personal.uniform(4, 9)
        self.actions_per_session = personal.uniform(2, 4)
        self.curve = normalise(curve)
        self.left = 0  # actions still to take in the current sitting

    def rate(self, t: datetime) -> float:
        """Expected sittings per hour at time t."""
        local = t.astimezone(TIMEZONE) - timedelta(hours=self.shift)
        h = local.hour + local.minute / 60 - 0.5  # each hour's weight sits at the middle of that hour
        i = math.floor(h)
        frac = h - i
        weight = self.curve[i % 24] * (1 - frac) + self.curve[(i + 1) % 24] * frac
        return self.sessions_per_day * weight

    def next_wait(self, now: datetime) -> float:
        """Seconds until this account's next action."""
        if self.left > 0:
            self.left -= 1
            return min(300.0, random.lognormvariate(math.log(SESSION_GAP_MEDIAN), 0.6))
        n = 1
        while random.random() > 1 / self.actions_per_session:
            n += 1
        self.left = n - 1
        return self._until_next_session(now)

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

    r = Rhythm("joe_tama", curve)
    assert Rhythm("joe_tama", curve).shift == r.shift
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
