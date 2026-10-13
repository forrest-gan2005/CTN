import asyncio
import json
import logging
import random
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional
from zoneinfo import ZoneInfo

log = logging.getLogger("collector")

COMPETITION_START = datetime(2026, 10, 19, 17, 0, tzinfo=ZoneInfo("Australia/Sydney"))
DEFAULT_DB = Path(__file__).parent / "data" / "npc.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS posts (
    id INTEGER PRIMARY KEY,
    author TEXT NOT NULL,
    content TEXT,
    parent_id INTEGER,
    created_at TEXT,          -- UTC ISO; NULL when the platform's value can't be parsed
    fetched_at TEXT NOT NULL, -- UTC ISO, when we first saw it
    ours INTEGER NOT NULL DEFAULT 0,
    raw TEXT NOT NULL         -- the post exactly as the API returned it
);
CREATE INDEX IF NOT EXISTS posts_author ON posts(author);
CREATE TABLE IF NOT EXISTS gaps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    low INTEGER,              -- newest stored id below the gap; NULL = back to the start of history
    high INTEGER NOT NULL     -- still need every post with id < high (and > low)
);
CREATE VIEW IF NOT EXISTS accounts AS
    SELECT author,
           COUNT(*) AS posts,
           MIN(created_at) AS first_post_at,
           MIN(fetched_at) AS first_fetched_at,
           MAX(ours) AS ours
    FROM posts GROUP BY author;
"""


def _iso(dt: datetime) -> str:
    # Fixed format so ISO strings sort chronologically in SQL.
    return dt.astimezone(timezone.utc).isoformat(timespec="microseconds")


def _parse_time(value) -> Optional[str]:
    """Normalise a platform timestamp to UTC ISO, or None if it's missing, naive or unparseable."""
    try:
        if isinstance(value, (int, float)):
            return _iso(datetime.fromtimestamp(value, timezone.utc))
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError, OverflowError):
        return None
    return _iso(dt) if dt.tzinfo else None


def _json_default(obj):
    return sorted(obj) if isinstance(obj, (set, frozenset)) else str(obj)


class NpcStore:
    def __init__(self, path=DEFAULT_DB, own_usernames: Iterable[str] = ()) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.executescript(SCHEMA)
        self.own = set(own_usernames)

    def add_posts(self, posts: list) -> int:
        """Store posts we haven't seen before; returns how many were new."""
        fetched_at = _iso(datetime.now(timezone.utc))
        rows = [(
            p["id"],
            p["author"],
            p.get("content"),
            p.get("parent_id"),
            _parse_time(p.get("created_at")),
            fetched_at,
            int(p["author"] in self.own),
            json.dumps(p, default=_json_default, ensure_ascii=False),
        ) for p in posts]
        before = self.db.total_changes
        with self.db:
            self.db.executemany("INSERT OR IGNORE INTO posts VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
        return self.db.total_changes - before

    def count(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM posts").fetchone()[0]

    def max_id(self) -> Optional[int]:
        return self.db.execute("SELECT MAX(id) FROM posts").fetchone()[0]

    def add_gap(self, low: Optional[int], high: int) -> None:
        with self.db:
            self.db.execute("INSERT INTO gaps (low, high) VALUES (?, ?)", (low, high))

    def gaps(self) -> list:
        return self.db.execute("SELECT id, low, high FROM gaps ORDER BY high DESC").fetchall()

    def update_gap(self, gap_id: int, high: int) -> None:
        with self.db:
            self.db.execute("UPDATE gaps SET high = ? WHERE id = ?", (high, gap_id))

    def delete_gap(self, gap_id: int) -> None:
        with self.db:
            self.db.execute("DELETE FROM gaps WHERE id = ?", (gap_id,))

    def whitelist(self, cutoff: datetime = COMPETITION_START) -> list:
        """Accounts that posted, or that we saw, before `cutoff`: competitors weren't on the platform yet."""
        c = _iso(cutoff)
        rows = self.db.execute(
            "SELECT author FROM accounts WHERE ours = 0 AND (first_post_at < ? OR first_fetched_at < ?) "
            "ORDER BY author",
            (c, c),
        )
        return [r[0] for r in rows]

    def stats(self) -> dict:
        q = self.db.execute
        return {
            "posts": self.count(),
            "accounts": q("SELECT COUNT(*) FROM accounts").fetchone()[0],
            "whitelist": len(self.whitelist()),
            "pending_gaps": len(self.gaps()),
            "unparsed_created_at": q("SELECT COUNT(*) FROM posts WHERE created_at IS NULL").fetchone()[0],
            "newest_fetch": q("SELECT MAX(fetched_at) FROM posts").fetchone()[0],
            "top_authors": q("SELECT author, posts FROM accounts ORDER BY posts DESC LIMIT 5").fetchall(),
            "latest": q("SELECT author, content FROM posts ORDER BY id DESC LIMIT 3").fetchall(),
        }


class Collector:
    def __init__(
        self,
        platform,
        store: NpcStore,
        page_size: int = 50,
        min_interval: float = 5.0,
        max_interval: float = 120.0,
    ) -> None:
        self.platform = platform
        self.store = store
        self.page_size = page_size
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.gaps = 0

    async def poll(self) -> tuple:
        """Fetch the newest page; if it doesn't reach back to what we already had, record the gap."""
        prev_max = self.store.max_id()
        page = await self.platform.feed_latest(n=self.page_size)
        new = self.store.add_posts(page)
        if page:
            lowest = min(p["id"] for p in page)
            if prev_max is None or lowest > prev_max:
                self.store.add_gap(prev_max, lowest)
                if prev_max is not None:
                    self.gaps += 1
        return new, len(page)

    async def fill_gaps(self) -> int:
        """Page backwards through every recorded gap, saving progress so a crash can resume mid-gap."""
        total = 0
        for gap_id, low, high in self.store.gaps():
            while True:
                page = await self.platform.feed_latest(n=self.page_size, before=high)
                total += self.store.add_posts(page)
                if not page or (low is not None and min(p["id"] for p in page) <= low):
                    self.store.delete_gap(gap_id)
                    break
                high = min(p["id"] for p in page)
                self.store.update_gap(gap_id, high)
                await asyncio.sleep(self.min_interval)
        return total

    async def run(self, until: Optional[float] = None) -> None:
        """Poll Latest and fill gaps forever (or until loop time `until`), adapting the poll rate."""
        loop = asyncio.get_running_loop()
        interval = self.min_interval * 4
        failures = 0
        while until is None or loop.time() < until:
            try:
                new, size = await self.poll()
                filled = await self.fill_gaps()
                if size and new == size:
                    interval = max(self.min_interval, interval / 2)
                elif new <= size // 4:
                    interval = min(self.max_interval, interval * 1.25)
                if new or filled:
                    log.info("+%d new, +%d from gaps (total %d), next poll in ~%.1fs",
                             new, filled, self.store.count(), interval)
                failures = 0
            except Exception:
                failures += 1
                wait = min(300.0, self.min_interval * 2 ** failures)
                log.exception("poll failed (%d in a row), retrying in %.0fs", failures, wait)
                await asyncio.sleep(wait)
                continue
            await asyncio.sleep(interval * random.uniform(0.8, 1.2))


async def _demo(seconds: float) -> None:
    from content import load_personas
    from fake_platform import FakePlatform

    p = FakePlatform()
    store = NpcStore(":memory:", own_usernames=[x["username"] for x in load_personas()])
    early = [f"npc_{i}" for i in range(40)]
    late = [f"late_npc_{i}" for i in range(10)]
    for i in range(300):
        await p.post(random.choice(early), f"pre-start post {i}")
    cutoff = datetime.now(timezone.utc)
    pre_start_authors = {post["author"] for post in p.posts.values()}

    async def live_traffic() -> None:
        while True:
            burst = 120 if random.random() < 0.05 else random.randint(0, 3)
            for _ in range(burst):
                await p.post(random.choice(early + late + ["joe_tama"]), "live post")
            await asyncio.sleep(random.uniform(0.05, 0.3))

    collector = Collector(p, store, page_size=50, min_interval=0.2, max_interval=2.0)
    traffic = asyncio.create_task(live_traffic())
    loop = asyncio.get_running_loop()
    await collector.run(until=loop.time() + seconds)
    traffic.cancel()
    await collector.poll()
    await collector.fill_gaps()

    wl = set(store.whitelist(cutoff))
    assert store.count() == len(p.posts), (store.count(), len(p.posts))
    assert wl == pre_start_authors, wl ^ pre_start_authors
    assert "joe_tama" not in wl and not any(a.startswith("late_") for a in wl)
    print(f"\ncollected {store.count()}/{len(p.posts)} posts, filled {collector.gaps} gaps, "
          f"whitelist {len(wl)} accounts")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    demo = sub.add_parser("demo", help="run against FakePlatform with simulated NPC traffic")
    demo.add_argument("--seconds", type=float, default=15)
    stats = sub.add_parser("stats", help="summarise a collected database")
    stats.add_argument("--db", default=str(DEFAULT_DB))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    if args.cmd == "demo":
        asyncio.run(_demo(args.seconds))
    else:
        for key, value in NpcStore(args.db).stats().items():
            print(f"{key}: {value}")
