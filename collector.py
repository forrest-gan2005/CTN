import asyncio
import json
import logging
import random
import re
import sqlite3
from collections import Counter
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
CREATE INDEX IF NOT EXISTS posts_author ON posts(author, id);
CREATE INDEX IF NOT EXISTS posts_created ON posts(created_at);
CREATE TABLE IF NOT EXISTS gaps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    low INTEGER,              -- newest stored id below the gap; NULL = back to the start of history
    high INTEGER NOT NULL     -- still need every post with id < high (and > low)
);
-- One row per account, kept up to date on insert: the competition will produce millions of
-- posts, and grouping them on every whitelist lookup would take seconds.
CREATE TABLE IF NOT EXISTS authors (
    author TEXT PRIMARY KEY,
    posts INTEGER NOT NULL,
    first_post_at TEXT,
    first_fetched_at TEXT NOT NULL,
    ours INTEGER NOT NULL
);
"""
PRE_START = "a.ours = 0 AND (IFNULL(a.first_post_at < ?, 0) OR a.first_fetched_at < ?)"
HASHTAG_RE = re.compile(r"#\w{2,30}")


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
        ids = [r[0] for r in rows]
        marks = ",".join("?" * len(ids))
        known = {r[0] for r in self.db.execute(f"SELECT id FROM posts WHERE id IN ({marks})", ids)} if ids else set()
        new = [r for r in rows if r[0] not in known]
        with self.db:
            self.db.executemany("INSERT OR IGNORE INTO posts VALUES (?, ?, ?, ?, ?, ?, ?, ?)", new)
            self.db.executemany(
                "INSERT INTO authors VALUES (?, 1, ?, ?, ?) ON CONFLICT(author) DO UPDATE SET "
                "posts = posts + 1, "
                "first_post_at = CASE WHEN excluded.first_post_at IS NULL THEN first_post_at "
                "WHEN first_post_at IS NULL OR excluded.first_post_at < first_post_at THEN excluded.first_post_at "
                "ELSE first_post_at END",
                [(r[1], r[4], r[5], r[6]) for r in new],
            )
        return len(new)

    def count(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM posts").fetchone()[0]

    def max_id(self) -> Optional[int]:
        return self.db.execute("SELECT MAX(id) FROM posts").fetchone()[0]

    def add_gap(self, low: Optional[int], high: int) -> None:
        with self.db:
            self.db.execute("INSERT INTO gaps (low, high) VALUES (?, ?)", (low, high))

    def gaps(self) -> list:
        return self.db.execute("SELECT id, low, high FROM gaps ORDER BY high DESC").fetchall()

    def gaps_above(self, post_id: int) -> int:
        """Recorded gaps that still miss posts newer than `post_id`."""
        return self.db.execute("SELECT COUNT(*) FROM gaps WHERE high - 1 > ?", (post_id,)).fetchone()[0]

    def update_gap(self, gap_id: int, high: int) -> None:
        with self.db:
            self.db.execute("UPDATE gaps SET high = ? WHERE id = ?", (high, gap_id))

    def delete_gap(self, gap_id: int) -> None:
        with self.db:
            self.db.execute("DELETE FROM gaps WHERE id = ?", (gap_id,))

    def whitelist(self, cutoff: datetime = COMPETITION_START) -> list:
        """Accounts that posted, or that we saw, before `cutoff`: competitors weren't on the platform yet."""
        c = _iso(cutoff)
        rows = self.db.execute(f"SELECT author FROM authors a WHERE {PRE_START} ORDER BY author", (c, c))
        return [r[0] for r in rows]

    def made_by(self, authors: Iterable[str]) -> list:
        """(id, author, content, parent_id, created_at) of every stored post by these accounts, oldest first."""
        authors = list(authors)
        if not authors:
            return []
        marks = ",".join("?" * len(authors))
        return self.db.execute(f"SELECT id, author, content, parent_id, created_at FROM posts WHERE author IN ({marks}) "
                               "ORDER BY id", authors).fetchall()

    def find_post(self, author: str, content: str, since: datetime, parent_id: Optional[int]) -> Optional[int]:
        row = self.db.execute("SELECT id FROM posts WHERE author = ? AND content = ? AND created_at >= ? "
                              "AND parent_id IS ? ORDER BY id LIMIT 1", (author, content, _iso(since), parent_id))
        found = row.fetchone()
        return found[0] if found else None

    def is_citizen(self, author: str, cutoff: datetime = COMPETITION_START) -> bool:
        c = _iso(cutoff)
        return self.db.execute(f"SELECT 1 FROM authors a WHERE a.author = ? AND {PRE_START}", (author, c, c)).fetchone() is not None

    def npc_posts(self, cutoff: datetime = COMPETITION_START, limit: Optional[int] = None) -> list:
        """(author, content, created_at, parent_id) by whitelisted citizens, oldest first; `limit` keeps the newest."""
        c = _iso(cutoff)
        rows = self.db.execute(
            "SELECT p.author, p.content, p.created_at, p.parent_id FROM posts p "
            f"JOIN authors a ON a.author = p.author WHERE {PRE_START} AND p.content IS NOT NULL "
            "ORDER BY p.id DESC LIMIT ?",
            (c, c, -1 if limit is None else limit),
        ).fetchall()
        return rows[::-1]

    def unknown_posts(self, cutoff: datetime = COMPETITION_START) -> list:
        """(author, content) for posts by accounts that are neither ours nor whitelisted."""
        c = _iso(cutoff)
        return self.db.execute(
            "SELECT p.author, p.content FROM posts p JOIN authors a ON a.author = p.author "
            f"WHERE a.ours = 0 AND NOT ({PRE_START}) AND p.content IS NOT NULL ORDER BY p.id",
            (c, c),
        ).fetchall()

    def recent(self, n: int = 30) -> list:
        """The newest posts as the API returned them, so bots can read the feed without extra API calls."""
        rows = self.db.execute("SELECT raw FROM posts ORDER BY id DESC LIMIT ?", (n,))
        return [json.loads(r[0]) for r in rows]

    def recent_citizen_posts(self, n: int = 30, cutoff: datetime = COMPETITION_START) -> list:
        """The newest posts by whitelisted citizens: bots will far outnumber them in Latest."""
        c = _iso(cutoff)
        rows = self.db.execute(
            f"SELECT p.raw FROM posts p JOIN authors a ON a.author = p.author WHERE {PRE_START} "
            "ORDER BY p.id DESC LIMIT ?",
            (c, c, n),
        )
        return [json.loads(r[0]) for r in rows]

    def recent_by(self, author: str, n: int = 3) -> list:
        """This account's newest post texts, to tailor a reply to the person."""
        rows = self.db.execute(
            "SELECT content FROM posts WHERE author = ? AND content IS NOT NULL ORDER BY id DESC LIMIT ?",
            (author, n),
        )
        return [r[0] for r in rows]

    def top_hashtags(self, since: datetime, n: int = 5, cutoff: datetime = COMPETITION_START) -> list:
        """Hashtags citizens used most since `since` (at least twice), most used first."""
        c = _iso(cutoff)
        rows = self.db.execute(
            f"SELECT p.content FROM posts p JOIN authors a ON a.author = p.author "
            f"WHERE p.created_at >= ? AND {PRE_START} AND p.content LIKE '%#%'",
            (_iso(since), c, c),
        )
        counts = Counter(tag.lower() for (content,) in rows for tag in set(HASHTAG_RE.findall(content)))
        return [tag for tag, k in counts.most_common(n) if k >= 2]

    def citizen_peak_hourly(self, since: datetime, cutoff: datetime = COMPETITION_START) -> int:
        """The most posts any whitelisted citizen made within one clock hour since `since`."""
        c = _iso(cutoff)
        row = self.db.execute(
            f"SELECT COUNT(*) AS k FROM posts p JOIN authors a ON a.author = p.author "
            f"WHERE p.created_at >= ? AND {PRE_START} GROUP BY p.author, substr(p.created_at, 1, 13) "
            "ORDER BY k DESC LIMIT 1",
            (_iso(since), c, c),
        ).fetchone()
        return row[0] if row else 0

    def posts_after(self, last_id: int) -> list:
        """(id, author, content, created_at) for posts newer than `last_id`, oldest first."""
        return self.db.execute(
            "SELECT id, author, content, created_at FROM posts WHERE id > ? ORDER BY id", (last_id,)).fetchall()

    def posts_since(self, since: datetime) -> list:
        """(id, author, content, created_at) for posts created at or after `since`, oldest first."""
        return self.db.execute(
            "SELECT id, author, content, created_at FROM posts WHERE created_at >= ? AND content IS NOT NULL "
            "ORDER BY created_at", (_iso(since),)).fetchall()

    def posts_by(self, authors: Iterable[str]) -> list:
        """(id, author, content) for every stored post by these accounts, newest first."""
        authors = list(authors)
        if not authors:
            return []
        marks = ",".join("?" * len(authors))
        return self.db.execute(
            f"SELECT id, author, content FROM posts WHERE author IN ({marks}) ORDER BY id DESC", authors).fetchall()

    def stats(self) -> dict:
        q = self.db.execute
        return {
            "posts": self.count(),
            "accounts": q("SELECT COUNT(*) FROM authors").fetchone()[0],
            "whitelist": len(self.whitelist()),
            "pending_gaps": len(self.gaps()),
            "unparsed_created_at": q("SELECT COUNT(*) FROM posts WHERE created_at IS NULL").fetchone()[0],
            "newest_fetch": q("SELECT MAX(fetched_at) FROM posts").fetchone()[0],
            "top_authors": q("SELECT author, posts FROM authors ORDER BY posts DESC LIMIT 5").fetchall(),
            "latest": q("SELECT author, content FROM posts ORDER BY id DESC LIMIT 3").fetchall(),
        }


BACKFILL_PAGES = 3  # history pages per round, after each fresh look at Latest


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

    async def fill_gaps(self, max_pages: Optional[int] = None, until: Optional[float] = None) -> int:
        """Page backwards through recorded gaps, newest gap first, saving progress after every page so a restart
        resumes where it stopped. At most `max_pages` pages, and none started at or after loop time `until`."""
        loop = asyncio.get_running_loop()
        total, pages = 0, 0
        for gap_id, low, high in sorted(self.store.gaps(), key=lambda g: g[2], reverse=True):
            while max_pages is None or pages < max_pages:
                if until is not None and loop.time() >= until:
                    return total
                page = await self.platform.feed_latest(n=self.page_size, before=high)
                pages += 1
                total += self.store.add_posts(page)
                if not page or (low is not None and min(p["id"] for p in page) <= low):
                    self.store.delete_gap(gap_id)
                    break
                high = min(p["id"] for p in page)
                self.store.update_gap(gap_id, high)
                await asyncio.sleep(self.min_interval)
            else:
                return total
        return total

    async def run(self, until: Optional[float] = None) -> None:
        """Poll Latest forever (or until loop time `until`), adapting the poll rate, and fill gaps a few pages
        per round, so a long history never keeps Latest waiting."""
        loop = asyncio.get_running_loop()
        interval = self.min_interval * 4
        failures = 0
        left = (lambda: float("inf")) if until is None else (lambda: until - loop.time())
        while left() > 0:
            try:
                new, size = await self.poll()
                filled = await self.fill_gaps(BACKFILL_PAGES, until)
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
                await asyncio.sleep(max(0.0, min(wait, left())))
                continue
            pause = self.min_interval if self.store.gaps() else interval * random.uniform(0.8, 1.2)
            await asyncio.sleep(max(0.0, min(pause, left())))


async def _backlog_demo() -> None:
    """A long history doesn't keep Latest waiting, survives a restart, and nothing is fetched after the deadline."""
    from fake_platform import FakePlatform

    class Watched(FakePlatform):
        calls = []  # (loop time, before) for every page fetched

        async def feed_latest(self, n: int = 10, before: Optional[int] = None) -> list:
            self.calls.append((asyncio.get_running_loop().time(), before))
            return await super().feed_latest(n, before)

    p = Watched()
    for i in range(3000):
        await p.post(f"npc_{i % 50}", f"old post {i}")
    store = NpcStore(":memory:")
    await Collector(p, store, page_size=10).poll()  # a first look leaves a gap back to the start of history

    async def arrivals() -> None:
        while True:
            await p.post("npc_new", "fresh")
            await asyncio.sleep(0.02)

    loop = asyncio.get_running_loop()
    traffic = asyncio.create_task(arrivals())
    p.calls.clear()
    until = loop.time() + 0.5
    await Collector(p, store, page_size=10, min_interval=0.02, max_interval=0.1).run(until)
    traffic.cancel()
    latest = [t for t, before in p.calls if before is None]
    assert all(t < until for t, _ in p.calls), "fetched after the deadline"
    assert len(latest) >= 5 and max(b - a for a, b in zip(latest, latest[1:])) < 0.2, "Latest went stale"
    assert store.gaps() and 0 < store.count() < 3000, "the history is long; it is filled bit by bit"
    resume_from = store.gaps()[0][2]
    p.calls.clear()
    await Collector(p, store, page_size=10, min_interval=0).fill_gaps(max_pages=1)  # a restart picks up the cursor
    assert p.calls[0][1] == resume_from
    t0 = loop.time()
    await Collector(p, store, min_interval=0.01).run(until=t0 - 1)  # an expired run fetches nothing
    assert len(p.calls) == 1
    print("backlog ok")


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
    assert {a for a, *_ in store.npc_posts(cutoff)} == wl
    counted = dict(store.db.execute("SELECT author, posts FROM authors"))
    for author in pre_start_authors:
        assert counted[author] == sum(post["author"] == author for post in p.posts.values())
    assert all(post["author"] in wl for post in store.recent_citizen_posts(20, cutoff))
    assert len(store.npc_posts(cutoff, limit=5)) == 5
    assert {a for a, _ in store.unknown_posts(cutoff)} == {a for a in late if any(
        post["author"] == a for post in p.posts.values())}
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
        asyncio.run(_backlog_demo())
        asyncio.run(_demo(args.seconds))
    else:
        for key, value in NpcStore(args.db).stats().items():
            print(f"{key}: {value}")
