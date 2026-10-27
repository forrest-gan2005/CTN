import asyncio
import json
import logging
import random
import re
import sqlite3
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional
from zoneinfo import ZoneInfo

from ctn import paths
from ctn.infra import newcomers
from ctn.infra.fake_platform import Incomplete
from ctn.text.gifs import embeds_from_raw

log = logging.getLogger("collector")

# No competitor can be on the platform before this: the opening webinar starts at 5pm and the game opens during it,
# so accounts seen earlier are the organisers' (citizens or their scripted fleet).
COMPETITION_START = datetime(2026, 10, 19, 17, 0, tzinfo=ZoneInfo("Australia/Sydney"))
# When the game is expected to open ("likely 6pm", the organisers on 18 October): our accounts register and act from
# then. Never before COMPETITION_START.
GAME_OPENS = datetime(2026, 10, 19, 18, 0, tzinfo=ZoneInfo("Australia/Sydney"))
DEFAULT_DB = paths.DATA / "npc.db"
# The account watcher's list (data/platform2026/watch_accounts.py): every public account, with when it registered.
ACCOUNTS_SEEN = paths.DATA / "platform2026" / "accounts_seen.json"

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
    ours INTEGER NOT NULL,
    verified INTEGER NOT NULL DEFAULT 0  -- the platform's badge: run by the organisers, never a competitor
);
-- Accounts there before the start that mostly post text many others post word for word: scripted, not citizens.
CREATE TABLE IF NOT EXISTS fleet (author TEXT PRIMARY KEY);
-- When accounts registered, from the platform's account list: one registered before the start is the organisers',
-- however late it first posts.
CREATE TABLE IF NOT EXISTS registered (author TEXT PRIMARY KEY, created_at TEXT NOT NULL);
-- Citizens who registered after the start, as the last refresh_newcomers labelled them (every username the account
-- went by); labelled_at is when it was first labelled.
CREATE TABLE IF NOT EXISTS newcomers (author TEXT PRIMARY KEY, author_id INTEGER, tier TEXT, labelled_at TEXT);
"""
# There before the start: no competitor could register yet. Citizens are those of them outside the fleet.
BEFORE_START = ("a.ours = 0 AND (IFNULL(a.first_post_at < ?, 0) OR a.first_fetched_at < ? "
                "OR a.author IN (SELECT author FROM registered WHERE created_at < ?))")
PRE_START = BEFORE_START + " AND a.author NOT IN (SELECT author FROM fleet)"
# Registered before the start but not seen posting yet: citizens too, so their first post is known for one.
REGISTERED_ONLY = "SELECT author FROM registered WHERE created_at < ? AND author NOT IN (SELECT author FROM authors)"
# Citizens who registered after the start (newcomers.py) are citizens too: the organisers keep adding them all
# competition. The writing-style curve, the posting peak Blue measures and "there before the start" stay on PRE_START.
NEWCOMERS = True
# The tiers taken for citizens (newcomers.label). Plain names ("C") are left out: YES swarms use them too, and
# about one in ten so labelled is a bot.
NEWCOMER_TIERS = ("A", "A-silent", "U")
# A text this long posted by this many accounts is a template; an account whose posts are half templates is scripted.
# On 16 Oct 151 such accounts arrived together (550 texts, 32 distinct, YES and NO alike); 30 of the 45 accounts
# there before were the same kind.
FLEET_MIN_CHARS = 20
FLEET_AUTHORS = 3
HASHTAG_RE = re.compile(r"#\w{2,30}")
# Some citizens tag every word ("#spent #the #morning"): common words and very short tags aren't trends.
TAG_STOP = set("the and for are but not you all any can had her was one our out day get has him his how man new "
               "now old see two way who did its let put say she too use this that with have from they will just been "
               "more what when your some them than then very also into like only over such time even most here much "
               "well back after again where would there their about which could other think".split())
FRACTION_RE = re.compile(r"\.(\d+)")


def newcomers_sql() -> str:
    """The usernames of citizens who registered after the start, as SQL without parameters."""
    return f"SELECT author FROM newcomers WHERE tier IN ({', '.join(repr(t) for t in NEWCOMER_TIERS)})"


def citizen_sql() -> str:
    """Whitelisted citizens, as a condition on `authors a` taking PRE_START's three parameters."""
    if not NEWCOMERS:
        return PRE_START
    return f"(({PRE_START}) OR (a.ours = 0 AND a.author IN ({newcomers_sql()})))"


def _iso(dt: datetime) -> str:
    # Fixed format so ISO strings sort chronologically in SQL.
    return dt.astimezone(timezone.utc).isoformat(timespec="microseconds")


def _parse_time(value) -> Optional[str]:
    """Normalise a platform timestamp to UTC ISO, or None if it's missing, naive or unparseable."""
    try:
        if isinstance(value, (int, float)):
            return _iso(datetime.fromtimestamp(value, timezone.utc))
        # The platform drops trailing zeros from the fraction ("12:17:10.06277Z"), which Python 3.9 can't parse.
        text = FRACTION_RE.sub(lambda m: "." + m.group(1)[:6].ljust(6, "0"), str(value).replace("Z", "+00:00"), 1)
        dt = datetime.fromisoformat(text)
    except (TypeError, ValueError, OverflowError):
        return None
    return _iso(dt) if dt.tzinfo else None


def accounts_seen(path=ACCOUNTS_SEEN) -> list:
    """The account watcher's list, each account as the platform gave it; empty without one."""
    try:
        listed = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return list(listed.values()) if isinstance(listed, dict) else []


def _json_default(obj):
    return sorted(obj) if isinstance(obj, (set, frozenset)) else str(obj)


LOCK_TIMEOUT = 60.0  # seconds a write waits for another connection's lock


class NpcStore:
    def __init__(self, path=DEFAULT_DB, own_usernames: Iterable[str] = ()) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        # A reader elsewhere (the dashboard) can hold the file for seconds; at startup that stopped live three times on
        # 27 October with "database is locked" after sqlite's default 5 s.
        self.db = sqlite3.connect(str(path), timeout=LOCK_TIMEOUT)
        self.db.executescript(SCHEMA)
        if "verified" not in {r[1] for r in self.db.execute("PRAGMA table_info(authors)")}:
            self.db.execute("ALTER TABLE authors ADD COLUMN verified INTEGER NOT NULL DEFAULT 0")
        self.own = set(own_usernames)
        self._repair_times()
        self.refresh_fleet()  # a database from before this table existed, or reopened after the start

    def _repair_times(self) -> None:
        """Posts stored when a short fraction in their time couldn't be parsed get their time now, and their
        authors their first post."""
        fixed = []
        for post_id, author, raw in self.db.execute("SELECT id, author, raw FROM posts WHERE created_at IS NULL"):
            try:
                when = _parse_time(json.loads(raw).get("created_at"))
            except (ValueError, AttributeError):
                continue
            if when:
                fixed.append((when, post_id, author))
        if not fixed:
            return
        with self.db:
            self.db.executemany("UPDATE posts SET created_at = ? WHERE id = ?", [(w, i) for w, i, _ in fixed])
            self.db.executemany("UPDATE authors SET first_post_at = ? WHERE author = ? "
                                "AND (first_post_at IS NULL OR ? < first_post_at)", [(w, a, w) for w, _, a in fixed])
        log.info("gave %d stored posts the time they were missing", len(fixed))

    def add_registrations(self, accounts: Iterable[dict]) -> int:
        """Store when accounts registered, from the platform's account list ({"username", "created_at"}); returns
        how many were new. A registration time never changes."""
        rows = [(a["username"], _parse_time(a.get("created_at"))) for a in accounts if a.get("username")]
        before = self.db.total_changes
        with self.db:
            self.db.executemany("INSERT OR IGNORE INTO registered VALUES (?, ?)", [r for r in rows if r[1]])
        return self.db.total_changes - before

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
        badges = self._badges()
        ids = [r[0] for r in rows]
        marks = ",".join("?" * len(ids))
        known = {r[0] for r in self.db.execute(f"SELECT id FROM posts WHERE id IN ({marks})", ids)} if ids else set()
        new = [r for r in rows if r[0] not in known]
        with self.db:
            self.db.executemany("INSERT OR IGNORE INTO posts VALUES (?, ?, ?, ?, ?, ?, ?, ?)", new)
            self.db.executemany(
                "INSERT INTO authors (author, posts, first_post_at, first_fetched_at, ours, verified) "
                "VALUES (?, 1, ?, ?, ?, ?) ON CONFLICT(author) DO UPDATE SET "
                "posts = posts + 1, verified = MAX(verified, excluded.verified), "
                "first_post_at = CASE WHEN excluded.first_post_at IS NULL THEN first_post_at "
                "WHEN first_post_at IS NULL OR excluded.first_post_at < first_post_at THEN excluded.first_post_at "
                "ELSE first_post_at END",
                [(r[1], r[4], r[5], r[6], int(bool(p.get("author_verified"))))
                 for r, p in zip(rows, posts) if r in new],
            )
            # A badge can appear later: any read of any post carrying it counts, a post seen before included.
            self.db.executemany("UPDATE authors SET verified = 1 WHERE author = ?",
                                [(p["author"],) for p in posts if p.get("author_verified")])
        start = _iso(COMPETITION_START)
        # A new badge takes its holder out of the template count, which can clear others too.
        if any(r[4] is not None and r[4] < start for r in new) or self._badges() != badges:
            self.refresh_fleet()
        return len(new)

    def _badges(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM authors WHERE verified = 1").fetchone()[0]

    def refresh_fleet(self) -> None:
        """Recount the scripted accounts from the posts made before the start."""
        rows = self.db.execute(
            "WITH pre AS (SELECT author, content FROM posts WHERE ours = 0 AND created_at < ? AND content IS NOT NULL "
            "  AND length(trim(content)) >= ? AND author NOT IN (SELECT author FROM authors WHERE verified = 1)), "
            "shared AS (SELECT content FROM pre GROUP BY content HAVING COUNT(DISTINCT author) >= ?) "
            "SELECT author FROM pre GROUP BY author "
            "HAVING SUM(content IN (SELECT content FROM shared)) >= 1 "
            "AND 2 * SUM(content IN (SELECT content FROM shared)) >= COUNT(*)",
            (_iso(COMPETITION_START), FLEET_MIN_CHARS, FLEET_AUTHORS)).fetchall()
        with self.db:
            self.db.execute("DELETE FROM fleet")
            self.db.executemany("INSERT INTO fleet VALUES (?)", rows)

    def fleet(self) -> set:
        return {r[0] for r in self.db.execute("SELECT author FROM fleet")}

    def count(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM posts").fetchone()[0]

    def max_id(self) -> Optional[int]:
        return self.db.execute("SELECT MAX(id) FROM posts").fetchone()[0]

    def replies_after(self, after: Optional[int], limit: int = 2000) -> tuple:
        """(newest id looked at, [(id, author, parent id)] of the replies stored with an id above `after`, oldest
        first, from the first `limit` posts there); with `after` None, (the newest id, [])."""
        if after is None:
            return self.max_id(), []
        rows = self.db.execute("SELECT id, author, parent_id FROM posts WHERE id > ? ORDER BY id LIMIT ?",
                               (after, limit)).fetchall()
        return (rows[-1][0] if rows else after), [r for r in rows if r[2] is not None]

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
        """Accounts that posted, that we saw or that registered before `cutoff`: competitors weren't on the platform
        yet."""
        c = _iso(cutoff)
        silent = f" UNION {newcomers_sql()}" if NEWCOMERS else ""
        rows = self.db.execute(
            f"SELECT author FROM authors a WHERE {citizen_sql()} UNION {REGISTERED_ONLY}{silent} ORDER BY author",
            (c, c, c, c))
        return [r[0] for r in rows if r[0] not in self.own]

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

    def verified(self) -> set:
        """Accounts the platform shows with its verified badge."""
        return {r[0] for r in self.db.execute("SELECT author FROM authors WHERE verified = 1")}

    def mark_verified(self, author: str) -> None:
        with self.db:
            changed = self.db.execute("UPDATE authors SET verified = 1 WHERE author = ? AND verified = 0",
                                      (author,)).rowcount
        if changed:
            self.refresh_fleet()

    def is_verified(self, author: str) -> bool:
        return self.db.execute("SELECT 1 FROM authors WHERE author = ? AND verified = 1", (author,)).fetchone() is not None

    def before_start(self, cutoff: datetime = COMPETITION_START) -> set:
        """Every account there before `cutoff`, the scripted fleet included: the organisers', never a competitor's."""
        c = _iso(cutoff)
        rows = self.db.execute(f"SELECT author FROM authors a WHERE {BEFORE_START} UNION {REGISTERED_ONLY}",
                               (c, c, c, c))
        return {r[0] for r in rows} - self.own

    def _registered_only(self, author: str, c: str) -> bool:
        return author not in self.own and self.db.execute(
            f"SELECT 1 FROM ({REGISTERED_ONLY}) WHERE author = ?", (c, author)).fetchone() is not None

    def was_before_start(self, author: str, cutoff: datetime = COMPETITION_START) -> bool:
        c = _iso(cutoff)
        return self.db.execute(f"SELECT 1 FROM authors a WHERE a.author = ? AND {BEFORE_START}",
                               (author, c, c, c)).fetchone() is not None or self._registered_only(author, c)

    def is_citizen(self, author: str, cutoff: datetime = COMPETITION_START) -> bool:
        c = _iso(cutoff)
        return (self.db.execute(f"SELECT 1 FROM authors a WHERE a.author = ? AND {PRE_START}",
                                (author, c, c, c)).fetchone() is not None or self._registered_only(author, c)
                or self.is_newcomer(author))

    def newcomers(self) -> set:
        """Citizens who registered after the start, as last labelled; none while NEWCOMERS is off."""
        if not NEWCOMERS:
            return set()
        return {r[0] for r in self.db.execute(newcomers_sql())} - self.own

    def is_newcomer(self, author: str) -> bool:
        return NEWCOMERS and author not in self.own and self.db.execute(
            f"SELECT 1 FROM ({newcomers_sql()}) WHERE author = ?", (author,)).fetchone() is not None

    def refresh_newcomers(self, known_bots: Iterable[str] = (), accounts: Iterable[dict] = (),
                          now: Optional[datetime] = None) -> Counter:
        """Label again the citizens who registered after the start (newcomers.label), from the account watcher's list
        (`accounts`, as accounts_seen gives it) and the posts stored, behaviour over the last WINDOW_HOURS; ours, the
        scripted fleet and `known_bots` (competitors, by username) never are. Returns how many of each tier."""
        now = now or datetime.now(timezone.utc)
        listed, names, ids = {}, defaultdict(set), {}  # id -> account; id -> every username; username -> id
        for account in accounts:
            try:
                key = int(account["account_id"])
            except (KeyError, TypeError, ValueError):
                continue
            listed[key] = account
            for old in account.get("names") or ():
                if old and old[0]:
                    names[key].add(old[0])
                    ids.setdefault(old[0], key)
        for key, account in listed.items():  # a username another account went by before is this one's now
            if account.get("username"):
                names[key].add(account["username"])
                ids[account["username"]] = key
        posted, latest, own = set(), {}, set(self.own)  # latest: id -> (newest post, its author)
        for author, ours, last, key in self.db.execute(
                "SELECT a.author, a.ours, p.id, json_extract(p.raw, '$.author_id') FROM authors a "
                "JOIN posts p ON p.id = (SELECT MAX(id) FROM posts WHERE author = a.author)"):
            if ours:
                own.add(author)
            if not isinstance(key, int):
                continue
            ids[author] = key
            names[key].add(author)
            posted.add(key)
            latest[key] = max(latest.get(key, (0, "")), (last, author))
        fleet = self.fleet()
        excluded = {ids[u] for u in own | fleet | set(known_bots) if u in ids}
        # No citizen there before the start ever changed its username; all 49 accounts seen doing so by 21 October were
        # competitors' (ducthang_004_01_2531 to cody_friedman_04, harbour_cafe_843e to mateo_lindqvist95).
        excluded |= {key for key, usernames in names.items() if len(usernames) > 1}
        vocab = newcomers.vocabulary((a.get("username") or "", a.get("display_name")) for key, a in listed.items()
                                     if key <= newcomers.LAST_PRE_START_ID and a.get("username") not in fleet)
        people = {key: (a.get("username") or "", a.get("display_name"), True) for key, a in listed.items()}
        people.update((key, (author, None, False)) for key, (_, author) in latest.items() if key not in listed)
        since = _iso(now - timedelta(hours=newcomers.WINDOW_HOURS))
        active = newcomers.activity(
            (key if isinstance(key, int) else ids.get(author, author), content, created_at)
            for key, author, content, created_at in self.db.execute(
                "SELECT json_extract(raw, '$.author_id'), author, content, created_at FROM posts "
                "WHERE created_at >= ?", (since,)))
        tiers = newcomers.label(people, posted, active, excluded, vocab)
        first = dict(self.db.execute("SELECT author, labelled_at FROM newcomers"))
        at = _iso(now)
        rows = [(author, key, tier, first.get(author, at)) for key, tier in tiers.items() for author in names[key]]
        with self.db:
            self.db.execute("DELETE FROM newcomers")
            self.db.executemany("INSERT OR REPLACE INTO newcomers VALUES (?, ?, ?, ?)", rows)
        return Counter(tiers.values())

    def npc_posts(self, cutoff: datetime = COMPETITION_START, limit: Optional[int] = None,
                  badged: bool = True) -> list:
        """(author, content, created_at, parent_id) by whitelisted citizens, oldest first; `limit` keeps the newest.
        badged=False leaves out verified accounts (papers, reporters, campaigners, officials): not how or when
        ordinary people here write."""
        c = _iso(cutoff)
        rows = self.db.execute(
            "SELECT p.author, p.content, p.created_at, p.parent_id FROM posts p "
            f"JOIN authors a ON a.author = p.author WHERE {PRE_START} AND p.content IS NOT NULL "
            f"{'' if badged else 'AND a.verified = 0 '}ORDER BY p.id DESC LIMIT ?",
            (c, c, c, -1 if limit is None else limit),
        ).fetchall()
        return rows[::-1]

    def unknown_posts(self, cutoff: datetime = COMPETITION_START) -> list:
        """(author, content) for posts by accounts that are neither ours nor whitelisted."""
        c = _iso(cutoff)
        return self.db.execute(
            "SELECT p.author, p.content FROM posts p JOIN authors a ON a.author = p.author "
            f"WHERE a.ours = 0 AND NOT ({citizen_sql()}) AND p.content IS NOT NULL ORDER BY p.id",
            (c, c, c),
        ).fetchall()

    def recent(self, n: int = 30) -> list:
        """The newest posts as the API returned them, so bots can read the feed without extra API calls."""
        rows = self.db.execute("SELECT raw FROM posts ORDER BY id DESC LIMIT ?", (n,))
        return [json.loads(r[0]) for r in rows]

    def recent_citizen_posts(self, n: int = 30, cutoff: datetime = COMPETITION_START) -> list:
        """The newest posts by whitelisted citizens: bots will far outnumber them in Latest."""
        c = _iso(cutoff)
        rows = self.db.execute(
            f"SELECT p.raw FROM posts p JOIN authors a ON a.author = p.author WHERE {citizen_sql()} "
            "ORDER BY p.id DESC LIMIT ?",
            (c, c, c, n),
        )
        return [json.loads(r[0]) for r in rows]

    def recent_from(self, authors: Iterable[str], since: datetime) -> list:
        """These accounts' posts created since `since`, as the API returned them, newest first."""
        authors = list(authors)
        if not authors:
            return []
        marks = ",".join("?" * len(authors))
        rows = self.db.execute(f"SELECT raw FROM posts WHERE author IN ({marks}) AND created_at >= ? ORDER BY id DESC",
                               authors + [_iso(since)])
        return [json.loads(r[0]) for r in rows]

    def depth(self, post_id: int, limit: int = 10) -> int:
        """How many messages deep a post sits in its thread (a post 1, a reply to it 2, ...), as far as the stored
        posts show, counting no further than `limit`."""
        depth, current = 0, post_id
        while current is not None and depth < limit:
            row = self.db.execute("SELECT parent_id FROM posts WHERE id = ?", (current,)).fetchone()
            if row is None:
                break
            depth += 1
            current = row[0]
        return depth

    def recent_by(self, author: str, n: int = 3) -> list:
        """This account's newest post texts, to tailor a reply to the person."""
        rows = self.db.execute(
            "SELECT content FROM posts WHERE author = ? AND content IS NOT NULL ORDER BY id DESC LIMIT ?",
            (author, n),
        )
        return [r[0] for r in rows]

    def said(self, words: list, since: datetime) -> Iterable[tuple]:
        """(author, content) of posts since `since` by accounts not ours with these words in this order, newest
        first, read as needed: a loose match (any apostrophe, anything in between) for the caller to check."""
        pattern = "%" + "%".join(w.replace("'", "_") for w in words) + "%"
        return self.db.execute("SELECT author, content FROM posts WHERE ours = 0 AND created_at >= ? AND content LIKE ? "
                               "ORDER BY id DESC", (_iso(since), pattern))

    def top_hashtags(self, since: datetime, n: int = 5, cutoff: datetime = COMPETITION_START) -> list:
        """Hashtags citizens used most since `since` (at least twice), most used first."""
        c = _iso(cutoff)
        rows = self.db.execute(
            f"SELECT p.content FROM posts p JOIN authors a ON a.author = p.author "
            f"WHERE p.created_at >= ? AND {citizen_sql()} AND p.content LIKE '%#%'",
            (_iso(since), c, c, c),
        )
        counts = Counter(tag.lower() for (content,) in rows for tag in set(HASHTAG_RE.findall(content))
                         if len(tag) > 4 and tag[1:].lower() not in TAG_STOP)
        return [tag for tag, k in counts.most_common(n) if k >= 2]

    def citizen_texts(self, since: datetime, cutoff: datetime = COMPETITION_START) -> list:
        """The texts of ordinary citizens' posts since `since`: verified accounts (papers, campaigners) left out."""
        c = _iso(cutoff)
        return [content for (content,) in self.db.execute(
            "SELECT p.content FROM posts p JOIN authors a ON a.author = p.author "
            f"WHERE p.created_at >= ? AND {citizen_sql()} AND a.verified = 0", (_iso(since), c, c, c)) if content]

    def citizen_posts_since(self, since: datetime, cutoff: datetime = COMPETITION_START) -> list:
        """(id, author, content, created_at) of ordinary citizens' posts and replies since `since`, newest first:
        verified accounts and the scripted fleet left out."""
        c = _iso(cutoff)
        return self.db.execute(
            "SELECT p.id, p.author, p.content, p.created_at FROM posts p JOIN authors a ON a.author = p.author "
            f"WHERE p.created_at >= ? AND {citizen_sql()} AND a.verified = 0 AND p.content IS NOT NULL "
            "ORDER BY p.id DESC", (_iso(since), c, c, c)).fetchall()

    def gif_embeds(self, limit: int = 5000) -> list:
        """The GIF embeds on the newest posts that aren't ours: the GIFs people here actually post."""
        rows = self.db.execute("SELECT raw FROM posts WHERE ours = 0 AND raw LIKE '%giphy%' ORDER BY id DESC LIMIT ?",
                               (limit,))
        return [e for e in embeds_from_raw(r[0] for r in rows) if e.get("kind") == "giphy"]

    def citizen_peak_hourly(self, since: datetime, cutoff: datetime = COMPETITION_START) -> int:
        """The most posts any whitelisted citizen made within one clock hour since `since`."""
        c = _iso(cutoff)
        row = self.db.execute(
            f"SELECT COUNT(*) AS k FROM posts p JOIN authors a ON a.author = p.author "
            f"WHERE p.created_at >= ? AND {PRE_START} GROUP BY p.author, substr(p.created_at, 1, 13) "
            "ORDER BY k DESC LIMIT 1",
            (_iso(since), c, c, c),
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

    def replies_since(self, since: datetime) -> list:
        """(id, author, parent_id) for replies by others created at or after `since`."""
        return self.db.execute(
            "SELECT id, author, parent_id FROM posts WHERE parent_id IS NOT NULL AND ours = 0 AND created_at >= ?",
            (_iso(since),)).fetchall()

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
            "registered before the start, not seen posting": len(self.db.execute(
                REGISTERED_ONLY, (_iso(COMPETITION_START),)).fetchall()),
            "scripted accounts left out": len(self.fleet()),
            "citizens who registered after the start": len(self.newcomers()),
            "pending_gaps": len(self.gaps()),
            "unparsed_created_at": q("SELECT COUNT(*) FROM posts WHERE created_at IS NULL").fetchone()[0],
            "newest_fetch": q("SELECT MAX(fetched_at) FROM posts").fetchone()[0],
            "top_authors": q("SELECT author, posts FROM authors ORDER BY posts DESC LIMIT 5").fetchall(),
            "latest": q("SELECT author, content FROM posts ORDER BY id DESC LIMIT 3").fetchall(),
        }


BACKFILL_PAGES = 3  # history pages per round, after each fresh look at Latest
# The feeds allow a team 10 reads a minute between them (SDK 0.5.0), the collector being the only reader: one
# every 7 s leaves room for probes and a backfill page.
FEED_INTERVAL = 7.0


class Collector:
    def __init__(
        self,
        platform,
        store: NpcStore,
        page_size: int = 50,
        min_interval: float = FEED_INTERVAL,
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
                try:
                    page = await self.platform.feed_latest(n=self.page_size, before=high)
                except Incomplete as e:  # nothing to keep yet, but it read this far: the gap starts there now
                    if e.below is not None and low is not None and e.below <= low:
                        self.store.delete_gap(gap_id)
                    elif e.below is not None and e.below < high:
                        self.store.update_gap(gap_id, e.below)
                    log.info("gap below #%s not finished this round: %s", high, e)
                    return total
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
    from ctn.infra.fake_platform import FakePlatform

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
    from ctn.infra.fake_platform import FakePlatform
    from ctn.text.content import load_personas

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
    paper = next(iter(wl))
    store.mark_verified(paper)
    assert paper in {a for a, *_ in store.npc_posts(cutoff)}
    assert {a for a, *_ in store.npc_posts(cutoff, badged=False)} == wl - {paper}, "a verified account as a citizen"
    assert {a for a, _ in store.unknown_posts(cutoff)} == {a for a in late if any(
        post["author"] == a for post in p.posts.values())}
    _fleet_check()
    _registered_check()
    _hashtag_check()
    _newcomers_check()
    print(f"\ncollected {store.count()}/{len(p.posts)} posts, filled {collector.gaps} gaps, "
          f"whitelist {len(wl)} accounts")


def _hashtag_check() -> None:
    """A citizen who tags every word doesn't make "#the" a trend; real tags still count. Thread depth counts up
    the stored parents. Others' posts with a phrase are found whatever the apostrophe, never ours or old ones."""
    store = NpcStore(":memory:")
    t0 = datetime(2026, 10, 19, 1, 0, tzinfo=timezone.utc)
    texts = ["#spent #the #morning at the wharf", "#the ferry again #H2D2", "#h2d2 worries me #the end", "#NHS #NHS"]
    store.add_posts([{"id": i, "author": f"citizen_{i}", "content": t, "created_at": _iso(t0)}
                     for i, t in enumerate(texts)])
    assert store.top_hashtags(t0 - timedelta(hours=1), cutoff=t0 + timedelta(hours=1)) == ["#h2d2"]
    assert sorted(store.citizen_texts(t0 - timedelta(hours=1), cutoff=t0 + timedelta(hours=1))) == sorted(texts)
    assert store.citizen_texts(t0 + timedelta(minutes=1), cutoff=t0 + timedelta(hours=1)) == [], "older ones aren't"
    store.add_posts([{"id": 100 + i, "author": "a", "content": "x", "created_at": _iso(t0),
                      "parent_id": 99 + i if i else None} for i in range(12)])
    assert [store.depth(100), store.depth(101), store.depth(107), store.depth(111)] == [1, 2, 8, 10]
    assert store.depth(5000) == 0 and store.depth(111, limit=4) == 4
    said = NpcStore(":memory:", own_usernames={"mate"})
    said.add_posts([{"id": 1, "author": "citizen_1", "content": "I’d rather show a card than shut the bakery",
                     "created_at": _iso(t0)},
                    {"id": 2, "author": "citizen_2", "content": "id rather not", "created_at": _iso(t0)},
                    {"id": 3, "author": "mate", "content": "I'd rather show a card", "created_at": _iso(t0)},
                    {"id": 4, "author": "citizen_3", "content": "I'd rather show a card", "created_at": _iso(t0 - timedelta(days=8))}])
    assert [a for a, _ in said.said(["i'd", "rather", "show", "a"], t0 - timedelta(days=7))] == ["citizen_1"]


def _newcomers_check() -> None:
    """Citizens who registered after the start join the whitelist and the citizens' feeds while NEWCOMERS is on:
    labelled from the account list and their own posts over the latest two days, keyed by account id; ours, known
    competitors, renamed accounts and busy or copying ones never are. The writing-style curve and "there before the
    start" stay on the citizens from before."""
    global NEWCOMERS, NEWCOMER_TIERS
    now = datetime(2026, 10, 21, 12, 0, tzinfo=timezone.utc)
    store = NpcStore(":memory:", own_usernames={"Kai_is_so_blue"})
    n = 0

    def post(key, author, content, hours_ago):
        nonlocal n
        n += 1
        return {"id": n, "author": author, "author_id": key, "content": content,
                "created_at": _iso(now - timedelta(hours=hours_ago))}

    pre = now - timedelta(days=3)
    listed = [{"account_id": key, "username": u, "display_name": d, "created_at": _iso(pre)} for key, u, d in [
        (300, "GolferRuMaViolet", "Ruth Mason"), (301, "chillin_with_linh", "Linh Tran"),
        (2401, "Nina_is_so_violet", "Nina Cole"), (2402, "BoSa37Falcon", "Bob Sands"),
        (2403, "david_fletcher", "David Fletcher"), (2404, "susan_perkins", "Susan Perkins"),
        (2405, "Moana_is_so_azure", "Moana Kalani"), (2406, "Tama_is_so_teal", "Tama Fale"),
        (2407, "anna_smith_21", "Anna Smith"), (2408, "ben_jones_22", "Ben Jones"),
        (2409, "Kai_is_so_blue", "Kai Blue"), (2410, "Rua_is_so_red", "Rua Hemi"),
        (2411, "Lena_is_so_lilac", "Lena Moss")]]
    swarm = "I kept flip-flopping on this, but a simple check beats another winter of closure."
    store.add_posts([post(300, "GolferRuMaViolet", "morning all", 60), post(301, "chillin_with_linh", "hi", 60)]
                    + [post(2401, "Nina_is_so_violet", f"ferry late again #ferrytime {i}", 2 + i) for i in range(2)]
                    + [post(2403, "david_fletcher", "markets on sunday?", 5)]
                    + [post(2405, "Moana_is_so_azure", f"vote no {i}", i) for i in range(40)]
                    + [post(2406, "Tama_is_so_teal", f"vote no {i}", 49 + i / 10) for i in range(40)]
                    + [post(2407, "anna_smith_21", swarm, 3), post(2408, "ben_jones_22", swarm, 3)]
                    + [post(2409, "Kai_is_so_blue", "ours", 3), post(2410, "Rua_is_so_red", "a rival", 3)]
                    + [post(2411, "lena_bot_77", "before a rename", 30), post(2411, "Lena_is_so_lilac", "after", 3)]
                    + [post(2412, "i_have_9_otters", "lovely harbour", 4), post(None, "Nopi_is_so_odd", "no id", 4)])
    tiers = store.refresh_newcomers({"Rua_is_so_red"}, listed, now=now)
    saved = NEWCOMERS, NEWCOMER_TIERS
    try:
        NEWCOMER_TIERS = ("A", "A-silent", "C", "U")
        assert tiers == Counter({"A": 2, "A-silent": 1, "C": 1, "U": 1}), tiers
        new = {"Nina_is_so_violet", "BoSa37Falcon", "david_fletcher", "Tama_is_so_teal", "i_have_9_otters"}
        assert store.newcomers() == new, store.newcomers()
        assert set(store.whitelist()) == new | {"GolferRuMaViolet", "chillin_with_linh"}, store.whitelist()
        assert store.is_citizen("Nina_is_so_violet") and store.is_citizen("BoSa37Falcon")
        assert not store.is_citizen("Moana_is_so_azure") and not store.is_citizen("lena_bot_77")
        day = now - timedelta(days=1)
        assert {a for _, a, _, _ in store.citizen_posts_since(day)} == {"Nina_is_so_violet", "david_fletcher",
                                                                         "i_have_9_otters"}
        assert "Nina_is_so_violet" in {p["author"] for p in store.recent_citizen_posts(100)}
        assert any("ferry late" in t for t in store.citizen_texts(day)) and store.top_hashtags(day) == ["#ferrytime"]
        unknown = {a for a, _ in store.unknown_posts()}
        assert "Moana_is_so_azure" in unknown and not unknown & new, unknown
        assert {a for a, *_ in store.npc_posts()} == {"GolferRuMaViolet", "chillin_with_linh"}, "the curve: before only"
        assert not store.was_before_start("Nina_is_so_violet") and "Nina_is_so_violet" not in store.before_start()
        first = dict(store.db.execute("SELECT author, labelled_at FROM newcomers"))
        store.refresh_newcomers({"Rua_is_so_red"}, listed, now=now + timedelta(hours=1))
        assert dict(store.db.execute("SELECT author, labelled_at FROM newcomers")) == first, "first labelled, kept"
        NEWCOMER_TIERS = ("A", "A-silent", "U")
        assert "david_fletcher" not in store.whitelist() and not store.is_citizen("david_fletcher")
        NEWCOMERS = False
        assert set(store.whitelist()) == {"GolferRuMaViolet", "chillin_with_linh"} and not store.newcomers()
        assert not store.is_citizen("Nina_is_so_violet")
        assert "Nina_is_so_violet" in {a for a, _ in store.unknown_posts()}
        assert store.citizen_posts_since(day) == []
    finally:
        NEWCOMERS, NEWCOMER_TIERS = saved


def _registered_check() -> None:
    """Times with a short fraction parse, old rows without one are repaired, and an account registered before the
    start is the organisers' however late it posts."""
    assert _parse_time("2026-10-18T12:17:10.06277Z") == "2026-10-18T12:17:10.062770+00:00"
    assert _parse_time("2026-10-18T12:17:10.6903+00:00") == "2026-10-18T12:17:10.690300+00:00"
    assert _parse_time("2026-10-18T12:17:10.1234567Z") == "2026-10-18T12:17:10.123456+00:00"
    assert _parse_time("2026-10-18T12:17:10Z") == "2026-10-18T12:17:10.000000+00:00"
    assert _parse_time("2026-10-18T12:17:10.5") is None and _parse_time(None) is None and _parse_time("soon") is None
    start = datetime(2026, 10, 19, 6, 0, tzinfo=timezone.utc)
    before, after = "2026-10-19T05:00:00.5Z", "2026-10-19T07:00:00.25Z"
    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/npc.db"
        s = NpcStore(path, own_usernames={"ours_early"})
        s.add_posts([{"id": 1, "author": "old_timer", "content": "morning all", "created_at": before},
                     {"id": 2, "author": "late_one", "content": "hi", "created_at": after}])
        with s.db:  # as stored before short fractions parsed
            s.db.execute("UPDATE posts SET created_at = NULL")
            s.db.execute("UPDATE authors SET first_post_at = NULL, first_fetched_at = '2026-10-19T08:00:00+00:00'")
        assert not s.is_citizen("old_timer", start)
        s.db.close()
        s = NpcStore(path, own_usernames={"ours_early"})
        assert s.db.execute("SELECT COUNT(*) FROM posts WHERE created_at IS NULL").fetchone()[0] == 0
        assert s.is_citizen("old_timer", start) and not s.is_citizen("late_one", start), "repaired from raw"
        listed = [{"username": "quiet_npc", "created_at": before}, {"username": "late_one", "created_at": after},
                  {"username": "poster_later", "created_at": "2026-10-19T04:00:00.123Z"},
                  {"username": "ours_early", "created_at": before}, {"username": "private_one"}, {"created_at": before}]
        assert s.add_registrations(listed) == 4 and s.add_registrations(listed) == 0
        assert s.is_citizen("quiet_npc", start) and s.was_before_start("quiet_npc", start)
        assert "quiet_npc" in s.whitelist(start) and "quiet_npc" in s.before_start(start)
        assert not s.was_before_start("late_one", start) and not s.was_before_start("private_one", start)
        assert "ours_early" not in s.whitelist(start) and "ours_early" not in s.before_start(start)
        assert not s.was_before_start("ours_early", start), "ours, whatever the list says"
        s.add_posts([{"id": 3, "author": "poster_later", "content": "first post, long after", "created_at": after}])
        assert s.is_citizen("poster_later", start) and "poster_later" in s.whitelist(start)
        assert {a for a, _ in s.unknown_posts(start)} == {"late_one"}
        assert {a for a, *_ in s.npc_posts(start)} == {"old_timer", "poster_later"}
    assert accounts_seen("/nonexistent/accounts_seen.json") == []


def _fleet_check() -> None:
    """Scripted accounts posting shared templates stay off the whitelist; a citizen who once echoes one stays on."""
    store = NpcStore(":memory:")
    templates = ["Good post. It reminds me how important it is to vote yes on the referendum.",
                 "Totally relevant to the referendum debate too. I hope you're voting no."]
    t0 = datetime(2026, 10, 16, 2, 20, tzinfo=timezone.utc)
    posts, n = [], 0

    def add(author, content, minutes=0):
        nonlocal n
        n += 1
        posts.append({"id": n, "author": author, "content": content, "created_at": _iso(t0 + timedelta(minutes=minutes))})

    for i in range(5):
        for t in templates:
            add(f"bold_otter{i}", t)
        add(f"bold_otter{i}", None)  # a repost has no text
    add("ana", templates[0])
    for k in range(3):
        add("ana", f"Ferry was late again, the {k}th time this month, and the kids missed the bell.")
    add("once", templates[1])  # a scripted account with a single post is still scripted
    add("brief", "lol")
    add("paper", templates[1])
    add("paper", templates[0])
    posts[-1]["author_verified"] = posts[-2]["author_verified"] = True
    add("late_bot", templates[0], minutes=60 * 24 * 7)  # after the start: its text doesn't count towards a fleet
    store.add_posts(posts)
    assert store.fleet() == {f"bold_otter{i}" for i in range(5)} | {"once"}, store.fleet()
    assert set(store.whitelist()) - {"late_bot"} == {"ana", "brief", "paper"}, store.whitelist()
    assert not store.is_citizen("bold_otter0") and store.is_citizen("ana")
    assert "bold_otter1" in {a for a, _ in store.unknown_posts()}
    # A badge arriving later, by any route, recounts: three template authors become two, so none is scripted.
    later = datetime(2026, 10, 20, tzinfo=timezone.utc)
    for route in ("reread", "mark", "new post"):
        s = NpcStore(":memory:")
        s.add_posts([{"id": i, "author": a, "content": templates[0], "created_at": _iso(t0)}
                     for i, a in enumerate("abc")])
        assert s.fleet() == set("abc")
        if route == "reread":
            s.add_posts([{"id": 0, "author": "a", "content": templates[0], "created_at": _iso(t0), "author_verified": True}])
        elif route == "mark":
            s.mark_verified("a")
        else:
            s.add_posts([{"id": 9, "author": "a", "content": "after the start", "created_at": _iso(later),
                          "author_verified": True}])
        assert s.fleet() == set() and set(s.whitelist()) == set("abc"), (route, s.fleet())
        assert not {a for a, _ in s.unknown_posts()} & set("abc"), route
    s = NpcStore(":memory:")
    s.add_posts([{"id": i, "author": a, "content": templates[0], "created_at": _iso(t0)} for i, a in enumerate("abcd")])
    s.mark_verified("a")
    assert s.fleet() == set("bcd"), "three unbadged authors still make a template"

    import tempfile
    path = Path(tempfile.mkdtemp()) / "npc.db"
    NpcStore(path).add_posts(posts)
    with sqlite3.connect(str(path)) as db:
        db.execute("DELETE FROM fleet")  # as in a database collected before the table existed
    assert NpcStore(path).fleet() == store.fleet(), "reopening rebuilds the list"


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
