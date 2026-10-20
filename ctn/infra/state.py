import json
import sqlite3
from collections import Counter
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Iterable, Optional
from zoneinfo import ZoneInfo

from ctn import paths

DEFAULT_DB = paths.DATA / "bots.db"
# Statuses of reports that reached the platform, or may have.
OUT = "('sent', 'upheld', 'unknown', 'hidden', 'stood')"
SYDNEY = ZoneInfo("Australia/Sydney")  # quota places are kept by Sydney date

SCHEMA = """
CREATE TABLE IF NOT EXISTS seen (
    username TEXT NOT NULL,
    post_id INTEGER NOT NULL,
    PRIMARY KEY (username, post_id)
);
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL,
    ts TEXT NOT NULL,
    type TEXT NOT NULL,
    post_id INTEGER,
    content TEXT,
    line TEXT,            -- persona line: persuade | mobilise
    topic_kind TEXT,      -- posts: issue | view | daily; replies: conversation | feed
    angle TEXT,           -- the specific angle/topic used
    target_author TEXT,
    target_class TEXT,    -- npc | unknown, who we engaged
    made_id INTEGER,      -- the post or reply we created
    model TEXT            -- which LLM wrote the content
);
CREATE INDEX IF NOT EXISTS history_user ON history(username);
CREATE INDEX IF NOT EXISTS history_target ON history(target_author, ts);
CREATE TABLE IF NOT EXISTS claims (
    key TEXT PRIMARY KEY,   -- post:<id> or follow:<username>; one of our accounts per target
    username TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS suspects (
    author TEXT PRIMARY KEY,   -- an account we're sure belongs to a competitor team
    reason TEXT NOT NULL,
    post_id INTEGER,           -- the post that gave it away
    found_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS reports (
    post_id INTEGER PRIMARY KEY,
    author TEXT NOT NULL,
    reason TEXT NOT NULL,
    reported_at TEXT NOT NULL,
    status TEXT NOT NULL,      -- sent (accepted, verdict pending) | upheld | rejected | unknown (may or may not
                               -- have arrived; never resent) | error (never arrived; retried) | failed (gave up)
                               -- | hidden (gone from view since: ruled inauthentic, or deleted) | stood (still
                               -- up after the ruling time: ruled genuine, as far as we can tell)
    attempts INTEGER NOT NULL DEFAULT 1,
    next_try TEXT              -- for errors: not before this
);
CREATE TABLE IF NOT EXISTS writes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL,    -- a post or reply, journalled before it is sent; one that may be live is never resent
    kind TEXT NOT NULL,
    parent_id INTEGER,
    content TEXT NOT NULL,
    sent_at TEXT NOT NULL,
    status TEXT NOT NULL,      -- sending | sent | not_sent | pending (may or may not be live) | found | not_found
    made_id INTEGER,
    error TEXT
);
-- A quota place that was used: by the write it went with (a reply to a journalist, a pitch), or, write_id 0, by a
-- like or repost whose request started. Kept while what used it may have gone out; given back once it didn't.
CREATE TABLE IF NOT EXISTS holds (
    key TEXT PRIMARY KEY,
    write_id INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS reply_picks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL,    -- a post chosen to be answered, whatever then became of the reply
    route TEXT NOT NULL,       -- conversation | lean-hit | npc-random | unknown-fallback
    post_id INTEGER NOT NULL,
    author TEXT,
    author_class TEXT NOT NULL, -- npc | unknown when chosen; what the author turned out to be is looked up later
    picked_at TEXT NOT NULL,   -- UTC
    outcome TEXT,              -- claim_failed | dropped (nothing publishable) | written (writes.status says the rest)
    write_id INTEGER
);
CREATE INDEX IF NOT EXISTS reply_picks_at ON reply_picks(picked_at);
-- Which competitor team an account belongs to, when a person could tell: reports are rationed per team.
CREATE TABLE IF NOT EXISTS rival_teams (
    author TEXT PRIMARY KEY,
    team TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS report_intents (
    post_id INTEGER PRIMARY KEY, -- a report on its way; one left here when the process stopped may have arrived
    author TEXT NOT NULL,
    reason TEXT NOT NULL,
    at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS engagement (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL,    -- our account that was engaged
    note_key TEXT NOT NULL,    -- the platform's notification id, so a redelivered one isn't counted twice
    kind TEXT NOT NULL,        -- like | repost | reply | follow | ...
    actor TEXT,
    post_id INTEGER,           -- as the notification gave it
    received_at TEXT NOT NULL,
    raw TEXT,                  -- the notification as received, for replay and debugging
    UNIQUE (username, note_key)
);
CREATE TABLE IF NOT EXISTS moderation (
    post_id INTEGER PRIMARY KEY, -- one of our posts the platform stopped showing normally
    username TEXT NOT NULL,
    status TEXT NOT NULL,        -- hidden | missing (suspected) | confirmed | restored
    first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS schedule (
    username TEXT PRIMARY KEY,   -- when a new account first comes online, kept across restarts
    start_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS inbox (
    username TEXT NOT NULL,
    post_id INTEGER NOT NULL,  -- a reply to one of our posts, waiting for an answer
    actor TEXT,
    received_at TEXT NOT NULL,
    PRIMARY KEY (username, post_id)
);
"""

# Columns returned by load_history, in order, matching Action's fields.
HISTORY_COLS = "ts, type, post_id, content, line, topic_kind, angle, target_author, target_class, made_id, model"


def note_key(note: dict) -> str:
    """A notification's identity: the platform's id, or its contents when it has none."""
    if note.get("id") is not None:
        return str(note["id"])
    return "|".join(str(note.get(k)) for k in ("kind", "actor", "post_id", "created_at"))


# Desk replies to the papers have quotas of their own and use none of an author's daily ones.
NOT_DESK = (" AND NOT EXISTS (SELECT 1 FROM claims c WHERE c.key = 'desk:' || history.post_id || ':' "
            "|| history.username)")


class StateStore:
    """Durable team memory: what each bot has handled and done, and what the team has claimed.

    Survives restarts so bots don't repeat replies, repeat themselves, forget their
    own history or lose citizens' replies mid-competition. Pass path=":memory:" for tests.
    """

    def __init__(self, path=DEFAULT_DB) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)

    def load_seen(self, username: str) -> set:
        return {r[0] for r in self.db.execute("SELECT post_id FROM seen WHERE username = ?", (username,))}

    def add_seen(self, username: str, post_id: int) -> None:
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO seen VALUES (?, ?)", (username, post_id))

    def load_history(self, username: str, limit: int = 50) -> list:
        rows = self.db.execute(
            f"SELECT {HISTORY_COLS} FROM history WHERE username = ? ORDER BY id DESC LIMIT ?",
            (username, limit),
        ).fetchall()
        return list(reversed(rows))

    def load_followed(self, username: str) -> set:
        rows = self.db.execute(
            "SELECT target_author FROM history WHERE username = ? AND type = 'follow'", (username,))
        return {r[0] for r in rows}

    def our_posts(self) -> list:
        """(username, content, ts) for every post and reply our accounts published."""
        return self.db.execute(
            "SELECT username, content, ts FROM history WHERE type IN ('post', 'reply') AND content IS NOT NULL"
        ).fetchall()

    def made_posts(self, since: str) -> list:
        """(username, platform post id, ts) for our posts and replies published since `since`."""
        return self.db.execute(
            "SELECT username, made_id, ts FROM history WHERE type IN ('post', 'reply') AND made_id IS NOT NULL "
            "AND ts >= ? ORDER BY ts", (since,)).fetchall()

    def moderation_status(self, post_id: int) -> Optional[str]:
        row = self.db.execute("SELECT status FROM moderation WHERE post_id = ?", (post_id,)).fetchone()
        return row[0] if row else None

    def set_moderation(self, post_id: int, username: str, status: str, ts: str) -> bool:
        """Record what the platform shows for one of our posts; True if that changed. Confirmed never reverts."""
        previous = self.moderation_status(post_id)
        if previous == status or previous == "confirmed":
            return False
        with self.db:
            self.db.execute("INSERT INTO moderation VALUES (?, ?, ?, ?, ?) ON CONFLICT(post_id) DO UPDATE SET "
                            "status = excluded.status, last_seen = excluded.last_seen",
                            (post_id, username, status, ts, ts))
        return True

    def moderation_counts(self) -> dict:
        """username -> {"suspected": posts hidden or missing, "confirmed": posts confirmed taken down}."""
        counts = {}
        for username, status, n in self.db.execute(
                "SELECT username, status, COUNT(*) FROM moderation WHERE status != 'restored' GROUP BY username, status"):
            kind = "confirmed" if status == "confirmed" else "suspected"
            counts.setdefault(username, {"suspected": 0, "confirmed": 0})[kind] += n
        return counts

    def planned_start(self, username: str) -> Optional[str]:
        row = self.db.execute("SELECT start_at FROM schedule WHERE username = ?", (username,)).fetchone()
        return row[0] if row else None

    def plan_start(self, username: str, start_at: str) -> None:
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO schedule VALUES (?, ?)", (username, start_at))

    def record(self, username: str, ts: str, action) -> None:
        with self.db:
            self.db.execute(
                "INSERT INTO history (username, ts, type, post_id, content, line, topic_kind, "
                "angle, target_author, target_class, made_id, model) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (username, ts, action.type, action.post_id, action.content, action.line, action.topic_kind,
                 action.angle, action.target_author, action.target_class, action.made_id, action.model),
            )

    def team_texts(self, since: str, exclude: str) -> list:
        """(username, text) of every post and reply the other accounts have made since UTC ISO time `since`,
        counting ones still being sent and ones that may be live; a write known never to have gone out doesn't."""
        return self.db.execute(
            "SELECT username, content FROM history WHERE type IN ('post', 'reply') AND ts >= ? AND username != ? "
            "AND content IS NOT NULL UNION ALL SELECT username, content FROM writes WHERE sent_at >= ? "
            "AND username != ? AND status IN ('sending', 'pending')", (since, exclude, since, exclude)).fetchall()

    def acted_on(self, username: str, kind: str, post_id: int) -> bool:
        """Whether this account has liked, reposted or replied to (`kind`) this post as far as history shows, or
        has a reply to it that went out or may have; history alone survives a restore that loses the quota place."""
        if self.db.execute("SELECT 1 FROM history WHERE username = ? AND type = ? AND post_id = ?",
                           (username, kind, post_id)).fetchone():
            return True
        return kind == "reply" and self.db.execute(
            "SELECT 1 FROM writes WHERE username = ? AND kind = 'reply' AND parent_id = ? "
            "AND status NOT IN ('not_sent', 'not_found')", (username, post_id)).fetchone() is not None

    def team_replies(self, post_id: int) -> list:
        """Texts of the replies our accounts have made to this post, oldest first, counting ones still being sent
        and ones that may be live."""
        return [r[0] for r in self.db.execute(
            "SELECT content, ts FROM history WHERE type = 'reply' AND post_id = ? AND content IS NOT NULL "
            "UNION ALL SELECT content, sent_at FROM writes WHERE kind = 'reply' AND parent_id = ? "
            "AND status IN ('sending', 'pending') ORDER BY 2", (post_id, post_id))]

    def team_posts(self, topics: tuple, since: str, exclude: str) -> list:
        """(username, post id, text) of the other accounts' published posts on these topics since UTC ISO time
        `since`, oldest first."""
        marks = ",".join("?" * len(topics))
        return self.db.execute(
            f"SELECT username, made_id, content FROM history WHERE type = 'post' AND topic_kind IN ({marks}) "
            "AND ts >= ? AND username != ? AND made_id IS NOT NULL ORDER BY ts", (*topics, since, exclude)).fetchall()

    def gif_texts(self, since: str) -> list:
        """(UTC ISO time, text) of our posts and replies since `since` with a GIF link, counting writes being sent
        or that may be live; a write known never to have gone out doesn't."""
        gif = "(content LIKE '%giphy%' OR content LIKE '%gph.is%')"
        return self.db.execute(
            f"SELECT ts, content FROM history WHERE type IN ('post', 'reply') AND ts >= ? AND {gif} "
            f"UNION ALL SELECT sent_at, content FROM writes WHERE sent_at >= ? AND status != 'not_sent' AND {gif}",
            (since, since)).fetchall()

    def replies_to(self, author: str, since: str) -> int:
        """How many replies the whole team has sent `author` since UTC ISO time `since`, leaving out the desk's."""
        return self.db.execute(
            "SELECT COUNT(*) FROM history WHERE type = 'reply' AND target_author = ? AND ts >= ?" + NOT_DESK,
            (author, since),
        ).fetchone()[0]

    def answered(self, made_id: int, author: str) -> bool:
        """Whether our post `made_id` was itself a reply to `author`: their reply to it carries a conversation on."""
        return self.db.execute("SELECT 1 FROM history WHERE type = 'reply' AND made_id = ? AND target_author = ?",
                               (made_id, author)).fetchone() is not None

    def claim(self, key: str, username: str) -> bool:
        """Atomically reserve a target for one bot; False if another of our accounts has it."""
        with self.db:
            cur = self.db.execute("INSERT OR IGNORE INTO claims VALUES (?, ?)", (key, username))
        return cur.rowcount == 1

    def reserve(self, prefix: str, username: str, limit: int, used: Iterable[str] = ()) -> Optional[str]:
        """Claim `prefix` + username: a quota shared by the team, one place per account. None if `limit` places
        are taken, counting the accounts in `used` (one each time listed) that used one without a key, or if this
        account has one already. Returns the key."""
        holders = self.holders(prefix)
        used = list(used)
        if username in holders or username in used:
            return None
        if len(holders) + sum(u not in holders for u in used) >= limit:
            return None
        key = prefix + username
        return key if self.claim(key, username) else None

    def holders(self, prefix: str) -> set:
        """The accounts holding a place in the quota `prefix`, as `reserve` counts them."""
        return {k[len(prefix):] for (k,) in self.db.execute(
            "SELECT key FROM claims WHERE substr(key, 1, ?) = ?", (len(prefix), prefix))}

    def replied_since(self, author: str, since: str) -> list:
        """The accounts that replied to `author` since UTC ISO time `since`, once per reply, leaving out the desk's."""
        return [r[0] for r in self.db.execute(
            "SELECT username FROM history WHERE type = 'reply' AND target_author = ? AND ts >= ?" + NOT_DESK,
            (author, since))]

    def release(self, key: str) -> None:
        with self.db:
            self.db.execute("DELETE FROM claims WHERE key = ?", (key,))
            self.db.execute("DELETE FROM holds WHERE key = ?", (key,))

    def attempt(self, key: str) -> None:
        """A like or repost holding this place is about to be sent: from now on it may have gone, even if this
        process dies before hearing back."""
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO holds VALUES (?, 0)", (key,))

    def claimed(self, keys: Iterable[str]) -> set:
        keys = list(keys)
        if not keys:
            return set()
        marks = ",".join("?" * len(keys))
        return {r[0] for r in self.db.execute(f"SELECT key FROM claims WHERE key IN ({marks})", keys)}

    def add_suspect(self, author: str, reason: str, post_id: int, found_at: str) -> bool:
        with self.db:
            cur = self.db.execute("INSERT OR IGNORE INTO suspects VALUES (?, ?, ?, ?)",
                                  (author, reason, post_id, found_at))
        return cur.rowcount == 1

    def update_suspect(self, author: str, reason: str) -> None:
        with self.db:
            self.db.execute("UPDATE suspects SET reason = ? WHERE author = ?", (reason, author))

    def suspect(self, author: str) -> Optional[tuple]:
        """(reason, post_id) for one account, read fresh: another process may have decided on it since."""
        return self.db.execute("SELECT reason, post_id FROM suspects WHERE author = ?", (author,)).fetchone()

    def escalate_suspect(self, author: str, reason: str, post_id: int, found_at: str, waiting: tuple,
                         prefix: str) -> bool:
        """Turn a suspect still waiting for a look (reason in `waiting`, or starting with `prefix`) into one caught
        outright by `post_id`, in one statement: a decision someone made meanwhile (cleared, written up) stands."""
        marks = ",".join("?" * len(waiting))
        with self.db:
            cur = self.db.execute(
                f"UPDATE suspects SET reason = ?, post_id = ?, found_at = ? WHERE author = ? "
                f"AND (substr(reason, 1, ?) = ? OR reason IN ({marks}))",
                (reason, post_id, found_at, author, len(prefix), prefix, *waiting))
        return cur.rowcount == 1

    def suspects(self) -> dict:
        """author -> (reason, post_id)."""
        return {a: (r, p) for a, r, p in self.db.execute("SELECT author, reason, post_id FROM suspects")}

    def record_report(self, post_id: int, author: str, reason: str, at: str, status: str,
                      next_try: Optional[str] = None) -> int:
        """Record a report's outcome; returns how many times it has been attempted."""
        row = self.db.execute("SELECT attempts FROM reports WHERE post_id = ?", (post_id,)).fetchone()
        attempts = (row[0] + 1) if row and status in ("error", "failed") else (row[0] if row else 1)
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO reports VALUES (?, ?, ?, ?, ?, ?, ?)",
                            (post_id, author, reason, at, status, attempts, next_try))
        return attempts

    def report_attempts(self, post_id: int) -> int:
        row = self.db.execute("SELECT attempts FROM reports WHERE post_id = ? AND status IN ('error', 'failed')",
                              (post_id,)).fetchone()
        return row[0] if row else 0

    def rejected_reasons(self) -> set:
        return {r[0] for r in self.db.execute("SELECT DISTINCT reason FROM reports WHERE status = 'rejected'")}

    def reported(self, now: str = "9999") -> set:
        """Posts not to report (again) now: everything except failures that never arrived and are due a retry,
        including reports on their way."""
        return {r[0] for r in self.db.execute(
            "SELECT post_id FROM reports WHERE status != 'error' OR next_try > ? "
            "UNION SELECT post_id FROM report_intents", (now,))}

    def author_reports(self) -> dict:
        """author -> reports on its posts that went out, may have, or are on their way."""
        return dict(self.db.execute(
            f"SELECT author, COUNT(*) FROM (SELECT post_id, author FROM reports WHERE status IN {OUT} "
            "UNION SELECT post_id, author FROM report_intents) GROUP BY author").fetchall())

    def set_team(self, author: str, team: str) -> None:
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO rival_teams VALUES (?, ?)", (author, team))

    def team_of(self, author: str) -> str:
        """The competitor team a person put this account in; on its own, the account is its own team."""
        row = self.db.execute("SELECT team FROM rival_teams WHERE author = ?", (author,)).fetchone()
        return row[0] if row else author

    def team_reports(self, since: str) -> dict:
        """team -> reports on its accounts' posts since UTC ISO time `since` that went out, may have, or are on
        their way."""
        return dict(self.db.execute(
            f"SELECT COALESCE(t.team, r.author), COUNT(*) FROM (SELECT post_id, author FROM reports WHERE status IN "
            f"{OUT} AND reported_at >= ? UNION SELECT post_id, author FROM report_intents WHERE at >= ?) r "
            "LEFT JOIN rival_teams t ON t.author = r.author GROUP BY 1", (since, since)).fetchall())

    def awaiting_ruling(self, before: str, limit: int) -> list:
        """(post_id, reported_at) of reports the platform took, sent before `before`, whose fate we haven't seen,
        least recently looked at first."""
        return self.db.execute("SELECT post_id, reported_at FROM reports WHERE status = 'sent' AND reported_at < ? "
                               "ORDER BY attempts, reported_at LIMIT ?", (before, limit)).fetchall()

    def mark_ruling(self, post_id: int, status: Optional[str]) -> None:
        """What became of a reported post: hidden, stood, or None for looked at and still pending."""
        with self.db:
            if status is None:
                self.db.execute("UPDATE reports SET attempts = attempts + 1 WHERE post_id = ?", (post_id,))
            else:
                self.db.execute("UPDATE reports SET status = ? WHERE post_id = ?", (status, post_id))



    def begin_report(self, post_id: int, author: str, reason: str, at: str) -> None:
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO report_intents VALUES (?, ?, ?, ?)", (post_id, author, reason, at))

    def end_report(self, post_id: int) -> None:
        """The report's request came back, one way or another; its outcome is recorded separately."""
        with self.db:
            self.db.execute("DELETE FROM report_intents WHERE post_id = ?", (post_id,))

    def begin_write(self, username: str, kind: str, parent_id: Optional[int], content: str, sent_at: str,
                    pick: Optional[int] = None, hold: Optional[str] = None) -> int:
        """Journal a post or reply before sending it, so one interrupted mid-request is neither lost nor resent.
        A reply's pick and the quota place it holds are tied to it in the same transaction; from then on the
        write's status is the pick's, and the place stays taken unless the write is known never to have gone."""
        with self.db:
            write = self.db.execute("INSERT INTO writes (username, kind, parent_id, content, sent_at, status) "
                                    "VALUES (?, ?, ?, ?, ?, 'sending')", (username, kind, parent_id, content, sent_at)
                                    ).lastrowid
            if pick is not None:
                self.db.execute("UPDATE reply_picks SET outcome = 'written', write_id = ? WHERE id = ?", (write, pick))
            if hold is not None:
                self.db.execute("INSERT OR REPLACE INTO holds VALUES (?, ?)", (hold, write))
            return write

    def _free_holds(self, write_id: int) -> None:
        self.db.execute("DELETE FROM claims WHERE key IN (SELECT key FROM holds WHERE write_id = ?)", (write_id,))
        self.db.execute("DELETE FROM holds WHERE write_id = ?", (write_id,))

    def pick(self, username: str, route: str, post_id: int, author: Optional[str], author_class: str,
             at: str) -> int:
        """Record why a reply is going to this post, before anything is claimed or written."""
        with self.db:
            return self.db.execute(
                "INSERT INTO reply_picks (username, route, post_id, author, author_class, picked_at) "
                "VALUES (?, ?, ?, ?, ?, ?)", (username, route, post_id, author, author_class, at)).lastrowid

    def made(self, post_id: int) -> bool:
        """Whether this post is one of ours."""
        return self.db.execute("SELECT 1 FROM history WHERE made_id = ?", (post_id,)).fetchone() is not None

    def drafting(self, post_id: int, since: str) -> int:
        """Replies to this post picked since UTC ISO time `since` and still being written."""
        return self.db.execute("SELECT COUNT(*) FROM reply_picks WHERE post_id = ? AND outcome IS NULL AND picked_at >= ?",
                               (post_id, since)).fetchone()[0]

    def end_pick(self, pick: int, outcome: str) -> None:
        """A pick that never became a write: claim_failed or dropped."""
        with self.db:
            self.db.execute("UPDATE reply_picks SET outcome = ? WHERE id = ? AND outcome IS NULL", (outcome, pick))

    def reply_mix(self, since: str, now_class=None) -> dict:
        """Reply targets picked since UTC `since`, by route, and how many of those replies are known to be live
        (sent, or found by reconciliation; each write counts once). Authors unknown when picked are also counted
        by what `now_class(author)` says they are now; the class at the time of picking is kept as it was."""
        rows = self.db.execute("SELECT p.route, p.author, p.author_class, p.outcome, w.status FROM reply_picks p "
                               "LEFT JOIN writes w ON w.id = p.write_id WHERE p.picked_at >= ?", (since,)).fetchall()
        mix = {"picked": Counter(), "live": Counter(), "outcomes": Counter(), "unknown_picked": 0, "unknown_live": 0,
               "unknown_now": Counter()}
        for route, author, author_class, outcome, status in rows:
            live = status in ("sent", "found")
            mix["picked"][route] += 1
            mix["live"][route] += live
            mix["outcomes"][status or outcome or "open"] += 1
            if author_class == "unknown":
                mix["unknown_picked"] += 1
                mix["unknown_live"] += live
                if now_class is not None:
                    mix["unknown_now"][now_class(author)] += 1
        return {k: (+v if isinstance(v, Counter) else v) for k, v in mix.items()}

    def upheld_authors(self) -> set:
        """Accounts with a report the platform upheld: competitors, as far as anyone can know."""
        return {r[0] for r in self.db.execute("SELECT DISTINCT author FROM reports WHERE status = 'upheld'")}

    def end_write(self, write_id: int, status: str, made_id: Optional[int] = None, error: Optional[str] = None) -> None:
        with self.db:
            self.db.execute("UPDATE writes SET status = ?, made_id = ?, error = ? WHERE id = ?",
                            (status, made_id, error, write_id))
            if status == "not_sent":
                self._free_holds(write_id)

    def recover(self) -> tuple:
        """At startup: writes and reports still on their way when the last process stopped may have landed.
        Writes become pending, to be reconciled; reports become unknown. Neither is resent.
        Returns how many of each."""
        with self.db:
            writes = self.db.execute("UPDATE writes SET status = 'pending', error = 'interrupted' "
                                     "WHERE status = 'sending'").rowcount
            reports = self.db.execute(
                "INSERT INTO reports (post_id, author, reason, reported_at, status) "
                "SELECT post_id, author, reason, at, 'unknown' FROM report_intents WHERE true "
                "ON CONFLICT(post_id) DO UPDATE SET status = 'unknown', reported_at = excluded.reported_at, "
                "next_try = NULL WHERE status = 'error'").rowcount  # an outcome recorded before the stop stands
            self.db.execute("DELETE FROM report_intents")
        self.free_unused_places()
        return writes, reports

    def free_unused_places(self) -> int:
        """At startup: quota places taken by a run that stopped before using them. A place is used once what it
        went with started out (holds); a place from before holds were kept is used if history or a write that
        may have gone out shows it. Returns how many were freed."""
        rows = self.db.execute(
            "SELECT key, username FROM claims WHERE substr(key, 1, 6) IN ('press:', 'pitch:', 'boost:') "
            "AND key NOT IN (SELECT key FROM holds)").fetchall()
        unused = [(key,) for key, username in rows if not self._place_used(key, username)]
        with self.db:
            self.db.executemany("DELETE FROM claims WHERE key = ?", unused)
        return len(unused)

    def _place_used(self, key: str, username: str) -> bool:
        q = lambda sql, *args: self.db.execute(sql, args).fetchone() is not None
        kind, rest = key.split(":", 1)
        if kind == "boost":
            action, post_id, _ = rest.split(":", 2)
            return q("SELECT 1 FROM history WHERE username = ? AND type = ? AND post_id = ?", username, action,
                     int(post_id))
        journalist, day = rest.split(":")[:2]
        since = datetime.combine(date.fromisoformat(day), time(0), SYDNEY).astimezone(timezone.utc).isoformat()
        live = "status NOT IN ('not_sent', 'not_found')"
        if kind == "pitch":
            return (q("SELECT 1 FROM history WHERE username = ? AND type = 'post' AND topic_kind = 'pitch' "
                      "AND angle = ? AND ts >= ?", username, journalist, since)
                    or q(f"SELECT 1 FROM writes WHERE username = ? AND kind = 'post' AND {live} AND sent_at >= ? "
                         "AND instr(lower(content), ?) > 0", username, since, f"@{journalist}".lower()))
        return (q("SELECT 1 FROM history WHERE username = ? AND type = 'reply' AND target_author = ? AND ts >= ?",
                  username, journalist, since)
                or q("SELECT 1 FROM writes WHERE username = ? AND kind = 'reply' AND status = 'pending' "
                     "AND sent_at >= ?", username, since))

    def unresolved(self, status: str = "pending") -> list:
        """(id, username, kind, parent_id, content, sent_at) of writes in this state; pending means we don't
        know whether it was published."""
        return self.db.execute("SELECT id, username, kind, parent_id, content, sent_at FROM writes "
                               "WHERE status = ? ORDER BY id", (status,)).fetchall()

    def resolve(self, write_id: int, status: str, made_id: Optional[int] = None) -> None:
        """Settle a write whose outcome was unknown. A found one gets its post id in history, so it is watched;
        if the process died before history was written, the history row is added now."""
        with self.db:
            self.db.execute("UPDATE writes SET status = ?, made_id = ? WHERE id = ?", (status, made_id, write_id))
            if status == "not_found":  # it never appeared: its place is free again
                self._free_holds(write_id)
            if made_id is not None:
                linked = self.db.execute(
                    "UPDATE history SET made_id = ? WHERE id = (SELECT h.id FROM history h JOIN writes w "
                    "ON h.username = w.username AND h.content = w.content AND h.ts >= w.sent_at "
                    "WHERE w.id = ? AND h.made_id IS NULL ORDER BY h.id LIMIT 1)", (made_id, write_id)).rowcount
                if not linked:
                    self.db.execute("INSERT INTO history (username, ts, type, post_id, content, made_id) "
                                    "SELECT username, sent_at, kind, parent_id, content, ? FROM writes WHERE id = ?",
                                    (made_id, write_id))

    def get_meta(self, key: str) -> Optional[str]:
        row = self.db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, value))

    def made_ids(self) -> set:
        return {r[0] for r in self.db.execute("SELECT made_id FROM history WHERE made_id IS NOT NULL")}

    def adopt(self, username: str, ts: str, parent_id: Optional[int], content: Optional[str], made_id: int) -> None:
        """Take into history a post of ours the platform shows but history lacks (after restoring an older backup):
        health then watches it, and a reply's target is claimed and seen, so it isn't answered twice."""
        kind = "reply" if parent_id is not None else "post"
        with self.db:
            self.db.execute("INSERT INTO history (username, ts, type, post_id, content, made_id) VALUES (?, ?, ?, ?, ?, ?)",
                            (username, ts, kind, parent_id, content, made_id))
            if parent_id is not None:
                self.db.execute("INSERT OR IGNORE INTO claims VALUES (?, ?)", (f"post:{parent_id}", username))
                self.db.execute("INSERT OR IGNORE INTO seen VALUES (?, ?)", (username, parent_id))
                self.db.execute("DELETE FROM inbox WHERE username = ? AND post_id = ?", (username, parent_id))

    def ingest_notifications(self, username: str, notes: list, replies: list, received_at: str) -> int:
        """Store notifications as engagement feedback and queue the replies to answer, all in one transaction;
        a notification seen before is ignored. Returns how many were new."""
        rows = [(username, note_key(n), n.get("kind") or "?", n.get("actor"), n.get("post_id"), received_at,
                 json.dumps(n, default=str, ensure_ascii=False)) for n in notes]
        with self.db:
            before = self.db.total_changes
            self.db.executemany("INSERT OR IGNORE INTO engagement (username, note_key, kind, actor, post_id, "
                                "received_at, raw) VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
            new = self.db.total_changes - before
            self.db.executemany("INSERT OR IGNORE INTO inbox VALUES (?, ?, ?, ?)",
                                [(username, n["post_id"], n.get("actor"), received_at) for n in replies])
        return new

    def action_counts(self, since: str) -> dict:
        rows = self.db.execute("SELECT type, COUNT(*) FROM history WHERE ts >= ? GROUP BY type", (since,))
        return dict(rows.fetchall())

    def engagement_counts(self, since: str) -> dict:
        rows = self.db.execute("SELECT kind, COUNT(*) FROM engagement WHERE received_at >= ? GROUP BY kind", (since,))
        return dict(rows.fetchall())

    def report_counts(self) -> dict:
        return dict(self.db.execute("SELECT status, COUNT(*) FROM reports GROUP BY status").fetchall())

    def inbox(self, username: str) -> list:
        """Waiting replies as (post_id, actor, received_at), oldest first."""
        return self.db.execute(
            "SELECT post_id, actor, received_at FROM inbox WHERE username = ? ORDER BY received_at",
            (username,),
        ).fetchall()

    def drop_inbox(self, username: str, post_id: int) -> None:
        with self.db:
            self.db.execute("DELETE FROM inbox WHERE username = ? AND post_id = ?", (username, post_id))


if __name__ == "__main__":
    from dataclasses import dataclass

    @dataclass
    class A:
        type: str
        post_id: Optional[int] = None
        content: Optional[str] = None
        line: Optional[str] = None
        topic_kind: Optional[str] = None
        angle: Optional[str] = None
        target_author: Optional[str] = None
        target_class: Optional[str] = None
        made_id: Optional[int] = None
        model: Optional[str] = None

    s = StateStore(":memory:")
    s.add_seen("joe", 5)
    s.add_seen("joe", 5)  # idempotent
    s.record("joe", "t1", A("reply", post_id=5, content="hi", line="persuade", target_author="npc1", target_class="npc"))
    s.record("joe", "t2", A("post", content="milk is up", line="persuade", topic_kind="daily", angle="prices",
                            made_id=77, model="fake"))
    s.record("joe", "t3", A("follow", target_author="npc1", target_class="npc"))
    s.record("rosa", "t4", A("reply", post_id=9, content="yo", target_author="npc1", target_class="npc"))
    assert s.load_seen("joe") == {5}
    hist = s.load_history("joe")
    assert [r[1] for r in hist] == ["reply", "post", "follow"]
    assert hist[0][7] == "npc1" and hist[1][5] == "daily" and hist[1][9] == 77 and hist[1][10] == "fake"
    assert s.load_followed("joe") == {"npc1"}
    assert s.replies_to("npc1", "t0") == 2 and s.replies_to("npc1", "t2") == 1

    assert s.claim("post:5", "joe")
    assert not s.claim("post:5", "rosa")
    assert s.claimed(["post:5", "post:6"]) == {"post:5"}
    s.release("post:5")
    assert s.claim("post:5", "rosa")

    r11, r10 = ({"id": 1, "kind": "reply", "post_id": 11, "actor": "npc1"},
                {"id": 2, "kind": "reply", "post_id": 10, "actor": "npc2"})
    assert s.ingest_notifications("joe", [r11], [r11], "2026-10-20T01:00:00+00:00") == 1
    assert s.ingest_notifications("joe", [r10, r11], [r10, r11], "2026-10-20T02:00:00+00:00") == 1
    assert [r[0] for r in s.inbox("joe")] == [11, 10]
    s.drop_inbox("joe", 11)
    assert [r[0] for r in s.inbox("joe")] == [10] and s.inbox("rosa") == []
    assert s.add_suspect("rival", "assistant talk", 3, "t") and not s.add_suspect("rival", "probe", 4, "t")
    assert s.suspects() == {"rival": ("assistant talk", 3)}
    s.record_report(3, "rival", "assistant talk", "t", "sent")
    s.record_report(3, "rival", "assistant talk", "t", "upheld")
    assert s.reported() == {3}
    like = {"kind": "like", "actor": "npc1", "post_id": 77, "created_at": "t"}  # no id: keyed by its contents
    assert s.ingest_notifications("joe", [like], [], "2026-10-20T03:00:00+00:00") == 1
    assert s.ingest_notifications("joe", [like], [], "2026-10-20T04:00:00+00:00") == 0
    assert s.engagement_counts("2026-10-20") == {"like": 1, "reply": 2} and s.report_counts() == {"upheld": 1}
    assert s.action_counts("t0")["reply"] == 2
    assert s.set_moderation(7, "joe", "hidden", "t1") and not s.set_moderation(7, "joe", "hidden", "t2")
    assert s.set_moderation(8, "joe", "missing", "t1") and s.moderation_counts() == {"joe": {"suspected": 2, "confirmed": 0}}
    assert s.set_moderation(7, "joe", "restored", "t3") and s.moderation_counts()["joe"]["suspected"] == 1
    assert s.set_moderation(8, "joe", "confirmed", "t4") and not s.set_moderation(8, "joe", "restored", "t5")
    assert s.moderation_counts() == {"joe": {"suspected": 0, "confirmed": 1}}
    s.plan_start("ana", "2026-10-19T08:00:00+00:00")
    s.plan_start("ana", "2026-10-20T08:00:00+00:00")
    assert s.planned_start("ana") == "2026-10-19T08:00:00+00:00" and s.planned_start("bob") is None
    s.record_report(4, "late_npc", "account: output label", "t", "rejected")
    assert s.rejected_reasons() == {"account: output label"}
    assert s.upheld_authors() == {"rival"}

    # Reply picks: what was chosen, and what of it went live, counted once, in the window it was picked in.
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        t = StateStore(f"{tmp}/bots.db")
        t.pick("joe", "npc-random", 50, "npc1", "npc", "2026-10-20T12:59:00+00:00")  # 23:59 in Sydney
        fresh = t.pick("joe", "unknown-fallback", 51, "ghost", "unknown", "2026-10-20T13:01:00+00:00")  # 00:01
        conv = t.pick("joe", "conversation", 52, "ghost2", "unknown", "2026-10-20T13:02:00+00:00")
        t.end_pick(t.pick("joe", "lean-hit", 53, "npc2", "npc", "2026-10-20T13:03:00+00:00"), "claim_failed")
        lost = t.begin_write("joe", "reply", 51, "a", "2026-10-20T13:01:05+00:00", pick=fresh)
        t.end_write(lost, "pending")  # its answer never came
        refused = t.begin_write("joe", "reply", 52, "b", "2026-10-20T13:02:05+00:00", pick=conv)
        t.end_write(refused, "not_sent")
        t.end_pick(conv, "dropped")  # a written pick keeps its write's outcome
        midnight = "2026-10-20T13:00:00+00:00"
        mix = t.reply_mix(midnight)
        assert mix["picked"] == {"unknown-fallback": 1, "conversation": 1, "lean-hit": 1} and mix["live"] == {}
        assert mix["outcomes"] == {"pending": 1, "not_sent": 1, "claim_failed": 1}, mix
        t.resolve(lost, "found", 900)
        t.resolve(lost, "found", 900)  # reconciled twice, still one live reply
        again = StateStore(f"{tmp}/bots.db")  # and a restart keeps all of it
        mix = again.reply_mix(midnight, now_class=lambda a: {"ghost": "npc"}.get(a, "unknown"))
        assert mix["live"] == {"unknown-fallback": 1} and (mix["unknown_picked"], mix["unknown_live"]) == (2, 1)
        assert mix["unknown_now"] == {"npc": 1, "unknown": 1}
        before = again.reply_mix("2026-10-20T12:00:00+00:00")
        assert before["picked"]["npc-random"] == 1 and before["outcomes"]["open"] == 1
    replies = StateStore(":memory:")
    replies.record("ana", "2026-10-20T13:00:00+00:00", A("reply", post_id=77, content="first"))
    sending = replies.begin_write("joe", "reply", 77, "second", "2026-10-20T13:01:00+00:00")
    gone = replies.begin_write("kai", "reply", 77, "never went", "2026-10-20T13:02:00+00:00")
    replies.end_write(gone, "not_sent")
    replies.begin_write("kai", "reply", 78, "elsewhere", "2026-10-20T13:03:00+00:00")
    assert replies.team_replies(77) == ["first", "second"], replies.team_replies(77)
    replies.end_write(sending, "pending")
    assert replies.team_replies(77) == ["first", "second"], "one that may be live still counts"
    assert replies.acted_on("ana", "reply", 77) and replies.acted_on("joe", "reply", 77)
    assert not replies.acted_on("kai", "reply", 77) and not replies.acted_on("ana", "like", 77)
    assert replies.reserve("boost:repost:77:", "ana", 2) and replies.reserve("boost:repost:77:", "joe", 2)
    assert replies.holders("boost:repost:77:") == {"ana", "joe"} and replies.holders("boost:repost:7:") == set()
    print("state ok")
