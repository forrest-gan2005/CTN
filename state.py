import sqlite3
from pathlib import Path
from typing import Iterable, Optional

DEFAULT_DB = Path(__file__).parent / "data" / "bots.db"

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
    status TEXT NOT NULL       -- sent | upheld | rejected | error
);
CREATE TABLE IF NOT EXISTS engagement (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL,    -- our account that was engaged
    kind TEXT NOT NULL,        -- like | repost | reply | follow | ...
    actor TEXT,
    post_id INTEGER,           -- as the notification gave it
    received_at TEXT NOT NULL
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

    def record(self, username: str, ts: str, action) -> None:
        with self.db:
            self.db.execute(
                "INSERT INTO history (username, ts, type, post_id, content, line, topic_kind, "
                "angle, target_author, target_class, made_id, model) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (username, ts, action.type, action.post_id, action.content, action.line, action.topic_kind,
                 action.angle, action.target_author, action.target_class, action.made_id, action.model),
            )

    def replies_to(self, author: str, since: str) -> int:
        """How many replies the whole team has sent `author` since UTC ISO time `since`."""
        return self.db.execute(
            "SELECT COUNT(*) FROM history WHERE type = 'reply' AND target_author = ? AND ts >= ?",
            (author, since),
        ).fetchone()[0]

    def claim(self, key: str, username: str) -> bool:
        """Atomically reserve a target for one bot; False if another of our accounts has it."""
        with self.db:
            cur = self.db.execute("INSERT OR IGNORE INTO claims VALUES (?, ?)", (key, username))
        return cur.rowcount == 1

    def release(self, key: str) -> None:
        with self.db:
            self.db.execute("DELETE FROM claims WHERE key = ?", (key,))

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

    def suspects(self) -> dict:
        """author -> (reason, post_id)."""
        return {a: (r, p) for a, r, p in self.db.execute("SELECT author, reason, post_id FROM suspects")}

    def record_report(self, post_id: int, author: str, reason: str, at: str, status: str) -> None:
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO reports VALUES (?, ?, ?, ?, ?)",
                            (post_id, author, reason, at, status))

    def rejected_reasons(self) -> set:
        return {r[0] for r in self.db.execute("SELECT DISTINCT reason FROM reports WHERE status = 'rejected'")}

    def reported(self) -> set:
        return {r[0] for r in self.db.execute("SELECT post_id FROM reports")}

    def record_engagement(self, username: str, notes: list, received_at: str) -> None:
        """Every notification, kept as the feedback for judging which tactics earn engagement."""
        with self.db:
            self.db.executemany(
                "INSERT INTO engagement (username, kind, actor, post_id, received_at) VALUES (?, ?, ?, ?, ?)",
                [(username, n.get("kind") or "?", n.get("actor"), n.get("post_id"), received_at) for n in notes],
            )

    def action_counts(self, since: str) -> dict:
        rows = self.db.execute("SELECT type, COUNT(*) FROM history WHERE ts >= ? GROUP BY type", (since,))
        return dict(rows.fetchall())

    def engagement_counts(self, since: str) -> dict:
        rows = self.db.execute("SELECT kind, COUNT(*) FROM engagement WHERE received_at >= ? GROUP BY kind", (since,))
        return dict(rows.fetchall())

    def report_counts(self) -> dict:
        return dict(self.db.execute("SELECT status, COUNT(*) FROM reports GROUP BY status").fetchall())

    def add_inbox(self, username: str, notes: list, received_at: str) -> None:
        with self.db:
            self.db.executemany(
                "INSERT OR IGNORE INTO inbox VALUES (?, ?, ?, ?)",
                [(username, n["post_id"], n.get("actor"), received_at) for n in notes],
            )

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

    s.add_inbox("joe", [{"post_id": 11, "actor": "npc1"}], "2026-10-20T01:00:00+00:00")
    s.add_inbox("joe", [{"post_id": 10, "actor": "npc2"}, {"post_id": 11, "actor": "npc1"}], "2026-10-20T02:00:00+00:00")
    assert [r[0] for r in s.inbox("joe")] == [11, 10]
    s.drop_inbox("joe", 11)
    assert [r[0] for r in s.inbox("joe")] == [10] and s.inbox("rosa") == []
    assert s.add_suspect("rival", "assistant talk", 3, "t") and not s.add_suspect("rival", "probe", 4, "t")
    assert s.suspects() == {"rival": ("assistant talk", 3)}
    s.record_report(3, "rival", "assistant talk", "t", "sent")
    s.record_report(3, "rival", "assistant talk", "t", "upheld")
    assert s.reported() == {3}
    s.record_engagement("joe", [{"kind": "like", "actor": "npc1", "post_id": 77}], "2026-10-20T03:00:00+00:00")
    assert s.engagement_counts("2026-10-20") == {"like": 1} and s.report_counts() == {"upheld": 1}
    assert s.action_counts("t0")["reply"] == 2
    s.record_report(4, "late_npc", "account: output label", "t", "rejected")
    assert s.rejected_reasons() == {"account: output label"}
    print("state ok")
