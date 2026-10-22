"""Anchor bank A/B (bot.ANCHOR_BANK): the hand-written anchors against the model's, since a given time. Each anchor
is followed by its best Trending rank (data/trending-history.jsonl), the team's likes, reposts and replies that
landed in its first LIFT_WINDOW (history in data/bots.db), and the replies to it from outside the team and from
whitelisted citizens, with a rough stance by bot.says_yes and bot.leans_no (data/npc.db). The hourly score can't
be split per anchor, so it isn't here. Read-only: every database is opened read-only, nothing is called.

    python3 -m tools.anchor_ab [since-UTC-ISO]     # from the first bank entry used, or the last day, by default
    python3 -m tools.anchor_ab 2026-10-22T16:00:00+00:00 --bots copy/bots.db --npc copy/npc.db
"""
import argparse
import json
import sqlite3
import statistics
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from ctn import paths
from ctn.infra.collector import NpcStore
from ctn.red.bot import ANCHOR_BANK_ENTRIES, leans_no, says_yes
from ctn.text.content import load_personas

LIFT_WINDOW = timedelta(minutes=25)  # rivals' top posts had about 20 replies in their first 25 minutes
LIFTS = ("like", "repost", "reply")
ARMS = ("bank", "llm")


def read_only(path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro", uri=True)


def arm(angle: Optional[str], model: Optional[str]) -> Optional[str]:
    """bank, llm, or None for an anchor neither wrote (one posted by hand)."""
    if (angle or "").startswith("bank:"):
        return "bank"
    return None if model == "manual" else "llm"


def stance(text: str) -> str:
    yes, no = says_yes(text or ""), leans_no(text or "")
    return "mixed" if yes and no else "yes" if yes else "no" if no else "other"


def best_ranks(path) -> dict:
    """Post id -> (best Trending rank, snapshots it was in)."""
    ranks = {}
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return ranks
    for line in lines:
        try:
            snapshot = json.loads(line)
        except ValueError:
            continue
        for item in snapshot.get("items") or []:
            best, seen = ranks.get(item.get("id"), (None, 0))
            rank = item.get("rank")
            ranks[item.get("id")] = (rank if best is None or (rank is not None and rank < best) else best, seen + 1)
    return ranks


def anchors(bots: sqlite3.Connection, since: str) -> list:
    """(arm, ts, username, made_id, angle) of our anchor posts since `since`, oldest first."""
    rows = bots.execute("SELECT ts, username, made_id, angle, model FROM history WHERE type = 'post' "
                        "AND topic_kind = 'anchor' AND ts >= ? ORDER BY ts", (since,)).fetchall()
    return [(arm(angle, model), ts, user, made_id, angle) for ts, user, made_id, angle, model in rows
            if arm(angle, model) is not None]


def lifts(bots: sqlite3.Connection, made_id: int, ts: str) -> Counter:
    end = (datetime.fromisoformat(ts) + LIFT_WINDOW).isoformat()
    return Counter(dict(bots.execute(
        "SELECT type, COUNT(*) FROM history WHERE post_id = ? AND type IN ('like', 'repost', 'reply') "
        "AND ts >= ? AND ts <= ? GROUP BY type", (made_id, ts, end)).fetchall()))


def replies(npc: sqlite3.Connection, made_id: int, team: set, citizens: set) -> tuple:
    """(direct replies from outside the team, Counter of citizens' replies by stance)."""
    rows = npc.execute("SELECT author, content FROM posts WHERE parent_id = ?", (made_id,)).fetchall()
    outside = [(a, c) for a, c in rows if a not in team]
    return len(outside), Counter(stance(c) for a, c in outside if a in citizens)


def default_since(bots: sqlite3.Connection) -> str:
    try:
        first = bots.execute("SELECT MIN(used_at) FROM bank_used").fetchone()[0]
    except sqlite3.OperationalError:  # a bots.db from before the bank
        first = None
    return first or (datetime.now(timezone.utc) - timedelta(days=1)).isoformat(timespec="seconds")


def report(bots_path, npc_path, trending_path, since: Optional[str]) -> None:
    bots, npc = read_only(bots_path), read_only(npc_path)
    since = since or default_since(bots)
    team = ({p["username"] for p in load_personas(status=None)}
            | {r[0] for r in bots.execute("SELECT DISTINCT username FROM history")}
            | {r[0] for r in npc.execute("SELECT author FROM authors WHERE ours = 1")})
    whitelist = NpcStore.__new__(NpcStore)  # NpcStore() writes on opening; only its read is used here
    whitelist.db, whitelist.own = npc, team
    citizens = set(whitelist.whitelist())
    ranks = best_ranks(trending_path)

    rows = []
    for which, ts, user, made_id, angle in anchors(bots, since):
        if made_id is None:  # never found on the platform: nothing to follow
            rows.append((which, ts, user, None, angle, None, 0, Counter(), 0, Counter()))
            continue
        best, seen = ranks.get(made_id, (None, 0))
        outside, cited = replies(npc, made_id, team, citizens)
        rows.append((which, ts, user, made_id, angle, best, seen, lifts(bots, made_id, ts), outside, cited))

    print(f"anchors since {since} (team lifts within {int(LIFT_WINDOW.total_seconds() // 60)} minutes; replies "
          "direct, as the collector saw them)\n")
    print(f"{'arm':<5} {'posted (UTC)':<20} {'account':<16} {'post':>7} {'best':>5} {'snaps':>5} {'like':>5} "
          f"{'rep.':>5} {'reply':>5} {'others':>6} {'cit.':>5} {'yes':>4} {'no':>4}  angle")
    for which, ts, user, made_id, angle, best, seen, lift, outside, cited in rows:
        print(f"{which:<5} {ts[:19]:<20} {user[:16]:<16} {made_id or '-':>7} {best or '-':>5} {seen:>5} "
              f"{lift['like']:>5} {lift['repost']:>5} {lift['reply']:>5} {outside:>6} {sum(cited.values()):>5} "
              f"{cited['yes']:>4} {cited['no']:>4}  {(angle or '')[:50]}")

    print("\nby arm (anchors found on the platform):")
    for which in ARMS:
        found = [r for r in rows if r[0] == which and r[3] is not None]
        if not found:
            print(f"  {which}: no anchors")
            continue
        ranked = [r[5] for r in found if r[5] is not None]
        mean = lambda values: statistics.mean(values) if values else 0.0
        stances = sum((r[9] for r in found), Counter())
        print(f"  {which}: {len(found)} anchors, {len(ranked)} on Trending"
              + (f" (best rank median {statistics.median(ranked):g}, top 10: {sum(r <= 10 for r in ranked)})"
                 if ranked else "")
              + f"; lifts each {mean([r[7]['like'] for r in found]):.1f} likes, "
              f"{mean([r[7]['repost'] for r in found]):.1f} reposts, {mean([r[7]['reply'] for r in found]):.1f} "
              f"replies; replies from outside the team {mean([r[8] for r in found]):.1f}, from citizens "
              f"{mean([sum(r[9].values()) for r in found]):.1f} (stance: {dict(stances) or 'none'})")
    lost = Counter(r[0] for r in rows if r[3] is None)
    if lost:
        print(f"  not found on the platform, left out: {dict(lost)}")

    try:
        used = bots.execute("SELECT id, username, used_at, outcome FROM bank_used ORDER BY used_at").fetchall()
    except sqlite3.OperationalError:
        used = []
    print(f"\nbank entries: {len(used)} of {len(ANCHOR_BANK_ENTRIES)} used "
          f"({dict(Counter(o.split(':')[0] for *_, o in used)) or 'none'})")
    for entry_id, user, at, outcome in used:
        if outcome != "drawn":
            print(f"  {entry_id} {user} {at[:19]}: {outcome}")
    taken = {r[0] for r in used}
    left = Counter(e["username"] for e in ANCHOR_BANK_ENTRIES if e["id"] not in taken)
    print(f"  left: {dict(sorted(left.items())) or 'none'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("since", nargs="?", help="UTC ISO time, e.g. 2026-10-22T16:00:00+00:00")
    parser.add_argument("--bots", default=paths.DATA / "bots.db")
    parser.add_argument("--npc", default=paths.DATA / "npc.db")
    parser.add_argument("--trending", default=paths.DATA / "trending-history.jsonl")
    args = parser.parse_args()
    since = args.since
    if since:
        when = datetime.fromisoformat(since.replace("Z", "+00:00"))
        since = (when if when.tzinfo else when.replace(tzinfo=timezone.utc)).astimezone(timezone.utc).isoformat()
    report(args.bots, args.npc, args.trending, since)
