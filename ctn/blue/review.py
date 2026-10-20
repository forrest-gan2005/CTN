"""A person's look at accounts Blue can't settle by rule: what is waiting for review, a sample of new accounts'
recent posts, and writing an account up (or clearing it). The running team reads the suspects table every round,
so a decision here takes effect without a restart. Nothing here talks to the platform: registration times come
from the account watcher's saved list."""
import argparse
import json
import random
import re
import tempfile
from datetime import datetime, timedelta, timezone
from typing import Optional

from ctn import paths
from ctn.blue.blue import (ACCOUNT_REPORTS, CLEARED, COPIED, MANUAL, PARAPHRASED, REVIEW, YES_LEAN_RE, BlueScanner,
                           waiting)
from ctn.infra.collector import COMPETITION_START, GAME_OPENS, NpcStore
from ctn.infra.state import StateStore
from ctn.text.content import load_personas
from ctn.text.guard import leans_no, plain

LINK_RE = re.compile(r"https?://\S+")
# The account watcher's list: every public account with the platform's registration time (none for private ones).
ACCOUNTS = paths.DATA / "platform2026" / "accounts_seen.json"
SHOWN = 200  # characters of each post shown


def _scanner(npc_store: NpcStore, state: StateStore, cutoff=COMPETITION_START) -> BlueScanner:
    return BlueScanner(npc_store, state, {p["username"] for p in load_personas(status=None)}, cutoff=cutoff)


def registered(path=ACCOUNTS) -> dict:
    """username -> registration time (UTC ISO), as far as the watcher has seen."""
    try:
        accounts = json.loads(open(path, encoding="utf-8").read()).values()
    except (OSError, ValueError):
        return {}
    return {a["username"]: a["created_at"] for a in accounts if a.get("username") and a.get("created_at")}


def before_opening(when: Optional[str], opens=GAME_OPENS) -> bool:
    """Registered before the game opened, when no team could: the organisers' account, whatever it posts."""
    # The platform's times are UTC with a varying number of fraction digits, which Python 3.9 can't parse.
    return bool(when) and datetime.fromisoformat(when[:19]).replace(tzinfo=timezone.utc) < opens


def _text(content: Optional[str]) -> str:
    return " ".join(LINK_RE.sub("[gif]", content or "").split())[:SHOWN]


def _lean(texts: list) -> str:
    no = sum(leans_no(t) and not YES_LEAN_RE.search(plain(t)) for t in texts)
    yes = sum(bool(YES_LEAN_RE.search(plain(t))) and not leans_no(t) for t in texts)
    return f"NO-ish {no}, YES-ish {yes}"


def describe(npc_store: NpcStore, author: str, posts: int, joined: Optional[dict] = None) -> list:
    rows = npc_store.db.execute("SELECT posts, first_post_at FROM authors WHERE author = ?", (author,)).fetchone()
    texts = [c for _, _, c in npc_store.posts_by([author])]
    when = (joined if joined is not None else registered()).get(author)
    lines = [f"{author}: {rows[0] if rows else 0} posts since {(rows[1] or '?')[:16] if rows else '?'}; "
             f"registered {when[:16] if when else '? (private, or not listed yet)'}; by pattern {_lean(texts)}"]
    lines += [f"    - {_text(t)}" for t in texts[:posts]]
    return lines


def queue(npc_store: NpcStore, state: StateStore, posts: int = 4) -> list:
    """Suspects not cleared: those waiting for review first, then the ones being reported."""
    out = []
    suspects = sorted(state.suspects().items(), key=lambda kv: (not waiting(kv[1][0]), kv[0]))
    for author, (reason, _) in suspects:
        if reason == CLEARED:
            continue
        out.append(f"[{reason}]")
        out += describe(npc_store, author, posts)
    return out or ["no suspects"]


def sample(npc_store: NpcStore, state: StateStore, minutes: float = 30, accounts: int = 25, posts: int = 5,
           cutoff=COMPETITION_START, now: Optional[datetime] = None, joined: Optional[dict] = None,
           opens=GAME_OPENS) -> list:
    """Accounts Blue may report (not there before the start, not verified, not ours, not registered before the
    game opened) that posted in the last `minutes`, a random `accounts` of them, with their newest posts."""
    scanner = _scanner(npc_store, state, cutoff)
    joined = registered() if joined is None else joined
    since = ((now or datetime.now(timezone.utc)) - timedelta(minutes=minutes)).isoformat()
    recent = [a for (a,) in npc_store.db.execute(
        "SELECT DISTINCT author FROM posts WHERE ours = 0 AND created_at >= ?", (since,))]
    possible = [a for a in recent if not scanner.off_limits(a)]
    open_ = [a for a in possible if not before_opening(joined.get(a), opens)]
    suspects = state.suspects()
    picked = random.sample(open_, min(accounts, len(open_)))
    out = [f"{len(open_)} accounts Blue may report posted in the last {minutes:g} minutes (another "
           f"{len(possible) - len(open_)} registered before the game opened, so citizens); showing {len(picked)}"]
    for author in sorted(picked):
        lines = describe(npc_store, author, posts, joined)
        status = suspects.get(author, ("",))[0]
        if status:
            lines[0] += f"  [already: {status}]"
        out += lines
    return out


def approve(npc_store: NpcStore, state: StateStore, author: str, why: str, cutoff=COMPETITION_START,
            joined: Optional[dict] = None, opens=GAME_OPENS, team: Optional[str] = None) -> str:
    """Write an account up: from then on its posts are reported (up to ACCOUNT_REPORTS), the report saying `why`."""
    why = " ".join(why.split()).rstrip(".")
    if len(why) < 20:
        raise SystemExit("say why in a sentence a moderator can check (it goes after 'because ...')")
    scanner = _scanner(npc_store, state, cutoff)
    if scanner.off_limits(author):
        raise SystemExit(f"{author} was there before the start, is verified or is ours: never reported")
    when = (registered() if joined is None else joined).get(author)
    if before_opening(when, opens):
        raise SystemExit(f"{author} registered at {when}, before the game opened: a citizen, never reported")
    newest = npc_store.posts_by([author])
    if not newest:
        raise SystemExit(f"no posts by {author} collected")
    reason = MANUAL + why
    if not state.add_suspect(author, reason, newest[0][0], datetime.now(timezone.utc).isoformat(timespec="seconds")):
        state.update_suspect(author, reason)
    if team:
        state.set_team(author, team)  # reports are rationed per competitor team a day
    return (f"{author} written up{f' as one of {team}' if team else ''}: up to {ACCOUNT_REPORTS} of its posts will "
            f"be reported because {why}")


def confirm(state: StateStore, author: str) -> str:
    """Send a rule's catch that waits for review as the rule found it (its giveaway post, its own detail)."""
    found = state.suspect(author)
    if found is None or not found[0].startswith(REVIEW):
        raise SystemExit(f"{author} isn't waiting for review")
    rule = found[0][len(REVIEW):]
    if rule in (COPIED, PARAPHRASED):
        raise SystemExit("copies need a written-up reason: use approve")
    state.update_suspect(author, rule)
    return f"{author} confirmed: post #{found[1]} will be reported ({rule})"


def clear(state: StateStore, author: str) -> str:
    """Never report this account."""
    if not state.add_suspect(author, CLEARED, 0, datetime.now(timezone.utc).isoformat(timespec="seconds")):
        state.update_suspect(author, CLEARED)
    return f"{author} cleared: never reported"


def _self_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=2)
        npc_store, state = NpcStore(f"{tmp}/npc.db"), StateStore(f"{tmp}/bots.db")
        old, new = (cutoff - timedelta(days=1)).isoformat(), (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
        npc_store.add_posts([
            {"id": 1, "author": "citizen", "content": "voting no, keep Kingstonia open", "created_at": old},
            {"id": 2, "author": "citizen", "content": "still no from me", "created_at": new},
            {"id": 3, "author": "late_one", "content": "Keep Kingstonia open! Vote no https://giphy.com/gifs/x", "created_at": new},
            {"id": 4, "author": "late_two", "content": "ferry late again", "created_at": new},
            {"id": 5, "author": "badge", "content": "news", "created_at": new, "author_verified": True}])
        opens = cutoff + timedelta(minutes=30)
        joined = {"late_one": (cutoff + timedelta(hours=1)).isoformat(), "late_two": None,
                  "quiet_citizen": (cutoff - timedelta(days=1)).isoformat().replace("+00:00", "Z")}
        npc_store.add_posts([{"id": 6, "author": "quiet_citizen", "content": "vote no lol", "created_at": new}])
        shown = "\n".join(sample(npc_store, state, minutes=30, cutoff=cutoff, joined=joined, opens=opens))
        assert "another 1 registered before the game opened" in shown, shown
        heads = {line.split(":")[0] for line in shown.splitlines()[1:] if not line.startswith(" ")}
        assert heads == {"late_one", "late_two"}, shown
        assert "[gif]" in shown and "NO-ish 1" in shown
        state.add_suspect("late_two", REVIEW + "copied across accounts", 4, new)
        assert any(line.startswith("late_two: 1 posts") and line.endswith("[already: review: copied across accounts]")
                   for line in sample(npc_store, state, minutes=30, cutoff=cutoff, joined=joined, opens=opens))
        for author in ("citizen", "badge", "quiet_citizen"):
            try:
                approve(npc_store, state, author, "it posts nothing but campaign slogans in the same words", cutoff,
                        joined, opens)
                raise AssertionError(f"{author} must never be written up")
            except SystemExit:
                pass
        approve(npc_store, state, "late_one", "it posts nothing but NO campaign slogans, with the same GIF each time.",
                cutoff, joined, opens, team="slogan_team")
        assert state.team_of("late_one") == "slogan_team" and state.team_of("late_two") == "late_two"
        assert state.suspects()["late_one"] == (MANUAL + "it posts nothing but NO campaign slogans, with the same "
                                                "GIF each time", 3)
        listed = queue(npc_store, state)
        assert listed[0].startswith("[review: ") and any("late_one" in line for line in listed)
        clear(state, "late_two")
        assert state.suspects()["late_two"][0] == CLEARED
        try:
            approve(npc_store, state, "late_two", "short", cutoff, joined, opens)
            raise AssertionError("a reason too short to check must be refused")
        except SystemExit:
            pass
    print("review ok")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("queue", help="suspects: those waiting for review first")
    s = sub.add_parser("sample", help="a random sample of new accounts' recent posts")
    s.add_argument("--minutes", type=float, default=30)
    s.add_argument("--accounts", type=int, default=25)
    s.add_argument("--posts", type=int, default=5)
    sh = sub.add_parser("show", help="one account's posts")
    sh.add_argument("author")
    sh.add_argument("--posts", type=int, default=30)
    a = sub.add_parser("approve", help="write an account up: its posts get reported, up to three")
    a.add_argument("author")
    a.add_argument("why", nargs="+", help="why, as it should read after 'because'")
    a.add_argument("--team", help="the competitor team it belongs to, when known (five reports a team a day)")
    cf = sub.add_parser("confirm", help="report a rule's catch as found (its giveaway post)")
    cf.add_argument("author")
    c = sub.add_parser("clear", help="never report this account")
    c.add_argument("author")
    sub.add_parser("self-test", help="check this tool, offline")
    args = parser.parse_args()
    if args.cmd == "self-test":
        return _self_test()
    npc_store, state = NpcStore(paths.DATA / "npc.db"), StateStore(paths.DATA / "bots.db")
    if args.cmd == "queue":
        print("\n".join(queue(npc_store, state)))
    elif args.cmd == "sample":
        print("\n".join(sample(npc_store, state, args.minutes, args.accounts, args.posts)))
    elif args.cmd == "show":
        print("\n".join(describe(npc_store, args.author, args.posts)))
    elif args.cmd == "approve":
        print(approve(npc_store, state, args.author, " ".join(args.why), team=args.team))
    elif args.cmd == "confirm":
        print(confirm(state, args.author))
    elif args.cmd == "clear":
        print(clear(state, args.author))


if __name__ == "__main__":
    main()
