import argparse
import random
from collections import Counter, deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

from semantic import Embedder

DATA = Path(__file__).parent / "data"
WINDOW = timedelta(minutes=10)  # two posts this close in time can be one script firing
SIMILAR = 0.75  # the same opinion in other words scores ~0.74; our own personas never went above 0.69
MIN_LINKS = 3  # one shared talking point is coincidence; three is a habit
MIN_GROUP = 3
MIN_CHARS = 30


def links(posts: Iterable[tuple], embedder: Embedder, window: timedelta = WINDOW,
          similar: float = SIMILAR) -> tuple:
    """Count, for each pair of accounts, how often they posted the same thing within `window` of each other.

    `posts` are (author, content, created_at as UTC ISO). Returns (Counter of sorted account pairs, an example
    pair of texts for each).
    """
    posts = sorted((p for p in posts if len(p[1] or "") >= MIN_CHARS), key=lambda p: p[2])
    if not posts:
        return Counter(), {}
    vectors = embedder.encode([content for _, content, _ in posts])
    recent = deque()  # (time, author, content, vector) inside the window
    counts, examples = Counter(), {}
    for (author, content, created_at), vector in zip(posts, vectors):
        t = datetime.fromisoformat(created_at)
        while recent and t - recent[0][0] > window:
            recent.popleft()
        if recent:
            scores = np.stack([r[3] for r in recent]) @ vector
            partners = {}
            for (_, other, text, _), score in zip(recent, scores):
                if other != author and score >= similar:
                    partners.setdefault(other, text)
            for other, text in partners.items():
                pair = tuple(sorted((author, other)))
                counts[pair] += 1
                examples.setdefault(pair, (text, content))
        recent.append((t, author, content, vector))
    return counts, examples


def groups(counts: Counter, min_links: int = MIN_LINKS, min_group: int = MIN_GROUP) -> list:
    """Accounts joined by pairs with at least `min_links` shared moments, largest group first."""
    parent = {}

    def root(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for (a, b), n in counts.items():
        if n >= min_links:
            parent[root(a)] = root(b)
    found = {}
    for a in parent:
        found.setdefault(root(a), set()).add(a)
    return sorted((g for g in found.values() if len(g) >= min_group), key=len, reverse=True)


def describe(group: set, counts: Counter, examples: dict) -> str:
    pairs = [(n, pair) for pair, n in counts.items() if set(pair) <= group]
    n, pair = max(pairs)
    a, b = examples[pair]
    return (f"{len(group)} accounts, {sum(n for n, _ in pairs)} shared moments: {', '.join(sorted(group))}\n"
            f"    e.g. {pair[0]}: {a[:100]!r}\n         {pair[1]}: {b[:100]!r}")


def _demo() -> None:
    from semantic import load_embedder

    embedder = load_embedder()
    assert embedder is not None, "pip install sentence-transformers"
    rng = random.Random(0)
    start = datetime(2026, 10, 20, 9, tzinfo=timezone.utc)
    talking_points = [
        "A door check is nothing compared to another lockdown, I'm voting for the vaccine pass.",
        "The tourism board says checks scare visitors but outbreaks are what really scared them off.",
        "Our hospitals barely coped in 2021, we need the vaccine requirement before H2D2 gets here.",
    ]
    rewordings = [
        "Honestly a check at the door beats another lockdown any day, so I'm voting for the pass.",
        "Visitors weren't scared off by checks, they were scared off by the outbreak, whatever tourism says.",
        "Hospitals nearly collapsed in 2021, so let's have the vaccine requirement ready before H2D2 arrives.",
    ]
    posts = []
    team = ["team_a", "team_b", "team_c", "team_d"]
    for k, (point, reworded) in enumerate(zip(talking_points, rewordings)):  # the team fires three times
        t = start + timedelta(hours=3 * k)
        for i, acct in enumerate(team):
            text = point if i % 2 == 0 else reworded
            posts.append((acct, text, (t + timedelta(minutes=rng.uniform(0, 8))).isoformat()))
    chatter = ["ferry was late again this morning", "who's going to regatta week", "rent went up again, ridiculous",
               "big swell this weekend, surf's up", "the reef looked rough on today's dive", "power's out on the east side"]
    for i in range(300):
        t = start + timedelta(minutes=rng.uniform(0, 600))
        posts.append((f"citizen_{i % 60}", f"{rng.choice(chatter)} {rng.choice(['lol', 'ugh', '', 'again'])}",
                      t.isoformat()))
    counts, examples = links(posts, embedder)
    found = groups(counts)
    assert found == [set(team)], found
    print(describe(found[0], counts, examples))
    print("coordination ok")


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find groups of accounts that keep posting the same thing "
                                                 "at the same time (for review; nothing is reported)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    rivals = sub.add_parser("rivals", help="accounts outside the whitelist, with citizens as a baseline")
    rivals.add_argument("--hours", type=float, default=24)
    rivals.add_argument("--db", default=str(DATA / "npc.db"))
    ours = sub.add_parser("ours", help="audit our own accounts: would a Blue team see us as one group?")
    ours.add_argument("--hours", type=float, default=24)
    ours.add_argument("--state", default=str(DATA / "bots.db"))
    sub.add_parser("demo")
    args = parser.parse_args()

    if args.cmd == "demo":
        _demo()
        raise SystemExit
    from semantic import load_embedder

    embedder = load_embedder()
    if embedder is None:
        raise SystemExit("pip install sentence-transformers")
    since = datetime.now(timezone.utc) - timedelta(hours=args.hours)
    if args.cmd == "ours":
        from state import StateStore

        mine = [(u, c, ts) for u, c, ts in StateStore(args.state).our_posts() if ts >= _iso(since)]
        counts, examples = links(mine, embedder)
        found = groups(counts)
        print(f"{len(mine)} of our posts; account pairs with a shared moment: {len(counts)}")
        for g in found:
            print(describe(g, counts, examples))
        if not found:
            print(f"no group of {MIN_GROUP}+ accounts with {MIN_LINKS}+ shared moments per pair")
    else:
        from collector import NpcStore
        from content import load_personas

        store = NpcStore(args.db, own_usernames={p["username"] for p in load_personas(status=None)})
        whitelist = set(store.whitelist())
        rows = [(a, c, ts) for _, a, c, ts in store.posts_since(since) if a not in store.own]
        counts, examples = links(rows, embedder)
        citizen_pairs = sum(1 for (a, b), n in counts.items() if n >= MIN_LINKS and a in whitelist and b in whitelist)
        rival = Counter({pair: n for pair, n in counts.items() if not set(pair) & whitelist})
        print(f"{len(rows)} posts in {args.hours:g}h; baseline: {citizen_pairs} citizen pairs with "
              f"{MIN_LINKS}+ shared moments (what coincidence alone produces)")
        for g in groups(rival):
            print(describe(g, rival, examples))
