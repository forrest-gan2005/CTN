"""Trending logger: every few minutes, the top of Legit Social's Trending feed (rank, post id, author, hot score,
whether the author is ours) is appended to data/trending-history.jsonl. Public reads only, no account used.

    python3 -m tools.trending_log [--every 300]   # run in the background
    python3 -m tools.trending_log --show [hours]  # our posts' best rank and time on Trending lately
"""
import argparse
import json
import sqlite3
import time
from datetime import datetime, timedelta, timezone

import httpx

from ctn import paths

API = "https://social.legitreal.com/api"
OUT = paths.DATA / "trending-history.jsonl"
TOP = 50


def our_ids() -> set:
    db = sqlite3.connect(f"file:{paths.DATA / 'accounts.db'}?mode=ro", uri=True)
    ids = {r[0] for r in db.execute("SELECT account_id FROM accounts WHERE account_id IS NOT NULL")}
    db.close()
    return ids


def snapshot(client: httpx.Client, names: dict, ours: set) -> dict:
    items = client.get(f"{API}/posts", params={"sort": "trending", "limit": TOP}).json().get("items") or []
    rows = []
    for i, p in enumerate(items):
        pid = p.get("repost_of_post_id") or p.get("post_id")
        author = p.get("author_id")
        if p.get("repost_of_post_id"):  # a repost there lifts the post it reposts
            author = client.get(f"{API}/posts/{pid}").json().get("author_id")
        if author not in names:
            names[author] = client.get(f"{API}/accounts/{author}").json().get("username")
        rows.append({"rank": p.get("rank_position") or i + 1, "id": pid, "author": names[author],
                     "ours": author in ours, "hot": p.get("global_hot_score"), "created": p.get("created_at")})
    return {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "items": rows}


def run(every: float) -> None:
    names, ours, refreshed = {}, our_ids(), time.monotonic()
    with httpx.Client(timeout=30) as client:
        while True:
            if time.monotonic() - refreshed > 600:  # accounts swapped in since
                ours, refreshed = our_ids(), time.monotonic()
            try:
                snap = snapshot(client, names, ours)
                with open(OUT, "a") as f:
                    f.write(json.dumps(snap) + "\n")
                mine = [(r["rank"], r["id"]) for r in snap["items"] if r["ours"]]
                print(snap["at"], f"ours on Trending: {mine}", flush=True)
            except (httpx.HTTPError, ValueError) as e:
                print(datetime.now(timezone.utc).isoformat(timespec="seconds"), "failed:", e, flush=True)
            time.sleep(every)


def show(hours: float) -> None:
    since = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    best, seen = {}, {}
    for line in open(OUT):
        snap = json.loads(line)
        if snap["at"] < since:
            continue
        for r in snap.get("items", []):
            if r.get("ours"):
                key = (r["id"], r["author"])
                best[key] = min(best.get(key, 99), r["rank"])
                seen.setdefault(key, []).append(snap["at"])
    for (pid, author), rank in sorted(best.items()):
        times = seen[(pid, author)]
        print(f"#{pid} {author:<20} best rank {rank:>2}  on Trending {times[0][11:16]}-{times[-1][11:16]} UTC "
              f"({len(times)} snapshots)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--every", type=float, default=300)
    parser.add_argument("--show", type=float, nargs="?", const=6)
    args = parser.parse_args()
    if args.show is not None:
        show(args.show)
    else:
        run(args.every)
