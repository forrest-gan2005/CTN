"""Citizen stance map: the judge model scores each whitelisted citizen's referendum posts (2 firm YES .. -2 firm NO)
and data/citizen_stance.json keeps each citizen's latest, for the team's reply targeting (run.py reads it at every
refresh). Incremental: posts already scored are kept in data/citizen_stance_posts.json.
Run: python3 -m tools.stance_map [--hours 48]"""
import argparse
import asyncio
import json
import os
import re
import sqlite3
from datetime import datetime, timedelta, timezone

from ctn import paths
from ctn.infra.collector import COMPETITION_START, _iso, citizen_sql
from ctn.infra.llm import load_env, make_llm
from ctn.red.bot import about_referendum

DATA = paths.DATA
STANCE_FILE = DATA / "citizen_stance.json"
POSTS_FILE = DATA / "citizen_stance_posts.json"
BATCH = 15
SYSTEM = ("You read posts by residents of Kingstonia about a referendum on requiring proof of H2D2 vaccination to enter "
          "public places. For each post, judge the writer's own position on that requirement right now: "
          "2 = firmly for it (voting YES), 1 = leaning YES or YES with conditions, 0 = undecided or torn, "
          "-1 = leaning NO or NO unless conditions are met, -2 = firmly against (voting NO), null = the post shows no "
          "position. Answer only with JSON lines, one object {\"i\": number, \"s\": score or null} per post.")


def citizen_posts(hours: float) -> list:
    """(id, author, created_at, content) of whitelisted citizens' referendum posts of the last `hours`."""
    db = sqlite3.connect(f"file:{DATA / 'npc.db'}?mode=ro", uri=True, timeout=60)
    start = _iso(COMPETITION_START)
    since = _iso(datetime.now(timezone.utc) - timedelta(hours=hours))
    rows = db.execute(f"SELECT p.id, p.author, p.created_at, p.content FROM posts p JOIN authors a ON a.author = p.author "
                      f"WHERE {citizen_sql()} AND p.created_at > ? ORDER BY p.id", (start, start, start, since)).fetchall()
    db.close()
    return [r for r in rows if about_referendum(r[3] or "")]


def parse(raw: str, n: int) -> dict:
    """index -> score from the model's JSON lines; anything unreadable is left out."""
    out = {}
    for m in re.findall(r"\{[^{}]*\}", raw or ""):
        try:
            o = json.loads(m)
            i, s = int(o["i"]), o.get("s")
        except (ValueError, KeyError, TypeError):
            continue
        if 0 <= i < n and (s is None or s in (-2, -1, 0, 1, 2)):
            out[i] = s
    return out


async def score(rows: list, scored: dict) -> None:
    llm = make_llm(os.environ.get("LLM_JUDGE") or os.environ.get("LLM"), temperature=0)
    gate = asyncio.Semaphore(4)

    async def one(chunk: list) -> None:
        prompt = "\n\n".join(f"[{k}] {(r[3] or '')[:500]}" for k, r in enumerate(chunk))
        async with gate:
            raw = await llm.generate(SYSTEM, prompt)
        for k, s in parse(raw, len(chunk)).items():
            scored[str(chunk[k][0])] = s

    await asyncio.gather(*(one(rows[i:i + BATCH]) for i in range(0, len(rows), BATCH)))


def latest(rows: list, scored: dict) -> dict:
    """author -> {"s": score of their newest scored post, "at": its time}."""
    out = {}
    for pid, author, at, _ in rows:
        s = scored.get(str(pid))
        if s is not None and (author not in out or at > out[author]["at"]):
            out[author] = {"s": s, "at": at}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=float, default=48)
    args = parser.parse_args()
    load_env()
    rows = citizen_posts(args.hours)
    scored = json.loads(POSTS_FILE.read_text()) if POSTS_FILE.exists() else {}
    todo = [r for r in rows if str(r[0]) not in scored]
    asyncio.run(score(todo, scored))
    keep = {str(r[0]) for r in rows}
    scored = {k: v for k, v in scored.items() if k in keep}
    POSTS_FILE.write_text(json.dumps(scored))
    stances = latest(rows, scored)
    tmp = STANCE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(stances))
    tmp.replace(STANCE_FILE)  # read whole or not at all by run.py
    print(f"{len(rows)} referendum posts, {len(todo)} newly scored, {len(stances)} citizens mapped")


def _check() -> None:
    assert parse('{"i": 0, "s": 1}\n{"i": 2, "s": null}\n{"i": 9, "s": 1}\n{"i": 1, "s": 7}', 3) == {0: 1, 2: None}
    rows = [(1, "ann", "2026-10-28T01:00", "x"), (2, "ann", "2026-10-28T02:00", "y"), (3, "bob", "2026-10-28T01:00", "z"),
            (4, "cat", "2026-10-28T01:00", "w")]
    assert latest(rows, {"1": 2, "2": -1, "3": None, "4": 0}) == {"ann": {"s": -1, "at": "2026-10-28T02:00"},
                                                                  "cat": {"s": 0, "at": "2026-10-28T01:00"}}
    print("stance map ok")


if __name__ == "__main__":
    if os.environ.get("STANCE_MAP_CHECK"):
        _check()
    else:
        main()
