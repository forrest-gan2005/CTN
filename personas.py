import argparse
import json
import random
import re
from collections import Counter

from content import CONTENT_DIR, LINES, load_personas

ARCHETYPES = CONTENT_DIR / "archetypes.json"
PERSONAS = CONTENT_DIR / "personas.json"
MAX_PER_ARCHETYPE = 3  # more accounts with the same job and story would start to look like a set


def _username(rng: random.Random, first: str, last: str) -> str:
    f, l = (re.sub(r"[^a-z]", "", s.lower()) for s in (first, last))
    return rng.choice([
        f"{f}_{l}", f"{f}{l[0]}{rng.randint(1, 99)}", f"{f}.{l}", f"{l}{f[0]}", f"{f}_{rng.randint(70, 99)}",
        f"its{f}", f"{f}{l[:3]}", f"{f}_{l[0]}",
    ])


def generate(n: int, line: str, status: str, existing: list, seed: int) -> list:
    """New personas built from job archetypes plus randomly drawn names, places, voices and views.

    Each archetype keeps a job's details consistent (its covid story, reasons and everyday topics);
    the random draws make accounts that share an archetype still read as different people.
    """
    pools = json.loads(ARCHETYPES.read_text(encoding="utf-8"))
    rng = random.Random(seed)
    usernames = {p["username"] for p in existing}
    names = {p["name"] for p in existing}
    jobs = Counter(p["job"] for p in existing)
    homes, weights = zip(*pools["homes"].items())
    made = []
    while len(made) < n:
        open_ = [a for a in pools["archetypes"] if jobs[a["job"]] < MAX_PER_ARCHETYPE]
        if not open_:
            raise SystemExit("every archetype is used up; add more to content/archetypes.json")
        least = min(jobs[a["job"]] for a in open_)
        arch = rng.choice([a for a in open_ if jobs[a["job"]] == least])
        first, last = rng.choice(pools["first_names"]), rng.choice(pools["last_names"])
        name = f"{first} {last}"
        username = _username(rng, first, last)
        if name in names or username in usernames:
            continue
        issues = rng.sample(sorted(pools["views"]), 2)
        made.append({
            "username": username,
            "name": name,
            "age": rng.randint(*arch["ages"]),
            "job": arch["job"],
            "home": rng.choice(arch["homes"]) if arch["homes"] else rng.choices(homes, weights)[0],
            "line": line,
            "status": status,
            "covid": rng.choice(arch["covid"]),
            "why_yes": rng.choice(arch["why_yes"]),
            "style": rng.choice(pools["styles"]),
            "angles": list(arch["angles"]),
            "topics": rng.sample(arch["topics"], 4),
            "views": [rng.choice(pools["views"][issue]) for issue in issues],
            "examples": [],
        })
        usernames.add(username)
        names.add(name)
        jobs[arch["job"]] += 1
    return made


def _save(personas: list) -> None:
    PERSONAS.write_text(json.dumps(personas, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    load_personas(status=None)  # fail fast if anything written is invalid


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grow and manage the persona library")
    sub = parser.add_subparsers(dest="cmd", required=True)
    add = sub.add_parser("add", help="generate new personas and append them")
    add.add_argument("--line", choices=sorted(LINES), required=True)
    add.add_argument("--n", type=int, required=True)
    add.add_argument("--status", choices=("active", "reserve"), default="active")
    add.add_argument("--seed", type=int, default=0)
    act = sub.add_parser("activate", help="move a reserve persona into play, e.g. to replace a banned account")
    act.add_argument("username")
    sub.add_parser("list", help="count personas by line and status")
    args = parser.parse_args()

    personas = json.loads(PERSONAS.read_text(encoding="utf-8"))
    if args.cmd == "add":
        made = generate(args.n, args.line, args.status, personas, args.seed)
        _save(personas + made)
        print(f"added {len(made)}: {', '.join(p['username'] for p in made)}")
    elif args.cmd == "activate":
        match = [p for p in personas if p["username"] == args.username]
        if not match:
            raise SystemExit(f"no persona {args.username!r}")
        match[0]["status"] = "active"
        _save(personas)
        print(f"{args.username} is active")
    else:
        counts = Counter((p["line"], p.get("status", "active")) for p in personas)
        for (line, status), k in sorted(counts.items()):
            print(f"{line:<12}{status:<9}{k}")
        print(f"total {len(personas)}")
