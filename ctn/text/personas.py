import argparse
import json
import random
import re
from collections import Counter

from ctn.text.content import CONTENT_DIR, LINES, load_personas, load_world
from ctn.text.guard import EMOJI_RE, copies, fresh_phrases

ARCHETYPES = CONTENT_DIR / "archetypes.json"
PERSONAS = CONTENT_DIR / "personas.json"
MAX_PER_ARCHETYPE = 3  # more accounts with the same job and story would start to look like a set
YEAR = 2026  # ages are as of the competition


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
            "age": rng.randint(*_age_range(arch)),
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


def _age_range(arch: dict) -> tuple:
    """Ages the job allows that also fit its 2021 story (e.g. old enough then to have had the job)."""
    lo, hi = arch["age_2021"]
    return max(arch["ages"][0], lo + YEAR - 2021), min(arch["ages"][1], hi + YEAR - 2021)


def check(personas: list) -> list:
    """Problems with personas built from an archetype: an age that doesn't fit the archetype's 2021 story.
    Hand-written personas, whose job matches no archetype, are left to whoever wrote them."""
    pools = json.loads(ARCHETYPES.read_text(encoding="utf-8"))
    by_job = {a["job"]: a for a in pools["archetypes"]}
    problems = []
    for p in personas:
        arch = by_job.get(p["job"])
        if arch is None:
            continue
        lo, hi = arch["age_2021"]
        then = p["age"] - (YEAR - 2021)  # during 2021 they were this or one less
        if then < lo or then - 1 > hi:
            problems.append(f"{p['username']}: {p['age']} now, about {then} in 2021, but the story needs {lo}-{hi}")
    return problems


STORY_WORDS = (250, 700)


def check_backstories(personas: list) -> list:
    """Problems with the key accounts' stories: too short or too long, and anything two accounts share that a
    reader of both could notice (a phrase the setting doesn't explain, a family name, a sign-off, an emoji), or
    a family name that is also one of our accounts' names; or a line of its own persona's angles, views or topics
    told in the same words (the writer is handed that line, so a post on it would read as a prompt leak)."""
    keyed = {p["username"]: p["backstory"] for p in personas if "backstory" in p}
    by_user = {p["username"]: p for p in personas}
    accounts = {p["name"].split()[0].lower() for p in personas}
    setting = load_world()
    problems, owners = [], {}
    for user, b in keyed.items():
        words = len(b["story"].split())
        if not STORY_WORDS[0] <= words <= STORY_WORDS[1]:
            problems.append(f"{user}: story is {words} words, want {STORY_WORDS[0]}-{STORY_WORDS[1]}")
        for field in ("angles", "views", "topics"):
            for line in by_user[user][field]:
                if copies(line, b["story"]):
                    problems.append(f"{user}: story tells its {field[:-1]} {line!r} in the same words")
        names = {m.lower() for m in re.findall(r"\b([A-Z][a-z]+) \(", b["people"])}
        for name in names & accounts:
            problems.append(f"{user}: {name.title()} in the family is also one of our accounts")
        signoffs = {s.lower() for h in b["voice"] if h.startswith("Sign-offs:") for s in re.findall(r'"([^"]+)"', h)}
        emojis = set(EMOJI_RE.findall(" ".join(b["voice"])))
        for kind, items in (("phrase", fresh_phrases(b["story"], setting)), ("name", names), ("sign-off", signoffs),
                            ("emoji", emojis)):
            for item in items:
                owners.setdefault((kind, item), set()).add(user)
    for (kind, item), users in sorted(owners.items()):
        if len(users) > 1:
            problems.append(f"{kind} {item!r} is in {', '.join(sorted(users))}")
    return problems


def _save(personas: list) -> None:
    problems = check(personas)
    if problems:
        raise SystemExit("not saved:\n" + "\n".join(problems))
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
    sub.add_parser("check", help="ages that don't fit an archetype's 2021 story")
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
    elif args.cmd == "check":
        merged = load_personas(status=None)
        problems = check(personas) + check_backstories(merged)
        print("\n".join(problems) or f"{len(personas)} personas ok, "
              f"{sum('backstory' in p for p in merged)} with backstories")
        raise SystemExit(1 if problems else 0)
    else:
        counts = Counter((p["line"], p.get("status", "active")) for p in personas)
        for (line, status), k in sorted(counts.items()):
            print(f"{line:<12}{status:<9}{k}")
        print(f"total {len(personas)}")
