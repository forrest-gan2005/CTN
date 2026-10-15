"""Can a model be our writer? Posts and replies written through the real Bot path, the drops and why, and optionally
simulated citizens reading them, all within a fixed number of calls. Nothing is posted: the platform is a fake.

    python3 -m tools.writer_eval --writer official:<model>                      # prints the plan only
    python3 -m tools.writer_eval --writer official:<model> --reader deepseek:deepseek-flash --go
    python3 -m tools.writer_eval test                                           # offline self-test

The judge defaults to the writer's model at temperature 0, as in team_llms(). Reports go to data/writer_eval/.
"""
import argparse
import asyncio
import json
import random
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from statistics import mean
from typing import Optional
from zoneinfo import ZoneInfo

from ctn import paths
from ctn.infra.fake_platform import FakePlatform
from ctn.infra.llm import make_llm
from ctn.infra.state import StateStore
from ctn.red.bot import NEWS_OUTLETS, VACCINE_ANGLES, Bot, Roster
from ctn.text.content import load_personas, load_world
from ctn.text.guard import wrap_external
from tools.content_ab import CONCERNS, readers

OUT = paths.DATA / "writer_eval"
# The referendum named outright; "vacc" alone is expected in vaccine posts.
VOTE_RE = re.compile(r"referendum|mandate|\bvot(e|ing)\b|proof of|\b2 nov|nov(ember)? 2\b", re.IGNORECASE)
LINES = ("persuade", "bridge", "mobilise", "background")
PER_LINE = 2
POST_KINDS = ("issue", "vaccine", "view", "daily")
# Citizens' posts to reply to: (author, lean as _lean would find it, text). Fixed so runs compare.
CITIZENS = [
    ("citizen_a", "no", "A vaccine pass to buy groceries? Not for me. The mandate goes too far. Voting no."),
    ("citizen_b", "no", "If they bring in door checks my sister's salon loses half her regulars overnight. "
                        "Who pays her rent then? No from me."),
    ("citizen_c", "no", "Had all my covid shots. This H2D2 one is brand new and nobody knows what it does in five "
                        "years. Not forcing that on my kids."),
    ("citizen_d", "undecided", "Lockdown nearly finished my shop, but proof of vaccination at the door sounds like a "
                               "nightmare too. Still don't know how I'm voting."),
    ("citizen_e", "undecided", "There isn't a single H2D2 case here and they want a vaccine requirement for every "
                               "public space. Bit much isn't it."),
    ("citizen_f", "yes", "Voting yes on the referendum. My mum is on chemo and can't have the jab."),
    ("citizen_g", "unknown", "Ferry was late again, third time this week. Anyone know what's going on?"),
    ("DailyKingstonia", "unknown", "Doctors' union says the island clinics are preparing for H2D2 as the referendum "
                                   "on proof of vaccination draws near."),
]
OPINIONS = ("referendum", "vaccines", "yearly")
LEANS = {
    "firm_no": "You are firmly voting NO in the referendum. You think proof of vaccination to enter shops, schools "
               "and transport goes too far, and you've heard most of the YES arguments already.",
    "lean_no": "You are leaning NO in the referendum: you got your covid shots, but proof of vaccination at the "
               "shops sits wrong with you. You could still be talked round, but not easily.",
    "undecided": "You haven't decided how to vote in the referendum: you remember how bad 2021 was, but you're "
                 "wary of the mandate.",
}
READ = (
    "{world}\n\n# Who you are\nYou are {name}, {age}, from {home}. You {job}.\n{lean}\n"
    "What weighs on you most: {concern}.\n\n# What to do\n"
    "You are scrolling Legit Social and see the post below. React as yourself, honestly. Most posts change nothing; "
    "a post can move you a little, not at all, or push you the other way.\n"
    "Three of your views can move, each on 0-100: \"referendum\" = the chance you vote YES; \"vaccines\" = how "
    "positive you feel about vaccines in general; \"yearly\" = how willing you are to get a vaccine every year.\n"
    'Answer with one JSON object and nothing else: {{"referendum": [before, after], "vaccines": [before, after], '
    '"yearly": [before, after], "annoyed": true or false, "sounds_like": "person" or "bot" or "campaign", '
    '"why": "one short sentence"}}. Before and after are whole numbers for how you stood just before and just '
    "after reading it."
)
# Readers that can't tell these apart can't judge the writer either.
CONTROLS = {
    "control_unrelated": "Saw a turtle off New Haven this morning. Made my week.",
    "control_no": "A vaccine pass just to buy groceries? Not for me. My body, my choice. Voting NO on the 2nd.",
    "control_yes": "I was a no for weeks. Then my mum's transplant team said she can't have the H2D2 jab, so she's "
                   "relying on the rest of us. I'm voting yes, not because anyone told me to, but for her.",
}


class Capped:
    """Passes calls through until the cap, then answers None so the rest of the run winds down."""

    def __init__(self, llm, budget: dict, label: str) -> None:
        self.llm, self.budget, self.label = llm, budget, label
        self.name = getattr(llm, "name", label)
        self.calls, self.empty, self.seconds, self.raw = 0, 0, 0.0, []

    def delay(self) -> float:
        return self.llm.delay() if hasattr(self.llm, "delay") else 0.0

    async def generate(self, system: str, prompt: str) -> Optional[str]:
        if self.budget["left"] <= 0:
            return None
        self.budget["left"] -= 1
        self.calls += 1
        t = time.monotonic()
        out = await self.llm.generate(system, prompt)
        self.seconds += time.monotonic() - t
        self.empty += not out
        self.raw.append(out)
        return out

    async def aclose(self) -> None:
        await self.llm.aclose()


def accounts(seed: str) -> list:
    everyone = load_personas()
    rng = random.Random(seed)
    picked = []
    for line in LINES:
        group = sorted((p for p in everyone if p["line"] == line), key=lambda p: p["username"])
        picked += rng.sample(group, min(PER_LINE, len(group)))
    return picked


def signoffs(persona: dict) -> list:
    voice = (persona.get("backstory") or {}).get("voice", [])
    return [s for h in voice if h.startswith("Sign-offs:") for s in re.findall(r'"([^"]+)"', h)]


def plan(n_readers: int) -> dict:
    posts = len(LINES) * PER_LINE * len(POST_KINDS)
    replies = len(CITIZENS)
    drafts = posts + replies
    return {"posts": posts, "replies": replies, "writer and judge calls (about)": drafts * 2,
            "reader calls (at most)": (drafts + len(CONTROLS)) * n_readers}


async def write(writer, judge, seed: str) -> list:
    world, everyone = load_world(), load_personas()
    roster = Roster(own={p["username"] for p in everyone}, npcs=[a for a, _, _ in CITIZENS])
    store = StateStore(":memory:")
    picked = accounts(seed)
    repliers = [p for p in picked if p["line"] in ("persuade", "bridge")]
    jobs = []
    for p in picked:
        rng = random.Random(f"{seed}:{p['username']}")
        topics = {"issue": rng.choice(p["angles"]), "vaccine": rng.choice(VACCINE_ANGLES),
                  "view": rng.choice(p["views"]), "daily": rng.choice(p["topics"])}
        jobs += [(p, "post", (kind, topics[kind]), None) for kind in POST_KINDS]
    for i, (author, lean, text) in enumerate(CITIZENS):
        target = {"id": 1000 + i, "author": author, "parent_id": None, "content": text}
        jobs.append((repliers[i % len(repliers)], "reply", ("feed", lean), target))
    out = []
    for p, kind, topic, target in jobs:
        if writer.budget["left"] <= 0:
            out.append({"persona": p["username"], "kind": kind, "topic": topic[1], "skipped": True})
            continue
        bot = Bot(p, FakePlatform(), writer, world, stance_llm=judge, roster=roster, store=store)
        before = len(writer.raw)
        random.seed(f"{seed}:{p['username']}:{kind}:{topic[1]}")
        action = await bot._compose(kind, topic=topic, target=target)
        text = action.content if action else None
        label = f"post_{topic[0]}" if kind == "post" else (
            "reply_news" if target["author"] in NEWS_OUTLETS else f"reply_{topic[1]}")
        out.append({"persona": p["username"], "line": p["line"], "kind": label, "topic": topic[1],
                    "parent": target["content"] if target else None, "text": text,
                    "dropped": dict(bot.rejected), "raw": writer.raw[before:],
                    "signed": [s for s in signoffs(p) if text and s.lower().rstrip(".!") in text.lower()],
                    "off_topic": bool(text) and topic[0] in ("vaccine", "view", "daily") and bool(VOTE_RE.search(text))})
    return out


def parse(answer: Optional[str]) -> Optional[dict]:
    m = re.search(r"\{.*\}", answer or "", re.S)
    try:
        d = json.loads(m.group(0)) if m else None
        for k in OPINIONS:
            a, b = d[k]
            if not all(isinstance(x, int) and 0 <= x <= 100 for x in (a, b)):
                return None
        return d
    except (ValueError, TypeError, KeyError):
        return None


async def read(reader, drafts: list, n_readers: int, seed: str) -> list:
    world = load_world()
    leans = list(LEANS)
    people = [{**r, "lean": leans[i % 3], "concern": list(CONCERNS)[i % len(CONCERNS)]}
              for i, r in enumerate(readers(n_readers, f"{seed}:readers"))]
    items = [(d["kind"], d["text"], d["parent"]) for d in drafts if d.get("text")]
    items += [(k, t, None) for k, t in CONTROLS.items()]

    async def one(person, kind, text, parent):
        shown = wrap_external("someone", text) if parent is None else (
            f"{wrap_external('citizen', parent)}\nA reply to it:\n{wrap_external('someone', text)}")
        system = READ.format(world=world, **{**person, "lean": LEANS[person["lean"]]})
        return {"lean": person["lean"], "kind": kind, "result": parse(await reader.generate(system, shown))}

    return list(await asyncio.gather(*(one(p, *it) for p in people for it in items)))


def report(args, drafts: list, reads: list, models: dict) -> str:
    done = [d for d in drafts if not d.get("skipped")]
    kept = [d for d in done if d["text"]]
    reasons = defaultdict(int)
    for d in done:
        for k, v in d["dropped"].items():
            reasons[k] += v
    lines = [f"# Writer eval {args.stamp}", "", f"writer `{args.writer}`, judge `{args.judge or args.writer}`, "
             f"reader `{args.reader or '-'}`", "", "## Writing", "",
             "| | calls | empty answers | avg seconds per call, queueing included |", "|---|---|---|---|"]
    for label, m in models.items():
        lines.append(f"| {label} | {m.calls} | {m.empty} | {m.seconds / m.calls if m.calls else 0:.1f} |")
    lines += ["", f"- kept {len(kept)}/{len(done)} drafts; skipped by the call cap {len(drafts) - len(done)}",
              f"- drop reasons: {dict(sorted(reasons.items(), key=lambda kv: -kv[1])) or '-'}",
              f"- sign-offs used: {sum(bool(d['signed']) for d in kept)}/{len(kept)}",
              f"- referendum in a post meant to leave it out: {sum(d['off_topic'] for d in kept)}",
              f"- average words: {mean(len(d['text'].split()) for d in kept) if kept else 0:.0f}", "",
              "| kind | kept | dropped |", "|---|---|---|"]
    for kind in sorted({d["kind"] for d in done}):
        g = [d for d in done if d["kind"] == kind]
        lines.append(f"| {kind} | {sum(bool(d['text']) for d in g)} | {sum(not d['text'] for d in g)} |")
    if reads:
        ok = [r for r in reads if r["result"]]
        lines += ["", "## Simulated readers", "", f"parsed {len(ok)}/{len(reads)}", "",
                  "| kind | n | " + " | ".join(OPINIONS) + " | moved NO on referendum | sounds like bot/campaign |",
                  "|---|---|---|---|---|---|---|"]
        by = defaultdict(list)
        for r in ok:
            by[r["kind"]].append(r)
        for kind in sorted(by, key=lambda k: (k.startswith("control"), k)):
            g = by[kind]
            shift = [mean(r["result"][o][1] - r["result"][o][0] for r in g) for o in OPINIONS]
            no = sum(r["result"]["referendum"][1] < r["result"]["referendum"][0] for r in g) / len(g)
            bot = sum(r["result"].get("sounds_like") in ("bot", "campaign") for r in g) / len(g)
            lines.append(f"| {kind} | {len(g)} | " + " | ".join(f"{s:+.1f}" for s in shift) + f" | {no:.0%} | {bot:.0%} |")
        ctl = {k: mean(r["result"]["referendum"][1] - r["result"]["referendum"][0] for r in by[k]) for k in CONTROLS
               if by[k]}
        sane = len(ctl) == 3 and ctl["control_no"] < 0 < ctl["control_yes"] and abs(ctl["control_unrelated"]) < 1
        lines += ["", "Readers pass the controls (NO < 0 < YES, unrelated near 0): " + ("yes" if sane else
                  "**no, so don't trust the shifts above**")]
    lines += ["", "## Drafts", ""]
    for d in done:
        parent = f" (to: {d['parent'][:70]}…)" if d["parent"] else ""
        lines.append(f"- {d['kind']} {d['persona']}{parent}: {d['text'] or 'DROPPED ' + str(d['dropped'])}")
    return "\n".join(lines) + "\n"


async def run(args, writer_llm, judge_llm, reader_llm) -> str:
    budget = {"left": args.max_calls}
    writer, judge = Capped(writer_llm, budget, "writer"), Capped(judge_llm, budget, "judge")
    models = {"writer": writer, "judge": judge}
    try:
        drafts = await write(writer, judge, args.seed)
        reads = []
        if reader_llm is not None:
            reader = Capped(reader_llm, {"left": args.max_reads}, "reader")
            models["reader"] = reader
            reads = await read(reader, drafts, args.readers, args.seed)
    finally:
        for m in {id(x.llm): x for x in models.values()}.values():
            await m.aclose()
    OUT.mkdir(parents=True, exist_ok=True)
    name = re.sub(r"[^A-Za-z0-9.-]+", "_", args.writer.split("?")[0])
    base = OUT / f"{args.stamp}-{name}"
    base.with_suffix(".json").write_text(json.dumps({"args": vars(args), "drafts": drafts, "reads": reads}, indent=1))
    text = report(args, drafts, reads, models)
    base.with_suffix(".md").write_text(text, encoding="utf-8")
    return str(base.with_suffix(".md"))


def main() -> None:
    if sys.argv[1:] == ["test"]:
        asyncio.run(_test())
        return
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--writer", required=True, help="model spec, as for make_llm")
    ap.add_argument("--judge", help="stance judge spec; default the writer's model at temperature 0")
    ap.add_argument("--reader", help="simulated citizens' model spec; omit to skip reading")
    ap.add_argument("--readers", type=int, default=6)
    ap.add_argument("--max-calls", type=int, default=120, help="writer and judge calls together")
    ap.add_argument("--max-reads", type=int, default=400)
    ap.add_argument("--seed", default="writer-eval")
    ap.add_argument("--go", action="store_true", help="actually call the models")
    args = ap.parse_args()
    args.stamp = datetime.now(ZoneInfo("Australia/Sydney")).strftime("%Y%m%d-%H%M")
    print(json.dumps(plan(args.readers if args.reader else 0), indent=1))
    print(f"caps: {args.max_calls} writer+judge calls, {args.max_reads} reader calls")
    if not args.go:
        print("plan only; add --go to call the models")
        return
    writer = make_llm(args.writer)
    judge = make_llm(args.judge or args.writer, temperature=0)
    reader = make_llm(args.reader) if args.reader else None
    print("saved", asyncio.run(run(args, writer, judge, reader)))


class _Stub:
    """Writer, judge or reader for the self-test, answering by what the prompt asks for."""

    def __init__(self, role: str) -> None:
        self.role, self.name = role, f"stub-{role}"

    def delay(self) -> float:
        return 0.0

    async def generate(self, system: str, prompt: str) -> Optional[str]:
        if self.role == "judge":
            return "SUPPORT"
        if self.role == "reader":
            if "turtle" in prompt:
                return '{"referendum": [40, 40], "vaccines": [50, 50], "yearly": [30, 30], "why": "nice"}'
            if "Voting NO" in prompt:
                return '{"referendum": [40, 37], "vaccines": [50, 50], "yearly": [30, 30], "why": "no"}'
            return '{"referendum": [40, 44], "vaccines": [50, 51], "yearly": [30, 30], "sounds_like": "person"}'
        if "post about the referendum" in prompt:
            return random.choice(["Voting yes on the 2nd so the clinic queue never looks like 2021 again.",
                                  "Proof of vaccination at the door beats months with the shop shut. Yes from me."])
        return random.choice(["Ferry was slow again but the sunset made up for it.",
                              "Garden finally has tomatoes. Small wins.", "Long shift, cold tea, good crew."])

    async def aclose(self) -> None:
        pass


async def _test() -> None:
    import tempfile
    from pathlib import Path
    global OUT
    OUT = Path(tempfile.mkdtemp())
    args = argparse.Namespace(writer="stub", judge=None, reader="stub", readers=3, max_calls=400, max_reads=400,
                              seed="t", stamp="test")
    path = await run(args, _Stub("writer"), _Stub("judge"), _Stub("reader"))
    text = Path(path).read_text()
    data = json.loads(Path(path).with_suffix(".json").read_text())
    assert len(data["drafts"]) == len(LINES) * PER_LINE * len(POST_KINDS) + len(CITIZENS)
    assert not any(d.get("skipped") for d in data["drafts"])
    assert "Readers pass the controls (NO < 0 < YES, unrelated near 0): yes" in text, text
    assert {d["kind"] for d in data["drafts"]} >= {"post_issue", "reply_no", "reply_news", "reply_unknown"}

    args.max_calls, args.stamp = 10, "cap"
    path = await run(args, _Stub("writer"), _Stub("judge"), None)
    data = json.loads(Path(path).with_suffix(".json").read_text())
    calls = sum(len(d.get("raw", [])) for d in data["drafts"])
    assert calls <= 10 and any(d.get("skipped") for d in data["drafts"]), calls
    assert "Simulated readers" not in Path(path).read_text()
    print("writer_eval ok")


if __name__ == "__main__":
    main()
