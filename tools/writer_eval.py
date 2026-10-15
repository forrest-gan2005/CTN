"""Can a model be our writer? Posts and replies written through the real Bot path, the drops and why, and optionally
simulated citizens reading them, all within a fixed number of calls. Nothing is posted: the platform is a fake.

    python3 -m tools.writer_eval --writer official:<model>                      # prints the plan only
    python3 -m tools.writer_eval --writer official:<model> --reader deepseek:deepseek-flash --go
    python3 -m tools.writer_eval test                                           # offline self-test

The judge defaults to the writer's model at temperature 0, as in team_llms(). Reports go to data/writer_eval/.
"""
import argparse
import asyncio
import contextvars
import json
import random
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from statistics import mean
from typing import Optional

import httpx
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
               "and transport goes too far.",
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


CALL = contextvars.ContextVar("writer_eval_call", default=None)  # requests sent and refused for one generate()


class Meter(httpx.AsyncBaseTransport):
    """Every request a model sends, retries and fallbacks included, counts against the cap; past it the request
    isn't sent and a local 499 comes back, which OpenAICompatLLM treats as a refusal of that one call."""

    def __init__(self, inner: httpx.AsyncBaseTransport, budget: dict) -> None:
        self.inner, self.budget = inner, budget

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        call = CALL.get()
        if self.budget["left"] <= 0:
            if call is not None:
                call["refused"] += 1
            return httpx.Response(499, request=request, text="writer_eval cap reached")
        self.budget["left"] -= 1
        self.budget["sent"] += 1
        if call is not None:
            call["sent"] += 1
        return await self.inner.handle_async_request(request)

    async def aclose(self) -> None:
        await self.inner.aclose()


def meter(llm, budget: dict) -> bool:
    """Counts requests at the HTTP layer where the model has one; False means count generate() calls instead."""
    parts = getattr(llm, "llms", [llm])
    http = [hasattr(m, "transport") for m in parts]
    if not any(http):
        return False
    if not all(http):  # one member's requests would go uncounted, and the cap with them
        raise ValueError(f"{getattr(llm, 'name', llm)!r} mixes API and local models; test one model at a time")
    for m in parts:
        m.transport = Meter(m.transport or httpx.AsyncHTTPTransport(), budget)
    return True


class Capped:
    """Tells apart answers, empty answers and calls the cap stopped, and counts the requests actually sent."""

    def __init__(self, llm, budget: dict, label: str) -> None:
        self.llm, self.budget, self.label = llm, budget, label
        self.metered = meter(llm, budget)
        self.name = getattr(llm, "name", label)
        self.calls, self.requests, self.empty, self.capped, self.seconds, self.raw = 0, 0, 0, 0, 0.0, []

    def delay(self) -> float:
        return self.llm.delay() if hasattr(self.llm, "delay") else 0.0

    async def generate(self, system: str, prompt: str) -> Optional[str]:
        return (await self.attempt(system, prompt))[0]

    async def attempt(self, system: str, prompt: str) -> tuple:
        """(answer, whether the cap stopped it)."""
        if self.budget["left"] <= 0:
            self.capped += 1
            return None, True
        call = {"sent": 0, "refused": 0}
        if not self.metered:
            self.budget["left"] -= 1
            self.budget["sent"] += 1
            call["sent"] = 1
        token = CALL.set(call)
        t = time.monotonic()
        try:
            out = await self.llm.generate(system, prompt)
        finally:
            CALL.reset(token)
        self.requests += call["sent"]
        if not out and call["refused"]:
            self.capped += 1
            return None, True
        self.calls += 1
        self.seconds += time.monotonic() - t
        self.empty += not out
        self.raw.append(out)
        return out, False

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
            "reader calls (at most)": (drafts + replies + len(CONTROLS)) * n_readers}


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
        label = f"post_{topic[0]}" if kind == "post" else (
            "reply_news" if target["author"] in NEWS_OUTLETS else f"reply_{topic[1]}")
        row = {"persona": p["username"], "line": p["line"], "kind": label, "topic": topic[1],
               "parent": target["content"] if target else None}
        if writer.budget["left"] < 2:  # a draft needs the writer and usually the judge
            out.append({**row, "status": "skipped"})
            continue
        bot = Bot(p, FakePlatform(), writer, world, stance_llm=judge, roster=roster, store=store)
        before, capped = len(writer.raw), writer.capped + judge.capped
        random.seed(f"{seed}:{p['username']}:{kind}:{topic[1]}")
        action = await bot._compose(kind, topic=topic, target=target)
        text = action.content if action else None
        status = "kept" if text else "incomplete" if writer.capped + judge.capped > capped else "dropped"
        out.append({**row, "status": status, "text": text, "dropped": dict(bot.rejected) if status == "dropped" else {},
                    "raw": writer.raw[before:],
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


def items(drafts: list) -> list:
    """What the readers see: our kept drafts, each reply's parent on its own (so a reply is judged by what it adds
    to its parent), and the controls."""
    out = [{"kind": d["kind"], "text": d["text"], "parent": d["parent"]} for d in drafts if d.get("text")]
    parents = sorted({d["parent"] for d in drafts if d.get("text") and d["parent"]})
    out += [{"kind": "parent_only", "text": t, "parent": None} for t in parents]
    out += [{"kind": k, "text": t, "parent": None} for k, t in CONTROLS.items()]
    return out


async def read(reader, shown_items: list, n_readers: int, seed: str) -> list:
    world = load_world()
    leans = list(LEANS)
    people = [{**r, "lean": leans[i % 3], "concern": list(CONCERNS)[i % len(CONCERNS)]}
              for i, r in enumerate(readers(n_readers, f"{seed}:readers"))]
    jobs = [(p, i) for p in people for i in range(len(shown_items))]
    random.Random(f"{seed}:order").shuffle(jobs)  # a cap that cuts the run short then cuts every lean alike

    async def one(person, i):
        it = shown_items[i]
        shown = wrap_external("someone" if it["kind"] != "parent_only" else "citizen", it["text"]) \
            if it["parent"] is None else (
            f"{wrap_external('citizen', it['parent'])}\nA reply to it:\n{wrap_external('someone', it['text'])}")
        system = READ.format(world=world, **{**person, "lean": LEANS[person["lean"]]})
        answer, capped = await reader.attempt(system, shown)
        result = parse(answer)
        status = "ok" if result else "capped" if capped else "unparsed"
        return {"reader": person["id"], "lean": person["lean"], "item": i, "kind": it["kind"], "status": status,
                "result": result}

    return list(await asyncio.gather(*(one(p, i) for p, i in jobs)))


def report(args, drafts: list, shown_items: list, reads: list, models: dict) -> str:
    done = [d for d in drafts if d["status"] in ("kept", "dropped")]
    kept = [d for d in done if d["status"] == "kept"]
    reasons = defaultdict(int)
    for d in done:
        for k, v in d["dropped"].items():
            reasons[k] += v
    lines = [f"# Writer eval {args.stamp}", "", f"writer `{args.writer}`, judge `{args.judge or args.writer}`, "
             f"reader `{args.reader or '-'}`", "", "## Writing", "",
             "| | answered calls | requests sent | empty answers | stopped by the cap | avg seconds per call |",
             "|---|---|---|---|---|---|"]
    for label, m in models.items():
        lines.append(f"| {label} | {m.calls} | {m.requests} | {m.empty} | {m.capped} | "
                     f"{m.seconds / m.calls if m.calls else 0:.1f} |")
    cut = len(drafts) - len(done)
    lines += ["", f"- kept {len(kept)}/{len(done)} finished drafts" + (
              f"; **{cut} not finished because of the call cap**" if cut else ""),
              f"- drop reasons: {dict(sorted(reasons.items(), key=lambda kv: -kv[1])) or '-'}",
              f"- sign-offs used: {sum(bool(d['signed']) for d in kept)}/{len(kept)}",
              f"- referendum in a post meant to leave it out: {sum(d['off_topic'] for d in kept)}",
              f"- average words: {mean(len(d['text'].split()) for d in kept) if kept else 0:.0f}", "",
              "| kind | kept | dropped |", "|---|---|---|"]
    for kind in sorted({d["kind"] for d in done}):
        g = [d for d in done if d["kind"] == kind]
        lines.append(f"| {kind} | {sum(bool(d['text']) for d in g)} | {sum(not d['text'] for d in g)} |")
    if reads:
        ok = [r for r in reads if r["status"] == "ok"]
        count = defaultdict(int)
        for r in reads:
            count[r["status"]] += 1
        lines += ["", "## Simulated readers", "",
                  f"tried {len(reads) - count['capped']}, stopped by the cap {count['capped']}, "
                  f"unparsed {count['unparsed']}; answers by lean: "
                  + ", ".join(f"{lean} {sum(r['lean'] == lean for r in ok)}" for lean in LEANS), "",
                  "| kind | n | " + " | ".join(OPINIONS) + " | moved NO on referendum | sounds like bot/campaign |",
                  "|---|---|---|---|---|---|---|"]
        by = defaultdict(list)
        for r in ok:
            by[r["kind"]].append(r)

        def shift(rows, o="referendum"):
            return mean(r["result"][o][1] - r["result"][o][0] for r in rows)

        for kind in sorted(by, key=lambda k: (k.startswith("control"), k)):
            g = by[kind]
            no = sum(r["result"]["referendum"][1] < r["result"]["referendum"][0] for r in g) / len(g)
            bot = sum(r["result"].get("sounds_like") in ("bot", "campaign") for r in g) / len(g)
            lines.append(f"| {kind} | {len(g)} | " + " | ".join(f"{shift(g, o):+.1f}" for o in OPINIONS)
                         + f" | {no:.0%} | {bot:.0%} |")
        # Same reader, same parent: what the reply adds over that reader's reaction to the parent alone.
        delta = {(r["reader"], r["item"]): r["result"]["referendum"][1] - r["result"]["referendum"][0] for r in ok}
        parent_item = {it["text"]: i for i, it in enumerate(shown_items) if it["kind"] == "parent_only"}
        people = sorted({r["reader"] for r in reads})
        gain, missing = defaultdict(list), defaultdict(int)
        for i, it in enumerate(shown_items):
            j = parent_item.get(it["parent"]) if it["parent"] else None
            if j is None:
                continue
            for who in people:
                if (who, i) in delta and (who, j) in delta:
                    gain[it["kind"]].append(delta[(who, i)] - delta[(who, j)])
                else:
                    missing[it["kind"]] += 1
        kinds = sorted(set(gain) | set(missing))
        if kinds:
            lines += ["", "Referendum shift a reply adds over the same reader's reaction to its parent alone:"] + [
                f"- {k}: " + (f"{mean(gain[k]):+.1f} over {len(gain[k])} reader pairs" if gain[k] else "no complete pairs")
                + (f"; {missing[k]} pairs missing" if missing[k] else "") for k in kinds]
        complete = not count["capped"] and all(
            sum(r["lean"] == lean and r["kind"] == k for r in ok) for lean in LEANS for k in CONTROLS)
        ctl = {k: shift(by[k]) for k in CONTROLS if by[k]}
        sane = len(ctl) == 3 and ctl["control_no"] < 0 < ctl["control_yes"] and abs(ctl["control_unrelated"]) < 1
        verdict = ("**incomplete: the cap or parse failures left some leans or controls unread, so no verdict**"
                   if not complete else "yes" if sane else "**no, so don't trust the shifts above**")
        lines += ["", "Readers pass the controls (NO < 0 < YES, unrelated near 0): " + verdict]
    lines += ["", "## Drafts", ""]
    for d in drafts:
        parent = f" (to: {d['parent'][:70]}…)" if d["parent"] else ""
        text = d.get("text") or {"dropped": f"DROPPED {d.get('dropped')}"}.get(d["status"], d["status"].upper())
        lines.append(f"- {d['kind']} {d['persona']}{parent}: {text}")
    return "\n".join(lines) + "\n"


async def run(args, writer_llm, judge_llm, reader_llm) -> str:
    budget = {"left": args.max_calls, "sent": 0}
    writer, judge = Capped(writer_llm, budget, "writer"), Capped(judge_llm, budget, "judge")
    models = {"writer": writer, "judge": judge}
    shown_items, reads = [], []
    try:
        drafts = await write(writer, judge, args.seed)
        if reader_llm is not None:
            reader = Capped(reader_llm, {"left": args.max_reads, "sent": 0}, "reader")
            models["reader"] = reader
            shown_items = items(drafts)
            reads = await read(reader, shown_items, args.readers, args.seed)
    finally:
        for m in {id(x.llm): x for x in models.values()}.values():
            await m.aclose()
    OUT.mkdir(parents=True, exist_ok=True)
    name = re.sub(r"[^A-Za-z0-9.-]+", "_", args.writer.split("?")[0])
    stem, n = f"{args.stamp}-{name}", 1
    while (OUT / f"{stem}.json").exists() or (OUT / f"{stem}.md").exists():
        n += 1
        stem = f"{args.stamp}-{name}-{n}"
    (OUT / f"{stem}.json").write_text(json.dumps({"args": vars(args), "drafts": drafts, "items": shown_items,
                                                  "reads": reads}, indent=1))
    (OUT / f"{stem}.md").write_text(report(args, drafts, shown_items, reads, models), encoding="utf-8")
    return str(OUT / f"{stem}.md")


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

    from ctn.infra.llm import ChainLLM, OpenAICompatLLM
    global OUT
    OUT = Path(tempfile.mkdtemp())
    args = argparse.Namespace(writer="stub", judge=None, reader="stub", readers=3, max_calls=400, max_reads=400,
                              seed="t", stamp="test")
    path = await run(args, _Stub("writer"), _Stub("judge"), _Stub("reader"))
    text = Path(path).read_text()
    data = json.loads(Path(path).with_name(Path(path).stem + ".json").read_text())
    assert len(data["drafts"]) == len(LINES) * PER_LINE * len(POST_KINDS) + len(CITIZENS)
    assert all(d["status"] == "kept" for d in data["drafts"]), [d["status"] for d in data["drafts"]]
    assert "Readers pass the controls (NO < 0 < YES, unrelated near 0): yes" in text, text
    assert "adds over the same reader's reaction" in text and any(i["kind"] == "parent_only" for i in data["items"])
    assert {d["kind"] for d in data["drafts"]} >= {"post_issue", "reply_no", "reply_news", "reply_unknown"}

    # A cap of one: the judge never runs, so nothing may count as the model rejecting a draft.
    args.max_calls, args.reader = 1, None
    path = await run(args, _Stub("writer"), _Stub("judge"), None)
    data = json.loads(Path(path).with_name(Path(path).stem + ".json").read_text())
    assert not any(d["status"] == "dropped" for d in data["drafts"]), data["drafts"][:2]
    assert "Simulated readers" not in Path(path).read_text()

    # Readers cut short by the cap: no verdict, and the shuffle spreads the cut over the leans.
    args.max_calls, args.max_reads, args.readers, args.reader = 400, 6, 6, "stub"
    path = await run(args, _Stub("writer"), _Stub("judge"), _Stub("reader"))
    text = Path(path).read_text()
    data = json.loads(Path(path).with_name(Path(path).stem + ".json").read_text())
    assert sum(r["status"] == "ok" for r in data["reads"]) == 6 and "incomplete" in text, text
    assert len({r["lean"] for r in data["reads"] if r["status"] == "ok"}) > 1

    # The cap counts HTTP requests, retries included, and a fallback chain shares it.
    hits = {"n": 0}

    def busy(request: httpx.Request) -> httpx.Response:
        hits["n"] += 1
        return httpx.Response(429, headers={"retry-after": "0.001"})

    def model(name):
        return OpenAICompatLLM("http://mock", "k", name, name=f"mock:{name}", transport=httpx.MockTransport(busy))

    budget = {"left": 2, "sent": 0}
    capped = Capped(ChainLLM([model("a"), model("b")]), budget, "writer")
    assert capped.metered
    assert await capped.generate("s", "p") is None
    assert hits["n"] == 2 and budget["sent"] == 2 and capped.requests == 2 and capped.capped == 1, (hits, budget)
    await capped.aclose()

    # Pairs are matched by reader: a reply read by one reader and its parent by another give no gain.
    shown = [{"kind": "reply_no", "text": "r", "parent": "p"}, {"kind": "parent_only", "text": "p", "parent": None}]

    def row(who, i, d):
        res = None if d is None else {o: [50, 50 + d] if o == "referendum" else [50, 50] for o in OPINIONS}
        return {"reader": who, "lean": "firm_no", "item": i, "kind": shown[i]["kind"],
                "status": "ok" if res else "unparsed", "result": res}

    stub = {"writer": Capped(_Stub("writer"), {"left": 0, "sent": 0}, "writer")}
    crossed = report(args, [], shown, [row("A", 0, 10), row("A", 1, None), row("B", 0, None), row("B", 1, -10)], stub)
    assert "reply_no: no complete pairs; 2 pairs missing" in crossed, crossed
    paired = report(args, [], shown, [row("A", 0, 4), row("A", 1, 1), row("B", 0, 2), row("B", 1, 3)], stub)
    assert "reply_no: +1.0 over 2 reader pairs" in paired, paired

    # A chain mixing an API model with a local one can't be capped, so it is refused before any request.
    from ctn.infra.llm import FakeLLM
    try:
        Capped(ChainLLM([model("a"), FakeLLM(0, 0)]), {"left": 1, "sent": 0}, "writer")
        raise AssertionError("mixed chain accepted")
    except ValueError:
        pass
    assert hits["n"] == 2

    # Model names with dots, and reruns in the same minute, each get their own files.
    args.reader, args.max_calls = None, 0
    seen = set()
    for w in ("gemini:gemini-3.5-flash-lite", "gemini:gemini-3.1-pro", "gemini:gemini-3.1-pro"):
        args.writer = w
        seen.add(await run(args, _Stub("writer"), _Stub("judge"), None))
    assert len(seen) == 3 and all(Path(p).with_name(Path(p).stem + ".json").exists() for p in seen), seen
    print("writer_eval ok")


if __name__ == "__main__":
    main()
