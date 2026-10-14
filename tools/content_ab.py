"""Content A/B on simulated readers (S3). The same personas and topics are written three ways (a personal
story, an official fact, or both); each post is then read on its own, in a fresh conversation, by NO-leaning
and undecided simulated citizens who are never told which version they got.

Only for weeding out versions that put readers off, slip out of character or get facts wrong: simulated
readers are not the competition's NPCs, a guard pass is not a fact check, and a writer and readers on the same
model share its biases. Fake models by default; a real run needs explicit models and stays inside hard caps
on calls, tokens and cost, never falls back to another model, and resumes from its cache if stopped. Real
calls share the provider's daily quota with the team, so don't run them during the competition.

    python3 -m tools.content_ab plan                    # what a run would call; nothing is sent
    python3 -m tools.content_ab run                     # fake writer and readers
    python3 -m tools.content_ab run --writer gemini:gemma-4-31b-it --judge gemini:gemini-3.5-flash-lite
    python3 -m tools.content_ab run --reveal-holdout    # also show the held-out readers (logged)
    python3 -m tools.content_ab test
"""
import argparse
import asyncio
import hashlib
import json
import random
import re
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ctn import paths
from ctn.infra.fake_platform import FakePlatform
from ctn.infra.llm import PRICES, PROVIDERS, ledger, make_llm
from ctn.infra.state import StateStore
from ctn.red.bot import Bot
from ctn.text.content import load_personas, load_world
from ctn.text.guard import output_problem, wrap_external

OUT = paths.DATA / "ab"
NO_CREDENTIALS = "Don't claim any qualification, job or expertise beyond who you are above."
VERSIONS = {
    "story": "Ground the post in one thing that happened to you or someone close to you, true to who you are "
             "above. No statistics and no official sources.",
    "fact": "Ground the post in one fact from the setting above, as it is stated there, and say who it comes from "
            "(the IHO, the vaccine trials, the doctors' union...). No personal story, and no numbers or details "
            "the setting doesn't give.",
    "both": "Use one short personal detail true to who you are above, and one fact from the setting above as it "
            "is stated there, saying who it comes from. No numbers or details the setting doesn't give.",
}
LEANS = {
    "no": "You are leaning NO on the referendum: you got your covid shots, but a vaccine pass at the door of the "
          "shops sits wrong with you.",
    "undecided": "You haven't decided how to vote in the referendum: you remember how bad 2021 was, but you're "
                 "wary of the mandate.",
}
# What the NO campaign and the setting give people to worry about.
CONCERNS = [
    "a two-tier society, with some people shut out of shops and schools",
    "people who can't work or keep a business going if they can't get in",
    "tourism and the jobs that depend on it",
    "a rule this big for a disease that isn't spreading here yet",
    "the vaccine being new, with its long-term effects still being studied",
    "whether it should be a personal choice rather than the government's",
]
READER = (
    "{world}\n\n# Who you are\nYou are {name}, {age}, from {home}. You {job}.\n{lean}\n"
    "What weighs on you most: {concern}.\n\n# What to do\n"
    "You are scrolling Legit Social and come across the post below. React as yourself, honestly: a post can "
    "move you a little, not at all, or push you the other way.\n"
    'Answer with JSON only: {{"before": n, "after": n, "annoyed": true or false, '
    '"sounds_like": "person" or "bot" or "campaign", "why": "one short sentence"}}. "before" and "after" are '
    "the chance, from 0 to 100, that you would vote YES in the referendum, before and after reading it; a post "
    "can move it by a point or two, or not at all."
)
JSON_RE = re.compile(r"\{.*\}", re.S)
PROMPT_VERSION = hashlib.sha256(json.dumps([VERSIONS, NO_CREDENTIALS, LEANS, CONCERNS, READER]).encode()
                                ).hexdigest()[:10]


class Stop(Exception):
    """A cap was reached; what was done so far is cached and a later run carries on from it."""


def check_spec(spec: str, max_cost: float, allow_unpriced: bool) -> None:
    """Refuse fallback chains, and paid or unpriced models the caps can't keep inside a budget."""
    if "," in spec:
        raise ValueError(f"{spec!r}: no fallback chains in an experiment; name one model")
    name, provider = spec.partition("?")[0], spec.partition(":")[0]
    if provider in ("fake", "ollama"):
        return
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider in {spec!r}")
    price = PRICES.get(name)
    if price is None and not allow_unpriced:
        raise ValueError(f"{name} has no known price, so its cost can't be capped; pass --allow-unpriced")
    if price and any(price) and max_cost <= 0:
        raise ValueError(f"{name} is paid; set --max-cost to allow it")


class Caller:
    """Every model call goes through here: answered from the cache if made before, otherwise counted against
    the caps first and cached as soon as it returns. A call that got no answer isn't cached, so a later run
    asks again."""

    def __init__(self, cache_path: Path, max_calls: int, max_tokens: int, max_cost: float, models: list) -> None:
        self.cache_path = cache_path
        self.cache = {}
        if cache_path.exists():
            for line in cache_path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                if row["text"] is not None:  # written by an earlier version of this tool
                    self.cache[row["key"]] = row["text"]
        self.max_calls, self.max_tokens, self.max_cost = max_calls, max_tokens, max_cost
        self.calls = 0
        self.models = models
        self.since = datetime.now(timezone.utc).isoformat(timespec="seconds")

    def used(self) -> tuple:
        if not self.models:
            return 0, 0.0
        rows = [r for r in ledger().totals(self.since) if r[0] in self.models]
        return sum(r[2] + r[3] for r in rows), sum(r[5] or 0.0 for r in rows)

    async def __call__(self, llm, system: str, prompt: str, same_as: Optional[str] = None) -> Optional[str]:
        """`same_as` stands in for the prompt in the cache key when the prompt varies for nothing (a random
        fence around a post)."""
        name = getattr(llm, "name", "model")
        key = hashlib.sha256(f"{name}\0{system}\0{prompt if same_as is None else same_as}".encode()).hexdigest()[:32]
        if key in self.cache:
            return self.cache[key]
        tokens, cost = self.used()
        if self.calls >= self.max_calls or tokens >= self.max_tokens or cost > self.max_cost:
            raise Stop(f"cap reached: {self.calls} calls, {tokens} tokens, ${cost:.2f}")
        self.calls += 1
        text = await llm.generate(system, prompt)
        if text is None:
            return None
        self.cache[key] = text = str(text)
        with self.cache_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "model": name, "text": text}) + "\n")
        return text


def scenarios(n: int, seed: str) -> list:
    """(persona, angle) pairs from the accounts that argue the referendum."""
    pool = sorted((p for p in load_personas(status=None) if p["line"] in ("persuade", "bridge")),
                  key=lambda p: p["username"])
    rng = random.Random(f"{seed}:scenarios")
    rng.shuffle(pool)
    return [(p, rng.choice(p["angles"])) for p in (pool * n)[:n]]


def readers(n: int, seed: str) -> list:
    """Simulated citizens, alternately leaning NO and undecided, each with one worry from the NO side."""
    with open(paths.CONTENT / "archetypes.json", encoding="utf-8") as f:
        arch = json.load(f)
    out = []
    for i in range(n):
        rng = random.Random(f"{seed}:reader:{i}")
        a = rng.choice(arch["archetypes"])
        home = rng.choice(a["homes"]) if a.get("homes") else rng.choices(list(arch["homes"]),
                                                                          list(arch["homes"].values()))[0]
        out.append({"id": f"r{i:02d}", "name": f"{rng.choice(arch['first_names'])} {rng.choice(arch['last_names'])}",
                    "age": rng.randint(*a["ages"]), "home": home, "job": a["job"],
                    "lean": "no" if i % 2 == 0 else "undecided", "concern": CONCERNS[i % len(CONCERNS)]})
    return out


def parse(answer: Optional[str]) -> Optional[dict]:
    m = JSON_RE.search(answer or "")
    try:
        v = json.loads(m.group(0)) if m else None
        before, after = int(v["before"]), int(v["after"])
    except (ValueError, TypeError, KeyError):
        return None
    if not (0 <= before <= 100 and 0 <= after <= 100) or v.get("sounds_like") not in ("person", "bot", "campaign"):
        return None
    return {"shift": after - before, "annoyed": bool(v.get("annoyed")), "sounds_like": v["sounds_like"],
            "why": str(v.get("why", ""))[:200]}


async def run(writer, judge, out: Path, n_scenarios: int, n_readers: int, holdout: int, seed: str, caller: Caller,
              reveal: bool = False, concurrency: int = 4) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    world, people = load_world(), readers(n_readers, seed)
    posts = []
    for i, (persona, angle) in enumerate(scenarios(n_scenarios, seed)):
        bot = Bot(persona, FakePlatform(), writer, world, store=StateStore(":memory:"))
        random.seed(f"{seed}:examples:{i}")  # the persona examples shown, the same for all three versions
        system = bot._system_prompt(*bot._examples(angle, reply=False))
        base = bot._user_prompt("post", None, ("issue", angle))
        for version, how in VERSIONS.items():
            raw = await caller(writer, system, f"{base}\n\n{how} {NO_CREDENTIALS}")
            text = bot._clean(raw)
            problem = "no answer" if raw is None else "cleaned away" if text is None else output_problem(text, system)
            posts.append({"scenario": i, "persona": persona["username"], "angle": angle, "version": version,
                          "text": text, "problem": problem})
    if all(p["problem"] == "no answer" for p in posts):
        raise Stop(f"{getattr(writer, 'name', 'the writer')} answered none of {len(posts)} prompts; check the model")
    jobs = [(p, r) for p in posts if p["problem"] is None for r in people]
    random.Random(f"{seed}:order").shuffle(jobs)
    sem = asyncio.Semaphore(concurrency)

    async def read(post: dict, reader: dict) -> dict:
        system = READER.format(world=world, **{**reader, "lean": LEANS[reader["lean"]]})
        async with sem:
            answer = await caller(judge, system, wrap_external("someone", post["text"]), same_as=post["text"])
        return {"scenario": post["scenario"], "version": post["version"], "reader": reader["id"],
                "held_out": int(reader["id"][1:]) >= n_readers - holdout, "result": parse(answer)}

    tasks = [asyncio.ensure_future(read(p, r)) for p, r in jobs]
    try:
        judged = await asyncio.gather(*tasks)
    except BaseException:  # a cap: don't leave the other readers running
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    results = {"prompt_version": PROMPT_VERSION, "seed": seed, "writer": getattr(writer, "name", "?"),
               "judge": getattr(judge, "name", "?"), "posts": posts, "readers": people, "judged": judged}
    (out / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    write_blind(posts, out, seed)
    (out / "report.md").write_text(report(results, reveal), encoding="utf-8")
    return results


def summary(posts: list, judged: list) -> dict:
    by = {}
    for v in VERSIONS:
        mine = [j for j in judged if j["version"] == v]
        ok = [j["result"] for j in mine if j["result"] is not None]
        by[v] = {"written": sum(p["version"] == v for p in posts),
                 "guard": Counter(p["problem"] for p in posts if p["version"] == v and p["problem"]),
                 "judged": len(mine), "unreadable": len(mine) - len(ok),
                 "mean_shift": round(sum(r["shift"] for r in ok) / len(ok), 2) if ok else None,
                 "backfire": sum(r["shift"] < 0 for r in ok), "annoyed": sum(r["annoyed"] for r in ok),
                 "bot_like": sum(r["sounds_like"] == "bot" for r in ok),
                 "campaign_like": sum(r["sounds_like"] == "campaign" for r in ok)}
    return by


def table(by: dict) -> str:
    rows = ["| version | written | guard failures | judged | unreadable | mean shift (/100) | backfire | annoyed | "
            "bot-like | campaign-like |", "|---|---|---|---|---|---|---|---|---|---|"]
    for v, s in by.items():
        rows.append(f"| {v} | {s['written']} | {dict(s['guard']) or '-'} | {s['judged']} | {s['unreadable']} | "
                    f"{s['mean_shift']} | {s['backfire']} | {s['annoyed']} | {s['bot_like']} | {s['campaign_like']} |")
    return "\n".join(rows)


def report(results: dict, reveal: bool) -> str:
    dev = [j for j in results["judged"] if not j["held_out"]]
    held = [j for j in results["judged"] if j["held_out"]]
    lines = [f"# Content A/B, prompts {results['prompt_version']}, seed {results['seed']}",
             f"Writer {results['writer']}, readers {results['judge']}. Simulated readers, not the competition's "
             "NPCs; a guard pass is not a fact check; the same model as writer and reader shares its biases. "
             "Use it to drop versions that backfire, annoy, read like a bot or fail the guard, not to predict "
             "points. Facts and persona consistency need a human read (blind.md).",
             "", "## Development readers", table(summary(results["posts"], dev))]
    if reveal:
        log = OUT / "holdout_reveals.jsonl"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                "prompt_version": results["prompt_version"]}) + "\n")
        seen = sum(json.loads(line)["prompt_version"] == results["prompt_version"]
                   for line in log.read_text(encoding="utf-8").splitlines())
        lines += ["", f"## Held-out readers (revealed {seen} time(s) for these prompts; don't tune on it)",
                  table(summary(results["posts"], held))]
    else:
        lines += ["", f"Held-out readers: {len(held)} judgments, hidden; pass --reveal-holdout to see them (logged)."]
    return "\n".join(lines) + "\n"


def write_blind(posts: list, out: Path, seed: str) -> None:
    """Every publishable post, shuffled, without its version, for a human to rate; the key goes elsewhere."""
    shown = [p for p in posts if p["problem"] is None]
    random.Random(f"{seed}:blind").shuffle(shown)
    personas = {p["username"]: p for p in load_personas(status=None)}
    lines = ["# Blind rating", "For each post: persuasive to an undecided person (1-5)? natural for this person "
             "(1-5)? any fact the setting doesn't give, or anything out of character? Don't open key.json first.", ""]
    key = {}
    for n, p in enumerate(shown, 1):
        who = personas[p["persona"]]
        key[f"P{n:02d}"] = p["version"]
        lines += [f"**P{n:02d}** {who['name']}, {who['age']}, {who['home']}; {who['job']}. Topic: {p['angle']}",
                  f"> {p['text']}", "", "persuasive: _ natural: _ problems: _", ""]
    (out / "blind.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "key.json").write_text(json.dumps(key, indent=1), encoding="utf-8")


def plan(n_scenarios: int, n_readers: int, cache: Caller, writer_name: str, judge_name: str) -> dict:
    """Upper bounds, assuming every post passes the guard; cached calls are free."""
    return {"writer_calls": n_scenarios * len(VERSIONS),
            "reader_calls": n_scenarios * len(VERSIONS) * n_readers, "cached": len(cache.cache),
            "writer": writer_name, "judge": judge_name}


class FakeWriter:
    """Writes by the version asked for, plus one post that fails the guard, so the pipeline is tested offline."""
    name = "fake:writer"

    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, system: str, prompt: str) -> Optional[str]:
        self.calls += 1
        angle = re.search(r"along this line: (.*?)\.\n", prompt + "\n")
        topic = angle.group(1)[:60] if angle else "the mandate"
        if self.calls == 3:
            return "More at www.example.com"
        if VERSIONS["story"] in prompt:
            return f"My mum spent a week on a ward in 2021, which is why {topic} matters to me."
        if VERSIONS["fact"] in prompt:
            return f"The IHO recommends vaccinating before the virus arrives; {topic}."
        return f"I lost shifts in 2021, and the IHO says vaccinate early, so {topic}."


class FakeReader:
    """Answers from a hash of the reader and the post; some answers are unreadable on purpose."""
    name = "fake:reader"

    def __init__(self) -> None:
        self.calls, self.prompts = 0, []

    async def generate(self, system: str, prompt: str) -> Optional[str]:
        self.calls += 1
        self.prompts.append((system, prompt))
        h = int(hashlib.sha256((system + prompt).encode()).hexdigest(), 16)
        if h % 11 == 0:
            return "I'd rather not say."
        before = 20 if "leaning NO" in system else 50
        return json.dumps({"before": before, "after": before + h % 21 - 10, "annoyed": h % 5 == 0,
                           "sounds_like": ("person", "bot", "campaign")[h % 7 % 3], "why": "fake"})


async def _test() -> None:
    def args_for(tmp: Path):
        return dict(out=tmp / "run", n_scenarios=4, n_readers=6, holdout=2, seed="t")

    # Specs a real run must refuse, before anything is sent.
    for spec, cost, unpriced in (("gemini:gemini-3.5-flash-lite,cerebras:gpt-oss-120b", 0, False),
                                 ("cerebras:gpt-oss-120b", 0, False), ("official:x", 5, False)):
        try:
            check_spec(spec, cost, unpriced)
            raise AssertionError(f"{spec} allowed")
        except ValueError:
            pass
    for spec in ("fake", "gemini:gemma-4-31b-it", "ollama:llama3.1:8b"):
        check_spec(spec, 0, False)
    check_spec("cerebras:gpt-oss-120b", 1.0, False)

    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        global OUT
        OUT = tmp / "ab"
        w, r = FakeWriter(), FakeReader()
        caller = Caller(tmp / "cache.jsonl", 10_000, 10**9, 0.0, [])
        assert plan(4, 6, caller, w.name, r.name) == {"writer_calls": 12, "reader_calls": 72, "cached": 0,
                                                      "writer": "fake:writer", "judge": "fake:reader"}
        assert w.calls == r.calls == 0, "planning sends nothing"
        res = await run(w, r, caller=caller, **args_for(tmp))
        bad = [p for p in res["posts"] if p["problem"]]
        assert w.calls == 12 and len(bad) >= 1 and all(p["problem"] == "link" for p in bad), bad
        assert r.calls == (12 - len(bad)) * 6 == len(res["judged"])
        for system, prompt in r.prompts:  # the readers never learn the version, and each sees one post
            assert not any(how in system + prompt for how in VERSIONS.values()) and prompt.count("<<<POST") == 1
        by = summary(res["posts"], res["judged"])
        assert sum(s["unreadable"] for s in by.values()) == sum(j["result"] is None for j in res["judged"]) > 0
        assert sum(s["guard"]["link"] for s in by.values()) == len(bad)
        text = (tmp / "run" / "report.md").read_text()
        assert "hidden" in text and "Held-out readers (revealed" not in text
        key = json.loads((tmp / "run" / "key.json").read_text())
        blind = (tmp / "run" / "blind.md").read_text()
        assert len(key) == 12 - len(bad) and not any(f"| {v} " in blind or v + ":" in blind for v in VERSIONS)

        # The same run again is answered from the cache: no new calls, the same results.
        w2, r2 = FakeWriter(), FakeReader()
        again = await run(w2, r2, caller=Caller(tmp / "cache.jsonl", 10_000, 10**9, 0.0, []), reveal=True,
                          **args_for(tmp))
        assert w2.calls == r2.calls == 0 and again["judged"] == res["judged"]
        assert "Held-out readers (revealed 1 time(s)" in (tmp / "run" / "report.md").read_text()

        # Stopped at a cap partway, then resumed: every call made exactly once.
        w3, r3 = FakeWriter(), FakeReader()
        capped = Caller(tmp / "cache2.jsonl", 20, 10**9, 0.0, [])
        try:
            await run(w3, r3, caller=capped, **args_for(tmp))
            raise AssertionError("the cap didn't stop the run")
        except Stop:
            pass
        assert capped.calls == 20 == w3.calls + r3.calls
        w4, r4 = FakeWriter(), FakeReader()
        await run(w4, r4, caller=Caller(tmp / "cache2.jsonl", 10_000, 10**9, 0.0, []), **args_for(tmp))
        assert w3.calls + w4.calls == 12 and r3.calls + r4.calls == (12 - len(bad)) * 6

        # No answer isn't cached: a writer that answers nothing stops the run, one that misses once is asked again.
        class Silent(FakeWriter):
            def __init__(self, misses) -> None:
                super().__init__()
                self.misses = misses

            async def generate(self, system: str, prompt: str) -> Optional[str]:
                answer = await super().generate(system, prompt)
                return None if self.calls in self.misses else answer

        silent, silent_cache = Silent(range(1, 13)), Caller(tmp / "cache3.jsonl", 10_000, 10**9, 0.0, [])
        try:
            await run(silent, FakeReader(), caller=silent_cache, **args_for(tmp))
            raise AssertionError("a run on no answers went on")
        except Stop:
            pass
        assert silent.calls == 12 and not silent_cache.cache and not (tmp / "cache3.jsonl").exists()
        once = Silent({2})
        first = await run(once, FakeReader(), caller=Caller(tmp / "cache4.jsonl", 10_000, 10**9, 0.0, []),
                          **args_for(tmp))
        assert [p["problem"] for p in first["posts"]].count("no answer") == 1
        again = FakeWriter()
        await run(again, FakeReader(), caller=Caller(tmp / "cache4.jsonl", 10_000, 10**9, 0.0, []), **args_for(tmp))
        assert again.calls == 1, again.calls
    print("content_ab ok")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cmd", choices=["plan", "run", "test"])
    parser.add_argument("--writer", default="fake")
    parser.add_argument("--judge", default="fake")
    parser.add_argument("--scenarios", type=int, default=6)
    parser.add_argument("--readers", type=int, default=18, help="including the held-out ones")
    parser.add_argument("--holdout", type=int, default=6)
    parser.add_argument("--seed", default="s3")
    parser.add_argument("--max-calls", type=int, default=None, help="default: the plan's upper bound")
    parser.add_argument("--max-tokens", type=int, default=400_000)
    parser.add_argument("--max-cost", type=float, default=0.0, help="USD; a paid model needs this above 0")
    parser.add_argument("--allow-unpriced", action="store_true")
    parser.add_argument("--reveal-holdout", dest="reveal", action="store_true")
    args = parser.parse_args()
    if args.cmd == "test":
        asyncio.run(_test())
        return
    for spec in (args.writer, args.judge):
        try:
            check_spec(spec, args.max_cost, args.allow_unpriced)
        except ValueError as e:
            parser.error(str(e))
    writer = FakeWriter() if args.writer == "fake" else make_llm(args.writer, temperature=0.9)
    judge = FakeReader() if args.judge == "fake" else make_llm(args.judge, temperature=0.2)
    names = [m.name for m in (writer, judge) if not isinstance(m, (FakeWriter, FakeReader))]
    cache_path = OUT / "cache.jsonl"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    caller = Caller(cache_path, 0, args.max_tokens, args.max_cost, names)
    p = plan(args.scenarios, args.readers, caller, getattr(writer, "name", "?"), getattr(judge, "name", "?"))
    caller.max_calls = args.max_calls if args.max_calls is not None else p["writer_calls"] + p["reader_calls"]
    print(f"plan (prompts {PROMPT_VERSION}): {p}; caps: {caller.max_calls} calls, {args.max_tokens} tokens, "
          f"${args.max_cost:.2f}")
    if args.cmd == "plan":
        return
    out = OUT / f"{datetime.now():%Y%m%d-%H%M%S}-{PROMPT_VERSION}"
    try:
        asyncio.run(run(writer, judge, out, args.scenarios, args.readers, args.holdout, args.seed, caller,
                        reveal=args.reveal))
    except Stop as e:
        print(f"stopped: {e}; run the same command again to carry on from the cache")
        return
    print(f"{caller.calls} new calls; report in {out / 'report.md'}, blind sheet in {out / 'blind.md'}")


if __name__ == "__main__":
    main()
