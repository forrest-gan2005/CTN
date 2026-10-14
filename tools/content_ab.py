"""Content A/B on simulated readers (S3). The same personas and topics are written three ways (a personal
story, an official fact, or both); each post is then read on its own, in a fresh conversation, by NO-leaning
and undecided simulated citizens who are never told which version they got. Fixed control posts (one that
answers the reader's worry, one about nothing, one that repeats a NO view) show whether the readers' answers
can tell posts apart at all.

Only for weeding out versions that put readers off, slip out of character or get facts wrong: simulated
readers are not the competition's NPCs, "addressed" and "action" are self-reports, a guard pass is not a fact
check, and a writer and readers on the same model share its biases. Fake models by default. A real run names
one model each (no fallbacks). The call cap is hard. The token and cost caps are estimated thresholds, not
guarantees: a call is sent only if what the ledger says the run has used, plus an estimate for every call
still in flight and for this one (a token per byte of its prompt, 64 more for the chat template, and the
model's max_tokens in full), stays within them. A provider that counts more input than that (its own
template, hidden tokens) or more output than max_tokens can take a run over, by up to what the estimates of
the calls already in flight missed; the next call then stops. A model without a known price has no cost
cap. Answers are cached, so a stopped run carries on where it was. Real calls share the provider's daily
quota with the team, so don't run them during the competition.

    python3 -m tools.content_ab plan                    # what a run would call; nothing is sent
    python3 -m tools.content_ab run                     # fake writer and readers
    python3 -m tools.content_ab run --writer gemini:gemini-3.5-flash-lite --judge gemini:gemini-3.5-flash-lite
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
from typing import Callable, Optional

from ctn import paths
from ctn.infra.fake_platform import FakePlatform
from ctn.infra.llm import PRICES, PROVIDERS, ledger, make_llm
from ctn.infra.state import StateStore
from ctn.red.bot import Bot
from ctn.text.content import load_personas, load_world
from ctn.text.guard import output_problem, wrap_external

OUT = paths.DATA / "ab"
VERSIONS = {
    "story": "Ground the post in one thing that happened to you or someone close to you, true to who you are "
             "above. No statistics and no official sources.",
    "fact": "Ground the post in one fact from the setting above, as it is stated there, and say who it comes from "
            "(the IHO, the vaccine trials, the doctors' union...). No personal story, and no numbers or details "
            "the setting doesn't give.",
    "both": "Use one short personal detail true to who you are above, and one fact from the setting above as it "
            "is stated there, saying who it comes from. No numbers or details the setting doesn't give.",
}
# The same words after every version, so none of them is told more than the others.
COMMON = ("Say it in your own words; don't reuse the wording above. "
          "Don't claim any qualification, job or expertise beyond who you are above.")
LEANS = {
    "no": "You are leaning NO on the referendum: you got your covid shots, but a vaccine pass at the door of the "
          "shops sits wrong with you.",
    "undecided": "You haven't decided how to vote in the referendum: you remember how bad 2021 was, but you're "
                 "wary of the mandate.",
}
# What the NO campaign and the setting give people to worry about, each with a post that answers it.
CONCERNS = {
    "a two-tier society, with some people shut out of shops and schools":
        "Nobody's forced to get it, and it only covers designated public spaces. A door check beats another "
        "lockdown, which shuts everyone out.",
    "people who can't work or keep a business going if they can't get in":
        "I run a cafe. Checking a pass at the door takes ten seconds; a lockdown closed us for months in 2021. "
        "That's why I'm voting yes.",
    "tourism and the jobs that depend on it":
        "Tourists don't book trips into an outbreak. Keeping H2D2 out keeps the bookings coming, so I'm voting yes.",
    "a rule this big for a disease that isn't spreading here yet":
        "The IHO says vaccinate before it arrives, not after. Once it's spreading here it's too late to get "
        "ahead of it.",
    "the vaccine being new, with its long-term effects still being studied":
        "The trials showed a strong immune response and no major safety problems so far. Not everything is known "
        "yet, but I'd rather that than covid again.",
    "whether it should be a personal choice rather than the government's":
        "Nobody is physically forced. You can still say no; you'd stay out of designated public spaces, that's all.",
}
CONTROLS = {"answers": None,  # the post for the reader's own worry
            "unrelated": "Saw a turtle off New Haven this morning. Made my week.",
            "repeats": "A vaccine pass just to buy groceries? Not for me. Voting NO."}
READER = (
    "{world}\n\n# Who you are\nYou are {name}, {age}, from {home}. You {job}.\n{lean}\n"
    "What weighs on you most: {concern}.\n\n# What to do\n"
    "You are scrolling Legit Social and come across the post below. React as yourself, honestly: a post can "
    "move you a little, not at all, or push you the other way.\n"
    'Answer with one JSON object and nothing else: {{"before": n, "after": n, "addressed": "yes" or "partly" or '
    '"no", "action": "like" or "reply" or "scroll", "annoyed": true or false, "sounds_like": "person" or "bot" '
    'or "campaign", "why": "one short sentence"}}. "before" and "after" are whole numbers: the chance, from 0 '
    "to 100, that you would vote YES in the referendum, before and after reading it; a post can move it by a "
    'point or two, or not at all. "addressed" is whether the post speaks to what weighs on you most; "action" '
    "is what you would do with it."
)
ENUMS = {"addressed": ("yes", "partly", "no"), "action": ("like", "reply", "scroll"),
         "sounds_like": ("person", "bot", "campaign")}
FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.S)
PROMPT_VERSION = hashlib.sha256(json.dumps([VERSIONS, COMMON, LEANS, CONCERNS, CONTROLS, READER]).encode()
                                ).hexdigest()[:10]


TEMPLATE_TOKENS = 64  # roles and markers a chat template adds around the messages, by estimate


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
    """Every model call goes through here: answered from the cache if made before (free), otherwise admitted
    only if the caps hold with this call and every call in flight counted at their estimate, and cached as
    soon as it answers. A call that got no answer isn't cached, so a later run asks again. `totals` reads the usage
    ledger (UsageLedger.totals); spend from earlier runs on the same cache is kept beside it and reported."""

    def __init__(self, cache_path: Path, max_calls: int, max_tokens: int, max_cost: float,
                 totals: Optional[Callable[[str], list]] = None) -> None:
        self.cache_path = cache_path
        self.cache = {}
        if cache_path.exists():
            for line in cache_path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                if row["text"] is not None:  # written by an earlier version of this tool
                    self.cache[row["key"]] = row["text"]
        self.spend_path = cache_path.with_name(cache_path.stem + ".spend.json")
        self.earlier = (json.loads(self.spend_path.read_text()) if self.spend_path.exists()
                        else {"runs": 0, "calls": 0, "tokens": 0, "cost": 0.0})
        self.max_calls, self.max_tokens, self.max_cost = max_calls, max_tokens, max_cost
        self.totals = totals or (lambda since: ledger().totals(since))
        self.since = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.calls, self.pending_tokens, self.pending_cost = 0, 0, 0.0
        self.models = set()

    def used(self) -> tuple:
        """Tokens and USD this run's models have used since it started, as far as the ledger has heard."""
        rows = [r for r in self.totals(self.since) if r[0] in self.models]
        return sum(r[2] + r[3] for r in rows), sum(r[5] or 0.0 for r in rows)

    @staticmethod
    def reservation(llm, name: str, system: str, prompt: str) -> tuple:
        """An estimate of what one call can use, not a bound: a token per byte of its prompt plus 64 for the
        chat template, and the model's max_tokens in full. Cost is None for a model without a known price."""
        prompt_tokens = len((system + prompt).encode()) + TEMPLATE_TOKENS
        answer = int(getattr(llm, "body", {}).get("max_tokens", 400))
        price = PRICES.get(name)
        cost = None if price is None else (prompt_tokens * price[0] + answer * price[1]) / 1e6
        return prompt_tokens + answer, cost

    def save(self) -> dict:
        tokens, cost = self.used()
        total = {"runs": self.earlier["runs"] + 1, "calls": self.earlier["calls"] + self.calls,
                 "tokens": self.earlier["tokens"] + tokens, "cost": round(self.earlier["cost"] + cost, 6)}
        if self.calls:
            self.spend_path.write_text(json.dumps(total))
        return total

    async def __call__(self, llm, system: str, prompt: str, same_as: Optional[str] = None) -> Optional[str]:
        """`same_as` stands in for the prompt in the cache key when the prompt varies for nothing (a random
        fence around a post)."""
        name = getattr(llm, "name", "model")
        key = hashlib.sha256(f"{name}\0{system}\0{prompt if same_as is None else same_as}".encode()).hexdigest()[:32]
        if key in self.cache:
            return self.cache[key]
        self.models.add(name)
        tokens, cost = self.used()
        want_tokens, want_cost = self.reservation(llm, name, system, prompt)
        if self.calls >= self.max_calls:
            raise Stop(f"call cap: {self.calls} of {self.max_calls}")
        if tokens + self.pending_tokens + want_tokens > self.max_tokens:
            raise Stop(f"token cap: {tokens} used, {self.pending_tokens} in flight, {want_tokens} more asked "
                       f"for, cap {self.max_tokens}")
        if want_cost and cost + self.pending_cost + want_cost > self.max_cost:
            raise Stop(f"cost cap: ${cost:.4f} used, ${self.pending_cost:.4f} in flight, ${want_cost:.4f} more "
                       f"asked for, cap ${self.max_cost:.2f}")
        self.calls += 1
        self.pending_tokens += want_tokens
        self.pending_cost += want_cost or 0.0
        try:
            text = await llm.generate(system, prompt)
        finally:
            self.pending_tokens -= want_tokens
            self.pending_cost -= want_cost or 0.0
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
    worries = list(CONCERNS)
    out = []
    for i in range(n):
        rng = random.Random(f"{seed}:reader:{i}")
        a = rng.choice(arch["archetypes"])
        home = rng.choice(a["homes"]) if a.get("homes") else rng.choices(list(arch["homes"]),
                                                                          list(arch["homes"].values()))[0]
        out.append({"id": f"r{i:02d}", "name": f"{rng.choice(arch['first_names'])} {rng.choice(arch['last_names'])}",
                    "age": rng.randint(*a["ages"]), "home": home, "job": a["job"],
                    "lean": "no" if i % 2 == 0 else "undecided", "concern": worries[i % len(worries)]})
    return out


def parse(answer: Optional[str]) -> Optional[dict]:
    """A reader's answer, or None if it isn't exactly one JSON object (a ```json fence around it is fine) with
    every field present and of the asked-for type: whole numbers 0-100 that aren't true/false, a real JSON
    true or false, the listed words, and a reason."""
    text = (answer or "").strip()
    fenced = FENCE_RE.match(text)
    try:
        v = json.loads(fenced.group(1) if fenced else text)
    except ValueError:
        return None
    if not isinstance(v, dict):
        return None
    numbers = [v.get("before"), v.get("after")]
    if not all(isinstance(n, int) and not isinstance(n, bool) and 0 <= n <= 100 for n in numbers):
        return None
    if not isinstance(v.get("annoyed"), bool) or any(v.get(k) not in ok for k, ok in ENUMS.items()):
        return None
    if not isinstance(v.get("why"), str) or not v["why"].strip():
        return None
    return {"before": v["before"], "after": v["after"], "shift": v["after"] - v["before"], "annoyed": v["annoyed"],
            **{k: v[k] for k in ENUMS}, "why": v["why"][:200]}


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
            raw = await caller(writer, system, f"{base}\n\n{how} {COMMON}")
            text = bot._clean(raw)
            problem = "no answer" if raw is None else "cleaned away" if text is None else output_problem(text, system)
            posts.append({"scenario": i, "persona": persona["username"], "angle": angle, "version": version,
                          "text": text, "problem": problem,
                          "base_hash": hashlib.sha256((system + base).encode()).hexdigest()[:10]})
    if all(p["problem"] == "no answer" for p in posts):
        raise Stop(f"{getattr(writer, 'name', 'the writer')} answered none of {len(posts)} prompts; check the model")
    dev = [r for r in people if int(r["id"][1:]) < n_readers - holdout]
    jobs = [(p, r) for p in posts if p["problem"] is None for r in people]
    jobs += [({"scenario": None, "version": f"control:{kind}", "text": text or CONCERNS[r["concern"]]}, r)
             for kind, text in CONTROLS.items() for r in dev]
    random.Random(f"{seed}:order").shuffle(jobs)
    sem = asyncio.Semaphore(concurrency)

    async def read(post: dict, reader: dict) -> dict:
        system = READER.format(world=world, **{**reader, "lean": LEANS[reader["lean"]]})
        async with sem:
            answer = await caller(judge, system, wrap_external("someone", post["text"]), same_as=post["text"])
        return {"scenario": post["scenario"], "version": post["version"], "reader": reader["id"],
                "held_out": reader not in dev, "result": parse(answer), "raw": answer}

    tasks = [asyncio.ensure_future(read(p, r)) for p, r in jobs]
    try:
        judged = await asyncio.gather(*tasks)
    except BaseException:  # a cap: don't leave the other readers running
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    results = {"prompt_version": PROMPT_VERSION, "seed": seed, "writer": getattr(writer, "name", "?"),
               "judge": getattr(judge, "name", "?"), "date": datetime.now(timezone.utc).date().isoformat(),
               "world_hash": hashlib.sha256(world.encode()).hexdigest()[:10],
               "posts": posts, "readers": people, "judged": judged}
    (out / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    write_blind(posts, out, seed)
    (out / "report.md").write_text(report(results, reveal), encoding="utf-8")
    return results


def summary(posts: list, judged: list, kinds) -> dict:
    by = {}
    for v in kinds:
        mine = [j for j in judged if j["version"] == v]
        ok = [j["result"] for j in mine if j["result"] is not None]
        written = sum(p["version"] == v for p in posts)
        failed = Counter(p["problem"] for p in posts if p["version"] == v and p["problem"])
        by[v] = {"written": written, "publishable": written - sum(failed.values()), "guard": failed,
                 "judged": len(mine), "unreadable": len(mine) - len(ok),
                 "mean_shift": round(sum(r["shift"] for r in ok) / len(ok), 2) if ok else None,
                 "backfire": sum(r["shift"] < 0 for r in ok), "annoyed": sum(r["annoyed"] for r in ok),
                 **{k: Counter(r[k] for r in ok) for k in ENUMS}}
    return by


def table(by: dict, drafts: bool = True) -> str:
    head = "| post | " + ("written | publishable | guard failures | " if drafts else "") + \
        "judged | unreadable | mean shift (/100) | backfire | annoyed | addressed y/p/n | like/reply/scroll | " \
        "person/bot/campaign |"
    rows = [head, "|" + "---|" * (head.count("|") - 1)]
    for v, s in by.items():
        counts = [s[k] for k in ENUMS]
        cells = [v] + ([s["written"], s["publishable"], dict(s["guard"]) or "-"] if drafts else []) + \
            [s["judged"], s["unreadable"], s["mean_shift"], s["backfire"], s["annoyed"]] + \
            ["/".join(str(c[w]) for w in ENUMS[k]) for k, c in zip(ENUMS, counts)]
        rows.append("| " + " | ".join(str(c) for c in cells) + " |")
    return "\n".join(rows)


def report(results: dict, reveal: bool) -> str:
    dev = [j for j in results["judged"] if not j["held_out"]]
    held = [j for j in results["judged"] if j["held_out"]]
    controls = [f"control:{k}" for k in CONTROLS]
    lines = [f"# Content A/B, prompts {results['prompt_version']}, seed {results['seed']}, {results['date']}",
             f"Writer {results['writer']}, readers {results['judge']}. Simulated readers, not the competition's "
             "NPCs; addressed and action are what the readers say, not persuasion or real engagement; a guard pass "
             "is not a fact check; the same model as writer and reader shares its biases. Use it to drop versions "
             "that backfire, annoy, read like a bot or fail the guard, not to predict points. Publishable counts "
             "keep the drafts the guard stopped; facts and persona consistency need a human read (blind.md).",
             "", "## Versions, development readers", table(summary(results["posts"], dev, VERSIONS)),
             "", "## Controls, development readers",
             "If these don't differ (answers > repeats and unrelated on addressed), the readers can't tell posts "
             "apart and the version table says nothing.",
             table(summary([], dev, controls), drafts=False)]
    if reveal:
        log = OUT / "holdout_reveals.jsonl"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                "prompt_version": results["prompt_version"]}) + "\n")
        seen = sum(json.loads(line)["prompt_version"] == results["prompt_version"]
                   for line in log.read_text(encoding="utf-8").splitlines())
        lines += ["", f"## Versions, held-out readers (revealed {seen} time(s) for these prompts; don't tune on it)",
                  table(summary(results["posts"], held, VERSIONS))]
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


def plan(n_scenarios: int, n_readers: int, holdout: int, cache: Caller, writer_name: str, judge_name: str) -> dict:
    """Upper bounds, assuming every post passes the guard; cached calls are free."""
    return {"writer_calls": n_scenarios * len(VERSIONS),
            "reader_calls": n_scenarios * len(VERSIONS) * n_readers + len(CONTROLS) * (n_readers - holdout),
            "cached": len(cache.cache), "writer": writer_name, "judge": judge_name}


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
    """Answers from a hash of the reader and the post, in a fence now and then; some answers are unreadable
    on purpose (prose, or the right fields with a wrong type)."""
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
        answer = {"before": before, "after": before + h % 21 - 10, "addressed": ENUMS["addressed"][h % 3],
                  "action": ENUMS["action"][h % 5 % 3], "annoyed": h % 5 == 0,
                  "sounds_like": ENUMS["sounds_like"][h % 7 % 3], "why": "fake"}
        if h % 13 == 0:
            answer["annoyed"] = "false"
        text = json.dumps(answer)
        return f"```json\n{text}\n```" if h % 4 == 0 else text


async def _budget_test(tmp: Path) -> None:
    """The caps with calls in flight, a cap met exactly, a call bigger than what's left, cached answers, and a
    provider counting more input than estimated: that run goes over, and the next call stops."""
    class Ledger:
        def __init__(self) -> None:
            self.rows = Counter()

        def totals(self, since: str) -> list:
            return [(m, 1, t, 0, 0, t * PRICES[m][0] / 1e6) for m, t in self.rows.items()]

    class Paid:
        """Holds every call until released, then charges 40 tokens: within its max_tokens, as a provider should."""
        name = "test:paid"
        body = {"max_tokens": 50}

        def __init__(self, book: Ledger) -> None:
            self.book, self.calls, self.release = book, 0, asyncio.Event()

        async def generate(self, system: str, prompt: str) -> Optional[str]:
            self.calls += 1
            await self.release.wait()
            self.book.rows[self.name] += 40
            return "ok"

    PRICES["test:paid"] = (10_000.0, 10_000.0)  # a cent a token, so cost caps bite
    try:
        book = Ledger()
        model = Paid(book)
        one = Caller.reservation(model, model.name, "s", "p0")  # (tokens, cost) one call is counted at
        caller = Caller(tmp / "b.jsonl", 100, one[0] * 2, 100.0, totals=book.totals)
        tasks = [asyncio.ensure_future(caller(model, "s", f"p{i}")) for i in range(6)]
        await asyncio.sleep(0.05)
        assert model.calls == 2 and caller.pending_tokens == one[0] * 2, "only what fits is let in, in flight"
        model.release.set()
        outcome = await asyncio.gather(*tasks, return_exceptions=True)
        assert outcome.count("ok") == 2 and sum(isinstance(o, Stop) for o in outcome) == 4
        assert caller.pending_tokens == 0 and book.rows["test:paid"] == 80 <= caller.max_tokens

        # Exactly at the cap, and a call bigger than what's left: refused before the model is asked.
        for used in (one[0] * 2, one[0] * 2 - one[0] + 1):
            book.rows["test:paid"] = used
            capped = Caller(tmp / "b2.jsonl", 100, one[0] * 2, 100.0, totals=book.totals)
            capped.models.add("test:paid")
            before = model.calls
            try:
                await capped(model, "s", "fresh prompt")
                raise AssertionError("over the token cap")
            except Stop:
                assert model.calls == before
        book.rows["test:paid"] = 0
        priced = Caller(tmp / "b3.jsonl", 100, 10**9, one[1] * 3, totals=book.totals)
        priced.models.add("test:paid")
        book.rows["test:paid"] = int(one[1] * 3 * 1e6 / PRICES["test:paid"][0])  # spent exactly the cost cap
        try:
            await priced(model, "s", "another")
            raise AssertionError("at the cost cap")
        except Stop:
            pass

        # Cached answers cost nothing, even with the caps used up; a free model isn't held to a $0 cost cap.
        assert await caller(model, "s", "p0") == "ok"
        free = Caller(tmp / "b4.jsonl", 10, 10**9, 0.0, totals=lambda since: [])
        assert await free(FakeReader(), "s", "p") is not None
        assert free.save()["calls"] == 1 and Caller(tmp / "b4.jsonl", 10, 10**9, 0.0).earlier["calls"] == 1

        # A provider that counts more than the estimate: the cap is a threshold, so this run goes over it, and
        # the next call is refused once the ledger shows the overrun.
        class Greedy(Paid):
            async def generate(self, system: str, prompt: str) -> Optional[str]:
                self.calls += 1
                self.book.rows[self.name] += Caller.reservation(self, self.name, system, prompt)[0] + 10
                return "ok"

        book.rows.clear()
        greedy = Greedy(book)
        estimate = Caller.reservation(greedy, greedy.name, "s", "p")[0]
        loose = Caller(tmp / "b5.jsonl", 10, estimate, 10**6, totals=book.totals)
        assert await loose(greedy, "s", "p") == "ok" and book.rows["test:paid"] == estimate + 10 > loose.max_tokens
        try:
            await loose(greedy, "s", "q")
            raise AssertionError("went on past the cap")
        except Stop:
            assert greedy.calls == 1
    finally:
        del PRICES["test:paid"]


def _parse_test() -> None:
    good = {"before": 30, "after": 32, "addressed": "partly", "action": "scroll", "annoyed": False,
            "sounds_like": "person", "why": "fair point"}
    assert parse(json.dumps(good))["annoyed"] is False and parse(json.dumps(good))["shift"] == 2
    assert parse(f"```json\n{json.dumps(good)}\n```") is not None
    for bad in ({**good, "before": 2.9, "after": 3.9}, {**good, "annoyed": "false"}, {**good, "before": True},
                {k: v for k, v in good.items() if k != "annoyed"}, {**good, "after": None}, {**good, "after": 101},
                {**good, "action": "share"}, {**good, "why": ""}):
        assert parse(json.dumps(bad)) is None, bad
    for text in ("[1, 2]", "null", f"{json.dumps(good)} {json.dumps(good)}", f"Sure! {json.dumps(good)}", "", None):
        assert parse(text) is None, text


async def _test() -> None:
    def args_for(tmp: Path):
        return dict(out=tmp / "run", n_scenarios=4, n_readers=6, holdout=2, seed="t")

    _parse_test()
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
        await _budget_test(tmp)
        w, r = FakeWriter(), FakeReader()
        caller = Caller(tmp / "cache.jsonl", 10_000, 10**9, 0.0, totals=lambda since: [])
        assert plan(4, 6, 2, caller, w.name, r.name) == {"writer_calls": 12, "reader_calls": 72 + 12, "cached": 0,
                                                         "writer": "fake:writer", "judge": "fake:reader"}
        assert w.calls == r.calls == 0, "planning sends nothing"
        res = await run(w, r, caller=caller, **args_for(tmp))
        bad = [p for p in res["posts"] if p["problem"]]
        assert w.calls == 12 and len(bad) >= 1 and all(p["problem"] == "link" for p in bad), bad
        assert r.calls == (12 - len(bad)) * 6 + len(CONTROLS) * 4 == len(res["judged"])
        for system, prompt in r.prompts:  # the readers never learn the version, and each sees one post
            assert not any(how in system + prompt for how in VERSIONS.values()) and prompt.count("<<<POST") == 1
            assert "control:" not in system + prompt
        by = summary(res["posts"], res["judged"], VERSIONS)
        assert sum(s["unreadable"] for s in by.values()) == sum(j["result"] is None for j in res["judged"]
                                                                if not j["version"].startswith("control")) > 0
        assert sum(s["guard"]["link"] for s in by.values()) == len(bad)
        assert sum(s["publishable"] for s in by.values()) == 12 - len(bad)
        shown = [j for j in res["judged"] if j["version"] == "control:answers"]
        assert {j["reader"] for j in shown} == {"r00", "r01", "r02", "r03"}, "controls go to development readers"
        text = (tmp / "run" / "report.md").read_text()
        assert "hidden" in text and "held-out readers (revealed" not in text and "## Controls" in text
        key = json.loads((tmp / "run" / "key.json").read_text())
        blind = (tmp / "run" / "blind.md").read_text()
        assert len(key) == 12 - len(bad) and not any(f"| {v} " in blind or v + ":" in blind for v in VERSIONS)

        # The same run again is answered from the cache: no new calls, the same results.
        w2, r2 = FakeWriter(), FakeReader()
        again = await run(w2, r2, caller=Caller(tmp / "cache.jsonl", 10_000, 10**9, 0.0, totals=lambda s: []),
                          reveal=True, **args_for(tmp))
        assert w2.calls == r2.calls == 0 and again["judged"] == res["judged"]
        assert "held-out readers (revealed 1 time(s)" in (tmp / "run" / "report.md").read_text()

        # Stopped at a cap partway, then resumed: every call made exactly once.
        w3, r3 = FakeWriter(), FakeReader()
        capped = Caller(tmp / "cache2.jsonl", 20, 10**9, 0.0, totals=lambda s: [])
        try:
            await run(w3, r3, caller=capped, **args_for(tmp))
            raise AssertionError("the cap didn't stop the run")
        except Stop:
            pass
        assert capped.calls == 20 == w3.calls + r3.calls
        w4, r4 = FakeWriter(), FakeReader()
        await run(w4, r4, caller=Caller(tmp / "cache2.jsonl", 10_000, 10**9, 0.0, totals=lambda s: []),
                  **args_for(tmp))
        assert w3.calls + w4.calls == 12 and r3.calls + r4.calls == (12 - len(bad)) * 6 + len(CONTROLS) * 4

        # No answer isn't cached: a writer that answers nothing stops the run, one that misses once is asked again.
        class Silent(FakeWriter):
            def __init__(self, misses) -> None:
                super().__init__()
                self.misses = misses

            async def generate(self, system: str, prompt: str) -> Optional[str]:
                answer = await super().generate(system, prompt)
                return None if self.calls in self.misses else answer

        silent = Silent(range(1, 13))
        silent_cache = Caller(tmp / "cache3.jsonl", 10_000, 10**9, 0.0, totals=lambda s: [])
        try:
            await run(silent, FakeReader(), caller=silent_cache, **args_for(tmp))
            raise AssertionError("a run on no answers went on")
        except Stop:
            pass
        assert silent.calls == 12 and not silent_cache.cache and not (tmp / "cache3.jsonl").exists()
        once = Silent({2})
        first = await run(once, FakeReader(), caller=Caller(tmp / "cache4.jsonl", 10_000, 10**9, 0.0,
                                                            totals=lambda s: []), **args_for(tmp))
        assert [p["problem"] for p in first["posts"]].count("no answer") == 1
        again = FakeWriter()
        await run(again, FakeReader(), caller=Caller(tmp / "cache4.jsonl", 10_000, 10**9, 0.0, totals=lambda s: []),
                  **args_for(tmp))
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
    cache_path = OUT / "cache.jsonl"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    caller = Caller(cache_path, 0, args.max_tokens, args.max_cost)
    p = plan(args.scenarios, args.readers, args.holdout, caller, getattr(writer, "name", "?"),
             getattr(judge, "name", "?"))
    caller.max_calls = args.max_calls if args.max_calls is not None else p["writer_calls"] + p["reader_calls"]
    e = caller.earlier
    print("token and cost caps are estimated thresholds (see --help): a provider counting more input than a token "
          "per byte plus 64, or more output than max_tokens, can take a run over them")
    print(f"plan (prompts {PROMPT_VERSION}): {p}\nthis run's caps: {caller.max_calls} calls, {args.max_tokens} "
          f"tokens, ${args.max_cost:.2f}; spent by earlier runs on this cache: {e['runs']} runs, {e['calls']} calls, "
          f"{e['tokens']} tokens, ${e['cost']:.4f} (not counted against this run's caps)")
    if args.cmd == "plan":
        return
    out = OUT / f"{datetime.now():%Y%m%d-%H%M%S}-{PROMPT_VERSION}"
    try:
        asyncio.run(run(writer, judge, out, args.scenarios, args.readers, args.holdout, args.seed, caller,
                        reveal=args.reveal))
    except Stop as e:
        print(f"stopped: {e}; run the same command again to carry on from the cache")
    finally:
        total = caller.save()
        print(f"this run: {caller.calls} new calls; all runs on this cache: {total['runs']} runs, {total['calls']} "
              f"calls, {total['tokens']} tokens, ${total['cost']:.4f}")
    if (out / "report.md").exists():
        print(f"report in {out / 'report.md'}, blind sheet in {out / 'blind.md'}")


if __name__ == "__main__":
    main()
