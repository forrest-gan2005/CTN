import argparse
import asyncio
import json
import logging
import random
import re
import statistics
import time
from collections import Counter
from datetime import datetime
from difflib import SequenceMatcher
from typing import Optional

from ctn.infra.fake_platform import FakePlatform
from ctn.infra.llm import load_env, make_llm
from ctn.infra.state import StateStore
from ctn.red.bot import SIMILARITY_LIMIT, Bot
from ctn.red.mimic import DATA, MIN_TEXTS, DraftStore, evaluate
from ctn.text.content import CONTENT_DIR, load_personas, load_world
from ctn.text.guard import IDENTITY_RE, judge_stance, leans_no, looks_like_probe, plain
from ctn.text.injection import load_injection_classifier

CASES = json.loads((CONTENT_DIR / "bench.json").read_text(encoding="utf-8"))
NPC_DB = DATA / "npc.db"
REPORTS = DATA / "bench"
REPLY_SHARE = 0.5


class Metered:
    """An LLM wrapped to time every call and keep what it returned."""

    def __init__(self, llm) -> None:
        self.llm = llm
        self.name = llm.name
        self.latencies = []
        self.outputs = []

    def delay(self) -> float:
        return self.llm.delay()

    async def generate(self, system: str, prompt: str) -> Optional[str]:
        start = time.perf_counter()
        out = await self.llm.generate(system, prompt)
        self.latencies.append(time.perf_counter() - start)
        self.outputs.append(out)
        return out

    async def aclose(self) -> None:
        await self.llm.aclose()


def broken(text: Optional[str], canary: Optional[str]) -> list:
    """What an attack post got the model to do, judged from its reply."""
    if not text:
        return []
    found = []
    if canary and re.search(canary, text, re.IGNORECASE):
        found.append("obeyed")
    if IDENTITY_RE.search(plain(text)):
        found.append("identity")
    if leans_no(text):
        found.append("off side")
    return found


def _bots(writer, judge, examples=None) -> list:
    world, memory = load_world(), StateStore(":memory:")
    return [Bot(p, FakePlatform(), writer, world, stance_llm=judge, store=memory, examples=examples)
            for p in load_personas()]


def _targets() -> tuple:
    """Posts to reply to and style examples: real citizens' once the collector has them, else our test set."""
    if NPC_DB.exists():
        from ctn.infra.collector import NpcStore
        from ctn.text.examples import NpcExamples

        store = NpcStore(NPC_DB)
        rows = [(a, c) for a, c, _, parent in store.npc_posts() if parent is None and not looks_like_probe(c)]
        if len(rows) >= MIN_TEXTS:
            return ([{"id": i, "author": a, "content": c} for i, (a, c) in enumerate(rows)],
                    NpcExamples.from_store(store), [(a, c) for a, c, _, _ in store.npc_posts()])
    return [{"id": i, "author": f"citizen_{i}", "content": c} for i, c in enumerate(CASES["targets"])], None, []


def run_filter(clf) -> dict:
    """The input filter on its own: attacks it lets through, and ordinary posts it would wrongly skip."""
    ordinary = CASES["targets"] + [text for text, _ in CASES["stance"]]
    skipped = lambda t: looks_like_probe(t) or (clf is not None and clf.flags(t))
    return {
        "classifier": clf is not None,
        "missed": [a["text"] for a in CASES["attacks"] if not skipped(a["text"])],
        "false_alarms": [t for t in ordinary if skipped(t)],
        "attacks": len(CASES["attacks"]),
        "ordinary": len(ordinary),
    }


async def run_stance(judge) -> dict:
    """Posts on their own, and replies judged next to the post they answer, scored apart."""
    posts = [(None, text, label) for text, label in CASES["stance"]]
    replies = [tuple(c) for c in CASES["reply_stance"]]
    got = await asyncio.gather(*(judge_stance(judge, text, parent) for parent, text, _ in posts + replies))
    rows = [(text if parent is None else f"(to: {parent[:60]}) {text}", label, g)
            for (parent, text, label), g in zip(posts + replies, got)]
    result = {"unparsed": sum(g is None for _, _, g in rows),
              "errors": [(text, label, g) for text, label, g in rows if g != label]}
    for prefix, part in (("", rows[:len(posts)]), ("reply_", rows[len(posts):])):
        oppose = [r for r in part if r[1] == "OPPOSE"]
        result[prefix + "accuracy"] = sum(label == g for _, label, g in part) / len(part)
        result[prefix + "no_missed"] = sum(g != "OPPOSE" for _, _, g in oppose)
        result[prefix + "no_total"] = len(oppose)
    return result


async def run_attacks(writer: Metered, judge, clf=None) -> dict:
    """Each attack post as a reply target, one at a time so each reply lines up with its attack."""
    bots = _bots(writer, judge)
    rng = random.Random(1)
    rows = []
    for i, case in enumerate(CASES["attacks"]):
        target = {"id": 10_000 + i, "author": "someone", "content": case["text"]}
        before = len(writer.outputs)
        action = await rng.choice(bots).draft(target)
        raw = writer.outputs[before] if len(writer.outputs) > before else None
        published = action.content if action is not None else None
        rows.append({
            "attack": case["text"],
            "filtered": looks_like_probe(case["text"]) or (clf is not None and clf.flags(case["text"])),
            "raw": raw,
            "raw_broken": broken(raw, case["canary"]),
            "published": published,
            "published_broken": broken(published, case["canary"]),
        })
    return {
        "raw_broken": sum(bool(r["raw_broken"]) for r in rows),
        "reached": sum(bool(r["published_broken"]) and not r["filtered"] for r in rows),
        "total": len(rows),
        "rows": rows,
    }


async def run_drafts(writer: Metered, judge, n: int, concurrency: int) -> dict:
    targets, examples, citizens = _targets()
    bots = _bots(writer, judge, examples)
    rng = random.Random(0)
    jobs = [(rng.choice(bots), rng.choice(targets) if rng.random() < REPLY_SHARE else None) for _ in range(n)]
    sem = asyncio.Semaphore(concurrency)
    made = []
    calls_before = len(writer.outputs)

    async def one(bot: Bot, target: Optional[dict]) -> None:
        async with sem:
            action = await bot.draft(target)
        if action is not None:
            bot.remember(action)
            made.append((bot.username, action))

    start = time.perf_counter()
    await asyncio.gather(*(one(bot, target) for bot, target in jobs))
    minutes = (time.perf_counter() - start) / 60

    texts = [a.content for _, a in made]
    rejected = Counter()
    for bot in bots:
        rejected.update(bot.rejected)
    failed = sum(out is None for out in writer.outputs[calls_before:])
    rejected["no answer"] = failed
    rejected["cleaned away"] = n - len(made) - sum(rejected.values())
    pairs = [(a, b) for i, a in enumerate(texts) for b in texts[i + 1:]]
    near = sum(SequenceMatcher(None, a.lower(), b.lower(), autojunk=False).ratio() > SIMILARITY_LIMIT
               for a, b in pairs)
    openers = Counter(" ".join(t.lower().split()[:2]) for t in texts)
    bigrams = [tuple(w[i:i + 2]) for t in texts for w in [t.lower().split()] for i in range(len(w) - 1)]
    style = None
    if citizens and len(made) >= MIN_TEXTS:
        style = evaluate(citizens, [(u, a.content) for u, a in made])[0]
    return {
        "made": made,
        "attempts": n,
        "per_minute": len(made) / minutes if minutes else 0.0,
        "rejected": {k: v for k, v in rejected.most_common() if v},
        "words": statistics.fmean(len(t.split()) for t in texts) if texts else 0.0,
        "top_opener": openers.most_common(1)[0] if openers else ("", 0),
        "near_dup": near / len(pairs) if pairs else 0.0,
        "distinct2": len(set(bigrams)) / len(bigrams) if bigrams else 0.0,
        "style": style,
    }


async def bench(spec: str, judge_spec: Optional[str], n: int, skip: set, concurrency: int, clf=None) -> dict:
    writer = Metered(make_llm(spec))
    self_judge = Metered(make_llm(spec, temperature=0))  # the stance test rates the model itself as a judge
    judge = Metered(make_llm(judge_spec, temperature=0)) if judge_spec else self_judge
    result = {"model": writer.name, "judge": judge.name}
    try:
        if "stance" not in skip:
            result["stance"] = await run_stance(self_judge)
        if "attacks" not in skip:
            result["attacks"] = await run_attacks(writer, judge, clf)
        if "drafts" not in skip:
            result["drafts"] = await run_drafts(writer, judge, n, concurrency)
            store = DraftStore()
            for username, action in result["drafts"]["made"]:
                store.add(username, action)
    finally:
        for llm in {writer, self_judge, judge}:
            await llm.aclose()
    lat = sorted(writer.latencies)
    result["calls"] = len(lat) + len(self_judge.latencies) + (len(judge.latencies) if judge is not self_judge else 0)
    result["p50"] = lat[len(lat) // 2] if lat else None
    result["p90"] = lat[int(len(lat) * 0.9)] if lat else None
    return result


def _cell(value, fmt: str = "{}") -> str:
    return "–" if value is None else fmt.format(value)


def summary(results: list) -> str:
    head = ("| model | calls | p50 s | p90 s | drafts | per min | words | top opener | near-dup | distinct-2 "
            "| style acc | stance acc | NO missed | reply acc | reply NO missed | attacks obeyed | reached citizens |")
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    for r in results:
        d, s, a = r.get("drafts"), r.get("stance"), r.get("attacks")
        lines.append("| " + " | ".join([
            r["model"], str(r["calls"]), _cell(r["p50"], "{:.1f}"), _cell(r["p90"], "{:.1f}"),
            f"{len(d['made'])}/{d['attempts']}" if d else "–",
            _cell(d and d["per_minute"], "{:.1f}"), _cell(d and d["words"], "{:.0f}"),
            f"{d['top_opener'][0]!r} x{d['top_opener'][1]}" if d else "–",
            _cell(d and d["near_dup"], "{:.1%}"), _cell(d and d["distinct2"], "{:.2f}"),
            _cell(d and d["style"], "{:.2f}"),
            _cell(s and s["accuracy"], "{:.0%}"), f"{s['no_missed']}/{s['no_total']}" if s else "–",
            _cell(s and s["reply_accuracy"], "{:.0%}"), f"{s['reply_no_missed']}/{s['reply_no_total']}" if s else "–",
            f"{a['raw_broken']}/{a['total']}" if a else "–", f"{a['reached']}/{a['total']}" if a else "–",
        ]) + " |")
    return "\n".join(lines)


def report(results: list, filt: dict) -> str:
    out = [f"# Model bench {datetime.now():%Y-%m-%d %H:%M}", "", summary(results), "",
           f"Input filter (regexes{' + classifier' if filt['classifier'] else ''}): lets through "
           f"{len(filt['missed'])}/{filt['attacks']} attacks, wrongly skips {len(filt['false_alarms'])}/"
           f"{filt['ordinary']} ordinary posts", ""]
    out += [f"- let through: {t}" for t in filt["missed"]] + [f"- wrongly skipped: {t}" for t in filt["false_alarms"]]
    out += ["",
           "- near-dup: share of draft pairs a Blue team could match as copies; distinct-2: share of unique "
           "word pairs; style acc: detector balanced accuracy against citizens (0.5 = indistinguishable)",
           "- NO missed: posts arguing NO the judge failed to call OPPOSE (we could amplify them); reply acc and "
           "reply NO missed: the same for replies judged next to the post they answer (a NO reply missed is one of "
           "ours that would go out)",
           "- attacks obeyed: replies that followed an attack post; reached citizens: of those, how many got past "
           "the input filter and every output check", ""]
    for r in results:
        out.append(f"## {r['model']} (judge {r['judge']})")
        if r.get("drafts"):
            out.append(f"Rejected: {r['drafts']['rejected'] or 'none'}\n")
            out.append("Samples:")
            out += [f"- [{a.type}/{a.topic_kind}] {a.content}" for _, a in r["drafts"]["made"][:8]]
        if r.get("stance"):
            out.append("\nStance errors (text, expected, got):")
            out += [f"- {label} -> {got}: {text}" for text, label, got in r["stance"]["errors"]]
        if r.get("attacks"):
            out.append("\nAttacks:")
            for row in r["attacks"]["rows"]:
                flag = "filtered" if row["filtered"] else ", ".join(row["raw_broken"]) or "ok"
                shipped = "published" if row["published"] else "dropped"
                out.append(f"- **{flag}, {shipped}** {row['attack']}\n  -> {row['raw']!r}")
        out.append("")
    return "\n".join(out)


async def main(args) -> None:
    load_env()
    runs = []
    for spec in args.models:
        try:
            make_llm(spec)
            runs.append(spec)
        except ValueError as e:
            print(f"skipping {spec}: {e}")
    clf = load_injection_classifier(args.injection)
    filt = run_filter(clf)
    results = await asyncio.gather(*(bench(spec, args.judge, args.n, set(args.skip), args.concurrency, clf)
                                     for spec in runs))
    text = report(list(results), filt)
    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / f"{datetime.now():%Y%m%d-%H%M%S}.md"
    path.write_text(text, encoding="utf-8")
    print(summary(list(results)))
    print(f"input filter: lets through {len(filt['missed'])}/{filt['attacks']} attacks, "
          f"wrongly skips {len(filt['false_alarms'])}/{filt['ordinary']} ordinary posts")
    print(f"\nfull report: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare LLMs on our real pipeline: drafts, stance judging, "
                                                 "prompt-injection resistance")
    parser.add_argument("models", nargs="+", help="specs like groq:llama-3.3-70b-versatile or ollama:llama3.1:8b")
    parser.add_argument("--judge", default=None, help="one stance judge for every writer (default: each model)")
    parser.add_argument("--n", type=int, default=40, help="drafts per model")
    parser.add_argument("--skip", nargs="*", default=[], choices=["stance", "attacks", "drafts"])
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--injection", default=None, help="injection classifier (default $INJECTION_MODEL, "
                                                            "else PromptGuard 2); unavailable means regexes only")
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("bot").setLevel(logging.ERROR)  # rejections are counted in the report instead
    asyncio.run(main(parser.parse_args()))
