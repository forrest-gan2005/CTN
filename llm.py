import asyncio
import json
import logging
import os
import random
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
from typing import Optional

import httpx

log = logging.getLogger("llm")

ENV_FILE = Path(__file__).parent / ".env"
MAX_WAIT = 20.0  # seconds; a model rate-limited or queued for longer hands the call to the next one
COOLDOWN = 60.0  # how long to skip a model that keeps failing
LOCKOUT = 3600.0  # bad key or unknown model: retrying won't help, so stop trying for a while
POLL = 1.0  # seconds; a caller waiting for its slot notices a new block at least this often

# provider: (OpenAI-compatible base URL, env var holding the key, free-tier requests per minute)
PROVIDERS = {
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY", 30),
    "cerebras": ("https://api.cerebras.ai/v1", "CEREBRAS_API_KEY", 5),  # trial gpt-oss-120b; qwen-3.8-27b allows 450
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY", 5),  # Flash Lite 15, Gemma 30
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", 20),
    "mistral": ("https://api.mistral.ai/v1", "MISTRAL_API_KEY", 60),
    "github": ("https://models.github.ai/inference", "GITHUB_TOKEN", 10),
    "official": (None, "CTN_LLM_KEY", None),  # URL goes in CTN_LLM_URL once the organisers publish it
}
RETRY_DELAY_RE = re.compile(r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"')
USAGE_DB = Path(__file__).parent / "data" / "usage.db"
GAME_TZ = ZoneInfo("Australia/Sydney")  # game days, and so daily budgets, turn over at Sydney midnight
# USD per million (input, output) tokens, for estimating spend; reasoning tokens bill as output.
# Third-party trackers, October 2026: Cerebras doesn't publish them in its docs.
PRICES = {
    "cerebras:gpt-oss-120b": (0.35, 0.75),
    "cerebras:qwen-3.8-27b": (0.40, 0.80),  # unpublished; priced like Cerebras' Qwen 3 32B to be safe
    # Free tier while no billing is linked to the Google project (paid would be $0.30 / $2.50).
    "gemini:gemini-3.5-flash-lite": (0.0, 0.0),
    "gemini:gemma-4-31b-it": (0.0, 0.0),
}


class UsageLedger:
    """Tokens every API call used, so spend can be watched and capped without the provider's console."""

    def __init__(self, path=USAGE_DB) -> None:
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.execute("CREATE TABLE IF NOT EXISTS usage (at TEXT NOT NULL, model TEXT NOT NULL, "
                        "prompt INTEGER NOT NULL, completion INTEGER NOT NULL, reasoning INTEGER NOT NULL)")

    def record(self, model: str, usage: dict) -> None:
        details = usage.get("completion_tokens_details") or {}
        with self.db:
            self.db.execute("INSERT INTO usage VALUES (?, ?, ?, ?, ?)",
                            (datetime.now(timezone.utc).isoformat(timespec="seconds"), model,
                             usage.get("prompt_tokens") or 0, usage.get("completion_tokens") or 0,
                             details.get("reasoning_tokens") or 0))

    def totals(self, since: str = "") -> list:
        """(model, calls, prompt tokens, completion tokens, reasoning tokens, estimated USD or None)."""
        rows = self.db.execute("SELECT model, COUNT(*), SUM(prompt), SUM(completion), SUM(reasoning) FROM usage "
                               "WHERE at >= ? GROUP BY model ORDER BY model", (since,)).fetchall()
        out = []
        for model, calls, prompt, completion, reasoning in rows:
            price = PRICES.get(model)
            cost = (prompt * price[0] + completion * price[1]) / 1e6 if price else None
            out.append((model, calls, prompt, completion, reasoning, cost))
        return out

    def spent(self, provider: str, since: str = "") -> float:
        """Estimated USD spent on one provider's models, which share its credit."""
        return sum(cost or 0.0 for model, *_, cost in self.totals(since) if model.startswith(provider + ":"))


def game_day_start() -> str:
    now = datetime.now(GAME_TZ)
    return now.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc).isoformat(timespec="seconds")


_LEDGER: Optional[UsageLedger] = None


def ledger() -> UsageLedger:
    global _LEDGER
    if _LEDGER is None:
        _LEDGER = UsageLedger()
    return _LEDGER


class Generated(str):
    """Model output that remembers which model wrote it, even when a fallback model did."""
    model: Optional[str] = None


def _generated(text: str, model: str) -> Generated:
    out = Generated(text)
    out.model = model
    return out


THINK_OPEN = re.compile(r"<(?:think|thought)>", re.IGNORECASE)
THINK_CLOSE = re.compile(r"</(?:think|thought)>", re.IGNORECASE)  # Qwen and DeepSeek say think, Gemma thought


def strip_thinking(text: str) -> str:
    """Drop a reasoning model's thinking block; a reply cut off mid-thought has no answer in it."""
    parts = THINK_CLOSE.split(text)
    if len(parts) > 1:
        return parts[-1].strip()
    if THINK_OPEN.search(text):
        return ""
    return text.strip()


def load_env(path: Path = ENV_FILE) -> None:
    """KEY=value lines from .env (kept out of git) into the environment, without overriding what is set."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


class Quota:
    """One provider's limit for one model, shared by every client using it: spaces requests evenly
    and remembers when the provider has told us to back off."""

    def __init__(self, rpm: Optional[float]) -> None:
        self.gap = 60.0 / rpm if rpm else 0.0
        self.next_slot = 0.0
        self.blocked_until = 0.0

    def delay(self) -> float:
        """Seconds before a request could go out; infinite while blocked for longer than a caller would wait."""
        now = time.monotonic()
        if self.blocked_until - now > MAX_WAIT:
            return float("inf")
        return max(0.0, self.blocked_until - now, self.next_slot - now)

    def block(self, seconds: float) -> None:
        """Back off for `seconds`; a shorter block never cuts an existing longer one short."""
        self.blocked_until = max(self.blocked_until, time.monotonic() + seconds)

    async def wait_turn(self, patience: float) -> bool:
        """Wait for this caller's request slot. False, and send nothing, if the provider has told us to back
        off for longer than `patience` seconds, whether before the wait or while it lasts."""
        give_up = time.monotonic() + patience
        while time.monotonic() < self.blocked_until:
            if self.blocked_until > give_up:
                return False
            await asyncio.sleep(min(POLL, self.blocked_until - time.monotonic()))
        slot = max(time.monotonic(), self.next_slot)
        self.next_slot = slot + self.gap
        while True:
            now = time.monotonic()
            if now < self.blocked_until:  # someone else was told to back off while we queued: start again
                return await self.wait_turn(max(0.0, give_up - now))
            if now >= slot:
                return True
            await asyncio.sleep(min(POLL, slot - now))


_QUOTAS = {}


def _quota(url: str, model: str, rpm: Optional[float]) -> Quota:
    return _QUOTAS.setdefault((url, model), Quota(rpm))


def _retry_after(resp: httpx.Response) -> float:
    header = resp.headers.get("retry-after")
    if header:
        try:
            return float(header)
        except ValueError:
            pass
    found = RETRY_DELAY_RE.search(resp.text)  # Gemini puts it in the body
    return float(found.group(1)) if found else 10.0


def _content(data: dict) -> str:
    content = data["choices"][0]["message"].get("content") or ""
    if isinstance(content, list):  # some providers split reasoning and answer into parts
        content = "".join(part.get("text", "") for part in content if part.get("type") == "text")
    return strip_thinking(content)


class OpenAICompatLLM:
    """Any chat-completions API in OpenAI's format: the free tiers of Groq, Cerebras, Gemini and others,
    and the organisers' LLM."""

    def __init__(self, base_url: str, api_key: str, model: str, name: str, rpm: Optional[float] = None,
                 temperature: Optional[float] = None, max_tokens: int = 400, extra: Optional[dict] = None,
                 max_concurrent: int = 8, timeout: float = 60,
                 transport: Optional[httpx.AsyncBaseTransport] = None, usage: Optional[UsageLedger] = None,
                 budget: Optional[float] = None, daily: Optional[float] = None) -> None:
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.name = name
        self.body = {"model": model, "max_tokens": max_tokens, **(extra or {})}
        if temperature is not None:
            self.body["temperature"] = temperature
        self.quota = _quota(self.url, model, rpm)
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.transport = transport
        self.usage = usage
        self.provider = name.partition(":")[0]
        self.budget = budget  # USD this provider may cost in all, and per game day; None means no cap
        self.daily = daily
        self._warned = False
        self._sem: Optional[asyncio.Semaphore] = None
        self._client: Optional[httpx.AsyncClient] = None

    def delay(self) -> float:
        return float("inf") if self.over_budget() else self.quota.delay()

    def over_budget(self) -> bool:
        if self.usage is None:
            return False
        over = ((self.budget is not None and self.usage.spent(self.provider) >= self.budget)
                or (self.daily is not None and self.usage.spent(self.provider, game_day_start()) >= self.daily))
        if over and not self._warned:
            log.error("%s budget reached (spent $%.2f in all, $%.2f today); falling back", self.name,
                      self.usage.spent(self.provider), self.usage.spent(self.provider, game_day_start()))
            self._warned = True
        return over

    async def generate(self, system: str, prompt: str) -> Optional[str]:
        if self._sem is None:
            self._sem = asyncio.Semaphore(self.max_concurrent)
            self._client = httpx.AsyncClient(timeout=self.timeout, transport=self.transport)
        body = {**self.body, "messages": [{"role": "system", "content": system},
                                          {"role": "user", "content": prompt}]}
        async with self._sem:
            for attempt in range(3):
                # Checked again after queueing: a limit or the budget may have been reached meanwhile.
                if self.over_budget() or not await self.quota.wait_turn(MAX_WAIT) or self.over_budget():
                    return None
                try:
                    resp = await self._client.post(self.url, headers=self.headers, json=body)
                except httpx.HTTPError as e:
                    log.warning("%s: %s", self.name, str(e) or type(e).__name__)  # a timeout has no message
                    await asyncio.sleep(2 ** attempt)
                    continue
                if resp.status_code == 429:
                    wait = _retry_after(resp)
                    self.quota.block(wait)  # for every caller of this model; a short one is sat out, not fallen back from
                    if wait > MAX_WAIT:
                        log.warning("%s rate-limited for %.0fs", self.name, wait)
                        return None
                    continue
                if resp.status_code >= 500:
                    await asyncio.sleep(2 ** attempt)
                    continue
                if resp.status_code in (401, 403, 404):
                    log.error("%s: HTTP %d %s", self.name, resp.status_code, resp.text[:200])
                    self.quota.block(LOCKOUT)
                    return None
                if resp.status_code >= 400:  # e.g. a content filter refusing this one prompt
                    log.warning("%s: HTTP %d %s", self.name, resp.status_code, resp.text[:200])
                    return None
                try:
                    data = resp.json()
                    if self.usage is not None and data.get("usage"):
                        self.usage.record(self.name, data["usage"])
                    text = _content(data)
                except (ValueError, KeyError, IndexError, TypeError, AttributeError):
                    log.warning("%s: unexpected response %s", self.name, resp.text[:200])
                    return None
                return _generated(text, self.name) if text else None
            self.quota.block(COOLDOWN)
            return None

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()


class OllamaLLM:
    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        max_concurrent: int = 4,
        timeout: float = 60,
        temperature: Optional[float] = None,
    ) -> None:
        self.base_url = (base_url or os.environ.get("OLLAMA_URL") or "http://localhost:11434").rstrip("/")
        self.model = model or os.environ.get("OLLAMA_MODEL") or "llama3.1:8b"
        self.name = f"ollama:{self.model}"  # recorded with every post, to compare models later
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        # Ollama silently drops the start of the prompt (our system prompt) past num_ctx.
        self.options = {"num_ctx": 4096}
        if temperature is not None:
            self.options["temperature"] = temperature
        # Created lazily: on Python 3.9 they bind to the event loop that exists when constructed.
        self._sem: Optional[asyncio.Semaphore] = None
        self._client: Optional[httpx.AsyncClient] = None

    def delay(self) -> float:
        return 0.0

    async def generate(self, system: str, prompt: str) -> Optional[str]:
        if self._sem is None:
            self._sem = asyncio.Semaphore(self.max_concurrent)
            self._client = httpx.AsyncClient(timeout=self.timeout)
        async with self._sem:
            try:
                resp = await self._client.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": prompt},
                        ],
                        "stream": False,
                        "options": self.options,
                    },
                )
                resp.raise_for_status()
                return strip_thinking(resp.json()["message"]["content"])
            except Exception:
                return None

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()


class ChainLLM:
    """Models in order of preference. A call goes to the first that isn't rate-limited or backed up,
    and falls through to the next if it fails, so one provider's limit or outage doesn't stop the team."""

    def __init__(self, llms: list) -> None:
        self.llms = llms
        self.name = "+".join(llm.name for llm in llms)

    def delay(self) -> float:
        return min(llm.delay() for llm in self.llms)

    async def generate(self, system: str, prompt: str) -> Optional[str]:
        # sorted() is stable: ready models keep their order, backed-up ones go last
        for llm in sorted(self.llms, key=lambda m: m.delay() > MAX_WAIT):
            out = await llm.generate(system, prompt)
            if out:
                return out if isinstance(out, Generated) else _generated(out, llm.name)
        return None

    async def aclose(self) -> None:
        for llm in self.llms:
            await llm.aclose()


class FakeLLM:
    LINES = [
        "Another slow morning, but the regulars still showed up.",
        "Queue out the door for the clinic again today.",
        "Rent is due and the till is half of what it was.",
        "Checking a pass takes ten seconds. Closing takes months.",
        "Heard the same rumour three times before lunch.",
    ]

    def __init__(self, min_delay: float = 0.5, max_delay: float = 2.0) -> None:
        self.name = "fake"
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.calls = 0
        self.total_wait = 0.0

    def delay(self) -> float:
        return 0.0

    async def generate(self, system: str, prompt: str) -> Optional[str]:
        delay = random.uniform(self.min_delay, self.max_delay)
        await asyncio.sleep(delay)
        self.calls += 1
        self.total_wait += delay
        return random.choice(self.LINES)

    async def aclose(self) -> None:
        pass


def _option(value: str):
    try:
        return json.loads(value)
    except ValueError:
        return value


def make_llm(spec: Optional[str] = None, temperature: Optional[float] = None):
    """The LLM a spec names: 'groq:llama-3.3-70b-versatile', 'ollama:llama3.1:8b', 'fake', ...

    Specs joined by commas fall back in that order. Options go after '?', e.g. '?rpm=5&budget=4.5&daily=0.4';
    rpm, max_tokens, budget and daily (USD caps on the provider's estimated spend, in all and per game day)
    are ours, anything else is sent to the provider. With no spec, $LLM is used.
    """
    load_env()
    spec = spec or os.environ.get("LLM") or "ollama"
    specs = [s.strip() for s in spec.split(",") if s.strip()]
    if len(specs) > 1:
        return ChainLLM([make_llm(s, temperature) for s in specs])
    spec, _, query = specs[0].partition("?")
    options = {k: _option(v) for k, _, v in (kv.partition("=") for kv in query.split("&") if kv)}
    provider, _, model = spec.partition(":")
    if provider == "fake":
        return FakeLLM(0.01, 0.05)
    if provider == "ollama":
        return OllamaLLM(model or None, temperature=temperature)
    if provider not in PROVIDERS:
        raise ValueError(f"unknown LLM provider {provider!r}; one of ollama, fake, {', '.join(PROVIDERS)}")
    base_url, key_var, rpm = PROVIDERS[provider]
    base_url = base_url or os.environ.get("CTN_LLM_URL")
    key = os.environ.get(key_var)
    if not model or not base_url or not key:
        raise ValueError(f"{spec!r} needs a model name, a base URL and ${key_var} (put keys in .env)")
    return OpenAICompatLLM(base_url, key, model, name=spec, rpm=options.pop("rpm", rpm),
                           temperature=temperature, max_tokens=options.pop("max_tokens", 400),
                           usage=ledger(), budget=options.pop("budget", None), daily=options.pop("daily", None),
                           extra=options)


def team_llms() -> tuple:
    """The writer ($LLM) and the stance judge ($LLM_JUDGE, else the writer's model at temperature 0)."""
    load_env()
    return make_llm(), make_llm(os.environ.get("LLM_JUDGE") or os.environ.get("LLM"), temperature=0)


async def _demo() -> None:
    fake = FakeLLM(0, 0)
    assert await fake.generate("you are joe", "write a post") in FakeLLM.LINES

    llm = OllamaLLM(base_url="http://127.0.0.1:1", model="nope")
    assert await llm.generate("s", "p") is None
    await llm.aclose()

    assert strip_thinking("<think>hmm, a post</think>\nFerry late again.") == "Ferry late again."
    assert strip_thinking("hmm, a post</think>Ferry late again.") == "Ferry late again."
    assert strip_thinking("<think>still thinking when the tokens ran out") == ""
    assert strip_thinking("<thought>*   Persona: nurse\n\"rent again\"</thought>rent went up again lol") == "rent went up again lol"

    calls = {"busy": 0, "good": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert request.headers["authorization"] == "Bearer k" and body["messages"][0]["role"] == "system"
        model = body["model"]
        calls[model] = calls.get(model, 0) + 1
        if model == "busy":  # daily limit used up
            return httpx.Response(429, headers={"retry-after": "3600"})
        if model == "flaky" and calls[model] == 1:
            return httpx.Response(429, text='{"error": {"details": [{"retryDelay": "0.01s"}]}}')
        if model == "badkey":
            return httpx.Response(401, text="invalid api key")
        usage = {"prompt_tokens": 2_000_000, "completion_tokens": 1_000_000,
                 "completion_tokens_details": {"reasoning_tokens": 400_000}}
        return httpx.Response(200, json={"choices": [{"message": {"content": f"<think>x</think> hi from {model}"}}],
                                         "usage": usage})

    mock = httpx.MockTransport(handler)
    make = lambda model, rpm=None: OpenAICompatLLM("http://mock/v1", "k", model, name=f"mock:{model}",
                                                   rpm=rpm, transport=mock)
    flaky = make("flaky")
    out = await flaky.generate("s", "p")
    assert out == "hi from flaky" and out.model == "mock:flaky" and calls["flaky"] == 2

    busy, good, bad = make("busy"), make("good"), make("badkey")
    chain = ChainLLM([bad, busy, good])
    out = await chain.generate("s", "p")
    assert out == "hi from good" and out.model == "mock:good"
    assert busy.delay() == float("inf") and bad.delay() == float("inf")
    await chain.generate("s", "p")
    assert calls["busy"] == 1 and calls["badkey"] == 1, "blocked models were tried again"

    paced = make("paced", rpm=600)  # one request every 0.1s
    start = time.monotonic()
    await asyncio.gather(*(paced.generate("s", "p") for _ in range(4)))
    assert 0.25 < time.monotonic() - start < 1.0
    second = make("paced", rpm=600)
    assert second.quota is paced.quota, "two clients on one model must share its limit"

    book = UsageLedger(":memory:")
    PRICES["mock:priced"] = (0.5, 1.0)  # 2M in + 1M out = $2 a call
    capped = OpenAICompatLLM("http://mock/v1", "k", "priced", name="mock:priced", transport=mock, usage=book,
                             budget=3.0)
    assert await capped.generate("s", "p") and await capped.generate("s", "p")
    assert book.spent("mock") == 4.0 and capped.over_budget() and capped.delay() == float("inf")
    assert await capped.generate("s", "p") is None and calls["priced"] == 2, "over budget, so no third call"
    model, n, prompt, completion, reasoning, cost = book.totals()[0]
    assert (model, n, reasoning, cost) == ("mock:priced", 2, 800_000, 4.0)
    daily = OpenAICompatLLM("http://mock/v1", "k", "priced", name="mock:priced", transport=mock, usage=book, daily=5.0)
    assert not daily.over_budget() and book.spent("mock", game_day_start()) == 4.0
    await capped.aclose()

    os.environ["GROQ_API_KEY"] = "test"
    groq = make_llm("groq:llama-3.3-70b-versatile?rpm=5&reasoning_effort=low&budget=4.5&daily=0.4", temperature=0)
    assert groq.quota.gap == 12 and groq.body["reasoning_effort"] == "low" and groq.body["temperature"] == 0
    assert (groq.budget, groq.daily) == (4.5, 0.4) and "budget" not in groq.body
    both = make_llm("fake, ollama:qwen3:8b")
    assert isinstance(both, ChainLLM) and both.name == "fake+ollama:qwen3:8b"
    for llm in (flaky, chain, paced):
        await llm.aclose()
    await _cooldown_demo()
    print("llm ok")


async def _cooldown_demo() -> None:
    """Callers already queued for a model's next slot respect a block or a budget reached meanwhile: a long 429
    sends nobody else and hands their calls to the fallback, a short one is sat out by everyone, and a later
    shorter block never shortens a longer one."""
    gate, sent, times = asyncio.Event(), asyncio.Event(), {}

    async def handler(request: httpx.Request) -> httpx.Response:
        model = json.loads(request.content)["model"]
        times.setdefault(model, []).append(time.monotonic())
        n = len(times[model])
        if model in ("gated", "spend") and n == 1:
            sent.set()
            await gate.wait()
        if model == "gated" and n == 1:
            return httpx.Response(429, headers={"retry-after": "120"})
        if model == "short" and n == 1:
            return httpx.Response(429, headers={"retry-after": "0.5"})
        usage = {"prompt_tokens": 2_000_000, "completion_tokens": 0} if model == "spend" else {}
        return httpx.Response(200, json={"choices": [{"message": {"content": f"hi from {model}"}}], "usage": usage})

    mock = httpx.MockTransport(handler)
    make = lambda model, rpm=None, **kw: OpenAICompatLLM("http://mock/v1", "k", model, name=f"mock:{model}", rpm=rpm,
                                                         transport=mock, **kw)

    # A long 429 arrives while the second caller waits for its slot (ten seconds away).
    first, second, spare = make("gated", rpm=6), make("gated", rpm=6), make("spare")
    start = time.monotonic()
    a = asyncio.ensure_future(first.generate("s", "p"))
    await sent.wait()
    b = asyncio.ensure_future(ChainLLM([second, spare]).generate("s", "p"))
    while second.quota.next_slot < start + 15:  # both slots taken: the second caller is queued
        await asyncio.sleep(0.01)
    gate.set()
    assert await a is None and await b == "hi from spare"
    assert len(times["gated"]) == 1, "a queued caller was sent during the block"
    assert time.monotonic() - start < POLL + 1, "the queued caller didn't notice the block promptly"

    # A short 429 holds back every caller of the model, then both go through without falling back.
    first, second = make("short", rpm=600), make("short", rpm=600)
    outs = await asyncio.gather(first.generate("s", "p"), second.generate("s", "p"))
    assert outs == ["hi from short"] * 2 and len(times["short"]) == 3
    assert all(t >= times["short"][0] + 0.5 for t in times["short"][1:]), "sent inside the shared block"

    # The budget is used up while the second caller is queued.
    gate.clear()
    sent.clear()
    PRICES["mock:spend"] = (1.0, 1.0)  # 2M tokens in = $2 a call
    book = UsageLedger(":memory:")
    first, second = (make("spend", rpm=120, usage=book, budget=1.0) for _ in range(2))
    start = time.monotonic()
    a = asyncio.ensure_future(first.generate("s", "p"))
    await sent.wait()
    b = asyncio.ensure_future(second.generate("s", "p"))
    while second.quota.next_slot < start + 0.9:
        await asyncio.sleep(0.01)
    gate.set()
    assert await a == "hi from spend" and await b is None and len(times["spend"]) == 1, "sent over budget"

    quota = Quota(None)
    quota.block(100)
    until = quota.blocked_until
    quota.block(1)
    assert quota.blocked_until == until and quota.delay() == float("inf")
    for llm in (first, second, spare):
        await llm.aclose()


def print_usage() -> None:
    book = ledger()
    for title, since in (("today (Sydney game day)", game_day_start()), ("in all", "")):
        print(f"{title}:")
        for model, calls, prompt, completion, reasoning, cost in book.totals(since):
            spend = f"${cost:.3f}" if cost is not None else "price unknown"
            print(f"  {model:<28}{calls:>6} calls {prompt:>11,} in {completion:>10,} out "
                  f"({reasoning:,} reasoning)  {spend}")


if __name__ == "__main__":
    import sys

    if sys.argv[1:] == ["usage"]:
        print_usage()
    else:
        asyncio.run(_demo())
