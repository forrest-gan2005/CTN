import asyncio
import json
import logging
import os
import random
import re
import time
from pathlib import Path
from typing import Optional

import httpx

log = logging.getLogger("llm")

ENV_FILE = Path(__file__).parent / ".env"
MAX_WAIT = 20.0  # seconds; a model rate-limited or queued for longer hands the call to the next one
COOLDOWN = 60.0  # how long to skip a model that keeps failing
LOCKOUT = 3600.0  # bad key or unknown model: retrying won't help, so stop trying for a while

# provider: (OpenAI-compatible base URL, env var holding the key, free-tier requests per minute)
PROVIDERS = {
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY", 30),
    "cerebras": ("https://api.cerebras.ai/v1", "CEREBRAS_API_KEY", 5),  # trial gpt-oss-120b; qwen-3.8-27b allows 450
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY", 10),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", 20),
    "mistral": ("https://api.mistral.ai/v1", "MISTRAL_API_KEY", 60),
    "github": ("https://models.github.ai/inference", "GITHUB_TOKEN", 10),
    "official": (None, "CTN_LLM_KEY", None),  # URL goes in CTN_LLM_URL once the organisers publish it
}
RETRY_DELAY_RE = re.compile(r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"')


class Generated(str):
    """Model output that remembers which model wrote it, even when a fallback model did."""
    model: Optional[str] = None


def _generated(text: str, model: str) -> Generated:
    out = Generated(text)
    out.model = model
    return out


def strip_thinking(text: str) -> str:
    """Drop a reasoning model's <think> block; a reply cut off mid-thought has no answer in it."""
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1].strip()
    if "<think>" in text:
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
        now = time.monotonic()
        if now < self.blocked_until:
            return float("inf")
        return max(0.0, self.next_slot - now)

    async def wait_turn(self) -> None:
        now = time.monotonic()
        slot = max(now, self.next_slot)
        self.next_slot = slot + self.gap
        if slot > now:
            await asyncio.sleep(slot - now)


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
                 transport: Optional[httpx.AsyncBaseTransport] = None) -> None:
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
        self._sem: Optional[asyncio.Semaphore] = None
        self._client: Optional[httpx.AsyncClient] = None

    def delay(self) -> float:
        return self.quota.delay()

    async def generate(self, system: str, prompt: str) -> Optional[str]:
        if self._sem is None:
            self._sem = asyncio.Semaphore(self.max_concurrent)
            self._client = httpx.AsyncClient(timeout=self.timeout, transport=self.transport)
        body = {**self.body, "messages": [{"role": "system", "content": system},
                                          {"role": "user", "content": prompt}]}
        async with self._sem:
            for attempt in range(3):
                if time.monotonic() < self.quota.blocked_until:
                    return None
                await self.quota.wait_turn()
                try:
                    resp = await self._client.post(self.url, headers=self.headers, json=body)
                except httpx.HTTPError as e:
                    log.warning("%s: %s", self.name, e or type(e).__name__)
                    await asyncio.sleep(2 ** attempt)
                    continue
                if resp.status_code == 429:
                    wait = _retry_after(resp)
                    if wait > MAX_WAIT:
                        log.warning("%s rate-limited for %.0fs", self.name, wait)
                        self.quota.blocked_until = time.monotonic() + wait
                        return None
                    await asyncio.sleep(wait)
                    continue
                if resp.status_code >= 500:
                    await asyncio.sleep(2 ** attempt)
                    continue
                if resp.status_code in (401, 403, 404):
                    log.error("%s: HTTP %d %s", self.name, resp.status_code, resp.text[:200])
                    self.quota.blocked_until = time.monotonic() + LOCKOUT
                    return None
                if resp.status_code >= 400:  # e.g. a content filter refusing this one prompt
                    log.warning("%s: HTTP %d %s", self.name, resp.status_code, resp.text[:200])
                    return None
                try:
                    text = _content(resp.json())
                except (ValueError, KeyError, IndexError, TypeError, AttributeError):
                    log.warning("%s: unexpected response %s", self.name, resp.text[:200])
                    return None
                return _generated(text, self.name) if text else None
            self.quota.blocked_until = time.monotonic() + COOLDOWN
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

    Specs joined by commas fall back in that order. Options go after '?', e.g. '?rpm=5&reasoning_effort=low';
    rpm and max_tokens are ours, anything else is sent to the provider. With no spec, $LLM is used.
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
                           temperature=temperature, max_tokens=options.pop("max_tokens", 400), extra=options)


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
        return httpx.Response(200, json={"choices": [{"message": {"content": f"<think>x</think> hi from {model}"}}]})

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

    os.environ["GROQ_API_KEY"] = "test"
    groq = make_llm("groq:llama-3.3-70b-versatile?rpm=5&reasoning_effort=low", temperature=0)
    assert groq.quota.gap == 12 and groq.body["reasoning_effort"] == "low" and groq.body["temperature"] == 0
    both = make_llm("fake, ollama:qwen3:8b")
    assert isinstance(both, ChainLLM) and both.name == "fake+ollama:qwen3:8b"
    for llm in (flaky, chain, paced):
        await llm.aclose()
    print("llm ok")


if __name__ == "__main__":
    asyncio.run(_demo())
