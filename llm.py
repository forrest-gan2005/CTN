import asyncio
import os
import random
from typing import Optional

import httpx


class OllamaLLM:
    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        max_concurrent: int = 4,
        timeout: float = 60,
    ) -> None:
        self.base_url = (base_url or os.environ.get("OLLAMA_URL") or "http://localhost:11434").rstrip("/")
        self.model = model or os.environ.get("OLLAMA_MODEL") or "llama3.1:8b"
        self.name = f"ollama:{self.model}"  # recorded with every post, to compare models later
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        # Created lazily: on Python 3.9 they bind to the event loop that exists when constructed.
        self._sem: Optional[asyncio.Semaphore] = None
        self._client: Optional[httpx.AsyncClient] = None

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
                        # Ollama silently drops the start of the prompt (our system prompt) past num_ctx.
                        "options": {"num_ctx": 4096},
                    },
                )
                resp.raise_for_status()
                return resp.json()["message"]["content"]
            except Exception:
                return None

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()


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

    async def generate(self, system: str, prompt: str) -> Optional[str]:
        delay = random.uniform(self.min_delay, self.max_delay)
        await asyncio.sleep(delay)
        self.calls += 1
        self.total_wait += delay
        return random.choice(self.LINES)

    async def aclose(self) -> None:
        pass


async def _demo() -> None:
    fake = FakeLLM(0, 0)
    assert await fake.generate("you are joe", "write a post") in FakeLLM.LINES

    llm = OllamaLLM(base_url="http://127.0.0.1:1", model="nope")
    assert await llm.generate("s", "p") is None
    await llm.aclose()

    env = OllamaLLM()
    print("base_url:", env.base_url, "model:", env.model)


if __name__ == "__main__":
    asyncio.run(_demo())
