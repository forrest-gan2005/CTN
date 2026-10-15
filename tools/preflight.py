"""Before going live: what `python3 run.py live` depends on, checked without registering, posting or reporting.

    python3 -m tools.preflight              every check; exit status 1 if anything would stop the launch
    python3 -m tools.preflight --llm        also sends the writer and the judge one short request each
    python3 -m tools.preflight --self-test  checks the checks, offline

Reads .env (printing no key or code), the personas, data/accounts.db and data/npc.db; GETs the platform's
/healthz and /feed/lock and the two papers' front pages, anonymously; loads the sentence model and the
injection classifier from the local cache only.
"""
import argparse
import asyncio
import email.utils
import os
import shutil
import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

import httpx

from ctn import paths
from ctn.blue.blue import MIN_CITIZENS
from ctn.infra import backup
from ctn.infra.collector import COMPETITION_START, NpcStore
from ctn.infra.llm import OllamaLLM, load_env, team_llms
from ctn.infra.news import NEWS_URL, SITES, ARTICLE_RE
from ctn.text.personas import load_personas
from run import API_URL, CLOCK_SKEW

TEAM_KEY_LEN = 16  # the accounts portal's team codes
MAX_ACTIVE = 100  # accounts a team may have active at once
MAX_USERNAME, MAX_DISPLAY_NAME = 32, 128  # the platform's register form
SKEW_WARN = 30  # seconds off the platform's clock worth a mention
MIN_FREE = 2 * 1024 ** 3  # bytes free on the data disk
MODEL_TIMEOUT = 300  # seconds for a local model to load

OK, WARN, FAIL = "ok", "WARN", "FAIL"


class Preflight:
    def __init__(self, data: Path = paths.DATA, api_url: Optional[str] = None,
                 transport: Optional[httpx.AsyncBaseTransport] = None,
                 clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc)) -> None:
        self.data = data
        self.api_url = api_url or os.environ.get("CTN_API_URL") or API_URL
        self.transport = transport
        self.clock = clock
        self.results = []

    def say(self, level: str, what: str, line: str) -> None:
        self.results.append((level, what, line))
        print(f"{level:>4}  {what}: {line}")

    def failed(self) -> list:
        return [r for r in self.results if r[0] == FAIL]

    def team_key(self) -> None:
        key = os.environ.get("CTN_TEAM_KEY")
        if not key:
            hint = " (CTN_BOT_KEY is the old name)" if os.environ.get("CTN_BOT_KEY") else ""
            self.say(FAIL, "team code", f"CTN_TEAM_KEY isn't set in .env{hint}; new accounts can't join the team")
        elif len(key) != TEAM_KEY_LEN:
            self.say(WARN, "team code", f"CTN_TEAM_KEY is {len(key)} characters; the portal's are {TEAM_KEY_LEN}")
        else:
            self.say(OK, "team code", "set")

    async def writers(self, ping: bool) -> None:
        try:
            llm, judge = team_llms()
        except ValueError as e:
            self.say(FAIL, "writers", str(e))
            return
        for role, model in (("writer", llm), ("judge", judge)):
            name = getattr(model, "name", type(model).__name__)
            if "fake" in name.lower():
                self.say(FAIL, role, "the fake model: set $LLM / $LLM_JUDGE in .env")
                continue
            parts = getattr(model, "llms", None) or [model]
            local = [part for part in parts if isinstance(part, OllamaLLM)]
            # a local model behind hosted ones is only the last fallback
            if not all([await self._ollama(role, part, FAIL if len(local) == len(parts) else WARN)
                        for part in local]) and len(local) == len(parts):
                continue
            if ping:
                try:
                    said = await model.generate("You are terse.", "Reply with the single word OK.")
                except Exception as e:
                    said, name = None, f"{name} ({str(e) or type(e).__name__})"
                self.say(OK if said else FAIL, role, f"{name} answered" if said else f"{name} gave nothing back")
            else:
                self.say(OK, role, f"{name} configured (not called; --llm sends one request)")
        for model in (llm, judge):
            await model.aclose()

    async def _ollama(self, role: str, model: OllamaLLM, level: str) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5, transport=self.transport) as client:
                resp = await client.get(f"{model.base_url}/api/tags")
                resp.raise_for_status()
                pulled = {m.get("name") for m in resp.json().get("models", [])}
        except (httpx.HTTPError, ValueError) as e:
            self.say(level, role, f"Ollama at {model.base_url} isn't answering ({str(e) or type(e).__name__})")
            return False
        if model.model not in pulled and f"{model.model}:latest" not in pulled:
            self.say(level, role, f"Ollama doesn't have {model.model}; pull it first")
            return False
        return True

    def personas(self) -> None:
        active, everyone = load_personas(), load_personas(status=None)
        self.say(OK if 0 < len(active) <= MAX_ACTIVE else FAIL, "personas",
                 f"{len(active)} active (at most {MAX_ACTIVE}), {len(everyone) - len(active)} others")
        names = [p["username"] for p in everyone]
        bad = sorted({n for n in names if not n.strip() or len(n) > MAX_USERNAME or names.count(n) > 1} |
                     {p["username"] for p in everyone if len(p.get("name") or "") > MAX_DISPLAY_NAME})
        if bad:
            self.say(FAIL, "personas", f"usernames over {MAX_USERNAME} characters, repeated, or display names "
                                       f"over {MAX_DISPLAY_NAME}: {bad}")

    def accounts(self) -> None:
        path = self.data / "accounts.db"
        active = {p["username"] for p in load_personas()}
        have = set()
        if path.exists():
            have = {r[0] for r in sqlite3.connect(path).execute("SELECT username FROM accounts")}
        new = len(active - have)
        self.say(OK, "accounts", f"{len(active & have)} active accounts have a password stored, {new} will be "
                                 "registered with the team code when the team starts")
        if have:
            self.say(WARN, "accounts", f"data/accounts.db holds {len(have)} passwords that can't be changed or "
                                       "recovered: keep a copy off this machine")
        if backup.marker(self.data):
            self.say(WARN, "accounts", "a restore is pending: the team catches up on the feed before it writes")

    async def platform(self) -> None:
        async with httpx.AsyncClient(base_url=self.api_url.rstrip("/"), timeout=15, transport=self.transport) as c:
            try:
                sent = self.clock()
                health = await c.get("/healthz")
                local = sent + (self.clock() - sent) / 2  # this machine's time when the server stamped its answer
                health.raise_for_status()
                status = health.json().get("status")
            except (httpx.HTTPError, ValueError) as e:
                self.say(FAIL, "platform", f"{self.api_url}/healthz: {str(e).splitlines()[0] if str(e) else e!r}")
                return
            self.say(OK if status == "ok" else WARN, "platform", f"healthz says {status!r}")
            self._clock(health.headers.get("date"), local)
            try:
                lock = await c.get("/feed/lock")
                lock.raise_for_status()
                locked = lock.json().get("locked")
            except (httpx.HTTPError, ValueError) as e:
                self.say(WARN, "feed lock", f"couldn't read it ({str(e).splitlines()[0] if str(e) else e!r})")
                return
            self.say(WARN if locked else OK, "feed lock", "locked: only NPCs and moderators can act" if locked
                     else "open")

    def _clock(self, date: Optional[str], local: datetime) -> None:
        if not date:
            self.say(WARN, "clock", "the platform sent no Date header to compare against")
            return
        skew = (local - email.utils.parsedate_to_datetime(date)).total_seconds()
        level = FAIL if abs(skew) > CLOCK_SKEW.total_seconds() else WARN if abs(skew) > SKEW_WARN else OK
        self.say(level, "clock", f"{skew:+.0f}s from the platform's; the start gate and finding our own posts "
                                 "after an outage go by this machine's clock")

    async def news(self) -> None:
        async with httpx.AsyncClient(base_url=NEWS_URL, timeout=15, transport=self.transport,
                                     follow_redirects=True) as c:
            for site, name in SITES.items():
                try:
                    resp = await c.get(f"/{site}/")
                    resp.raise_for_status()
                except httpx.HTTPError as e:
                    self.say(WARN, "papers", f"{name}: {str(e).splitlines()[0] if str(e) else e!r}")
                    continue
                links = set(ARTICLE_RE.findall(resp.text))
                self.say(OK if links else WARN, "papers", f"{name}: {len(links)} articles listed"
                         + ("" if links else "; has the page changed?"))

    def whitelist(self) -> None:
        path = self.data / "npc.db"
        started = self.clock() >= COMPETITION_START
        if not path.exists():
            self.say(FAIL if started else WARN, "whitelist",
                     "the collector hasn't run: start `run.py live` well before the start, so the citizens "
                     "posting before it are known (the bots wait for the start on their own)")
            return
        citizens = len(NpcStore(path).whitelist())
        level = OK if citizens >= MIN_CITIZENS else WARN
        self.say(level, "whitelist", f"{citizens} accounts seen before the start; Blue reports nothing below "
                                     f"{MIN_CITIZENS}")

    def models(self, run=subprocess.run) -> None:
        env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
        checks = (("sentence model", "from ctn.text.semantic import Embedder; Embedder().encode(['ok'])",
                   "every meaning-level check (copies across our accounts, Blue's paraphrase rules) is off"),
                  ("injection classifier", "import sys; from ctn.text.injection import load_injection_classifier; "
                   "sys.exit(load_injection_classifier() is None)", "probes are spotted by the regexes alone"))
        for what, code, without in checks:
            try:
                done = run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                           timeout=MODEL_TIMEOUT, cwd=paths.ROOT)
                loaded, why = done.returncode == 0, "doesn't load offline"
            except subprocess.TimeoutExpired:
                loaded, why = False, f"didn't load within {MODEL_TIMEOUT}s"
            except OSError as e:
                loaded, why = False, f"couldn't be tried ({type(e).__name__})"
            self.say(OK if loaded else WARN, what, "loads from the local cache" if loaded else f"{why}: {without}")

    def disk(self) -> None:
        free = shutil.disk_usage(self.data if self.data.exists() else self.data.parent).free
        self.say(OK if free >= MIN_FREE else WARN, "disk", f"{free / 1024 ** 3:.1f} GB free for data/")

    def start(self) -> None:
        left = COMPETITION_START - self.clock()
        self.say(OK, "start", f"{COMPETITION_START:%a %d %b %H:%M %Z}; " +
                 (f"{left.days}d {left.seconds // 3600}h {left.seconds % 3600 // 60}m to go"
                  if left.total_seconds() > 0 else "under way"))


async def main(ping: bool) -> int:
    load_env()
    check = Preflight()
    check.start()
    check.team_key()
    await check.writers(ping)
    check.personas()
    check.accounts()
    await check.platform()
    await check.news()
    check.whitelist()
    check.models()
    check.disk()
    failed = check.failed()
    print(f"\n{'NOT READY: ' + str(len(failed)) + ' to fix' if failed else 'ready'}"
          f" ({sum(r[0] == WARN for r in check.results)} warnings)")
    return 1 if failed else 0


async def _demo() -> None:
    """Offline: no .env, no data/usage.db, mock platform and papers, an injected clock and subprocess runner."""
    import tempfile

    from ctn.infra import llm

    clock = {"now": datetime(2026, 10, 18, 6, tzinfo=timezone.utc)}
    server = {"ahead": 0.0, "latency": 0.0}  # the platform's clock against this machine's; a request's duration

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/healthz":
            clock["now"] += timedelta(seconds=server["latency"] / 2)
            stamped = clock["now"].timestamp() + server["ahead"]
            clock["now"] += timedelta(seconds=server["latency"] / 2)
            return httpx.Response(200, json={"status": "ok"},
                                  headers={"date": email.utils.formatdate(stamped, usegmt=True)})
        if path == "/api/feed/lock":
            return httpx.Response(200, json={"locked": False})
        if path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "llama3.1:8b"}]})
        if path == "/daily-kingstonia/":
            return httpx.Response(200, text='<a href="/daily-kingstonia/articles/poll">')
        if path == "/kingstonia-herald/":
            return httpx.Response(200, text="<p>new layout</p>")
        return httpx.Response(404)

    def preflight(data: Path) -> Preflight:
        return Preflight(data, "https://legit.test/api", httpx.MockTransport(handler), clock=lambda: clock["now"])

    async def clock_check(ahead: float, elapsed: float = 0, latency: float = 0) -> list:
        server.update(ahead=ahead, latency=latency)
        check = preflight(data)
        clock["now"] += timedelta(seconds=elapsed)  # earlier checks took this long
        await check.platform()
        return [r[0] for r in check.results if r[1] == "clock"]

    saved = {k: os.environ.get(k) for k in ("CTN_TEAM_KEY", "CTN_BOT_KEY", "LLM", "LLM_JUDGE", "GROQ_API_KEY")}
    saved_env, saved_ledger = llm.load_env, llm._LEDGER
    llm.load_env, llm._LEDGER = lambda *a, **k: None, llm.UsageLedger(":memory:")
    try:
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)
            check = preflight(data)
            os.environ.pop("CTN_TEAM_KEY", None)
            os.environ["CTN_BOT_KEY"] = "x"
            check.team_key()
            assert check.failed() and "old name" in check.failed()[0][2], check.results
            os.environ["CTN_TEAM_KEY"] = "a" * TEAM_KEY_LEN
            os.environ["LLM"], os.environ["LLM_JUDGE"] = "fake", "ollama:llama3.1:8b"
            check = preflight(data)
            check.team_key()
            await check.writers(ping=False)
            assert [r[1] for r in check.failed()] == ["writer"], check.results  # a fake writer can't go live
            os.environ["LLM_JUDGE"] = "ollama:qwen3:8b"
            check.results.clear()
            await check.writers(ping=False)
            assert any("pull it" in r[2] for r in check.failed()), check.results
            assert not any(r[1] == "judge" and r[0] == OK for r in check.results), "a missing model shown as fine"
            os.environ["GROQ_API_KEY"], os.environ["LLM_JUDGE"] = "test", "groq:some-model,ollama:qwen3:8b"
            check.results.clear()
            await check.writers(ping=False)
            assert not [r for r in check.failed() if r[1] == "judge"], "a missing last fallback stopped the launch"
            assert any(r[1] == "judge" and r[0] == WARN for r in check.results), check.results
            check.results.clear()
            await check.platform()
            await check.news()
            check.whitelist()
            assert not check.failed(), check.results
            assert any(r[1] == "papers" and r[0] == WARN for r in check.results), "a front page with no links"
            assert any(r[1] == "whitelist" and r[0] == WARN for r in check.results), "no collector yet"

            assert await clock_check(0, elapsed=40) == [OK], "time spent on earlier checks counted as skew"
            assert await clock_check(0, elapsed=400) == [OK], "time spent on earlier checks counted as skew"
            assert await clock_check(0, latency=70) == [OK], "a slow answer counted as skew"
            assert await clock_check(-400) == [FAIL] and await clock_check(400) == [FAIL]  # this machine is off
            assert await clock_check(-45) == [WARN] and await clock_check(10) == [OK]

            check = Preflight(data, clock=lambda: datetime(2026, 10, 20, tzinfo=timezone.utc))
            check.whitelist()
            assert [r[1] for r in check.failed()] == ["whitelist"], "started without the collector"
            check.results.clear()
            check.models(run=lambda *a, **k: subprocess.CompletedProcess(a, 1))
            assert {r[0] for r in check.results} == {WARN}
            tried = []

            def hangs(args, **kwargs):
                tried.append(args)
                if len(tried) == 1:
                    raise subprocess.TimeoutExpired(args, kwargs["timeout"])
                raise FileNotFoundError(args[0])
            check.results.clear()
            check.models(run=hangs)  # the first hangs, the second can't start: both warnings, nothing raised
            assert len(tried) == 2 and [r[0] for r in check.results] == [WARN, WARN], check.results
            check.results.clear()
            check.personas()
            check.accounts()
            check.disk()
            assert not check.failed(), check.results
    finally:
        llm.load_env, llm._LEDGER = saved_env, saved_ledger
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    print("preflight ok")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--llm", action="store_true", help="send the writer and the judge one short request each")
    parser.add_argument("--self-test", action="store_true", help="check the checks, offline")
    args = parser.parse_args()
    if args.self_test:
        asyncio.run(_demo())
    else:
        sys.exit(asyncio.run(main(args.llm)))
