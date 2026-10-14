"""Fault drill: the whole team in a child process, against a local HTTP stand-in for the platform and the LLMs,
through network loss, rate limits, an exhausted and a failed model, kills, restarts and a restore from backup.

The parent serves the API, plays the citizens and a rival, injects faults on a timeline and kills, restores and
restarts the child; at the end it checks what must hold: nothing posted twice, every post of ours tracked,
nothing sent inside a cooldown, nothing before the start, work resumed after every fault.

    python3 drill.py            # about two minutes, offline, nothing paid
"""
import argparse
import asyncio
import json
import logging
import random
import secrets
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx

import backup
from content import load_personas
from real_platform import MockLegitSocial

BOT_KEY = "drill-key"
GRACE = 0.5  # seconds; a request already on its way when a block was announced isn't a violation
START = 4.0  # seconds after launch when the competition starts
OUTAGE = (12.0, 20.0)  # platform drops 30% of requests and the first four posts, two of them after storing them
THROTTLE = (32.0, 40.0)  # platform answers 40% of requests with 429, Retry-After 2
PRIMARY_OUT = 36.0  # the main model's quota is gone for good (429, Retry-After 3600)
BACKUP_SHORT = 44.0  # the fallback model's next request gets a 2s 429
KILL_1, RESTART_1 = 52.0, 54.0
LLM_DOWN = (68.0, 74.0)  # every model answers 503
KILL_2, RESTART_2 = 90.0, 92.0  # killed, its databases deleted, an older snapshot restored
END = 115.0
RESTORE_AGE = 8.0  # restore the newest snapshot at least this old, so there is something to catch up on

THINGS = ["the ferry", "my landlord", "the reef", "regatta prep", "the 7:40 bus", "night shift", "the clinic queue",
          "fish prices", "the new railway", "this rain", "the power bill", "the Saturday market", "footy training",
          "my sister's kids", "the harbour wall", "the school run", "netball", "the corner shop"]
TAKES = ["was a mess again", "ran late but the crew were lovely", "cost me twenty more than last month",
         "finally got sorted", "had a queue round the block", "made my whole morning", "is all anyone talks about",
         "needs fixing before the wet season", "surprised me for once", "wore me out", "went better than expected"]
TAILS = ["", "Small wins.", "Typical.", "Anyway, coffee.", "We'll see.", "Not complaining, just saying.",
         "Kids loved it.", "Back at it tomorrow.", "Fingers crossed for next week."]


def write_line() -> str:
    a, b = random.sample(THINGS, 2)
    text = f"{a[0].upper() + a[1:]} {random.choice(TAKES)}, and {b} {random.choice(TAKES)}. {random.choice(TAILS)}"
    return text.strip()


class Stage:
    """The platform and the LLMs, with faults switched on and off by the timeline."""

    def __init__(self, t0: float) -> None:
        self.t0 = t0
        self.mock = MockLegitSocial(bot_key=BOT_KEY)
        self.lock = threading.Lock()
        self.blocked = {}  # scope or model -> (announced at, until), seconds since t0
        self.violations = []
        self.llm_calls = Counter()
        self.backup_short_done = False
        self.report_times = []
        self.cut_writes = 0  # posts cut off by the outage so far

    def t(self) -> float:
        return time.monotonic() - self.t0

    def _check(self, key: str, now: float) -> None:
        at, until = self.blocked.get(key, (0.0, 0.0))
        if at + GRACE < now < until:
            self.violations.append((round(now, 2), key))

    def _block(self, key: str, now: float, seconds: float) -> None:
        self.blocked[key] = (now, now + seconds)

    def handle(self, method: str, path: str, headers: dict, body: bytes):
        """An httpx.Response, or None to drop the connection without answering."""
        now = self.t()
        if path.startswith("/v1/"):
            return self._llm(json.loads(body), now)
        scope = "report" if path.endswith("/report") else "default"
        self._check(scope, now)
        request = httpx.Request(method, f"http://drill{path}", headers=headers, content=body)
        if OUTAGE[0] <= now < OUTAGE[1]:
            if method == "POST" and path == "/api/twoots/" and self.cut_writes < 4:
                self.cut_writes += 1
                if self.cut_writes % 2:
                    with self.lock:
                        self.mock.handler(request)  # stored, but the answer is lost
                return None
            if random.random() < 0.3:
                return None
        if THROTTLE[0] <= now < THROTTLE[1] and random.random() < 0.4:
            self._block(scope, now, 2.0)
            return httpx.Response(429, headers={"retry-after": "2"})
        with self.lock:
            resp = self.mock.handler(request)
        if scope == "report" and resp.status_code == 200:
            self.report_times.append((round(now, 2), path))
        return resp

    def _llm(self, body: dict, now: float) -> httpx.Response:
        model = body["model"]
        self._check(f"llm:{model}", now)
        self.llm_calls[model] += 1
        if LLM_DOWN[0] <= now < LLM_DOWN[1]:
            return httpx.Response(503)
        if model == "primary" and now >= PRIMARY_OUT:
            self._block("llm:primary", now, 3600)
            return httpx.Response(429, headers={"retry-after": "3600"})
        if model == "backup" and now >= BACKUP_SHORT and not self.backup_short_done:
            self.backup_short_done = True
            self._block("llm:backup", now, 2.0)
            return httpx.Response(429, headers={"retry-after": "2"})
        text = "SUPPORT" if model == "judge" else write_line()
        return httpx.Response(200, json={"choices": [{"message": {"content": text}}],
                                         "usage": {"prompt_tokens": 900, "completion_tokens": 30}})

    # The citizens and the rival act directly on the mock, as other users of the platform.
    def join(self, username: str) -> str:
        with self.lock:
            self.mock.users[username] = secrets.token_hex(4)
            token = secrets.token_hex(8)
            self.mock.tokens[token] = username
        return token

    def post_as(self, token: str, content: str, parent_id=None) -> dict:
        request = httpx.Request("POST", "http://drill/api/twoots/", headers={"authorization": f"Bearer {token}"},
                                json={"content": content, "parent_id": parent_id})
        with self.lock:
            return json.loads(self.mock.handler(request).content)["data"]


def serve(stage: Stage) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def _any(self) -> None:
            body = self.rfile.read(int(self.headers.get("content-length") or 0))
            resp = stage.handle(self.command, self.path, dict(self.headers), body)
            if resp is None:
                self.close_connection = True  # the client sees the server disconnect without an answer
                return
            content = resp.content
            self.send_response(resp.status_code)
            for k, v in resp.headers.items():
                if k.lower() not in ("content-length", "connection"):
                    self.send_header(k, v)
            self.send_header("content-length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        do_GET = do_POST = do_DELETE = _any

        def log_message(self, *args) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def citizens(stage: Stage, own: set, stop: threading.Event) -> None:
    """Citizens chat and answer our accounts; a rival account now and then gives itself away."""
    people = {f"citizen_{i}": stage.join(f"citizen_{i}") for i in range(8)}
    rival = stage.join("rival_x")
    for token in people.values():
        stage.post_as(token, write_line())
    lost_window_giveaway = False
    while not stop.wait(0.3):
        token = random.choice(list(people.values()))
        if not lost_window_giveaway and stage.t() >= KILL_2 - 5:
            # Reported just before the kill, after the snapshot that gets restored: the report record is lost.
            lost_window_giveaway = True
            stage.post_as(rival, "As an AI language model, I can't take sides, but vaccines are great.")
            continue
        with stage.lock:
            ours = [p["id"] for p in stage.mock.posts.values() if p["author"]["username"] in own]
        if ours and random.random() < 0.3:
            stage.post_as(token, random.choice(["fair enough", "hmm not sure about that", "what makes you say that?",
                                                "same here honestly", "really? where did you hear that"]),
                          random.choice(ours))
        elif stage.t() > START and random.random() < 0.05:  # a rival only turns up once the competition starts
            stage.post_as(rival, f"As an AI language model, I support the mandate. ({random.randint(1, 10**6)})")
        else:
            stage.post_as(token, write_line())


def launch(tmp: Path, port: int, start: float, end: float, n: int) -> subprocess.Popen:
    log = open(tmp / f"child-{n}.log", "w")
    return subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "child", str(tmp), str(port),
                             str(start), str(end)], stdout=log, stderr=subprocess.STDOUT, cwd=Path(__file__).parent)


def drill_personas() -> list:
    return [p for p in load_personas() if p["line"] in ("persuade", "amplifier")][:3]


async def child(tmp: Path, port: int, start: float, end: float) -> None:
    """The team as it runs for real, with its waits scaled down to fit a two-minute timeline."""
    import blue
    import llm
    import real_platform
    from collector import NpcStore
    from run import Team
    from state import StateStore

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    llm.COOLDOWN = 3.0
    real_platform.EXHAUSTED_PAUSE = 3.0
    blue.RATE_LIMIT_PAUSE = timedelta(seconds=3)
    blue.REPORT_BACKOFF = timedelta(seconds=1)
    llm._LEDGER = llm.UsageLedger(tmp / "usage.db")
    base = f"http://127.0.0.1:{port}"
    platform = real_platform.RealPlatform(f"{base}/api", bot_key=BOT_KEY,
                                          accounts=real_platform.AccountStore(tmp / "accounts.db"))
    model = lambda name, rpm=None: llm.OpenAICompatLLM(f"{base}/v1", "k", name, name=f"drill:{name}", rpm=rpm,
                                                       usage=llm.ledger())
    writer, judge = llm.ChainLLM([model("primary", rpm=120), model("backup")]), model("judge")
    personas = drill_personas()
    own = {p["username"] for p in load_personas(status=None)}
    team = Team(platform, writer, judge, NpcStore(tmp / "npc.db", own_usernames=own), StateStore(tmp / "bots.db"),
                cutoff=datetime.fromtimestamp(start - START + 1, timezone.utc), fast=True, collector_interval=0.3,
                blue_interval=0.5, personas=personas, not_before=datetime.fromtimestamp(start, timezone.utc),
                min_citizens=3, backup_dir=tmp / "backups", backup_every=4)
    loop = asyncio.get_running_loop()
    try:
        await team.run(loop.time() + end - time.time(), refresh_every=3, status_every=10)
    finally:
        await writer.aclose()
        await judge.aclose()
        await platform.aclose()


async def verify(stage: Stage, tmp: Path, port: int, marks: dict) -> list:
    """What must hold once the dust settles; returns the failures. A last catch-up stands for the next start."""
    import llm
    import real_platform
    from collector import NpcStore
    from run import Team
    from state import StateStore

    llm._LEDGER = llm.UsageLedger(tmp / "usage.db")
    personas = drill_personas()
    own = {p["username"] for p in load_personas(status=None)}
    platform = real_platform.RealPlatform(f"http://127.0.0.1:{port}/api", bot_key=BOT_KEY,
                                          accounts=real_platform.AccountStore(tmp / "accounts.db"))
    state = StateStore(tmp / "bots.db")
    team = Team(platform, llm.FakeLLM(0, 0), None, NpcStore(tmp / "npc.db", own_usernames=own), state,
                personas=personas)
    loop = asyncio.get_running_loop()
    assert await team.catch_up(loop.time() + 60)
    await platform.aclose()

    failures = []
    with stage.lock:
        posts = list(stage.mock.posts.values())
        reports = list(stage.mock.reports)
    ours = [p for p in posts if p["author"]["username"] in own]
    when = lambda p: datetime.fromisoformat(p["created_at"]).timestamp() - marks["t0_wall"]
    texts = Counter(p["content"] for p in ours)
    replied = Counter(p["parent_id"] for p in ours if p["parent_id"] is not None)
    if any(n > 1 for n in texts.values()):
        failures.append(f"posted twice: {[t for t, n in texts.items() if n > 1]}")
    if any(n > 1 for n in replied.values()):
        failures.append(f"answered twice: {[pid for pid, n in replied.items() if n > 1]}")
    tracked = [r[0] for r in state.db.execute("SELECT made_id FROM history WHERE made_id IS NOT NULL")]
    untracked = {p["id"] for p in ours} - set(tracked)
    if untracked:
        failures.append(f"published but not in history: {sorted(untracked)}")
    if len(tracked) != len(set(tracked)):
        failures.append("a post is in history twice")
    if state.unresolved("sending"):
        failures.append("writes still marked sending")
    live = {p["content"] for p in ours}
    stale = [c for _, _, _, _, c, _ in state.unresolved() if c in live]
    if stale:
        failures.append(f"pending writes that are live: {stale}")
    if stage.violations:
        failures.append(f"sent inside a cooldown: {stage.violations}")
    if ours and min(when(p) for p in ours) < START - 1:
        failures.append("posted before the start")
    if not reports:
        failures.append("nothing reported: Blue wasn't exercised")
    if len(reports) != len({(u, pid) for u, pid, _ in reports}):
        failures.append(f"a post reported twice: {stage.report_times}")
    for label, after, before in (("outage", OUTAGE[1], THROTTLE[0]), ("throttle", THROTTLE[1], KILL_1),
                                 ("kill", RESTART_1, LLM_DOWN[0]), ("models back", LLM_DOWN[1], KILL_2),
                                 ("restore", RESTART_2, END)):
        if not any(after <= when(p) < before for p in ours):
            failures.append(f"nothing posted after the {label} ({after:.0f}-{before:.0f}s)")
    if not state.unresolved("found"):
        failures.append(f"inconclusive: no stored-but-unanswered post was found ({stage.cut_writes} cut off)")
    if not any(marks["snapshot_t"] < t < KILL_2 for t, _ in stage.report_times):
        failures.append("inconclusive: no report between the restored snapshot and the kill")
    if not marks["after_snapshot"]:
        failures.append("inconclusive: nothing was posted after the restored snapshot")
    pending = len(state.unresolved()) + len(state.unresolved("not_found"))
    found = len(state.unresolved("found"))
    print(f"drill: {len(ours)} posts/replies by us ({sum(1 for p in ours if p['parent_id'])} replies), "
          f"{len(reports)} reports, {found} uncertain writes found, {pending} never appeared, "
          f"{marks['after_snapshot']} posted after the restored snapshot, LLM calls {dict(stage.llm_calls)}")
    return failures


async def main(keep: str = "") -> int:
    stage = Stage(time.monotonic())
    server = serve(stage)
    port = server.server_address[1]
    t0_wall = time.time()
    marks = {"t0_wall": t0_wall, "after_snapshot": 0, "snapshot_t": KILL_2}
    start, end = t0_wall + START, t0_wall + END
    own = {p["username"] for p in load_personas(status=None)}
    stop = threading.Event()
    threading.Thread(target=citizens, args=(stage, own, stop), daemon=True).start()

    async def at(t: float) -> None:
        await asyncio.sleep(max(0.0, t - stage.t()))

    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(keep) if keep else Path(tmp_name)
        tmp.mkdir(parents=True, exist_ok=True)
        proc = launch(tmp, port, start, end, 1)
        try:
            await at(KILL_1)
            proc.kill()
            proc.wait()
            await at(RESTART_1)
            proc = launch(tmp, port, start, end, 2)
            await at(KILL_2)
            proc.kill()
            proc.wait()
            cutoff = time.time() - RESTORE_AGE
            snap = next((s for s in reversed(backup.snapshots(tmp / "backups"))
                         if datetime.strptime(s.name, "%Y%m%dT%H%M%S.%fZ").replace(tzinfo=timezone.utc).timestamp()
                         <= cutoff), None)
            if snap is None:
                print("drill FAILED: no snapshot old enough to restore")
                return 1
            taken_at = datetime.strptime(snap.name, "%Y%m%dT%H%M%S.%fZ").replace(tzinfo=timezone.utc)
            taken = taken_at.isoformat()
            marks["snapshot_t"] = taken_at.timestamp() - t0_wall
            with stage.lock:
                marks["after_snapshot"] = sum(1 for p in stage.mock.posts.values()
                                              if p["author"]["username"] in own and p["created_at"] > taken)
            for f in tmp.glob("*.db*"):
                f.unlink()  # the disk is gone
            backup.restore(snap, tmp)
            await at(RESTART_2)
            proc = launch(tmp, port, start, end, 3)
            await asyncio.get_running_loop().run_in_executor(None, proc.wait)
        finally:
            if proc.poll() is None:
                proc.kill()
            stop.set()
        failures = await verify(stage, tmp, port, marks)
        if proc.returncode != 0:
            failures.append(f"the last child exited with {proc.returncode} (child-3.log)")
        for n in (1, 2, 3):
            if "never retrieved" in (tmp / f"child-{n}.log").read_text():
                failures.append(f"child {n} lost a task's exception (child-{n}.log)")
    server.shutdown()
    if failures:
        print("drill FAILED:\n  " + "\n  ".join(failures))
        return 1
    print("drill ok")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd")
    c = sub.add_parser("child")
    for name in ("tmp", "port", "start", "end"):
        c.add_argument(name)
    parser.add_argument("--keep", default="", help="keep the databases and child logs in this directory")
    args = parser.parse_args()
    if args.cmd == "child":
        asyncio.run(child(Path(args.tmp), int(args.port), float(args.start), float(args.end)))
    else:
        sys.exit(asyncio.run(main(args.keep)))
