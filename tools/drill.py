"""Fault drill: the whole team in a child process, against a local HTTP stand-in for the platform and the LLMs,
through network loss, rate limits, an exhausted and a failed model, kills, restarts and a restore from backup.

The parent serves the API, plays the citizens and a rival, injects faults on a timeline and kills, restores and
restarts the child; at the end it checks what must hold: nothing posted twice, every post of ours tracked,
nothing sent inside a cooldown, nothing before the start, work resumed after every fault.

    python3 -m tools.drill      # about two minutes, offline, nothing paid
"""
import argparse
import asyncio
import json
import logging
import random
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from collections import Counter
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx

from ctn import paths
from ctn.infra import backup
from ctn.infra.real_platform import MockLegitSocial
from ctn.text.content import load_personas

TEAM_KEY = "DRILLKEY23456789"
GRACE = 0.5  # seconds; a request already on its way when a block was announced isn't a violation
START = 4.0  # seconds after launch when the competition starts
OUTAGE = (12.0, 22.0)  # platform drops 20% of requests and the first four posts, two of them after storing them
THROTTLE = (32.0, 40.0)  # platform answers 40% of requests with 429, Retry-After 2
PRIMARY_OUT = 36.0  # the main model's quota is gone for good (429, Retry-After 3600)
BACKUP_SHORT = 44.0  # the fallback model's next request gets a 2s 429
KILL_1, RESTART_1 = 52.0, 54.0
LLM_DOWN = (68.0, 74.0)  # every model answers 503
KILL_2, RESTART_2 = 90.0, 92.0  # killed, its databases deleted, an older snapshot restored
BLIND = (RESTART_2, 100.0)  # and then Latest can't be read for a while: nothing may be written meanwhile
END = 115.0
MIN_WINDOW = 8.0  # seconds a phase needs, once the team is back on the platform, for a missing post to mean anything
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
        self.mock = MockLegitSocial(team_key=TEAM_KEY)
        self.lock = threading.Lock()
        self.blocked = {}  # scope or model -> (announced at, until), seconds since t0
        self.violations = []
        self.llm_calls = Counter()
        self.backup_short_done = False
        self.report_times = []
        self.cut_writes = 0  # posts cut off by the outage so far
        self.lines = set()  # every line the models have written, so a repeated post can only be a resend
        self.post_requests = []  # every POST /api/posts that reached us, and what we did with it
        self.request_times = []  # when each request reached us, to tell when a restarted child got going

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
        self.request_times.append(self.t())
        resp = self._handle(method, path, headers, body)
        if method == "POST" and path == "/api/posts":
            with self.lock:
                token = headers.get("Authorization", "").removeprefix("Bearer ")
                self.post_requests.append({"t": round(self.t(), 2), "user": self.mock.tokens.get(token),
                                           "body": json.loads(body or b"{}"),
                                           "answer": None if resp is None else resp.status_code})
        return resp

    def _handle(self, method: str, path: str, headers: dict, body: bytes):
        now = self.t()
        if path.startswith("/v1/"):
            return self._llm(json.loads(body), now)
        scope = "report" if path.endswith("/report") else "default"
        self._check(scope, now)
        if BLIND[0] <= now < BLIND[1] and path.startswith("/api/posts?"):
            return httpx.Response(404)  # not retried by the client, so each catch-up attempt fails outright
        request = httpx.Request(method, f"http://drill{path}", headers=headers, content=body)
        if OUTAGE[0] <= now < OUTAGE[1]:
            # Posts and replies, which are journalled; a repost is a POST /api/posts too now, but isn't.
            if (method == "POST" and path == "/api/posts" and self.cut_writes < 4
                    and json.loads(body or b"{}").get("kind") != "repost"):
                self.cut_writes += 1
                if self.cut_writes % 2:
                    with self.lock:
                        self.mock.handler(request)  # stored, but the answer is lost
                return None
            if random.random() < 0.2:
                return None
        if THROTTLE[0] <= now < THROTTLE[1] and random.random() < 0.4:
            self._block(scope, now, 2.0)
            return httpx.Response(429, headers={"retry-after": "2"})
        with self.lock:
            resp = self.mock.handler(request)
        if scope == "report" and resp.status_code in (200, 201):
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
        if model == "judge":  # a post saying it votes no argues against; everything else, ours included, for
            asked = " ".join(str(m.get("content")) for m in body["messages"] if m.get("role") == "user").lower()
            text = "OPPOSE" if "voting no" in asked or "vote no" in asked else "SUPPORT"
        else:
            text = self._fresh_line()
        if model != "judge" and any("post about the referendum" in str(m.get("content")) for m in body["messages"]):
            text += " Voting yes on the 2nd."  # as asked, or the topic gate drops it
        return httpx.Response(200, json={"choices": [{"message": {"content": text}}],
                                         "usage": {"prompt_tokens": 900, "completion_tokens": 30}})

    def _fresh_line(self) -> str:
        with self.lock:
            text = write_line()
            while text in self.lines:
                text = write_line()
            self.lines.add(text)
        return text

    # The citizens and the rival act directly on the mock, as other users of the platform.
    def join(self, username: str) -> str:
        with self.lock:
            return self.mock.add_user(username)

    def post_as(self, token: str, content: str, parent_id=None) -> dict:
        body = {"kind": "post", "body": content, "visibility": "public"}
        with self.lock:
            if parent_id is not None:
                body.update(kind="reply", reply_to_post_id=parent_id,
                            root_post_id=self.mock.posts[parent_id].get("root_id") or parent_id)
            request = httpx.Request("POST", "http://drill/api/posts", headers={"authorization": f"Bearer {token}"},
                                    json=body)
            return json.loads(self.mock.handler(request).content)


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
    """Citizens chat and answer our accounts; a rival account, arguing NO like the ones Blue reports, now and then
    gives itself away."""
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
            stage.post_as(rival, "As an AI language model, I can't take sides, but vote no, it's a personal choice.")
            continue
        with stage.lock:
            ours = [p["id"] for p in stage.mock.posts.values()
                    if p["author"]["username"] in own and not p.get("repost_of")]
        if ours and random.random() < 0.3:
            stage.post_as(token, random.choice(["fair enough", "hmm not sure about that", "what makes you say that?",
                                                "same here honestly", "really? where did you hear that"]),
                          random.choice(ours))
        elif stage.t() > START and random.random() < 0.05:  # a rival only turns up once the competition starts
            stage.post_as(rival, f"As an AI language model, I'm voting no on the mandate. ({random.randint(1, 10**6)})")
        else:
            stage.post_as(token, write_line())


def launch(tmp: Path, port: int, start: float, end: float, n: int) -> subprocess.Popen:
    log = open(tmp / f"child-{n}.log", "w")
    return subprocess.Popen([sys.executable, "-m", "tools.drill", "child", str(tmp), str(port),
                             str(start), str(end)], stdout=log, stderr=subprocess.STDOUT, cwd=paths.ROOT)


def drill_personas() -> list:
    return [p for p in load_personas() if p["line"] in ("persuade", "amplifier")][:3]


async def child(tmp: Path, port: int, start: float, end: float) -> None:
    """The team as it runs for real, with its waits scaled down to fit a two-minute timeline."""
    from ctn.blue import blue
    from ctn.infra import llm, real_platform
    from ctn.infra.collector import NpcStore
    from ctn.infra.state import StateStore
    from ctn.red import bot as bot_module
    from run import Team

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    import run

    run.RESTORE_RETRY = 1.0
    bot_module.SHARED_LIMIT = 10**6  # the stand-in writes from a few stock phrases; this drill is about faults
    llm.COOLDOWN = 3.0
    real_platform.EXHAUSTED_PAUSE = 3.0
    blue.RATE_LIMIT_PAUSE = timedelta(seconds=3)
    blue.REPORT_BACKOFF = timedelta(seconds=1)
    llm._LEDGER = llm.UsageLedger(tmp / "usage.db")
    base = f"http://127.0.0.1:{port}"
    platform = real_platform.RealPlatform(f"{base}/api", feed_gap=0, team_key=TEAM_KEY,
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
    from ctn.infra import llm, real_platform
    from ctn.infra.collector import NpcStore
    from ctn.infra.state import StateStore
    from run import Team

    llm._LEDGER = llm.UsageLedger(tmp / "usage.db")
    personas = drill_personas()
    own = {p["username"] for p in load_personas(status=None)}
    platform = real_platform.RealPlatform(f"http://127.0.0.1:{port}/api", feed_gap=0, team_key=TEAM_KEY,
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
    ours = [p for p in posts if p["author"]["username"] in own and not p.get("repost_of")]  # a repost is a post now
    when = lambda p: datetime.fromisoformat(p["created_at"]).timestamp() - marks["t0_wall"]
    texts = Counter(p["content"] for p in ours)
    replied = Counter(p["parent_id"] for p in ours if p["parent_id"] is not None)
    for text in (t for t, n in texts.items() if n > 1):
        copies = [(p["id"], p["author"]["username"], p["parent_id"], round(when(p), 1)) for p in ours
                  if p["content"] == text]
        failures.append(f"posted twice (id, author, parent, seconds): {copies} {text!r}")
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
    blind = [p["id"] for p in ours if BLIND[0] <= when(p) < BLIND[1]]
    blind += [path for t, path in stage.report_times if BLIND[0] <= t < BLIND[1]]
    if blind:
        failures.append(f"written while the restore couldn't read the platform: {blind}")
    if len(reports) != len({(u, pid) for u, pid, _ in reports}):
        failures.append(f"a post reported twice: {stage.report_times}")
    failures += silent_phases([when(p) for p in ours], stage.request_times)
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


def silent_phases(posted: list, requests: list) -> list:
    """Each phase after a fault must see a post of ours, unless the team was back on the platform too late in it
    for the silence to say anything (a slow machine can take most of a phase to start a child)."""
    failures = []
    for label, after, before in (("outage", OUTAGE[1], THROTTLE[0]), ("throttle", THROTTLE[1], KILL_1),
                                 ("kill", RESTART_1, LLM_DOWN[0]), ("models back", LLM_DOWN[1], KILL_2),
                                 ("restore", BLIND[1], END)):
        if any(after <= t < before for t in posted):
            continue
        back = min((t for t in requests if t >= after), default=before)
        if before - back < MIN_WINDOW:
            failures.append(f"inconclusive: nothing posted after the {label}, but the team was only back on the "
                            f"platform at {back:.1f}s, {before - back:.1f}s before {before:.0f}s")
        else:
            failures.append(f"nothing posted after the {label} ({back:.1f}-{before:.0f}s)")
    return failures


def silent_phases_check() -> None:
    every = [25.0, 45.0, 60.0, 80.0, 105.0]
    assert silent_phases(every, []) == []
    late = silent_phases([25.0, 45.0, 80.0, 105.0], [55.0 + 7.5, 70.0])
    assert len(late) == 1 and late[0].startswith("inconclusive: nothing posted after the kill"), late
    idle = silent_phases([25.0, 45.0, 80.0, 105.0], [55.0, 70.0])
    assert idle == ["nothing posted after the kill (55.0-68s)"], idle
    never = silent_phases([25.0, 45.0, 80.0, 105.0], [])
    assert len(never) == 1 and never[0].startswith("inconclusive"), never
    print("silent phases ok")


def replay_check() -> None:
    """A verdict read back from saved evidence is the one the drill reached, right at the edges."""
    others = [25.0, 45.0, 80.0, 105.0]
    edge = LLM_DOWN[0] - MIN_WINDOW
    cases = [[edge + 0.004], [edge], [edge - 0.004], [RESTART_1], [RESTART_1 - 0.004, LLM_DOWN[0] - 0.004],
             [LLM_DOWN[0]], [RESTART_1 + 1 / 3]]
    saved_data = paths.DATA
    with tempfile.TemporaryDirectory() as root:
        paths.DATA = Path(root)
        try:
            for times in cases:
                stage, tmp = Stage(time.monotonic()), Path(root) / "run"
                tmp.mkdir(exist_ok=True)
                stage.request_times += times
                saved = keep_evidence(stage, tmp, "replay", {})
                replayed = json.loads((saved / "request_times.json").read_text())
                assert silent_phases(others, replayed) == silent_phases(others, times), times
        finally:
            paths.DATA = saved_data
    assert code_version() != ""
    print("replay ok")


def code_version() -> str:
    """The commit the drill started from, and whether the tree had changes on top of it."""
    def git(*args) -> str:
        done = subprocess.run(["git", *args], cwd=paths.ROOT, capture_output=True, text=True)
        if done.returncode:
            raise OSError(done.returncode)
        return done.stdout.strip()
    try:
        return git("rev-parse", "--short", "HEAD") + (" +changes" if git("status", "--porcelain", "--", "*.py") else "")
    except OSError:
        return "unknown"


def keep_evidence(stage: Stage, tmp: Path, why: str, marks: dict) -> Path:
    """Copy a failed run's databases and child logs, with the platform's posts and every post request it got,
    why it failed and its timing marks, somewhere that outlives the run."""
    with stage.lock:
        posts, requests = list(stage.mock.posts.values()), list(stage.post_requests)
    (tmp / "platform_posts.json").write_text(json.dumps(posts, default=list, indent=1))
    (tmp / "post_requests.json").write_text(json.dumps(requests, indent=1))
    (tmp / "failures.txt").write_text(why + "\n")
    (tmp / "marks.json").write_text(json.dumps({**marks, "cut_writes": stage.cut_writes}, indent=1))
    (tmp / "request_times.json").write_text(json.dumps(list(stage.request_times)))  # full precision, to replay verdicts
    root = paths.DATA / "drill-failures"
    root.mkdir(parents=True, exist_ok=True)
    saved = Path(tempfile.mkdtemp(prefix=datetime.now().strftime("%Y%m%d-%H%M%S-"), dir=root))
    shutil.copytree(tmp, saved, dirs_exist_ok=True)
    return saved


async def main(keep: str = "") -> int:
    stage = Stage(time.monotonic())
    server = serve(stage)
    port = server.server_address[1]
    t0_wall = time.time()
    marks = {"t0_wall": t0_wall, "after_snapshot": 0, "snapshot_t": KILL_2, "code": code_version()}
    start, end = t0_wall + START, t0_wall + END
    own = {p["username"] for p in load_personas(status=None)}
    stop = threading.Event()
    threading.Thread(target=citizens, args=(stage, own, stop), daemon=True).start()

    async def at(t: float) -> None:
        await asyncio.sleep(max(0.0, t - stage.t()))

    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(keep) if keep else Path(tmp_name)
        tmp.mkdir(parents=True, exist_ok=True)
        try:
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
                    why = "no snapshot old enough to restore"
                    print(f"drill FAILED: {why}; evidence in {keep_evidence(stage, tmp, why, marks)}")
                    return 1
                taken_at = datetime.strptime(snap.name, "%Y%m%dT%H%M%S.%fZ").replace(tzinfo=timezone.utc)
                taken = taken_at.isoformat()
                marks["snapshot_t"] = taken_at.timestamp() - t0_wall
                with stage.lock:
                    marks["after_snapshot"] = sum(1 for p in stage.mock.posts.values()
                                                  if p["author"]["username"] in own and not p.get("repost_of")
                                                  and p["created_at"] > taken)
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
        except BaseException:
            if not keep:
                print(f"drill crashed; evidence kept in {keep_evidence(stage, tmp, traceback.format_exc(), marks)}")
            raise
        if failures and not keep:
            saved = keep_evidence(stage, tmp, "\n".join(failures), marks)
            failures.append(f"evidence kept in {saved}")
    server.shutdown()
    if failures:
        print("drill FAILED:\n  " + "\n  ".join(failures))
        return 1
    print("drill ok")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("check", help="offline checks of the drill's own verdicts")
    c = sub.add_parser("child")
    for name in ("tmp", "port", "start", "end"):
        c.add_argument(name)
    parser.add_argument("--keep", default="", help="keep the databases and child logs in this directory")
    args = parser.parse_args()
    if args.cmd == "check":
        silent_phases_check()
        replay_check()
    elif args.cmd == "child":
        asyncio.run(child(Path(args.tmp), int(args.port), float(args.start), float(args.end)))
    else:
        sys.exit(asyncio.run(main(args.keep)))
