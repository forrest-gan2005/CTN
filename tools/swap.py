"""Replace a suspended account with a reserve persona of the same line, then register the reserve.

    python3 -m tools.swap emisha                 pick a reserve of emisha's line, show what would change
    python3 -m tools.swap emisha --apply         mark emisha banned, the reserve active, and register it
    python3 -m tools.swap emisha --reserve zoem95 --apply
    python3 -m tools.swap --self-test

The live process reads the personas only when it starts: restart it afterwards. Registering uses the team code in
.env and prints neither it nor any password. Run it from the checkout live runs from: the new account's password
is kept in that checkout's data/accounts.db, and live can't log in with one kept anywhere else.
"""
import argparse
import asyncio
import json
import random
import sys
import tempfile
from pathlib import Path

from ctn.text.content import CONTENT_DIR

PERSONAS = CONTENT_DIR / "personas.json"


def pick(personas: list, banned: str, reserve=None) -> dict:
    """The reserve persona to bring in for `banned`: the one named, or a random reserve of the same line."""
    gone = next((p for p in personas if p["username"] == banned), None)
    if gone is None:
        raise SystemExit(f"no persona {banned}")
    if gone.get("status", "active") == "reserve":
        raise SystemExit(f"{banned} is a reserve, not one of the team")
    if gone.get("replaced_by"):
        raise SystemExit(f"{banned} was already replaced by {gone['replaced_by']}")
    pool = [p for p in personas if p.get("status") == "reserve" and p["line"] == gone["line"]]
    if reserve is not None:
        pool = [p for p in pool if p["username"] == reserve]
        if not pool:
            raise SystemExit(f"{reserve} is not a reserve of the {gone['line']} line")
    if not pool:
        raise SystemExit(f"no reserve left for the {gone['line']} line")
    return random.choice(pool)


def swap(path: Path, banned: str, reserve: str) -> None:
    personas = json.loads(path.read_text(encoding="utf-8"))
    for p in personas:
        if p["username"] == banned:
            p["status"], p["replaced_by"] = "banned", reserve
        elif p["username"] == reserve:
            p["status"] = "active"
    path.write_text(json.dumps(personas, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


async def register(persona: dict) -> None:
    from run import real_platform

    platform = real_platform()
    try:
        await platform.ensure_account(persona["username"], persona["name"])
    finally:
        await platform.aclose()


def _self_test() -> None:
    personas = [{"username": "a", "line": "bridge"}, {"username": "b", "line": "bridge", "status": "reserve"},
                {"username": "c", "line": "persuade", "status": "reserve"}]
    assert pick(personas, "a")["username"] == "b"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "personas.json"
        path.write_text(json.dumps(personas))
        swap(path, "a", "b")
        status = {p["username"]: p.get("status", "active") for p in json.loads(path.read_text())}
        assert status == {"a": "banned", "b": "active", "c": "reserve"}, status
        again = json.loads(path.read_text())
        try:
            pick(again, "a")
        except SystemExit:
            pass
        else:
            raise AssertionError("an account is replaced once")
    assert pick([{**personas[0], "status": "banned"}] + personas[1:], "a")["username"] == "b", "marked banned already"
    for bad in (lambda: pick(personas, "x"), lambda: pick(personas, "a", "c"), lambda: pick(personas, "b"),
                lambda: pick([{"username": "a", "line": "mobilise"}], "a")):
        try:
            bad()
        except SystemExit:
            continue
        raise AssertionError("should have refused")
    print("swap ok")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("banned", nargs="?")
    parser.add_argument("--reserve")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return _self_test()
    if not args.banned:
        parser.error("name the suspended account")
    personas = json.loads(PERSONAS.read_text(encoding="utf-8"))
    new = pick(personas, args.banned, args.reserve)
    print(f"{args.banned} -> {new['username']} ({new['line']}): {new['name']}, {new['age']}, {new['home']}, "
          f"who {new['job']}")
    if not args.apply:
        print("dry run: add --apply to switch and register")
        return
    from ctn import paths

    if not (paths.DATA / "accounts.db").exists():
        raise SystemExit(f"no {paths.DATA / 'accounts.db'}: run this from the checkout live runs from")
    swap(PERSONAS, args.banned, new["username"])
    asyncio.run(register(new))
    print(f"{new['username']} registered and active; restart live to bring it in")


if __name__ == "__main__":
    sys.exit(main())
