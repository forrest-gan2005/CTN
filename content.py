import json
from pathlib import Path
from typing import Optional

CONTENT_DIR = Path(__file__).parent / "content"

PERSONA_FIELDS = {
    "username", "name", "age", "job", "home", "line", "covid", "why_yes",
    "style", "angles", "topics", "views", "examples",
}
# persuade: argues YES from their own life; mobilise: gets YES voters to turn up and keeps the issue on minds;
# bridge: has weighed both sides and draws NO-leaning and undecided citizens over; background: an ordinary
# citizen who rarely talks referendum; amplifier: mostly likes, reposts and follows.
LINES = {"persuade", "mobilise", "bridge", "background", "amplifier"}
STATUSES = {"active", "reserve"}


def load_world() -> str:
    return "\n\n".join(
        (CONTENT_DIR / name).read_text(encoding="utf-8").strip() for name in ("world.md", "rules.md", "arguments.md")
    )


def load_vaccine_angles() -> list:
    return json.loads((CONTENT_DIR / "archetypes.json").read_text(encoding="utf-8"))["vaccine_angles"]


def load_personas(status: Optional[str] = "active") -> list:
    """Load and validate the persona library, failing fast on a hand-editing mistake.

    Reserve personas wait to replace banned accounts; pass status=None to get every persona.
    """
    personas = json.loads((CONTENT_DIR / "personas.json").read_text(encoding="utf-8"))
    seen = set()
    for p in personas:
        if p.get("status", "active") not in STATUSES:
            raise ValueError(f"persona {p['username']!r} has unknown status {p['status']!r}")
        missing = PERSONA_FIELDS - p.keys()
        if missing:
            raise ValueError(f"persona {p.get('username')!r} is missing {sorted(missing)}")
        if p["line"] not in LINES:
            raise ValueError(f"persona {p['username']!r} has unknown line {p['line']!r}")
        if p["username"] in seen:
            raise ValueError(f"duplicate username {p['username']!r}")
        seen.add(p["username"])
    return [p for p in personas if status is None or p.get("status", "active") == status]
