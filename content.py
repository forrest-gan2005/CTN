import json
from pathlib import Path

CONTENT_DIR = Path(__file__).parent / "content"

PERSONA_FIELDS = {
    "username", "name", "age", "job", "home", "line", "covid", "why_yes",
    "style", "angles", "topics", "examples",
}
LINES = {"persuade", "mobilise"}


def load_world() -> str:
    return "\n\n".join(
        (CONTENT_DIR / name).read_text(encoding="utf-8").strip() for name in ("world.md", "rules.md", "arguments.md")
    )


def load_personas() -> list:
    """Load and validate the persona library, failing fast on a hand-editing mistake."""
    personas = json.loads((CONTENT_DIR / "personas.json").read_text(encoding="utf-8"))
    seen = set()
    for p in personas:
        missing = PERSONA_FIELDS - p.keys()
        if missing:
            raise ValueError(f"persona {p.get('username')!r} is missing {sorted(missing)}")
        if p["line"] not in LINES:
            raise ValueError(f"persona {p['username']!r} has unknown line {p['line']!r}")
        if p["username"] in seen:
            raise ValueError(f"duplicate username {p['username']!r}")
        seen.add(p["username"])
    return personas
