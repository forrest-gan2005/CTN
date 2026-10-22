import json
import re
from typing import Optional

from ctn import paths

CONTENT_DIR = paths.CONTENT

PERSONA_FIELDS = {
    "username", "name", "age", "job", "home", "line", "covid", "why_yes",
    "style", "angles", "topics", "views", "examples",
}
# persuade: argues YES from their own life; mobilise: gets YES voters to turn up and keeps the issue on minds;
# bridge: has weighed both sides and draws NO-leaning and undecided citizens over; background: an ordinary
# citizen who rarely talks referendum; amplifier: mostly likes, reposts and follows.
LINES = {"persuade", "mobilise", "bridge", "background", "amplifier"}
STATUSES = {"active", "reserve", "banned"}  # banned: suspended by the platform, kept so the team still knows it
# How replies and case posts meet a NO worry: "granted" names what is fair in it first, as through 21 October 17:50;
# "exp3" names it without granting it. Granted evenings moved few views toward YES too (20 October 21:00-01:00: 0-3
# an hour, with 3-5 moving away), so the two are compared at the same hours of the day before either is dropped.
WORDING = "exp3"
ARGUMENTS = {"granted": "arguments-granted.md", "exp3": "arguments.md"}
ARGUMENTS_REPLY = "arguments-reply.md"  # exp3's, for replies to citizens: see reply_world


def load_world() -> str:
    return "\n\n".join(
        (CONTENT_DIR / name).read_text(encoding="utf-8").strip()
        for name in ("world.md", "rules.md", ARGUMENTS[WORDING])
    )


def reply_world(world: str) -> str:
    """The world with the answers a reply to a citizen gives (bot.STORY_REPLIES): no "a NO rules out", no "the people
    decide", no "not on the ballot"."""
    arguments = (CONTENT_DIR / ARGUMENTS[WORDING]).read_text(encoding="utf-8").strip()
    return world.replace(arguments, (CONTENT_DIR / ARGUMENTS_REPLY).read_text(encoding="utf-8").strip())


def without_arguments(world: str) -> str:
    """The world without the answers to NO worries, for writing that leaves the referendum out."""
    arguments = (CONTENT_DIR / ARGUMENTS[WORDING]).read_text(encoding="utf-8").strip()
    return world.replace("\n\n" + arguments, "")


def load_vaccine_angles() -> list:
    return json.loads((CONTENT_DIR / "archetypes.json").read_text(encoding="utf-8"))["vaccine_angles"]


def load_journalists() -> dict:
    """The papers' journalists on Legit Social (journalists.json): username -> {"name", "paper", "role", "cares",
    "angles"}. Citizens read the papers daily, and the journalists write from what they discuss here."""
    journalists = json.loads((CONTENT_DIR / "journalists.json").read_text(encoding="utf-8"))
    for username, j in journalists.items():
        if not all(j.get(k) for k in ("name", "paper", "role", "cares", "angles")):
            raise ValueError(f"journalist {username!r} needs a name, paper, role, what they care about and angles")
    return journalists


def load_backstories() -> dict:
    """The key accounts' own stories (backstories.md): username -> {"story", "people", "voice"}."""
    text = (CONTENT_DIR / "backstories.md").read_text(encoding="utf-8")
    stories = {}
    for block in re.split(r"^## ", text, flags=re.M)[1:]:
        username, _, body = block.partition("\n")
        story, _, rest = body.partition("\nPeople: ")
        people, _, voice = rest.partition("\nVoice:\n")
        habits = [line[2:].strip() for line in voice.splitlines() if line.startswith("- ")]
        if not (story.strip() and people.strip() and habits):
            raise ValueError(f"backstory for {username!r} needs a story, a People line and Voice habits")
        paragraphs = [" ".join(par.split()) for par in re.split(r"\n\s*\n", story.strip())]
        stories[username.strip()] = {"story": "\n\n".join(paragraphs), "people": people.strip(), "voice": habits}
    return stories


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
    stories = load_backstories()
    unknown = stories.keys() - seen
    if unknown:
        raise ValueError(f"backstories for unknown accounts {sorted(unknown)}")
    for p in personas:
        if p["username"] in stories:
            p["backstory"] = stories[p["username"]]
    return [p for p in personas if status is None or p.get("status", "active") == status]
