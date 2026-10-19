"""The NO worries citizens raise most at the moment, matched to the ones content/arguments.md answers, so a post that
answers the other side answers what people here are actually saying."""
import re
from collections import Counter
from typing import Iterable

from ctn.text.guard import leans_no

# Each worry arguments.md answers: a short label in plain words (never its quote: arguments.md is in the system
# prompt, and a post repeating eight of its words is dropped as a prompt leak), and the words that mark it.
WORRIES = {
    "two-tier": ("that it splits people into two tiers", r"two[- ]tier|second[- ]class|segregat"),
    "jobs": ("lost jobs and businesses", r"\bjobs?\b|\bbusiness(es)?\b|livelihood"),
    "tourism": ("harm to tourism", r"touris|visitors?\b"),
    "not here": ("that the virus isn't even here yet",
                 r"not (even )?here( yet)?|no cases|no sustained|data,? not dread"),
    "powers": ("giving the government too much power", r"power grab|government power|overreach|\bcontrol\b"),
    "temporary": ("that the rules must be temporary", r"sunset|temporary|end date|never go away|permanent"),
    "id card": ("that the ID cards could be used to track people", r"\bid cards?\b|\btrack(ed|ing)?\b|surveillance"),
    "which places": ("that the list of places isn't out", r"which (places|venues)|venues? list|list of venues"),
    "kids": ("what it means for kids", r"\bkids?\b|\bchildren\b"),
    "too new": ("that the vaccine is too new", r"too new|rushed|long[- ]term|side effects?|untested"),
    "choice": ("that it should be a personal choice", r"personal choice|my body|\bchoice\b|coerc"),
}
# arguments.md answers these by agreeing (limits are fair; the list should be published), and a post built on a
# concession reads as doubt: they are never the one a post is told to answer.
CONCEDED = {"temporary", "which places"}
PATTERNS = {name: re.compile(rx, re.IGNORECASE) for name, (_, rx) in WORRIES.items()}
# Words like "overreach" or "two-tier" make leans_no say NO by themselves, so a YES post granting the worry counted.
SAYS_YES = re.compile(r"\b(vot(e|ing)|tick(ing)?|say(ing)?)\s+yes\b|\byes\s+(from me|vote)\b", re.IGNORECASE)


def top_worries(texts: Iterable[str], n: int = 3, at_least: int = 3) -> list:
    """(name, label) of the worries the most NO-leaning posts among `texts` raise, most raised first, leaving out
    the CONCEDED ones; one raised by fewer than `at_least` posts doesn't count."""
    counts = Counter(name for text in texts if leans_no(text) and not SAYS_YES.search(text)
                     for name, pattern in PATTERNS.items() if pattern.search(text) and name not in CONCEDED)
    return [(name, WORRIES[name][0]) for name, k in counts.most_common(n) if k >= at_least]


if __name__ == "__main__":
    from ctn.text.content import load_world
    from ctn.text.guard import copies

    world = load_world()
    for name, (label, _) in WORRIES.items():
        assert not copies(f"People say {label}, and I disagree with them on this one.", world), name
    no = ["Voting no, the new ID card means surveillance and tracking.",
          "Voting no because the ID cards will be used to track us.",
          "I'm voting no, there's surveillance in this ID card plan.",
          "Voting no: a sunset clause or nothing, emergency rules never go away.",
          "I'm voting no, there's no sunset clause and it could be permanent.",
          "Vote no. Without an end date this is forever, and it's permanent.",
          "I'm voting no, it should be a personal choice."]
    yes = ["I'm voting YES, though I get the fear of a two-tier island."] * 5
    assert top_worries(no + yes) == [("id card", WORRIES["id card"][0])], top_worries(no + yes)
    assert top_worries(no, at_least=1)[0][0] == "id card" and "temporary" not in dict(top_worries(no, at_least=1)), \
        "a worry answered only by agreeing is never handed out"
    assert top_worries(yes) == [], "a YES post's mention of a worry isn't one raised"
    print("worries ok")
