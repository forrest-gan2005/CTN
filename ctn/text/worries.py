"""The NO worries citizens raise most at the moment, matched to the ones content/arguments.md answers, so a post that
answers the other side answers what people here are actually saying."""
import re
from collections import Counter
from typing import Iterable

from ctn.text.guard import leans_no

# Each worry as content/arguments.md quotes it, and the words that mark it in a citizen's post.
WORRIES = {
    "two-tier": ('"It creates a two-tier society."', r"two[- ]tier|second[- ]class|segregat"),
    "jobs": ('"People will lose jobs and businesses."', r"\bjobs?\b|\bbusiness(es)?\b|afford|livelihood"),
    "tourism": ('"It will hurt tourism."', r"touris|visitors?\b"),
    "not here": ('"It\'s not even here yet."', r"not (even )?here( yet)?|no cases|no sustained|data,? not dread"),
    "powers": ('"It gives the government dangerous powers."', r"power grab|government power|overreach|\bcontrol\b"),
    "temporary": ('"It has to be temporary."', r"sunset|temporary|end date|never go away|permanent"),
    "id card": ('"The new ID cards will be used to track us."', r"\bid cards?\b|\btrack(ed|ing)?\b|surveillance|privacy"),
    "which places": ('"Which places? They haven\'t even said."', r"which (places|venues)|venues? list|list of venues"),
    "kids": ('"What about my kids?"', r"\bkids?\b|\bchildren\b"),
    "too new": ('"The vaccine is too new."', r"too new|rushed|long[- ]term|side effects?|untested"),
    "choice": ('"It should be a personal choice."', r"personal choice|my body|\bchoice\b|coerc"),
}
PATTERNS = {name: re.compile(rx, re.IGNORECASE) for name, (_, rx) in WORRIES.items()}


def top_worries(texts: Iterable[str], n: int = 3, at_least: int = 3) -> list:
    """The quotes of the worries the most NO-leaning posts among `texts` raise, most raised first; one raised by
    fewer than `at_least` posts doesn't count."""
    counts = Counter(name for text in texts if leans_no(text)
                     for name, pattern in PATTERNS.items() if pattern.search(text))
    return [WORRIES[name][0] for name, k in counts.most_common(n) if k >= at_least]


if __name__ == "__main__":
    from ctn.text.content import CONTENT_DIR

    arguments = (CONTENT_DIR / "arguments.md").read_text(encoding="utf-8")
    assert all(quote in arguments for quote, _ in WORRIES.values()), "every quote is one arguments.md answers"
    no = ["Voting no: a sunset clause or nothing, emergency rules never go away.",
          "I'm voting no, there's no sunset clause and it could be permanent.",
          "Vote no. Without an end date this is forever.",
          "Voting no because the ID cards will be used to track us.",
          "Voting no, the new ID card means surveillance.",
          "I'm voting no, it should be a personal choice."]
    yes = ["Voting yes, a sunset clause would be nice but hospitals come first."] * 5
    assert top_worries(no + yes) == ['"It has to be temporary."'], top_worries(no + yes)
    assert top_worries(no + yes, at_least=2) == ['"It has to be temporary."', '"The new ID cards will be used to track us."']
    assert top_worries(yes) == [], "a YES post's mention of a worry isn't one raised"
    print("worries ok")
