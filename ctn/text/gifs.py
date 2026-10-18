"""Reaction GIFs, the way citizens post them: a giphy link at the end of the text, which the platform turns into an
embed. The writer names what it would search for; the pick is among the GIFs citizens have already posted, by
Giphy's own tags, so every link is one the platform has shown before."""
import json
import logging
import random
import re
import time
from datetime import datetime
from typing import Iterable, Optional

log = logging.getLogger("gifs")

# Share of drafts the writer is asked for a GIF search: citizens put one on about two replies in five.
GIF_SHARE = {"reply": 0.4, "post": 0.3}
TOP = 5  # the pick is random among the best few matches, as a person scrolling search results would
REUSE_AFTER = 24 * 3600.0  # seconds before our team posts the same GIF again
GIF_PROMPT = ('Then add one more line: "GIF: " and the two or three words you would type into a GIF search for a '
              'reaction that fits your post (for example "GIF: thumbs up" or "GIF: tired cat"), or "GIF: none" if a '
              "GIF would feel out of place.")
# The writer's GIF line: "GIF:" with a colon, opening a line or following the end of a sentence, and running to
# the end of the text. Anything looser would cut ordinary words ("GIF-making"); those drafts go as leftovers.
GIF_LINE_RE = re.compile(r"(?:^|\n|(?<=[.!?\u2026)])[ \t]+)[\s\[(*]*gif(?:\s+search)?\s*\**\s*:\s*\**\s*([^\n]{0,40}?)"
                         r"[\s\"'\u201d\])*.!]*$", re.IGNORECASE)
NONE = {"", "none", "no", "n/a", "na", "nothing", "no gif", "skip"}
GIF_WORD_RE = re.compile(r"\bgifs?\b", re.IGNORECASE)  # anything about GIFs left in the text gives the game away
GIPHY_URL_RE = re.compile(r"https?://(?:[a-z0-9-]+\.)?(?:giphy\.com|gph\.is)/\S+", re.IGNORECASE)
STOP = {"a", "an", "the", "of", "and", "or", "to", "my", "your", "is", "it", "in", "on", "for", "gif", "gifs"}


def split_gif(text: Optional[str]) -> tuple:
    """(the text without the writer's GIF line, the search it asked for or None)."""
    if text is None:
        return None, None
    match = GIF_LINE_RE.search(text)
    if not match:
        return text, None
    query = match.group(1).strip(" \"'\u201c\u201d*.!()[]").lower()
    return text[:match.start()].rstrip(), (None if query in NONE else query)


def without_gifs(text: str) -> str:
    return GIPHY_URL_RE.sub("", text or "").strip()


def _words(text: str) -> set:
    return {w for w in re.findall(r"[a-z0-9']+", text.lower()) if w not in STOP}


class GifPool:
    """The GIFs seen on the platform, searchable by their tags; shared by the whole team."""

    def __init__(self, embeds: Iterable[dict] = ()) -> None:
        self.gifs = {}  # url -> (tags as phrases, words of tags and title)
        self.used = {}  # url -> when our team last posted it (time.time())
        self.off = None  # why GIFs were switched off, if they were
        self.update(embeds)

    def update(self, embeds: Iterable[dict]) -> None:
        for e in embeds:
            url = e.get("url") or ""
            if e.get("kind") != "giphy" or not GIPHY_URL_RE.fullmatch(url):
                continue
            tags = {str(t).lower().strip() for t in e.get("tags") or [] if str(t).strip()}
            title = re.sub(r"\s+GIF(\s+by\s+.*)?$", "", e.get("title") or "", flags=re.IGNORECASE)
            self.gifs[url] = (tags, _words(" ".join(tags) + " " + title))

    def remember(self, rows: Iterable[tuple]) -> None:
        """Take in (UTC ISO time, text) of our posts, so a restart still keeps the GIFs used lately apart."""
        for ts, text in rows:
            try:
                when = datetime.fromisoformat(ts).timestamp()
            except (TypeError, ValueError):
                continue
            for url in GIPHY_URL_RE.findall(text or ""):
                self.used[url] = max(self.used.get(url, when), when)

    def ready(self) -> bool:
        return self.off is None and bool(self.gifs)

    def disable(self, why: str) -> None:
        if self.off is None:
            self.off = why
            log.error("GIFs switched off: %s", why)

    def pick(self, query: Optional[str], now: Optional[float] = None) -> Optional[str]:
        """A GIF for this search: random among the TOP best by tags, none our team posted lately; None if nothing
        matches a word of it."""
        if not query or not self.ready():
            return None
        now = time.time() if now is None else now
        wanted = _words(query)
        scored = []
        for url, (tags, words) in self.gifs.items():
            if now - self.used.get(url, -REUSE_AFTER) < REUSE_AFTER:
                continue
            score = 3 * (query in tags) + len(wanted & words)
            if score:
                scored.append((score, random.random(), url))
        if not scored:
            return None
        url = random.choice(sorted(scored, reverse=True)[:TOP])[2]
        self.used[url] = now
        return url


def embeds_from_raw(raws: Iterable[str]) -> list:
    out = []
    for raw in raws:
        try:
            embed = json.loads(raw).get("embed")
        except (ValueError, AttributeError):
            continue
        if isinstance(embed, dict):
            out.append(embed)
    return out


def _demo() -> None:
    cases = [
        ("Ferry late again.\nGIF: eye roll", "Ferry late again.", "eye roll"),
        ("Ferry late again. GIF: eye roll", "Ferry late again.", "eye roll"),
        ('Ferry late again.\n\nGIF: "eye roll"', "Ferry late again.", "eye roll"),
        ("Ferry late again.\n(GIF: eye roll)", "Ferry late again.", "eye roll"),
        ("Ferry late again.\n**GIF:** eye roll", "Ferry late again.", "eye roll"),
        ("Ferry late again.\nGIF: none", "Ferry late again.", None),
        ("Ferry late again.\nGIF:", "Ferry late again.", None),
        ("Ferry late again.", "Ferry late again.", None),
        ("Gift shop's closed again", "Gift shop's closed again", None),
        ("Good game though, ref bought a round. GIF: shrug", "Good game though, ref bought a round.", "shrug"),
        # words, not a GIF line: left whole, for the leftover check to drop the draft
        ("I love GIF-making after work.", "I love GIF-making after work.", None),
        ("That GIF-worthy ferry entrance made my day.", "That GIF-worthy ferry entrance made my day.", None),
        ("I meant the GIF - not the policy.", "I meant the GIF - not the policy.", None),
        ("Saw a great gif: the cat one", "Saw a great gif: the cat one", None),
    ]
    for text, left, query in cases:
        assert split_gif(text) == (left, query), (text, split_gif(text))
    assert GIF_WORD_RE.search("found a gif of it") and not GIF_WORD_RE.search("a gift for mum")
    pool = GifPool([
        {"kind": "giphy", "url": "https://giphy.com/gifs/a1", "title": "Thumbs Up GIF by X", "tags": ["thumbs up", "ok"]},
        {"kind": "giphy", "url": "https://giphy.com/gifs/a2", "title": "Mr Bean Thumbs Up GIF", "tags": ["thumbs up"]},
        {"kind": "giphy", "url": "https://giphy.com/gifs/a3", "title": "Facepalm GIF", "tags": ["facepalm", "ugh"]},
        {"kind": "giphy", "url": "https://giphy.com/gifs/a4", "title": "Tired Cat GIF", "tags": ["tired", "cat"]},
        {"kind": "article", "url": "https://sites.legitreal.com/x", "title": "News"},
        {"kind": "giphy", "url": "https://evil.example/gifs/a5", "title": "Thumbs", "tags": ["thumbs up"]},
    ])
    assert set(pool.gifs) == {"https://giphy.com/gifs/a1", "https://giphy.com/gifs/a2", "https://giphy.com/gifs/a3",
                              "https://giphy.com/gifs/a4"}
    first = pool.pick("thumbs up", now=0.0)
    assert first in ("https://giphy.com/gifs/a1", "https://giphy.com/gifs/a2")
    second = pool.pick("thumbs up", now=1.0)  # the other one: ours used the first today
    assert second in ("https://giphy.com/gifs/a1", "https://giphy.com/gifs/a2") and second != first
    assert pool.pick("thumbs up", now=2.0) is None  # both used today, and nothing else fits
    assert pool.pick("thumbs up", now=REUSE_AFTER + 1.0) in (first, second)  # a day on, fine again
    assert pool.pick("so tired", now=0.0) == "https://giphy.com/gifs/a4"
    assert pool.pick("celebration", now=0.0) is None and pool.pick(None) is None
    pool.disable("the platform refused a post with a GIF")
    assert pool.pick("facepalm", now=0.0) is None and not pool.ready()
    assert without_gifs("so true https://giphy.com/gifs/9Ai5dIk8xvBm0") == "so true"
    again = GifPool([{"kind": "giphy", "url": "https://giphy.com/gifs/a1", "tags": ["ok"]}])
    again.remember([("2026-10-20T00:00:00+00:00", "fine https://giphy.com/gifs/a1"), (None, "x"), ("bad", "y")])
    noon = datetime.fromisoformat("2026-10-20T12:00:00+00:00").timestamp()
    assert again.pick("ok", now=noon) is None and again.pick("ok", now=noon + REUSE_AFTER) is not None
    assert embeds_from_raw(['{"embed": {"kind": "giphy"}}', "{}", "not json"]) == [{"kind": "giphy"}]
    print("gifs ok")


if __name__ == "__main__":
    _demo()
