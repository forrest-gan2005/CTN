"""The in-game newspapers on sites.legitreal.com (Daily Kingstonia, Kingstonia Herald): what has been published,
so the writers know what people here are reading. The organisers said news pages may be read; every article
page has a .json twin. Articles are listed ahead of their publication time and may change before it, so an
article reaches the writers only as fetched after it came out, and is fetched again for a while after that.
"""
import asyncio
import json
import logging
import re
import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import httpx

from ctn import paths

log = logging.getLogger("news")

NEWS_URL = "https://sites.legitreal.com"
SITES = {"daily-kingstonia": "Daily Kingstonia", "kingstonia-herald": "Kingstonia Herald"}
ARTICLE_RE = re.compile(r'href="(/(?:daily-kingstonia|kingstonia-herald)/articles/[a-z0-9-]+)"')
DEFAULT_DB = paths.DATA / "news.db"
BRIEF_DAYS = 3  # how far back the writers' news brief reaches
BRIEF_MAX = 6  # articles in it, newest first
SETTLE = timedelta(hours=6)  # an article is fetched again on every refresh until this long after it came out
PAUSE = 1.0  # seconds between article fetches
SYDNEY = ZoneInfo("Australia/Sydney")
SHARE_DAYS = 2  # how old a story a post may still link to
# Stories a YES post may link to: the flu, the vaccine, health, the vote, or everyday island news a post can start
# from (another team's posts linking a poll, the first case, the venue-list story and the ferry repairs held the top
# of Trending on 21 October). Left out: opinion, party politics (rules.md) and stories making NO's case (SHARE_NO_RE).
SHARE_RE = re.compile(r"h2d2|vaccin|hospital|health|clinic|nurse|\bwards?\b|intensive care|referendum|\bpoll|"
                      r"touris|ferr(y|ies)|\bschools?\b", re.IGNORECASE)
SHARE_SKIP_RE = re.compile(r"privacy|\bid cards?\b|guardrail|blank cheque|hawthorne|castillo|\bpart(y|ies)\b|"
                           r"assembly|\bvote no\b|launch\w*[^.]{0,20}\bno campaign|"
                           # rules.md knows one case: any story of more isn't one to link until it's updated
                           r"\b(second|third|new|more|further) (h2d2 )?cases?\b|contacts? (have )?test(ed)? positive|test\w* positive|"
                           r"cluster|(community|local) transmission[^.]{0,20}\b(confirmed|found|detected)",
                           re.IGNORECASE)
# Nor one whose headline or summary carries a NO argument: on 22 October anchors linked the "vaccines forever"
# professor and the tourism operators' call for "clarity" on which places, putting NO's case under a YES post.
SHARE_NO_RE = re.compile(r"\bforever\b|permanent|\bboosters?\b|pharma\w*|blank cheque|guardrail|sunset clause|"
                         r"\bin the dark\b|without a map|clarity|definitions?\b|\bspecify\b|which (venues|places)|"
                         r"privacy|surveillance|overreach|"
                         # a poll with NO ahead: linked under a YES post it still tells readers which way the island leans
                         r"\bno\b[^.]{0,30}\b(lead|edge|ahead)|(support|opposition)[^.]{0,20}\b(no\b|against|to h2d2)",
                         re.IGNORECASE)
# Stories an anchor draws more often than the rest (NEWS_ANCHOR_PREFER in bot.py): the first case, the nurses arriving.
SHARE_PREFER_RE = re.compile(r"\bfirst (h2d2 )?case\b|\b25 (new )?nurses\b|nursing arrivals", re.IGNORECASE)
SHARE_CATEGORIES = {"opinion"}
FRACTION_RE = re.compile(r"\.\d+")


def _at(value: str) -> datetime:
    return datetime.fromisoformat(value)


def published(value) -> Optional[str]:
    """UTC ISO of when an article came out. The papers give Sydney wall-clock time marked "Z": the story listed
    for 18:00Z on 19 October was new at 18:01 Sydney time, when the game opened, and an opinion piece listed for
    14:22Z was live at 03:24 UTC, two minutes after its journalist posted on the platform. An explicit offset is
    taken as given."""
    if not isinstance(value, str) or not value:
        return None
    text = FRACTION_RE.sub("", value[:-1] if value.endswith("Z") else value)
    try:
        when = datetime.fromisoformat(text)
    except ValueError:
        return None
    return (when if when.tzinfo else when.replace(tzinfo=SYDNEY)).astimezone(timezone.utc).isoformat(timespec="seconds")


class NewsStore:
    def __init__(self, path=DEFAULT_DB) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.execute("CREATE TABLE IF NOT EXISTS articles (url TEXT PRIMARY KEY, site TEXT NOT NULL, "
                        "title TEXT NOT NULL, summary TEXT, published_at TEXT, fetched_at TEXT NOT NULL, raw TEXT)")
        self._retime()

    def _retime(self) -> None:
        """Articles stored when their time was read as UTC get the right one, from the page as fetched."""
        fixed = []
        for url, stored, raw in self.db.execute("SELECT url, published_at, raw FROM articles WHERE raw IS NOT NULL"):
            try:
                when = published(json.loads(raw).get("published_at"))
            except (ValueError, AttributeError):
                continue
            if when != stored:
                fixed.append((when, url))
        with self.db:
            self.db.executemany("UPDATE articles SET published_at = ? WHERE url = ?", fixed)

    def known(self, url: str) -> bool:
        return self.db.execute("SELECT 1 FROM articles WHERE url = ?", (url,)).fetchone() is not None

    def add(self, site: str, article: dict, fetched_at: datetime) -> None:
        """Stores an article, replacing what an earlier fetch of the same page said."""
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO articles VALUES (?, ?, ?, ?, ?, ?, ?)",
                            (article["url"], site, article["title"], article.get("summary"),
                             published(article.get("published_at")), fetched_at.isoformat(timespec="seconds"),
                             json.dumps(article, ensure_ascii=False)))

    def remove(self, url: str) -> None:
        with self.db:
            self.db.execute("DELETE FROM articles WHERE url = ?", (url,))

    def unsettled(self, now: datetime) -> list:
        """(site, url) of articles that have come out but weren't fetched since SETTLE after that."""
        rows = self.db.execute("SELECT site, url, published_at, fetched_at FROM articles "
                               "WHERE published_at IS NOT NULL ORDER BY published_at").fetchall()
        return [(site, url) for site, url, published, fetched in rows
                if _at(published) <= now and _at(fetched) < _at(published) + SETTLE]

    def recent(self, now: datetime, days: int = BRIEF_DAYS, n: int = BRIEF_MAX) -> list:
        """(site, title, summary, published_at) published in the last `days` and not after `now`, as fetched
        after it came out (what was listed ahead of time may have changed), newest first, one per story: both
        papers run each story at the same minute, and which paper's take is kept alternates."""
        rows = self.db.execute(
            "SELECT site, title, summary, published_at, fetched_at FROM articles WHERE published_at <= ? "
            "AND published_at >= ? ORDER BY published_at DESC, site",
            (now.isoformat(), (now - timedelta(days=days)).isoformat())).fetchall()
        stories = {}
        for site, title, summary, published, fetched in rows:
            if _at(fetched) >= _at(published):
                stories.setdefault(published, []).append((site, title, summary, published))
        picked = [takes[sum(map(ord, published)) % len(takes)] for published, takes in stories.items()]
        return picked[:n]


    def shareable(self, now: datetime, days: int = SHARE_DAYS) -> list:
        """(url, site, title, summary) of the stories a YES post may link to (see SHARE_RE), published in
        the last `days` and fetched after they came out, newest first, one take per story."""
        rows = self.db.execute(
            "SELECT url, site, title, summary, published_at, fetched_at, raw FROM articles WHERE published_at <= ? "
            "AND published_at >= ? ORDER BY published_at DESC, site",
            (now.isoformat(), (now - timedelta(days=days)).isoformat())).fetchall()
        stories = {}
        for url, site, title, summary, when, fetched, raw in rows:
            category = (json.loads(raw).get("category") if raw else None) or ""
            text = f"{title} {summary or ''}"
            if (_at(fetched) >= _at(when) and category.lower() not in SHARE_CATEGORIES and SHARE_RE.search(text)
                    and not SHARE_SKIP_RE.search(text) and not SHARE_NO_RE.search(text)):
                stories.setdefault(when, []).append((NEWS_URL + url, site, title, summary or ""))
        return [takes[sum(map(ord, when)) % len(takes)] for when, takes in stories.items()]


class News:
    def __init__(self, store: NewsStore, base_url: str = NEWS_URL,
                 transport: Optional[httpx.AsyncBaseTransport] = None, pause: float = PAUSE) -> None:
        self.store = store
        self.pause = pause
        self.clock = lambda: datetime.now(timezone.utc)
        self.sleep = asyncio.sleep
        self.client = httpx.AsyncClient(base_url=base_url, timeout=20, transport=transport, follow_redirects=True)

    async def aclose(self) -> None:
        await self.client.aclose()

    async def refresh(self) -> int:
        """Read both front pages, fetch every article not seen before and every one still settling; returns
        how many were fetched. A page that fails is tried again on the next refresh."""
        due = {}
        for site in SITES:
            try:
                resp = await self.client.get(f"/{site}/")
                resp.raise_for_status()
            except httpx.HTTPError as e:
                log.warning("couldn't read the %s front page (%s)", SITES[site],
                            str(e).splitlines()[0] if str(e) else type(e).__name__)
                continue
            links = sorted(set(ARTICLE_RE.findall(resp.text)))
            if not links:
                log.warning("no article links on the %s front page: has it changed?", SITES[site])
            due.update((url, site) for url in links if not self.store.known(url))
        due.update((url, site) for site, url in self.store.unsettled(self.clock()))
        fetched = 0
        for url, site in due.items():
            asked = self.clock()  # what comes back is no newer than this, however long the answer takes
            try:
                page = await self.client.get(f"{url}.json")
                if page.status_code == 404 and self.store.known(url):
                    self.store.remove(url)  # taken down
                    log.info("%s was taken down", url)
                    continue
                page.raise_for_status()
                article = page.json()["article"]
            except (httpx.HTTPError, ValueError, KeyError) as e:
                log.warning("couldn't fetch %s (%s)", url, str(e).splitlines()[0] if str(e) else type(e).__name__)
                continue
            finally:
                await self.sleep(self.pause)
            self.store.add(site, {**article, "url": article.get("url") or url}, asked)
            fetched += 1
        return fetched

    def shareable(self, now: Optional[datetime] = None) -> list:
        return self.store.shareable(now or self.clock())

    def brief(self, now: Optional[datetime] = None) -> str:
        """The latest published stories, for the writers' context; empty when there are none."""
        rows = self.store.recent(now or self.clock())
        if not rows:
            return ""
        listed = "\n".join(f"- {SITES.get(site, site)}, {_at(when).astimezone(SYDNEY):%Y-%m-%d}: {title}. "
                           f"{summary or ''}".rstrip() for site, title, summary, when in rows)
        return ("In the local papers lately (people here may have read these; bring one up only when it fits, as "
                "something you read, in your own words, adding nothing the story doesn't say):\n" + listed)


async def _demo() -> None:
    front = {"daily-kingstonia": '<a href="/daily-kingstonia/articles/poll">x</a>'
                                 '<a href="/daily-kingstonia/articles/later">y</a>',
             "kingstonia-herald": '<a href="/kingstonia-herald/articles/ward">z</a>'}
    articles = {
        "/daily-kingstonia/articles/poll": {"title": "Early poll favours No", "summary": "Only 12 people asked.",
                                            "published_at": "2026-10-17T07:00:00Z"},
        "/daily-kingstonia/articles/later": {"title": "Not out yet", "summary": "Scheduled.",
                                             "published_at": "2026-10-25T07:00:00Z"},
        "/kingstonia-herald/articles/ward": {"title": "Hospital short of nurses", "summary": "Wards stretched.",
                                             "published_at": "2026-10-18T07:00:00Z"},
    }
    calls, down = [], set()

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        calls.append(path)
        site = path.strip("/")
        if site in front:
            return httpx.Response(200, text=front[site])
        if path in down:
            return httpx.Response(503)
        if path.endswith(".json") and path[:-5] in articles:
            return httpx.Response(200, json={"article": {**articles[path[:-5]], "url": path[:-5]}})
        return httpx.Response(404)

    def at(day: int, hour: int) -> datetime:  # Sydney wall-clock time, as the papers give it, in UTC as we keep it
        return datetime(2026, 10, day, hour, tzinfo=SYDNEY).astimezone(timezone.utc)

    assert published("2026-10-19T18:00:00Z") == "2026-10-19T07:00:00+00:00"
    assert published("2026-10-19T14:22:00.5Z") == "2026-10-19T03:22:00+00:00"
    assert published("2026-10-19T07:00:00+00:00") == "2026-10-19T07:00:00+00:00", "an explicit offset stands"
    assert published("2026-11-02T12:00:00Z") == "2026-11-02T01:00:00+00:00"
    assert published(None) is None and published("") is None and published("soon") is None

    def jsons() -> int:
        return sum(c.endswith(".json") for c in calls)

    news = News(NewsStore(":memory:"), transport=httpx.MockTransport(handler), pause=0)
    news.clock = lambda: at(15, 0)  # everything is listed days ahead
    assert await news.refresh() == 3 and await news.refresh() == 0 and jsons() == 3, "an article is fetched once"
    assert news.brief(at(19, 12)) == "", "what was fetched before it came out isn't trusted"

    # The poll story changes and the ward story moves back a day before either comes out; at 17 08:00 the
    # poll story is fetched afresh, the ward one isn't due yet.
    articles["/daily-kingstonia/articles/poll"].update(title="Final poll: No ahead in a tiny sample")
    articles["/kingstonia-herald/articles/ward"]["published_at"] = "2026-10-19T07:00:00Z"
    news.clock = lambda: at(17, 8)
    assert await news.refresh() == 1 and "Final poll" in news.brief() and "Early poll" not in news.brief()
    news.clock = lambda: at(18, 8)  # its old time: fetched, found moved, still not shown
    assert await news.refresh() == 2 and "Hospital" not in news.brief()
    down.add("/kingstonia-herald/articles/ward.json")
    news.clock = lambda: at(19, 8)  # due, but the fetch fails: not shown, and tried again
    assert await news.refresh() == 0 and "Hospital" not in news.brief()
    down.clear()
    assert await news.refresh() == 1 and "Hospital short of nurses" in news.brief()
    assert news.brief().index("Hospital") < news.brief().index("Final poll"), "newest first"
    assert news.store.db.execute("SELECT COUNT(*) FROM articles").fetchone()[0] == 3

    news.clock = lambda: at(20, 8)  # once fetched SETTLE after it came out, an article is left alone
    assert await news.refresh() == 1
    before = jsons()
    assert await news.refresh() == 0 and jsons() == before
    news.store.add("kingstonia-herald", {"url": "/kingstonia-herald/articles/poll", "title": "Poll: No ahead",
                                         "summary": "Twelve asked.", "published_at": "2026-10-17T07:00:00Z"},
                   at(17, 9))
    brief = news.brief(at(19, 12))
    assert ("Final poll" in brief) != ("Poll: No ahead" in brief), "one take per story"
    assert "Not out yet" not in brief, "an article listed ahead of its time isn't news yet"
    assert news.brief(at(30, 0)) == "", "old news drops out"
    news.store.add("kingstonia-herald", {"url": "/kingstonia-herald/articles/staff",
                                         "title": "Hospital warns of nurse shortfall ahead of H2D2",
                                         "summary": "Wards are stretched.", "category": "Health",
                                         "published_at": "2026-10-19T09:00:00Z"}, at(19, 10))
    news.store.add("daily-kingstonia", {"url": "/daily-kingstonia/articles/column", "title": "The H2D2 vote is a trap",
                                        "category": "Opinion", "published_at": "2026-10-19T09:00:00Z"}, at(19, 10))
    news.store.add("daily-kingstonia", {"url": "/daily-kingstonia/articles/venues", "title": "Operators seek clarity "
                                        "on referendum venues", "category": "Business",
                                        "published_at": "2026-10-19T08:00:00Z"}, at(19, 10))
    shared = news.store.shareable(at(19, 12))
    urls = [s[0] for s in shared]
    assert urls[0] == NEWS_URL + "/kingstonia-herald/articles/staff", shared
    assert not any(u.endswith("/column") or u.endswith("/venues") for u in urls), "no opinion, no NO argument"
    for title, category in (("Hawthorne backs H2D2 vaccine requirement", "Health"),
                            ("Kingstonia University professor warns of 'vaccines forever' risk", "Health"),
                            ("Kingstonia University professor warns of 'permanent' vaccine contracts", "Health"),
                            ("Tourism operators demand clarity on 'public space' definitions ahead of referendum",
                             "Business"),
                            ("Tourism operators seek clarity on 'public space' definitions ahead of referendum",
                             "National"),
                            ("Voting in the Dark: The Referendum Without a Map", "National"),
                            ("H2D2 referendum lacks the guardrails we need", "Health"),
                            ("Tourism Board launches Vote No campaign", "Business"),
                            ("No campaign takes slight lead in latest H2D2 referendum poll", "Politics"),
                            ("Early Poll Shows Strong Support for 'No' Vote in H2D2 Referendum", "Politics"),
                            ("Early Polling Indicates Strong Opposition to H2D2 Vaccine Mandate", "Politics"),
                            ("Second H2D2 case confirmed as contacts test positive", "Health"),
                            ("ID cards raise privacy fears over H2D2 records", "National"),
                            ("Reef survey shows coral recovery", "Environment")):
        news.store.add("daily-kingstonia", {"url": "/daily-kingstonia/articles/x", "title": title,
                                            "category": category, "published_at": "2026-10-19T09:30:00Z"}, at(19, 10))
        assert all(not u.endswith("/x") for u, *_ in news.store.shareable(at(19, 12))), title
    for title, summary, category in (
            ("New nursing arrivals to bolster hospital staffing", "Minister Tomas Reyes-Hale announced 25 new nurses "
             "will arrive over four months to assist Kingstonia City and Crown Harbour hospitals.", "Health"),
            ("Regional recruitment agreement to bring 25 nurses to Kingstonian hospitals", "Health Minister Tomas "
             "Reyes-Hale announced that 25 nurses will arrive via a Pacific recruitment agreement to address staffing "
             "shortages in Kingstonia City and Crown Harbour.", "Health"),
            ("First H2D2 case confirmed in Kingstonia via international traveller", "Health Ministry confirms first "
             "H2D2 case in a traveller; no evidence of community transmission reported.", "Health"),
            ("Outer-island ferry service reduced as engine repairs begin", "Ferry service to the outer islands faces "
             "a two-week period of reduced capacity due to urgent engine repairs.", "Business"),
            ("Advisory panel finds H2D2 vaccine shows strong immune response despite data gaps", "A medicines "
             "advisory panel reports that while the H2D2 vaccine shows strong immune responses, uncertainties remain "
             "regarding transmission prevention and long-term side effects.", "Health"),
            ("Referendum poll shows nation deadlocked as undecided voters hold balance of power", "The latest "
             "national poll has Yes and No level, with nearly 30% of voters still undecided.", "Politics")):
        news.store.add("daily-kingstonia", {"url": "/daily-kingstonia/articles/y", "title": title, "summary": summary,
                                            "category": category, "published_at": "2026-10-19T09:40:00Z"}, at(19, 10))
        assert any(u.endswith("/y") for u, *_ in news.store.shareable(at(19, 12))), title
        assert bool(SHARE_PREFER_RE.search(title)) == ("nurs" in title or "First" in title), title
    assert news.store.shareable(at(22, 12)) == [], "a story only two days old at most"

    del articles["/kingstonia-herald/articles/ward"]  # taken down while still settling
    news.store.add("kingstonia-herald", {"url": "/kingstonia-herald/articles/ward", "title": "Hospital short of nurses",
                                         "published_at": "2026-10-20T07:00:00Z"}, at(20, 7))
    assert await news.refresh() == 0 and "Hospital" not in news.brief()

    # Asked a second before it came out, answered (or paused after) a second after: still not trusted.
    articles["/daily-kingstonia/articles/ferry"] = {"title": "Ferry draft", "published_at": "2026-10-21T07:00:00Z"}
    front["daily-kingstonia"] += '<a href="/daily-kingstonia/articles/ferry">'
    clock = {"now": at(21, 7) - timedelta(seconds=1)}
    news.clock = lambda: clock["now"]

    async def slow(_: float) -> None:
        clock["now"] = at(21, 7) + timedelta(seconds=1)
    news.sleep = slow
    assert await news.refresh() == 1 and "Ferry" not in news.brief(), "a draft passed for the published story"
    assert await news.refresh() == 1 and "Ferry draft" in news.brief(), "asked again once it was out"

    seen = []
    handler_ = logging.Handler()
    handler_.emit = seen.append
    log.addHandler(handler_)
    front["kingstonia-herald"] = "<p>new layout</p>"
    await news.refresh()
    log.removeHandler(handler_)
    assert any("front page" in r.getMessage() for r in seen), "a front page with no links goes unremarked"
    await news.aclose()

    with tempfile.TemporaryDirectory() as tmp:  # stored when the papers' times were read as UTC
        old = NewsStore(f"{tmp}/news.db")
        old.add("daily-kingstonia", {"url": "/daily-kingstonia/articles/q", "title": "Tech jobs",
                                     "published_at": "2026-10-19T18:00:00Z"}, at(19, 18))
        with old.db:
            old.db.execute("UPDATE articles SET published_at = '2026-10-19T18:00:00+00:00'")
        old.db.close()
        store = NewsStore(f"{tmp}/news.db")
        assert [r[0] for r in store.db.execute("SELECT published_at FROM articles")] == ["2026-10-19T07:00:00+00:00"]
        brief = News(store).brief(at(19, 19))
        assert "Daily Kingstonia, 2026-10-19: Tech jobs" in brief, brief
        store.add("kingstonia-herald", {"url": "/kingstonia-herald/articles/m", "title": "Morning edition",
                                        "published_at": "2026-10-20T09:00:00Z"}, at(20, 9))
        assert "Kingstonia Herald, 2026-10-20: Morning edition" in News(store).brief(at(20, 10)), "Sydney's date"
    print("news ok")


if __name__ == "__main__":
    asyncio.run(_demo())
