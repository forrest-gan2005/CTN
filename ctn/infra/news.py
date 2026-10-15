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
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import httpx

from ctn import paths
from ctn.infra.real_platform import iso

log = logging.getLogger("news")

NEWS_URL = "https://sites.legitreal.com"
SITES = {"daily-kingstonia": "Daily Kingstonia", "kingstonia-herald": "Kingstonia Herald"}
ARTICLE_RE = re.compile(r'href="(/(?:daily-kingstonia|kingstonia-herald)/articles/[a-z0-9-]+)"')
DEFAULT_DB = paths.DATA / "news.db"
BRIEF_DAYS = 3  # how far back the writers' news brief reaches
BRIEF_MAX = 6  # articles in it, newest first
SETTLE = timedelta(hours=6)  # an article is fetched again on every refresh until this long after it came out
PAUSE = 1.0  # seconds between article fetches


def _at(value: str) -> datetime:
    return datetime.fromisoformat(value)


class NewsStore:
    def __init__(self, path=DEFAULT_DB) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.execute("CREATE TABLE IF NOT EXISTS articles (url TEXT PRIMARY KEY, site TEXT NOT NULL, "
                        "title TEXT NOT NULL, summary TEXT, published_at TEXT, fetched_at TEXT NOT NULL, raw TEXT)")

    def known(self, url: str) -> bool:
        return self.db.execute("SELECT 1 FROM articles WHERE url = ?", (url,)).fetchone() is not None

    def add(self, site: str, article: dict, fetched_at: datetime) -> None:
        """Stores an article, replacing what an earlier fetch of the same page said."""
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO articles VALUES (?, ?, ?, ?, ?, ?, ?)",
                            (article["url"], site, article["title"], article.get("summary"),
                             iso(article.get("published_at")), fetched_at.isoformat(timespec="seconds"),
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

    def brief(self, now: Optional[datetime] = None) -> str:
        """The latest published stories, for the writers' context; empty when there are none."""
        rows = self.store.recent(now or self.clock())
        if not rows:
            return ""
        listed = "\n".join(f"- {SITES.get(site, site)}, {published[:10]}: {title}. {summary or ''}".rstrip()
                           for site, title, summary, published in rows)
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

    def at(day: int, hour: int) -> datetime:
        return datetime(2026, 10, day, hour, tzinfo=timezone.utc)

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
    print("news ok")


if __name__ == "__main__":
    asyncio.run(_demo())
