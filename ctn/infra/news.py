"""The in-game newspapers on sites.legitreal.com (Daily Kingstonia, Kingstonia Herald): what has been published,
so the writers know what people here are reading. The organisers said news pages may be read; every article
page has a .json twin. Articles are listed ahead of their publication time, so only published ones are used.
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
PAUSE = 1.0  # seconds between article fetches


class NewsStore:
    def __init__(self, path=DEFAULT_DB) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.execute("CREATE TABLE IF NOT EXISTS articles (url TEXT PRIMARY KEY, site TEXT NOT NULL, "
                        "title TEXT NOT NULL, summary TEXT, published_at TEXT, fetched_at TEXT NOT NULL, raw TEXT)")

    def known(self, url: str) -> bool:
        return self.db.execute("SELECT 1 FROM articles WHERE url = ?", (url,)).fetchone() is not None

    def add(self, site: str, article: dict) -> None:
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO articles VALUES (?, ?, ?, ?, ?, ?, ?)",
                            (article["url"], site, article["title"], article.get("summary"),
                             iso(article.get("published_at")),
                             datetime.now(timezone.utc).isoformat(timespec="seconds"),
                             json.dumps(article, ensure_ascii=False)))

    def recent(self, now: datetime, days: int = BRIEF_DAYS, n: int = BRIEF_MAX) -> list:
        """(site, title, summary, published_at) published in the last `days` and not after `now`, newest first,
        one per story: both papers run each story at the same minute, and which paper's take is kept alternates."""
        rows = self.db.execute(
            "SELECT site, title, summary, published_at FROM articles WHERE published_at <= ? AND published_at >= ? "
            "ORDER BY published_at DESC, site", (now.isoformat(), (now - timedelta(days=days)).isoformat())).fetchall()
        stories = {}
        for row in rows:
            stories.setdefault(row[3], []).append(row)
        picked = [takes[sum(map(ord, published)) % len(takes)] for published, takes in stories.items()]
        return picked[:n]


class News:
    def __init__(self, store: NewsStore, base_url: str = NEWS_URL,
                 transport: Optional[httpx.AsyncBaseTransport] = None, pause: float = PAUSE) -> None:
        self.store = store
        self.pause = pause
        self.client = httpx.AsyncClient(base_url=base_url, timeout=20, transport=transport, follow_redirects=True)

    async def aclose(self) -> None:
        await self.client.aclose()

    async def refresh(self) -> int:
        """Read both front pages and fetch any article not seen before; returns how many were new."""
        new = 0
        for site in SITES:
            resp = await self.client.get(f"/{site}/")
            resp.raise_for_status()
            for url in sorted(set(ARTICLE_RE.findall(resp.text))):
                if self.store.known(url):
                    continue
                page = await self.client.get(f"{url}.json")
                page.raise_for_status()
                article = page.json()["article"]
                self.store.add(site, {**article, "url": article.get("url") or url})
                new += 1
                await asyncio.sleep(self.pause)
        if new:
            log.info("%d new article(s) in the papers", new)
        return new

    def brief(self, now: Optional[datetime] = None) -> str:
        """The latest published stories, for the writers' context; empty when there are none."""
        now = now or datetime.now(timezone.utc)
        rows = self.store.recent(now)
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
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        calls.append(path)
        site = path.strip("/")
        if site in front:
            return httpx.Response(200, text=front[site])
        if path.endswith(".json") and path[:-5] in articles:
            return httpx.Response(200, json={"article": {**articles[path[:-5]], "url": path[:-5]}})
        return httpx.Response(404)

    news = News(NewsStore(":memory:"), transport=httpx.MockTransport(handler), pause=0)
    assert await news.refresh() == 3 and await news.refresh() == 0
    news.store.add("kingstonia-herald", {"url": "/kingstonia-herald/articles/poll", "title": "Poll: No ahead",
                                         "summary": "Twelve asked.", "published_at": "2026-10-17T07:00:00Z"})
    assert sum(c.endswith(".json") for c in calls) == 3, "an article is fetched once"
    brief = news.brief(datetime(2026, 10, 19, 12, tzinfo=timezone.utc))
    assert "Hospital short of nurses" in brief
    assert brief.index("Hospital") < brief.index("oll"), "newest first"
    assert ("Early poll favours No" in brief) != ("Poll: No ahead" in brief), "one take per story"
    assert "Not out yet" not in brief, "an article listed ahead of its time isn't news yet"
    assert news.brief(datetime(2026, 10, 30, tzinfo=timezone.utc)) == "", "old news drops out"
    assert news.brief(datetime(2026, 10, 16, tzinfo=timezone.utc)) == ""
    await news.aclose()
    print("news ok")


if __name__ == "__main__":
    asyncio.run(_demo())
