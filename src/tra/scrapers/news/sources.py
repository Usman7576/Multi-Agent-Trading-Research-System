"""News sources: RSS (recent articles with text) and GDELT (historical headlines without text).

RSS feeds hold only days to weeks of articles, so they are collected from now on. History comes from the GDELT
DOC API, which gives title, URL and the time GDELT first saw the article. Both use the time an article became
public as ``known_at``; article pages themselves are not scraped.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from email.utils import parsedate_to_datetime

import pandas as pd
from bs4 import BeautifulSoup

from tra.scrapers.common.drift import DriftError
from tra.scrapers.common.http import HttpClient, HttpError

log = logging.getLogger(__name__)

RSS_FEEDS = {
    "brecorder": "https://www.brecorder.com/feeds/business-finance",
    "dawn": "https://www.dawn.com/feeds/business",
    "tribune": "https://tribune.com.pk/feed/business",
}
GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
COLUMNS = ["doc_id", "source", "title", "text", "url", "known_at", "fetched_at", "language"]


def doc_id(url: str) -> str:
    return hashlib.sha256(url.strip().encode()).hexdigest()[:16]


def _clean(html: str) -> str:
    return " ".join(BeautifulSoup(html or "", "lxml").get_text(" ").split())


def parse_rss(xml: bytes | str, source: str, fetched_at: pd.Timestamp) -> pd.DataFrame:
    """Articles from one RSS feed. Items without a usable publication time are dropped, never guessed."""
    soup = BeautifulSoup(xml, "xml")
    if soup.find("rss") is None and soup.find("feed") is None:
        raise DriftError(f"{source}: not an RSS or Atom document")
    rows = []
    for item in soup.find_all(["item", "entry"]):
        title = item.find("title")
        link = item.find("link")
        date = item.find("pubDate") or item.find("published") or item.find("updated")
        url = (link.get("href") or link.get_text() if link else "").strip()
        if not (title and url and date):
            continue
        try:
            published = pd.Timestamp(parsedate_to_datetime(date.get_text().strip()))
        except (TypeError, ValueError):
            try:
                published = pd.Timestamp(date.get_text().strip())
            except ValueError:
                continue
        if published.tzinfo is None:
            continue
        body = (
            item.find("encoded") or item.find("content") or item.find("description") or item.find("summary")
        )
        rows.append(
            {
                "doc_id": doc_id(url),
                "source": source,
                "title": _clean(title.get_text()),
                "text": _clean(body.get_text()) if body else "",
                "url": url,
                "known_at": published.tz_convert("UTC"),
                "fetched_at": fetched_at,
                "language": "English",
            }
        )
    return pd.DataFrame(rows, columns=COLUMNS)


def parse_gdelt(body: str, fetched_at: pd.Timestamp) -> pd.DataFrame:
    """Headlines from a GDELT DOC API response. ``seendate`` is when GDELT first saw the article."""
    articles = json.loads(body, strict=False).get("articles", [])
    rows = []
    for a in articles:
        seen = pd.to_datetime(a.get("seendate", ""), format="%Y%m%dT%H%M%SZ", utc=True, errors="coerce")
        if pd.isna(seen) or not a.get("url") or not a.get("title"):
            continue
        rows.append(
            {
                "doc_id": doc_id(a["url"]),
                "source": a.get("domain", "gdelt"),
                "title": " ".join(a["title"].split()),
                "text": "",
                "url": a["url"],
                "known_at": seen,
                "fetched_at": fetched_at,
                "language": a.get("language", ""),
            }
        )
    return pd.DataFrame(rows, columns=COLUMNS)


def fetch_gdelt(
    http: HttpClient,
    query: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    attempts: int = 4,
    wait: float = 15.0,
) -> str:
    """One GDELT query. A throttled request comes back as plain text with HTTP 200, so it is retried."""
    params = {
        "query": query,
        "mode": "artlist",
        "maxrecords": "250",
        "format": "json",
        "sort": "dateasc",
        "startdatetime": start.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end.strftime("%Y%m%d%H%M%S"),
    }
    for _ in range(attempts):
        body = http.get_text(GDELT_URL, params=params, use_cache=False)
        if body.lstrip().startswith("{"):
            return body
        if not body.strip():
            return "{}"
        log.warning("GDELT throttled or refused %r: %s", query, body[:80])
        time.sleep(wait)
    raise HttpError(f"GDELT refused {query!r} after {attempts} attempts")


def company_search_name(name: str) -> str:
    """'Lucky Cement Limited' -> 'Lucky Cement': the form newspapers use."""
    short = re.sub(r"\(.*?\)", " ", name)
    previous = None
    while previous != short:
        previous = short
        short = re.sub(r"[\s,]+(limited|ltd\.?|company|co\.?|corporation|corp\.?)\s*$", "", short, flags=re.I)
    return " ".join(short.split())
