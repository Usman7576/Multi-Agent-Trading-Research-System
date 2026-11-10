"""News collection CLI.

    uv run python -m tra.scrapers.news.run --dataset rss
    uv run python -m tra.scrapers.news.run --dataset history --since 2019 --symbols kse100

Both are incremental: history keeps a watermark per (company, year), and articles are keyed by URL and symbol.
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from tra.config import Settings
from tra.data.lake import ParquetLake
from tra.scrapers.common.http import HttpClient, HttpError
from tra.scrapers.news.mapper import SymbolMapper
from tra.scrapers.news.sources import (
    RSS_FEEDS,
    GdeltQueryRejected,
    company_search_name,
    fetch_gdelt,
    parse_gdelt,
    parse_rss,
)
from tra.scrapers.psx.client import USER_AGENT
from tra.state import StateDB

log = logging.getLogger(__name__)
MARKET = "PSX"
WATERMARK = "news_gdelt"
UNSEARCHABLE = "unsearchable"


def link_symbols(articles: pd.DataFrame, mapper: SymbolMapper) -> pd.DataFrame:
    """One row per (article, symbol) named in the title or text. ``in_title`` marks the strong links."""
    rows = []
    for a in articles.itertuples(index=False):
        title_hits = mapper.symbols_in(a.title)
        for symbol in title_hits | mapper.symbols_in(a.text or ""):
            rows.append({**a._asdict(), "symbol": symbol, "in_title": symbol in title_hits})
    return pd.DataFrame(rows)


def company_names(lake: ParquetLake, symbols: list[str] | None = None) -> dict[str, str]:
    snap = lake.read("symbols", MARKET)
    snap = snap[snap["snapshot_date"] == snap["snapshot_date"].max()]
    snap = snap[
        ~snap["is_debt"].astype(bool) & ~snap["is_etf"].astype(bool) & ~snap["symbol"].str.contains("-")
    ]
    names = dict(zip(snap["symbol"], snap["name"], strict=True))
    return {s: names[s] for s in symbols if s in names} if symbols else names


def collect_rss(http: HttpClient, lake: ParquetLake) -> int:
    mapper = SymbolMapper(company_names(lake))
    now = pd.Timestamp.now(tz="UTC")
    total = 0
    for source, url in RSS_FEEDS.items():
        try:
            articles = parse_rss(http.get_bytes(url, use_cache=False), source, now)
        except HttpError as exc:  # one feed being down must not stop the others
            log.warning("RSS %s failed: %s", source, exc)
            continue
        linked = link_symbols(articles, mapper)
        if not linked.empty:
            lake.upsert("news", MARKET, linked)
        total += len(linked)
        log.info("RSS %s: %d articles, %d company links", source, len(articles), len(linked))
    return total


def collect_history(
    http: HttpClient, lake: ParquetLake, state: StateDB, symbols: list[str], first_year: int, last_year: int
) -> int:
    """Historical headlines per company and year from Pakistani outlets. Only headlines naming the company
    are kept as links: GDELT also returns articles that merely mention it somewhere."""
    names = company_names(lake, symbols)
    mapper = SymbolMapper(company_names(lake))
    total = 0
    for symbol, name in names.items():
        query = f'"{company_search_name(name)}" sourcecountry:pakistan'
        for year in range(first_year, last_year + 1):
            key = f"{symbol}:{year}"
            if state.get_watermark(WATERMARK, key) in ("done", UNSEARCHABLE):
                continue
            start = pd.Timestamp(year=year, month=1, day=1)
            end = pd.Timestamp(year=year, month=12, day=31, hour=23, minute=59, second=59)
            try:
                body = fetch_gdelt(http, query, start, end)
            except GdeltQueryRejected as exc:  # the same for every year, and for every later run
                log.warning("Skipping %s: %s", symbol, exc)
                for later in range(year, last_year + 1):
                    state.set_watermark(WATERMARK, f"{symbol}:{later}", UNSEARCHABLE)
                break
            except HttpError as exc:  # no watermark, so the next run tries this year again
                log.warning("GDELT %s failed: %s", key, exc)
                continue
            linked = link_symbols(parse_gdelt(body, pd.Timestamp.now(tz="UTC")), mapper)
            if not linked.empty:
                lake.upsert("news", MARKET, linked)
                total += len(linked)
            state.set_watermark(WATERMARK, key, "done")
        log.info("News history %s done (%d links so far)", symbol, total)
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="News collection")
    parser.add_argument("--dataset", choices=["rss", "history"], required=True)
    parser.add_argument("--since", type=int, default=2019)
    parser.add_argument("--until", type=int, default=None)
    parser.add_argument("--symbols", default="kse100", help="Comma-separated symbols, or 'kse100'")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = Settings()
    lake = ParquetLake(settings.data_dir)
    if args.dataset == "rss":
        collect_rss(HttpClient(USER_AGENT, requests_per_second=1.0), lake)
        return
    if args.symbols == "kse100":
        snap = lake.read("index_constituents", MARKET)
        symbols = sorted(snap.loc[snap["snapshot_date"] == snap["snapshot_date"].max(), "symbol"])
    else:
        symbols = args.symbols.split(",")
    # GDELT asks for one request per 5 seconds, but still answers 429 at 6, so stay well below that
    http = HttpClient(USER_AGENT, requests_per_second=1 / 15, backoff_seconds=30.0, timeout_seconds=60)
    state = StateDB(settings.data_dir / "state.sqlite")
    until = args.until or pd.Timestamp.now(tz="UTC").year
    collect_history(http, lake, state, symbols, args.since, until)


if __name__ == "__main__":
    main()
