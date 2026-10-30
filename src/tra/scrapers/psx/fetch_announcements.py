"""Company announcements (with PDF links) and payouts for a list of PSX symbols.

Announcements are listed newest first, so an incremental run pages until it reaches an announcement that is
already stored. ``known_at`` is the announcement's own timestamp (Karachi time converted to UTC).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import pandas as pd

from tra.data.lake import ParquetLake
from tra.scrapers.psx.client import PSXClient
from tra.scrapers.psx.parsers.announcements import parse_announcements, parse_company_payouts

log = logging.getLogger(__name__)
MARKET = "PSX"
PAGE_SIZE = 50


def fetch_announcements(
    client: PSXClient, lake: ParquetLake, symbols: Iterable[str], max_pages: int = 60
) -> int:
    known: set[str] = set()
    if lake.exists("announcements", MARKET):
        known = set(lake.read("announcements", MARKET)["announcement_id"].astype(str))
    added = 0
    for symbol in symbols:
        frames = []
        for page in range(max_pages):
            html = client.post(
                "/announcements",
                {
                    "type": "C",
                    "symbol": symbol,
                    "query": "",
                    "count": str(PAGE_SIZE),
                    "offset": str(page * PAGE_SIZE),
                    "date_from": "",
                    "date_to": "",
                    "page": "annc",
                },
            )
            df, total = parse_announcements(html, source=f"announcements {symbol} p{page}")
            df = df[df["symbol"] == symbol]
            new = df[~df["announcement_id"].astype(str).isin(known)]
            frames.append(new)
            known |= set(new["announcement_id"].astype(str))
            last_page = len(df) < PAGE_SIZE or (page + 1) * PAGE_SIZE >= total
            if len(new) < len(df) or last_page:
                break
        batch = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if not batch.empty:
            batch["market"] = MARKET
            lake.upsert("announcements", MARKET, batch)
            added += len(batch)
        log.info("Announcements %s: %d new", symbol, len(batch))
    return added


def fetch_payouts(client: PSXClient, lake: ParquetLake, symbols: Iterable[str]) -> int:
    frames = []
    for symbol in symbols:
        html = client.post("/company/payouts", {"symbol": symbol})
        frames.append(parse_company_payouts(html, symbol, source=f"payouts {symbol}"))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not df.empty:
        df["known_at"] = df["announced_at"]
        lake.upsert("payouts", MARKET, df)
    return len(df)
