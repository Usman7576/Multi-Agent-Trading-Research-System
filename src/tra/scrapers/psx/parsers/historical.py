"""Parsers for the PSX ``/historical`` endpoint. Pure functions: no network access.

Two layouts exist: date-wise (every symbol for one session, with LDCP) and symbol-month (one symbol or
index, every session of a month).
"""

from __future__ import annotations

import pandas as pd
from bs4 import BeautifulSoup

from tra.scrapers.common.drift import DriftError, require_headers
from tra.scrapers.common.parse_number import parse_number

DATEWISE = {
    "SYMBOL": "symbol",
    "LDCP": "ldcp",
    "OPEN": "open",
    "HIGH": "high",
    "LOW": "low",
    "CLOSE": "close",
    "VOLUME": "volume",
}
SYMBOL_MONTH = {
    "DATE": "date",
    "OPEN": "open",
    "HIGH": "high",
    "LOW": "low",
    "CLOSE": "close",
    "VOLUME": "volume",
}


def _table(html: str, source: str):
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", id="historicalTable") or soup.find("table", class_="tbl")
    if table is None:
        raise DriftError(f"{source}: historical table not found")
    headers = [th.get_text(" ", strip=True) for th in table.select("thead th")]
    return table, headers


def parse_datewise(html: str, source: str = "historical date") -> pd.DataFrame:
    """Every symbol's bar for one session. An empty frame means the market was closed that day."""
    table, headers = _table(html, source)
    index_of = require_headers(headers, set(DATEWISE), source)
    records = []
    for tr in table.select("tbody tr"):
        cells = tr.find_all("td")
        if len(cells) != len(headers):
            raise DriftError(f"{source}: row has {len(cells)} cells, header has {len(headers)}")
        rec: dict[str, object] = {"security_type": tr.get("data-type", "")}
        for header, col in DATEWISE.items():
            cell = cells[index_of[header]]
            raw = cell.get("data-value")
            text = raw if isinstance(raw, str) else cell.get_text(" ", strip=True)
            rec[col] = text.strip() if col == "symbol" else parse_number(text)
        records.append(rec)
    df = pd.DataFrame(records, columns=["security_type", *DATEWISE.values()])
    if df["symbol"].duplicated().any():
        raise DriftError(f"{source}: duplicate symbols")
    return df


def parse_symbol_month(html: str, source: str = "historical symbol") -> pd.DataFrame:
    """One symbol's (or index's) bars for a month, oldest first."""
    table, headers = _table(html, source)
    index_of = require_headers(headers, set(SYMBOL_MONTH), source)
    records = []
    for tr in table.select("tbody tr"):
        cells = tr.find_all("td")
        if not cells:
            continue
        rec: dict[str, object] = {}
        for header, col in SYMBOL_MONTH.items():
            text = cells[index_of[header]].get_text(" ", strip=True)
            rec[col] = pd.to_datetime(text, format="%b %d, %Y") if col == "date" else parse_number(text)
        records.append(rec)
    df = pd.DataFrame(records, columns=list(SYMBOL_MONTH.values()))
    return df.sort_values("date").reset_index(drop=True)
