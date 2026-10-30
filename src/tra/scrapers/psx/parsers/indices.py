"""Parser for the PSX Data Portal index-constituents page (``/indices/<INDEX>``). No network access."""

from __future__ import annotations

import pandas as pd
from bs4 import BeautifulSoup, Tag

from tra.scrapers.common.drift import DriftError, require_headers
from tra.scrapers.common.parse_number import parse_number

# page header -> (output column, numeric?)
COLUMNS: dict[str, tuple[str, bool]] = {
    "SYMBOL": ("symbol", False),
    "NAME": ("name", False),
    "LDCP": ("ldcp", True),
    "CURRENT": ("current", True),
    "IDX WTG (%)": ("idx_weight_pct", True),
    "IDX POINT": ("idx_points", True),
    "VOLUME": ("volume", True),
    "FREEFLOAT (M)": ("free_float_shares", True),
    "MARKET CAP (M)": ("market_cap_pkr", True),
}


def _cell_value(cell: Tag, numeric: bool) -> object:
    # ``data-order`` holds the unrounded value (e.g. free float in shares, not millions)
    raw = cell.get("data-order")
    text = raw if isinstance(raw, str) else cell.get_text(" ", strip=True)
    return parse_number(text) if numeric else text.strip()


def parse_index_constituents(html: str, source: str = "psx index page") -> pd.DataFrame:
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", class_="tbl")
    if table is None:
        raise DriftError(f"{source}: constituents table not found")
    headers = [th.get_text(" ", strip=True) for th in table.select("thead th")]
    index_of = require_headers(headers, set(COLUMNS), source)

    records = []
    for tr in table.select("tbody tr"):
        cells = tr.find_all("td")
        if len(cells) != len(headers):
            raise DriftError(f"{source}: row has {len(cells)} cells, header has {len(headers)}")
        records.append({out: _cell_value(cells[index_of[h]], num) for h, (out, num) in COLUMNS.items()})

    df = pd.DataFrame(records)
    if df.empty:
        raise DriftError(f"{source}: no constituent rows")
    if df["symbol"].duplicated().any():
        raise DriftError(f"{source}: duplicate symbols {df.loc[df.symbol.duplicated(), 'symbol'].tolist()}")
    return df
