"""PSX daily prices, one request per session for every listed symbol.

Date-wise requests return all symbols that traded that day, including ones later delisted, so the price
history is free of survivorship bias. Weekdays with an empty table are recorded as market holidays, which
gives the PSX trading calendar. Raw responses are archived gzip-compressed; a watermark per date makes the
job resumable and incremental. Index levels (KSE-100) come from the symbol-month layout.
"""

from __future__ import annotations

import gzip
import logging
from collections.abc import Iterable
from datetime import date
from pathlib import Path

import pandas as pd

from tra.data.lake import ParquetLake
from tra.scrapers.psx.client import PSXClient
from tra.scrapers.psx.parsers.historical import parse_datewise, parse_symbol_month
from tra.state import StateDB

log = logging.getLogger(__name__)
MARKET = "PSX"
TIMEZONE = "Asia/Karachi"
# PSX closes by 15:30 (Mon-Thu) and 16:30 (Fri) local time; treat a bar as known at 17:00 every day.
KNOWN_AT_LOCAL = pd.Timedelta(hours=17)
WATERMARK = "psx_prices_datewise"


def session_known_at(dates: pd.Series) -> pd.Series:
    local = pd.to_datetime(dates).dt.tz_localize(TIMEZONE) + KNOWN_AT_LOCAL
    return local.dt.tz_convert("UTC")


def weekdays(start: date, end: date) -> pd.DatetimeIndex:
    return pd.bdate_range(start, end)


def _archive(raw_dir: Path, sub: str, name: str, text: str) -> Path:
    path = raw_dir / sub / f"{name}.html.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write(text)
    return path


def bars_from_datewise(day: pd.Timestamp, table: pd.DataFrame) -> pd.DataFrame:
    bars = table.rename(columns={"symbol": "ticker"}).copy()
    bars["date"] = day
    bars["known_at"] = session_known_at(bars["date"])
    bars["market"] = MARKET
    return bars


def fetch_datewise(
    client: PSXClient,
    lake: ParquetLake,
    state: StateDB,
    raw_dir: Path,
    days: Iterable[pd.Timestamp],
    flush_every: int = 40,
) -> int:
    """Download each session not yet fetched. Returns the number of trading sessions stored."""
    pending_bars: list[pd.DataFrame] = []
    pending_sessions: list[dict[str, object]] = []
    stored = 0

    def flush() -> None:
        nonlocal stored
        if pending_sessions:
            lake.upsert("sessions", MARKET, pd.DataFrame(pending_sessions))
        if pending_bars:
            lake.upsert("prices_daily", MARKET, pd.concat(pending_bars, ignore_index=True))
            stored += len(pending_bars)
        for s in pending_sessions:
            state.set_watermark(WATERMARK, str(s["date"].date()), "done")  # type: ignore[union-attr]
        pending_bars.clear()
        pending_sessions.clear()

    for day in days:
        key = str(day.date())
        if state.get_watermark(WATERMARK, key) == "done":
            continue
        html = client.post("/historical", {"date": key})
        _archive(raw_dir, f"historical/{day.year}", key, html)
        table = parse_datewise(html, source=f"historical {key}")
        is_open = not table.empty
        pending_sessions.append({"date": day, "is_trading_day": is_open, "n_symbols": len(table)})
        if is_open:
            pending_bars.append(bars_from_datewise(day, table))
        if len(pending_sessions) >= flush_every:
            flush()
            log.info("PSX prices stored up to %s", key)
    flush()
    return stored


def fetch_index_history(
    client: PSXClient, lake: ParquetLake, raw_dir: Path, index: str, start: date, end: date
) -> pd.DataFrame:
    """Daily index levels (e.g. KSE100) month by month, stored in prices_daily under the index code.

    Incremental: only the month of the latest stored level onwards is fetched again.
    """
    first = pd.Timestamp(start)
    if lake.exists("prices_daily", MARKET):
        stored = lake.read("prices_daily", MARKET)
        stored = stored.loc[stored["ticker"] == index, "date"]
        if not stored.empty:
            first = max(first, stored.max().replace(day=1))
    frames = []
    for month_start in pd.date_range(first.replace(day=1), end, freq="MS"):
        html = client.post(
            "/historical", {"month": str(month_start.month), "year": str(month_start.year), "symbol": index}
        )
        _archive(raw_dir, f"index/{index}", month_start.strftime("%Y-%m"), html)
        frames.append(parse_symbol_month(html, source=f"{index} {month_start:%Y-%m}"))
    df = pd.concat(frames, ignore_index=True).drop_duplicates("date")
    df = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))]
    df["ticker"] = index
    df["ldcp"] = df["close"].shift(1)  # first row of an incremental batch is fixed by the adjustment step
    df["security_type"] = "index"
    df["known_at"] = session_known_at(df["date"])
    df["market"] = MARKET
    lake.upsert("prices_daily", MARKET, df)
    return df


def fetch_symbols(
    client: PSXClient, lake: ParquetLake, fetched_at: pd.Timestamp | None = None
) -> pd.DataFrame:
    """Snapshot of every listed symbol with its sector and security flags."""
    fetched_at = fetched_at or pd.Timestamp.now(tz="UTC")
    df = pd.DataFrame(client.get_json("/symbols"))
    df = df.rename(columns={"sectorName": "sector", "isETF": "is_etf", "isDebt": "is_debt"})
    df["snapshot_date"] = fetched_at.tz_convert(TIMEZONE).normalize().tz_localize(None)
    df["known_at"] = fetched_at
    lake.upsert("symbols", MARKET, df)
    return df
