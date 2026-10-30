"""Daily snapshot of PSX index constituents (KSE-100 by default).

Past index membership is not published in a usable form, so history that is not recorded is lost. Run this
every trading day; each snapshot is archived raw and stored point-in-time (``known_at`` = fetch time).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from tra.data.lake import ParquetLake
from tra.scrapers.common.http import HttpClient
from tra.scrapers.psx.parsers.indices import parse_index_constituents
from tra.state import StateDB

log = logging.getLogger(__name__)

BASE_URL = "https://dps.psx.com.pk/indices/{index}"
TIMEZONE = "Asia/Karachi"
MARKET = "PSX"
# Sanity bounds: a KSE-100 snapshot with far fewer names or weights not summing to ~100% means bad data.
EXPECTED_COUNT = {"KSE100": (95, 105)}


def snapshot_constituents(
    client: HttpClient,
    lake: ParquetLake,
    state: StateDB,
    raw_dir: Path,
    index: str = "KSE100",
    fetched_at: pd.Timestamp | None = None,
) -> pd.DataFrame:
    fetched_at = fetched_at or pd.Timestamp.now(tz="UTC")
    snapshot_date = fetched_at.tz_convert(TIMEZONE).normalize().tz_localize(None)
    html = client.get_text(BASE_URL.format(index=index), use_cache=False)

    raw_path = raw_dir / "index_constituents" / snapshot_date.strftime("%Y-%m-%d") / f"{index}.html"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(html, encoding="utf-8")

    df = parse_index_constituents(html, source=f"{index} {snapshot_date.date()}")
    validate_snapshot(df, index)
    df.insert(0, "index", index)
    df.insert(1, "snapshot_date", snapshot_date)
    df["fetched_at"] = fetched_at
    df["known_at"] = fetched_at
    df["raw_path"] = raw_path.as_posix()
    lake.upsert("index_constituents", MARKET, df)
    state.set_watermark("psx_index_constituents", index, snapshot_date.strftime("%Y-%m-%d"))
    log.info("%s snapshot %s: %d constituents", index, snapshot_date.date(), len(df))
    return df


def validate_snapshot(df: pd.DataFrame, index: str) -> None:
    lo, hi = EXPECTED_COUNT.get(index, (1, 10_000))
    if not lo <= len(df) <= hi:
        raise ValueError(f"{index}: {len(df)} constituents, expected {lo}-{hi}")
    total = df["idx_weight_pct"].sum()
    if not 95.0 <= total <= 105.0:
        raise ValueError(f"{index}: index weights sum to {total:.2f}%, expected ~100%")
