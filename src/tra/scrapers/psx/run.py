"""PSX scraper CLI.

    uv run python -m tra.scrapers.psx.run --dataset constituents            # daily KSE-100 snapshot
    uv run python -m tra.scrapers.psx.run --dataset symbols                 # symbol / sector list
    uv run python -m tra.scrapers.psx.run --dataset prices --since 2015-01-01
    uv run python -m tra.scrapers.psx.run --dataset index --since 2015-01-01
    uv run python -m tra.scrapers.psx.run --dataset daily                   # everything incremental

All datasets are incremental: re-running only fetches what is missing.
"""

from __future__ import annotations

import argparse
import logging
from datetime import date

import pandas as pd

from tra.config import Settings
from tra.data.lake import ParquetLake
from tra.scrapers.common.http import HttpClient
from tra.scrapers.psx.client import USER_AGENT, PSXClient
from tra.scrapers.psx.fetch_constituents import snapshot_constituents
from tra.scrapers.psx.fetch_prices import fetch_datewise, fetch_index_history, fetch_symbols, weekdays
from tra.state import StateDB

DATASETS = ["constituents", "symbols", "prices", "index", "daily"]


def main() -> None:
    parser = argparse.ArgumentParser(description="PSX scraper")
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--index", action="append", help="Index code, repeatable (default KSE100)")
    parser.add_argument("--since", type=date.fromisoformat, default=date(2015, 1, 1))
    parser.add_argument("--until", type=date.fromisoformat, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    settings = Settings()
    http = HttpClient(USER_AGENT, requests_per_second=1.0, timeout_seconds=90)
    client = PSXClient(http)
    lake = ParquetLake(settings.data_dir)
    state = StateDB(settings.data_dir / "state.sqlite")
    raw = settings.data_dir / "raw" / "psx"
    until = args.until or pd.Timestamp.now(tz="Asia/Karachi").date()
    indices = args.index or ["KSE100"]

    if args.dataset in {"constituents", "daily"}:
        for index in indices:
            snapshot_constituents(http, lake, state, raw, index)
    if args.dataset in {"symbols", "daily"}:
        fetch_symbols(client, lake)
    if args.dataset in {"prices", "daily"}:
        fetch_datewise(client, lake, state, raw, weekdays(args.since, until))
    if args.dataset in {"index", "daily"}:
        for index in indices:
            fetch_index_history(client, lake, raw, index, args.since, until)


if __name__ == "__main__":
    main()
