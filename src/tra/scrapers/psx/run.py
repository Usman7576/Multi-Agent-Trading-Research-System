"""PSX scraper CLI.

    uv run python -m tra.scrapers.psx.run --dataset constituents --index KSE100

More datasets (prices, announcements, payouts, reports) are added in Milestone 2.
"""

from __future__ import annotations

import argparse
import logging

from tra.config import Settings
from tra.data.lake import ParquetLake
from tra.scrapers.common.http import HttpClient
from tra.scrapers.psx.fetch_constituents import snapshot_constituents
from tra.state import StateDB

USER_AGENT = (
    "Usman7576-trading-research/0.1 (+https://github.com/Usman7576/Multi-Agent-Trading-Research-System)"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="PSX scraper")
    parser.add_argument("--dataset", choices=["constituents"], required=True)
    parser.add_argument("--index", action="append", help="Index code, repeatable (default KSE100)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    settings = Settings()
    client = HttpClient(USER_AGENT, requests_per_second=1.0)
    lake = ParquetLake(settings.data_dir)
    state = StateDB(settings.data_dir / "state.sqlite")
    if args.dataset == "constituents":
        for index in args.index or ["KSE100"]:
            snapshot_constituents(client, lake, state, settings.data_dir / "raw" / "psx", index)


if __name__ == "__main__":
    main()
