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

from tra.config import Settings, load_market
from tra.data.lake import ParquetLake
from tra.data.psx_adjust import adjust_prices
from tra.data.psx_financials import results_to_facts
from tra.data.quality import check_prices
from tra.data.universe import candidate_tickers
from tra.llm.local import DEFAULT_MODEL, LocalLLM
from tra.scrapers.common.http import HttpClient
from tra.scrapers.psx.client import USER_AGENT, PSXClient
from tra.scrapers.psx.fetch_announcements import fetch_announcements, fetch_payouts
from tra.scrapers.psx.fetch_constituents import snapshot_constituents
from tra.scrapers.psx.fetch_prices import fetch_datewise, fetch_index_history, fetch_symbols, weekdays
from tra.scrapers.psx.fetch_reports import process_reports, result_announcements
from tra.state import StateDB

DATASETS = [
    "constituents",
    "symbols",
    "prices",
    "index",
    "adjust",
    "announcements",
    "payouts",
    "reports",
    "daily",
]


def universe_candidates(lake: ParquetLake) -> list[str]:
    """Stocks ever among the liquidity-universe candidates (for announcements and reports)."""
    market = load_market("PSX")
    prices = lake.read("prices_daily", "PSX")
    prices = prices[(prices["ticker"] != market.benchmark) & (prices["security_type"] == "equity")]
    assert market.universe_rule is not None
    return candidate_tickers(prices, market.universe_rule)


def adjust(lake: ParquetLake, state: StateDB) -> None:
    """Total-return adjustment from LDCP, corporate-event detection and quality checks."""
    prices = lake.read("prices_daily", "PSX")
    index = prices[prices["ticker"] == "KSE100"].copy()
    index["adj_close"] = index["close"]
    bars, events = adjust_prices(prices[prices["ticker"] != "KSE100"])
    bars = pd.concat([bars, index], ignore_index=True)
    sessions = pd.DatetimeIndex(sorted(bars["date"].unique()))
    issues = check_prices(bars[bars["security_type"] == "equity"], sessions, max_abs_return=0.25)
    # thinly traded PSX stocks miss sessions all the time: only hard errors are logged, not gaps
    state.log_quality("PSX", "prices_daily", issues[issues["check"] != "missing_session"])
    lake.upsert("prices_daily", "PSX", bars)
    lake.upsert("corporate_events", "PSX", events)
    logging.getLogger(__name__).info(
        "Adjusted %d bars; %d corporate events; quality %s",
        len(bars),
        len(events),
        issues.groupby(["check", "severity"]).size().to_dict(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="PSX scraper")
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--index", action="append", help="Index code, repeatable (default KSE100)")
    parser.add_argument("--since", type=date.fromisoformat, default=date(2015, 1, 1))
    parser.add_argument("--until", type=date.fromisoformat, default=None)
    parser.add_argument("--llm", action="store_true", help="Also extract report figures with the local LLM")
    parser.add_argument(
        "--symbols", default=None, help="Comma-separated symbols, or 'kse100' (default: universe)"
    )
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
    if args.dataset in {"adjust", "daily"}:
        adjust(lake, state)

    def symbols() -> list[str]:
        if args.symbols == "kse100":
            snap = lake.read("index_constituents", "PSX")
            return sorted(snap.loc[snap["snapshot_date"] == snap["snapshot_date"].max(), "symbol"])
        return args.symbols.split(",") if args.symbols else universe_candidates(lake)

    if args.dataset in {"announcements", "daily"}:
        fetch_announcements(client, lake, symbols())
    if args.dataset in {"payouts", "daily"}:
        fetch_payouts(client, lake, symbols())
    if args.dataset in {"reports", "daily"}:
        llm = None
        model = settings.data_dir / "models" / DEFAULT_MODEL
        if args.llm and model.exists():
            llm = LocalLLM(model, settings.data_dir / "cache" / "llm.sqlite")
        todo = result_announcements(lake, symbols(), str(args.since))
        process_reports(client, lake, raw, todo, llm)
        facts = results_to_facts(lake.read("psx_results", "PSX"))
        if not facts.empty:
            lake.upsert("fundamentals", "PSX", facts)


if __name__ == "__main__":
    main()
