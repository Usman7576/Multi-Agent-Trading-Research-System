"""US ingestion: daily prices and corporate actions (yfinance) and fundamentals (SEC EDGAR).

    uv run python -m tra.data.ingest_us --prices --fundamentals

yfinance ``Close`` is split-adjusted and ``Adj Close`` is split- and dividend-adjusted, both retroactively.
Returns and price ratios are unaffected by that scaling; dollar volume uses ``close * volume``, which is
split-invariant. This is documented as a known limitation of free data.
"""

from __future__ import annotations

import argparse
import logging
from datetime import date

import pandas as pd
import yfinance as yf

from tra.config import Settings, load_market
from tra.data.calendar import session_closes
from tra.data.edgar import fetch_company_facts, parse_company_facts, ticker_to_cik
from tra.data.lake import ParquetLake
from tra.data.quality import check_prices, raise_on_errors
from tra.scrapers.common.http import HttpClient
from tra.state import StateDB

log = logging.getLogger(__name__)
MARKET = "US"
RENAME = {"Open": "open", "High": "high", "Low": "low", "Close": "close", "Adj Close": "adj_close",
          "Volume": "volume"}


def download_prices(tickers: list[str], start: date, end: date, calendar: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (prices_daily, corporate_actions) frames for ``tickers``."""
    closes = session_closes(calendar, start, end)
    bars, actions = [], []
    for ticker in tickers:
        # yfinance treats ``end`` as exclusive
        hist = yf.Ticker(ticker).history(
            start=start, end=end + pd.Timedelta(days=1), auto_adjust=False, actions=True
        )
        if hist.empty:
            log.warning("No price data for %s", ticker)
            continue
        hist.index = pd.DatetimeIndex(hist.index.tz_localize(None)).normalize()
        frame = hist.rename(columns=RENAME)[list(RENAME.values())].copy()
        frame["ticker"] = ticker
        frame["date"] = frame.index
        frame["known_at"] = closes.reindex(frame.index).to_numpy()
        bars.append(frame.reset_index(drop=True))
        for col, kind in [("Dividends", "dividend"), ("Stock Splits", "split")]:
            ev = hist.loc[hist[col] != 0, col]
            if not ev.empty:
                actions.append(pd.DataFrame({
                    "ticker": ticker, "ex_date": ev.index, "type": kind, "value": ev.to_numpy(),
                    # The ex-date event is public by that session's close at the latest.
                    "known_at": closes.reindex(ev.index).to_numpy(),
                }))
    prices = pd.concat(bars, ignore_index=True)
    prices["market"] = MARKET
    acts = pd.concat(actions, ignore_index=True) if actions else pd.DataFrame(
        columns=["ticker", "ex_date", "type", "value", "known_at"])
    return prices, acts


def ingest_prices(lake: ParquetLake, state: StateDB, start: date, end: date) -> None:
    market = load_market(MARKET)
    tickers = market.tickers + [market.benchmark]
    prices, actions = download_prices(tickers, start, end, market.calendar)
    sessions = session_closes(market.calendar, start, end).index
    issues = check_prices(prices, sessions)
    state.log_quality(MARKET, "prices_daily", issues)
    log.info("Quality: %s", issues.groupby(["check", "severity"]).size().to_dict() if len(issues) else "clean")
    raise_on_errors(issues)
    n = lake.upsert("prices_daily", MARKET, prices)
    lake.upsert("corporate_actions", MARKET, actions)
    first = prices.groupby("ticker")["date"].min()
    universe = pd.DataFrame({
        "ticker": market.tickers, "sector": [market.sectors[t] for t in market.tickers],
        "start_date": [first.get(t, pd.NaT) for t in market.tickers], "end_date": pd.NaT,
        "source": "configs/markets/us.yaml (current constituents; survivorship bias documented)",
    })
    lake.upsert("universe", MARKET, universe)
    log.info("prices_daily now has %d rows", n)


def ingest_fundamentals(lake: ParquetLake, settings: Settings) -> None:
    if not settings.sec_user_agent:
        raise SystemExit("Set SEC_USER_AGENT in .env (your name and email) before downloading from EDGAR")
    client = HttpClient(settings.sec_user_agent, requests_per_second=5, cache_dir=settings.data_dir / "cache" / "sec")
    ciks = ticker_to_cik(client)
    frames = []
    for ticker in load_market(MARKET).tickers:
        cik = ciks.get(ticker.replace(".", "-"))
        if cik is None:
            log.warning("No CIK for %s", ticker)
            continue
        frames.append(parse_company_facts(fetch_company_facts(client, cik), ticker))
        log.info("EDGAR facts for %s: %d", ticker, len(frames[-1]))
    facts = pd.concat(frames, ignore_index=True)
    facts["market"] = MARKET
    lake.upsert("fundamentals", MARKET, facts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prices", action="store_true")
    parser.add_argument("--fundamentals", action="store_true")
    parser.add_argument("--start", type=date.fromisoformat, default=date(2014, 1, 2))
    parser.add_argument("--end", type=date.fromisoformat, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = Settings()
    lake = ParquetLake(settings.data_dir)
    state = StateDB(settings.data_dir / "state.sqlite")
    end = args.end or (pd.Timestamp.now(tz="America/New_York").normalize() - pd.Timedelta(days=1)).date()
    if args.prices:
        ingest_prices(lake, state, args.start, end)
    if args.fundamentals:
        ingest_fundamentals(lake, settings)


if __name__ == "__main__":
    main()
