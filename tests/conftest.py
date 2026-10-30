"""Synthetic, deterministic fixture data so tests run offline in seconds."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tra.data.calendar import end_of_local_day_utc, session_closes
from tra.data.lake import ParquetLake

TICKERS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
SECTORS = {"AAA": "Tech", "BBB": "Tech", "CCC": "Energy", "DDD": "Energy", "EEE": "Health", "FFF": "Health"}
BENCH = "SPY"


def make_prices(start: str = "2016-01-04", end: str = "2020-12-31", seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    closes = session_closes("XNYS", start, end)
    n = len(closes)
    market = rng.normal(0.0003, 0.01, n)
    frames = []
    for i, ticker in enumerate([*TICKERS, BENCH]):
        beta = 1.0 if ticker == BENCH else 0.6 + 0.15 * i
        idio = 0.0 if ticker == BENCH else rng.normal(0.0001 * (i - 2), 0.015, n)
        ret = beta * market + idio
        close = 50.0 * (1 + i) * np.exp(np.cumsum(ret))
        open_ = close * np.exp(rng.normal(0, 0.004, n))
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.005, n)))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.005, n)))
        frames.append(
            pd.DataFrame(
                {
                    "ticker": ticker,
                    "date": closes.index,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "adj_close": close,
                    "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
                    "known_at": closes.to_numpy(),
                    "market": "US",
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def make_fundamentals() -> pd.DataFrame:
    rows = []
    for i, ticker in enumerate(TICKERS):
        for year in range(2015, 2021):
            filed = pd.Timestamp(f"{year + 1}-02-15")
            for metric, value in [
                ("net_income", 100.0 + 10 * i + year - 2015),
                ("equity", 1000.0 + 50 * i),
                ("revenue", 1000.0 + 100 * i + 20 * (year - 2015)),
                ("liabilities", 500.0 + 20 * i),
                ("assets", 1500.0 + 70 * i),
                ("shares_outstanding", 10.0),
            ]:
                instant = metric in {"equity", "liabilities", "assets", "shares_outstanding"}
                rows.append(
                    {
                        "ticker": ticker,
                        "metric": metric,
                        "concept": metric,
                        "priority": 0,
                        "value": value,
                        "period_start": pd.NaT if instant else pd.Timestamp(f"{year}-01-01"),
                        "period_end": pd.Timestamp(f"{year}-12-31"),
                        "form": "10-K",
                        "fy": year,
                        "fp": "FY",
                        "accn": f"{ticker}-{year}",
                        "filed": filed,
                    }
                )
    df = pd.DataFrame(rows)
    df["known_at"] = end_of_local_day_utc(df["filed"], "America/New_York")
    df["market"] = "US"
    return df


@pytest.fixture(scope="session")
def lake(tmp_path_factory) -> ParquetLake:
    root = tmp_path_factory.mktemp("lake")
    lake = ParquetLake(root)
    prices = make_prices()
    lake.upsert("prices_daily", "US", prices)
    lake.upsert("fundamentals", "US", make_fundamentals())
    first = prices.groupby("ticker")["date"].min()
    lake.upsert(
        "universe",
        "US",
        pd.DataFrame(
            {
                "ticker": TICKERS,
                "sector": [SECTORS[t] for t in TICKERS],
                "start_date": [first[t] for t in TICKERS],
                "end_date": pd.NaT,
                "source": "fixture",
            }
        ),
    )
    return lake
