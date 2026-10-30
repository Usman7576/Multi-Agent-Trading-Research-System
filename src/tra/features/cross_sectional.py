"""Cross-sectional transforms: compare stocks with each other on the same date."""

from __future__ import annotations

import pandas as pd


def rank_by_date(panel: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Percentile ranks in (0, 1] across tickers for each date. NaNs stay NaN.

    Ranking removes market-wide drift and makes features with different scales comparable.
    """
    cols = columns or list(panel.columns)
    return panel[cols].groupby(level="date").rank(pct=True)


def zscore_by_date(s: pd.Series) -> pd.Series:
    g = s.groupby(level="date")
    return (s - g.transform("mean")) / g.transform("std")


def market_context(index_prices: pd.Series, stock_returns: pd.DataFrame) -> pd.DataFrame:
    """Date-level context shared by all stocks: index trend and volatility, cross-sectional dispersion."""
    r = index_prices.pct_change()
    return pd.DataFrame(
        {
            "mkt_ret_21": index_prices / index_prices.shift(21) - 1,
            "mkt_vol_21": r.rolling(21).std() * 252**0.5,
            "dispersion_21": stock_returns.std(axis=1).rolling(21).mean(),
        }
    )
