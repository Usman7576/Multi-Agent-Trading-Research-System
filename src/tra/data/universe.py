"""Point-in-time liquidity universe.

On each date a stock is a member if it is among the ``top_n`` ordinary shares by average traded value over
the previous ``window`` sessions, trades above ``min_price`` and has at least ``min_history`` sessions of
history. Everything is computed from data up to that date, and delisted stocks are included while they
traded, so the universe carries no knowledge of which companies later survived.
"""

from __future__ import annotations

import pandas as pd

from tra.config import UniverseRule


def liquidity_membership(prices: pd.DataFrame, rule: UniverseRule) -> pd.Series:
    """Boolean Series indexed by (date, ticker): True where the stock is in the universe that day."""
    close = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    volume = prices.pivot(index="date", columns="ticker", values="volume").reindex_like(close)
    traded_value = (close * volume).fillna(0.0)
    avg_value = traded_value.rolling(rule.window, min_periods=rule.window // 2).mean()
    history = close.notna().cumsum()
    eligible = (close >= rule.min_price) & (history >= rule.min_history) & close.notna()
    rank = avg_value.where(eligible).rank(axis=1, ascending=False, method="first")
    members = (rank <= rule.top_n).stack()
    members.index = members.index.set_names(["date", "ticker"])
    return members[members].astype(bool)


def candidate_tickers(prices: pd.DataFrame, rule: UniverseRule) -> list[str]:
    """Stocks that were ever among the ``candidates`` most liquid on any date (features are computed only
    for these, to keep the panel small). This uses the whole sample, but only to decide what to compute,
    never who is in the universe on a given day."""
    wide = UniverseRule(**{**rule.model_dump(), "top_n": rule.candidates})
    return sorted(liquidity_membership(prices, wide).index.get_level_values("ticker").unique())
