"""Build the research dataset (features, ranks, labels) from data known as of one moment.

All inputs come from ``PITStore``. Every feature on date D uses only rows known by D's close, which the
feature-shift test verifies by rebuilding the panel from truncated data and comparing.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from tra.config import MarketConfig
from tra.data.calendar import session_closes
from tra.data.pit_store import PITStore
from tra.features import fundamental, technical
from tra.features.cross_sectional import market_context, rank_by_date
from tra.features.labels import forward_excess_returns


@dataclass(frozen=True)
class ResearchData:
    prices: pd.DataFrame  # long prices of the universe and benchmark
    raw: pd.DataFrame  # (date, ticker) raw features
    ranked: pd.DataFrame  # (date, ticker) percentile-ranked signal features + date context
    labels: pd.DataFrame  # (date, ticker) forward returns and label_date
    context: pd.DataFrame  # date-level market context
    signal_columns: list[str]


def build_research_data(
    pit: PITStore, market: MarketConfig, as_of: pd.Timestamp, horizon: int, start: str | None = None
) -> ResearchData:
    tickers = market.tickers
    prices = pit.prices(market.name, as_of, tickers=[*tickers, market.benchmark], start=start)
    stock_px = prices[prices["ticker"].isin(tickers)]
    closes = session_closes(market.calendar, prices["date"].min(), prices["date"].max())

    tech = technical.technical_features(stock_px)
    signal_cols = list(technical.SIGNAL_COLUMNS)
    raw = tech
    try:
        facts = pit.fundamentals(market.name, as_of, tickers)
    except FileNotFoundError:
        facts = pd.DataFrame()
    if not facts.empty:
        try:
            actions = pit.corporate_actions(market.name, as_of)
        except FileNotFoundError:
            actions = pd.DataFrame(columns=["ticker", "ex_date", "type", "value"])
        splits = actions[actions["type"] == "split"][["ticker", "ex_date", "value"]]
        splits = splits.assign(ex_date=pd.to_datetime(splits["ex_date"]))
        fund = fundamental.fundamental_features(facts, stock_px, closes, splits)
        raw = tech.join(fund, how="left")
        signal_cols += fundamental.SIGNAL_COLUMNS

    adj = stock_px.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    bench = prices[prices["ticker"] == market.benchmark].set_index("date")["adj_close"].sort_index()
    context = market_context(bench, adj.pct_change())

    ranked = rank_by_date(raw, signal_cols)
    ranked = ranked.join(context, on="date")
    labels = forward_excess_returns(adj, horizon)
    return ResearchData(prices, raw, ranked, labels, context, signal_cols)
