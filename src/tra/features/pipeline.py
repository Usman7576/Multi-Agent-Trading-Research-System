"""Build the research dataset (features, ranks, labels) from data known as of one moment.

All inputs come from ``PITStore``. Every feature on date D uses only rows known by D's close, which the
feature-shift test verifies by rebuilding the panel from truncated data and comparing.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from tra.config import MarketConfig
from tra.data.pit_store import PITStore
from tra.data.universe import candidate_tickers, liquidity_membership
from tra.features import fundamental, technical
from tra.features.cross_sectional import market_context, rank_by_date
from tra.features.labels import forward_excess_returns


@dataclass(frozen=True)
class ResearchData:
    prices: pd.DataFrame  # long prices of the candidate stocks and the benchmark
    raw: pd.DataFrame  # (date, ticker) raw features for every candidate
    ranked: pd.DataFrame  # (date, ticker) percentile ranks within that day's universe + date context
    labels: pd.DataFrame  # (date, ticker) forward returns and label_date (universe members only)
    context: pd.DataFrame  # date-level market context
    signal_columns: list[str]
    members: pd.Series  # (date, ticker) -> True for universe members on that date
    sectors: dict[str, str]


def _stock_prices(pit: PITStore, market: MarketConfig, as_of: pd.Timestamp, start: str | None):
    """Prices of the stocks to compute features for, the benchmark series, membership and sectors."""
    if market.universe_rule is None:
        tickers = market.tickers
        prices = pit.prices(market.name, as_of, tickers=[*tickers, market.benchmark], start=start)
        stock_px = prices[prices["ticker"].isin(tickers)]
        members = pd.Series(True, index=pd.MultiIndex.from_frame(stock_px[["date", "ticker"]]))
        return prices, stock_px, members, market.sectors

    prices = pit.prices(market.name, as_of, start=start)
    equities = prices[prices["ticker"] != market.benchmark]
    if "security_type" in equities:
        equities = equities[equities["security_type"] == "equity"]
    sectors: dict[str, str] = {}
    try:
        sym = pit.symbols(market.name, as_of)
        sym = sym[~sym["is_debt"].astype(bool) & ~sym["is_etf"].astype(bool)]
        sectors = dict(zip(sym["symbol"], sym["sector"].replace("", "Unknown"), strict=True))
        equities = equities[equities["ticker"].isin(sectors) | ~equities["ticker"].isin(set(sym["symbol"]))]
    except FileNotFoundError:
        pass
    keep = candidate_tickers(equities, market.universe_rule)
    stock_px = equities[equities["ticker"].isin(keep)]
    members = liquidity_membership(stock_px, market.universe_rule)
    bench = prices[prices["ticker"] == market.benchmark]
    return pd.concat([stock_px, bench], ignore_index=True), stock_px, members, sectors


def build_research_data(
    pit: PITStore, market: MarketConfig, as_of: pd.Timestamp, horizon: int, start: str | None = None
) -> ResearchData:
    prices, stock_px, members, sectors = _stock_prices(pit, market, as_of, start)
    tickers = sorted(stock_px["ticker"].unique())
    # session date -> moment its bars became known (works for any market, including derived calendars)
    closes = prices.groupby("date")["known_at"].max()

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
        signal_cols += [c for c in fundamental.SIGNAL_COLUMNS if raw[c].notna().any()]

    adj = stock_px.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    bench = prices[prices["ticker"] == market.benchmark].set_index("date")["adj_close"].sort_index()
    context = market_context(bench, adj.pct_change())

    in_universe = raw.index.isin(members.index)
    ranked = rank_by_date(raw[in_universe], signal_cols)
    ranked = ranked.join(context, on="date")
    member_panel = members.unstack("ticker").reindex(index=adj.index, columns=adj.columns)
    labels = forward_excess_returns(adj, horizon, member_panel.fillna(False).astype(bool))
    return ResearchData(prices, raw, ranked, labels, context, signal_cols, members, sectors)
