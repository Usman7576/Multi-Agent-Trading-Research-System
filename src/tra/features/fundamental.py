"""Fundamental features with point-in-time visibility.

For every ticker we replay its filings in the order they became public (``known_at``). At each filing
moment we compute a snapshot (TTM net income, TTM revenue and its value a year earlier, latest equity,
liabilities, shares) from facts known at that moment only, then carry the snapshot forward to every
trading day until the next filing. Valuation ratios use the market price of that day.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DURATION_METRICS = ["net_income", "revenue"]
INSTANT_METRICS = ["equity", "liabilities", "liabilities_and_equity", "assets", "shares_outstanding"]


def _months(start: pd.Series, end: pd.Series) -> pd.Series:
    return ((end - start).dt.days / 30.44).round()


def ttm(facts: pd.DataFrame, cutoff: pd.Timestamp | None = None) -> float:
    """Trailing-twelve-month value of one duration metric from 10-K/10-Q facts.

    Uses the annual figure if it is the latest period; otherwise annual + current YTD - prior-year YTD.
    Only periods ending on or before ``cutoff`` are considered (used for "TTM one year ago").
    """
    f = facts.dropna(subset=["period_start"])
    if cutoff is not None:
        f = f[f["period_end"] <= cutoff]
    if f.empty:
        return np.nan
    f = f.assign(months=_months(f["period_start"], f["period_end"]))
    latest_end = f["period_end"].max()
    latest = f[f["period_end"] == latest_end]
    annual = latest[latest["months"].between(11, 13)]
    if not annual.empty:
        return float(annual["value"].iloc[0])

    ytd = latest[latest["months"].isin([3, 6, 9])].sort_values("months", ascending=False)
    annuals = f[f["months"].between(11, 13) & (f["period_end"] < latest_end)]
    if not ytd.empty and not annuals.empty:
        months = ytd["months"].iloc[0]
        prev_annual = annuals.loc[annuals["period_end"].idxmax()]
        prior_end = latest_end - pd.DateOffset(years=1)
        prior = f[(f["months"] == months) & ((f["period_end"] - prior_end).abs() <= pd.Timedelta(days=20))]
        if (latest_end - prev_annual["period_end"]).days < 370 and not prior.empty:
            return float(prev_annual["value"] + ytd["value"].iloc[0] - prior["value"].iloc[0])

    # Fall back to the latest annual figure if it is recent enough
    if not annuals.empty:
        last = annuals.loc[annuals["period_end"].idxmax()]
        if (latest_end - last["period_end"]).days <= 270:
            return float(last["value"])
    return np.nan


def _latest_instant(facts: pd.DataFrame) -> tuple[float, pd.Timestamp | None]:
    if facts.empty:
        return np.nan, None
    row = facts.loc[facts["period_end"].idxmax()]
    return float(row["value"]), row["period_end"]


def filing_snapshots(facts: pd.DataFrame) -> pd.DataFrame:
    """One row per (ticker, known_at) with the fundamentals known at that moment."""
    rows = []
    for ticker, tf in facts.groupby("ticker"):
        tf = tf.sort_values("known_at")
        for moment in tf["known_at"].unique():
            known = tf[tf["known_at"] <= moment]
            # latest filed value per (metric, period), best concept first
            known = known.sort_values(["priority", "known_at"], ascending=[True, False]).drop_duplicates(
                ["metric", "period_start", "period_end"]
            )
            by_metric = {m: g for m, g in known.groupby("metric")}
            empty = known.iloc[0:0]
            snap: dict[str, object] = {"ticker": ticker, "known_at": moment}
            for m in DURATION_METRICS:
                g = by_metric.get(m, empty)
                snap[f"{m}_ttm"] = ttm(g)
                ends = g["period_end"]
                snap[f"{m}_ttm_1y"] = ttm(g, ends.max() - pd.Timedelta(days=345)) if len(g) else np.nan
            for m in INSTANT_METRICS:
                value, period_end = _latest_instant(by_metric.get(m, empty))
                snap[m] = value
                if m == "shares_outstanding":
                    snap["shares_as_of"] = period_end
            rows.append(snap)
    return pd.DataFrame(rows)


def fundamental_features(
    facts: pd.DataFrame, prices: pd.DataFrame, closes_utc: pd.Series, splits: pd.DataFrame
) -> pd.DataFrame:
    """Daily fundamental features indexed by (date, ticker).

    ``closes_utc`` maps each session date to its close timestamp; a filing is usable on a date only if it
    was known by that day's close. ``splits`` (ticker, ex_date, value) is used to rebuild the unadjusted
    price and to keep share counts consistent across splits when computing market value.
    """
    if facts.empty:
        return pd.DataFrame()
    snaps = filing_snapshots(facts).sort_values("known_at")
    px = prices[["ticker", "date", "close"]].copy()
    px["close_utc"] = px["date"].map(closes_utc)
    px = px.dropna(subset=["close_utc"]).sort_values("close_utc")
    merged = pd.merge_asof(
        px, snaps, left_on="close_utc", right_on="known_at", by="ticker", direction="backward"
    ).sort_values(["ticker", "date"])

    raw_close = merged["close"] * _future_split_factor(merged, splits, "date")
    shares = merged["shares_outstanding"] * _splits_between(merged, splits, "shares_as_of", "date")
    mcap = raw_close * shares
    liabilities = merged["liabilities"].fillna(merged["liabilities_and_equity"] - merged["equity"])
    equity = merged["equity"].where(merged["equity"] > 0)

    out = pd.DataFrame(
        {
            "date": merged["date"],
            "ticker": merged["ticker"],
            "earnings_yield": merged["net_income_ttm"] / mcap,
            "book_to_market": merged["equity"] / mcap,
            "sales_to_price": merged["revenue_ttm"] / mcap,
            "roe": merged["net_income_ttm"] / equity,
            "debt_to_equity": liabilities / equity,
            "revenue_growth": merged["revenue_ttm"] / merged["revenue_ttm_1y"] - 1,
            "days_since_filing": (merged["close_utc"] - merged["known_at"]).dt.days,
        }
    )
    return out.set_index(["date", "ticker"]).sort_index().replace([np.inf, -np.inf], np.nan)


def _future_split_factor(df: pd.DataFrame, splits: pd.DataFrame, date_col: str) -> pd.Series:
    """Product of split ratios with ex_date after each row's date (undoes retroactive split adjustment)."""
    out = pd.Series(1.0, index=df.index)
    for row in splits.itertuples(index=False):
        mask = (df["ticker"] == row.ticker) & (df[date_col] < row.ex_date)
        out[mask] *= row.value
    return out


def _splits_between(df: pd.DataFrame, splits: pd.DataFrame, start_col: str, end_col: str) -> pd.Series:
    """Product of split ratios with start < ex_date <= end (share counts reported before a split)."""
    out = pd.Series(1.0, index=df.index)
    for row in splits.itertuples(index=False):
        mask = (df["ticker"] == row.ticker) & (df[start_col] < row.ex_date) & (row.ex_date <= df[end_col])
        out[mask] *= row.value
    return out


SIGNAL_COLUMNS = [
    "earnings_yield",
    "book_to_market",
    "sales_to_price",
    "roe",
    "debt_to_equity",
    "revenue_growth",
]
