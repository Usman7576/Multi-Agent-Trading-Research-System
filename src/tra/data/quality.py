"""Data-quality checks run after every ingestion.

Each check returns rows of (check, severity, ticker, date, detail). ``error`` issues stop the pipeline;
``warn`` issues are logged for review (e.g. a 40% move is often an unadjusted corporate action).
"""

from __future__ import annotations

import pandas as pd

COLUMNS = ["check", "severity", "ticker", "date", "detail"]


class DataQualityError(RuntimeError):
    def __init__(self, issues: pd.DataFrame) -> None:
        self.issues = issues
        super().__init__(f"{len(issues)} data-quality errors, first: {issues.iloc[0].to_dict()}")


def check_prices(prices: pd.DataFrame, sessions: pd.DatetimeIndex, max_abs_return: float = 0.40) -> pd.DataFrame:
    """Validate a long prices_daily frame against the exchange calendar."""
    issues: list[pd.DataFrame] = []

    def add(mask: pd.Series, check: str, severity: str, detail: pd.Series | str) -> None:
        if mask.any():
            bad = prices.loc[mask]
            issues.append(pd.DataFrame({
                "check": check, "severity": severity, "ticker": bad["ticker"].to_numpy(),
                "date": bad["date"].to_numpy(),
                "detail": detail[mask].to_numpy() if isinstance(detail, pd.Series) else detail,
            }))

    add(prices.duplicated(["ticker", "date"], keep=False), "duplicate_bar", "error", "duplicate (ticker, date)")
    price_cols = ["open", "high", "low", "close", "adj_close"]
    add((prices[price_cols] <= 0).any(axis=1), "non_positive_price", "error", "price <= 0")
    add(prices["volume"] < 0, "negative_volume", "error", "volume < 0")
    add(prices[price_cols + ["volume"]].isna().any(axis=1), "missing_value", "warn", "NaN in OHLCV")
    tol = 1e-6
    add(prices["high"] < prices[["open", "close"]].max(axis=1) * (1 - tol), "high_below_body", "error",
        "high < max(open, close)")
    add(prices["low"] > prices[["open", "close"]].min(axis=1) * (1 + tol), "low_above_body", "error",
        "low > min(open, close)")
    add(prices["known_at"].isna(), "missing_known_at", "error", "known_at is null")

    ordered = prices.sort_values(["ticker", "date"])
    ret = ordered.groupby("ticker")["adj_close"].pct_change()
    big = ret.abs() > max_abs_return
    add(big.reindex(prices.index, fill_value=False), "extreme_return", "warn",
        ret.reindex(prices.index).map(lambda r: f"daily return {r:+.1%}"))

    session_set = set(pd.DatetimeIndex(sessions).normalize())
    add(~prices["date"].isin(session_set), "non_session_date", "error", "date is not a trading session")

    for ticker, g in prices.groupby("ticker"):
        expected = [d for d in sessions if g["date"].min() <= d <= g["date"].max()]
        missing = sorted(set(expected) - set(g["date"]))
        if missing:
            issues.append(pd.DataFrame({
                "check": "missing_session", "severity": "warn", "ticker": ticker,
                "date": missing, "detail": "no bar on a trading session",
            }))

    return pd.concat(issues, ignore_index=True) if issues else pd.DataFrame(columns=COLUMNS)


def raise_on_errors(issues: pd.DataFrame) -> None:
    errors = issues[issues["severity"] == "error"]
    if not errors.empty:
        raise DataQualityError(errors)
