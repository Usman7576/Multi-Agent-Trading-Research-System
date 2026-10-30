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


def check_prices(
    prices: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    max_abs_return: float = 0.40,
    check_missing_sessions: bool = True,
    close_in_range: bool = True,
) -> pd.DataFrame:
    """Validate a long prices_daily frame against the exchange calendar.

    ``check_missing_sessions`` can be turned off for markets where thinly traded stocks routinely skip
    sessions (PSX), where gaps are normal rather than data errors. ``close_in_range=False`` checks only the
    open against high/low, for exchanges whose official close is not the last trade (PSX's closing-price
    rule can put a thin stock's close outside the day's traded range).
    """
    issues: list[pd.DataFrame] = []

    def add(mask: pd.Series, check: str, severity: str, detail: pd.Series | str) -> None:
        if mask.any():
            bad = prices.loc[mask]
            issues.append(
                pd.DataFrame(
                    {
                        "check": check,
                        "severity": severity,
                        "ticker": bad["ticker"].to_numpy(),
                        "date": bad["date"].to_numpy(),
                        "detail": detail[mask].to_numpy() if isinstance(detail, pd.Series) else detail,
                    }
                )
            )

    add(
        prices.duplicated(["ticker", "date"], keep=False),
        "duplicate_bar",
        "error",
        "duplicate (ticker, date)",
    )
    price_cols = ["open", "high", "low", "close", "adj_close"]
    add((prices[price_cols] <= 0).any(axis=1), "non_positive_price", "error", "price <= 0")
    add(prices["volume"] < 0, "negative_volume", "error", "volume < 0")
    add(prices[price_cols + ["volume"]].isna().any(axis=1), "missing_value", "warn", "NaN in OHLCV")
    tol = 1e-6
    body = ["open", "close"] if close_in_range else ["open"]
    add(
        prices["high"] < prices[body].max(axis=1) * (1 - tol),
        "high_below_body",
        "error",
        "high < max(open, close)",
    )
    add(
        prices["low"] > prices[body].min(axis=1) * (1 + tol),
        "low_above_body",
        "error",
        "low > min(open, close)",
    )
    add(prices["known_at"].isna(), "missing_known_at", "error", "known_at is null")

    ordered = prices.sort_values(["ticker", "date"])
    ret = ordered.groupby("ticker")["adj_close"].pct_change().reindex(prices.index)
    big = ret.abs() > max_abs_return
    add(
        big,
        "extreme_return",
        "warn",
        ret.where(big).map(lambda r: f"daily return {r:+.1%}", na_action="ignore"),
    )

    sessions = pd.DatetimeIndex(sessions).normalize()
    add(~prices["date"].isin(set(sessions)), "non_session_date", "error", "date is not a trading session")

    if check_missing_sessions:
        issues.append(_missing_sessions(prices, sessions))

    issues = [i for i in issues if not i.empty]
    return pd.concat(issues, ignore_index=True) if issues else pd.DataFrame(columns=COLUMNS)


def _missing_sessions(prices: pd.DataFrame, sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """Sessions with no bar between a ticker's first and last bar (vectorised over a presence matrix)."""
    present = (
        pd.DataFrame({"date": prices["date"], "ticker": prices["ticker"], "v": True})
        .drop_duplicates(["date", "ticker"])
        .pivot(index="date", columns="ticker", values="v")
        .reindex(sessions)
        .notna()
    )
    started = present.cummax()
    not_ended = present[::-1].cummax()[::-1]
    missing = (started & not_ended & ~present).stack()
    missing = missing[missing]
    return pd.DataFrame(
        {
            "check": "missing_session",
            "severity": "warn",
            "ticker": missing.index.get_level_values(1),
            "date": missing.index.get_level_values(0),
            "detail": "no bar on a trading session",
        }
    )


def raise_on_errors(issues: pd.DataFrame) -> None:
    errors = issues[issues["severity"] == "error"]
    if not errors.empty:
        raise DataQualityError(errors)
