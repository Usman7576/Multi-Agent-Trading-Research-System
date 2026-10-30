"""Corporate-action adjustment for PSX prices from the exchange's own reference price (LDCP).

On an ex-date (cash dividend, bonus or right issue) PSX sets LDCP, the reference price the day's change is
measured against, to the ex-adjusted previous close. So:

* the daily return including the corporate action is ``close / ldcp - 1``;
* the adjustment factor of an ex-date is ``ldcp / previous close`` (1.0 on ordinary days).

``adj_close`` is built *forward* by compounding daily returns from each symbol's first bar, so it never
depends on events after the date in question (unlike back-adjusted series that change retroactively).
Detected ex-dates are stored as ``corporate_events`` and validated against announced payouts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ROUNDING_TOL = 0.002  # LDCP vs previous close differences below 0.2% are price rounding, not events
FACE_VALUE = 10.0  # PSX par value for most ordinary shares (Rs 10); payouts are quoted as % of par


def adjust_prices(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add ``adj_close`` to PSX bars and return (bars, corporate_events)."""
    bars = prices.sort_values(["ticker", "date"]).copy()
    prev_close = bars.groupby("ticker")["close"].shift(1)
    ldcp = bars["ldcp"].where(bars["ldcp"] > 0, prev_close)
    # The first bar of a symbol (and any missing LDCP) has no reference: no return that day.
    gross = (bars["close"] / ldcp).where(ldcp.notna() & (ldcp > 0), 1.0)
    first_close = bars.groupby("ticker")["close"].transform("first")
    bars["adj_close"] = first_close * gross.groupby(bars["ticker"]).cumprod()

    factor = ldcp / prev_close
    is_event = prev_close.notna() & ((factor - 1).abs() > ROUNDING_TOL)
    events = pd.DataFrame(
        {
            "ticker": bars.loc[is_event, "ticker"],
            "ex_date": bars.loc[is_event, "date"],
            "prev_close": prev_close[is_event],
            "ldcp": ldcp[is_event],
            "factor": factor[is_event],
            "known_at": bars.loc[is_event, "known_at"] if "known_at" in bars else pd.NaT,
        }
    ).reset_index(drop=True)
    events["implied_cash_dividend"] = events["prev_close"] - events["ldcp"]
    events["implied_bonus_pct"] = (1 / events["factor"] - 1) * 100
    return bars, events


def expected_factor(kind: str, percent: float, prev_close: float, face_value: float = FACE_VALUE) -> float:
    """Ex-date factor implied by an announced payout (cash dividend or bonus; rights need the issue price)."""
    if kind == "dividend":
        return (prev_close - percent / 100 * face_value) / prev_close
    if kind == "bonus":
        return 1 / (1 + percent / 100)
    return np.nan


def validate_against_payouts(
    events: pd.DataFrame, payouts: pd.DataFrame, sessions: pd.DatetimeIndex, window: int = 6
) -> pd.DataFrame:
    """Match each announced payout to a detected ex-date in the ``window`` sessions before book closure.

    Returns one row per payout with the matched event (if any) and how close the detected factor is to the
    factor implied by the announcement.
    """
    rows = []
    sessions = pd.DatetimeIndex(sessions).sort_values()
    by_ticker = {t: g for t, g in events.groupby("ticker")}
    for p in payouts.itertuples(index=False):
        if pd.isna(p.book_closure_from):
            continue
        end = sessions.searchsorted(p.book_closure_from)
        start_day = sessions[max(0, end - window)]
        cands = by_ticker.get(p.symbol, events.iloc[0:0])
        cands = cands[(cands["ex_date"] >= start_day) & (cands["ex_date"] <= p.book_closure_from)]
        if cands.empty:
            rows.append({**p._asdict(), "matched": False})
            continue
        ev = cands.iloc[-1]
        exp = expected_factor(p.kind, p.percent, ev["prev_close"])
        rows.append(
            {
                **p._asdict(),
                "matched": True,
                "ex_date": ev["ex_date"],
                "detected_factor": ev["factor"],
                "expected_factor": exp,
                "factor_error": ev["factor"] - exp,
            }
        )
    return pd.DataFrame(rows)
