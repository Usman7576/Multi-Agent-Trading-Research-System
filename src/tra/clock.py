"""Single source of "now".

Every component that needs the current time asks a Clock. In a backtest the SimClock is moved forward one
session at a time; in paper trading the LiveClock returns wall-clock time. Nothing else may call
``datetime.now()`` or ``pd.Timestamp.now()``.
"""

from __future__ import annotations

from typing import Protocol

import pandas as pd


class Clock(Protocol):
    def now(self) -> pd.Timestamp: ...


class SimClock:
    """Simulated clock. Time only moves forward."""

    def __init__(self, start: pd.Timestamp | str) -> None:
        self._now = _utc(pd.Timestamp(start))

    def now(self) -> pd.Timestamp:
        return self._now

    def set(self, ts: pd.Timestamp | str) -> None:
        ts = _utc(pd.Timestamp(ts))
        if ts < self._now:
            raise ValueError(f"SimClock cannot move backwards: {ts} < {self._now}")
        self._now = ts


class LiveClock:
    def now(self) -> pd.Timestamp:
        return pd.Timestamp.now(tz="UTC")


def _utc(ts: pd.Timestamp) -> pd.Timestamp:
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
