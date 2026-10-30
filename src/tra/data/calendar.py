"""Exchange trading calendars and session-close timestamps (UTC)."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pandas_market_calendars as mcal


def session_closes(calendar: str, start: date | str, end: date | str) -> pd.Series:
    """Map each session date to its close time in UTC (handles holidays and early closes)."""
    schedule = mcal.get_calendar(calendar).schedule(start_date=start, end_date=end)
    closes = pd.to_datetime(schedule["market_close"], utc=True)
    closes.index = pd.DatetimeIndex(schedule.index).normalize()
    closes.index.name = "date"
    return closes.rename("close_utc")


def end_of_local_day_utc(days: pd.Series, timezone: str) -> pd.Series:
    """23:59:59 local time on each date, in UTC.

    Used for sources that give a date but no time (e.g. EDGAR ``filed``). A document dated D is treated as
    known only after D ends locally, so it can first be used for the decision at the close of D+1.
    """
    local = pd.to_datetime(days).dt.tz_localize(timezone) + pd.Timedelta(hours=23, minutes=59, seconds=59)
    return local.dt.tz_convert("UTC")
