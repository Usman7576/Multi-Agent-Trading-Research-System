"""Point-in-time data access: the heart of backtest correctness.

Every stored row carries ``known_at``: the UTC moment it became public (a price bar is known at that
session's close; an EDGAR fact at the end of its filing day). Every read takes an ``as_of`` timestamp and
returns only rows with ``known_at <= as_of``. If a clock is attached, asking for a moment later than the
clock's "now" raises ``LookAheadError``, so a backtest can never query the future by accident.
"""

from __future__ import annotations

import duckdb
import pandas as pd

from tra.clock import Clock
from tra.data.lake import ParquetLake


class LookAheadError(RuntimeError):
    pass


class PITStore:
    def __init__(self, lake: ParquetLake, clock: Clock | None = None) -> None:
        self._lake = lake
        self._clock = clock
        self._con = duckdb.connect()
        # DuckDB returns timestamps in the session time zone; keep everything in UTC
        self._con.execute("SET TimeZone = 'UTC'")

    # ------------------------------------------------------------------ prices
    def prices(
        self,
        market: str,
        as_of: pd.Timestamp,
        tickers: list[str] | None = None,
        start: pd.Timestamp | str | None = None,
    ) -> pd.DataFrame:
        """Daily bars known by ``as_of``, sorted by (ticker, date)."""
        as_of = self._check(as_of)
        sql = f"SELECT * FROM {self._src('prices_daily', market)} WHERE known_at <= $as_of"
        params: dict[str, object] = {"as_of": as_of}
        if start is not None:
            sql += " AND date >= $start"
            params["start"] = pd.Timestamp(start)
        if tickers is not None:
            sql += " AND list_contains($tickers, ticker)"
            params["tickers"] = list(tickers)
        df = self._con.execute(sql + " ORDER BY ticker, date", params).df()
        df["date"] = pd.to_datetime(df["date"])
        return df

    def price_panel(
        self,
        market: str,
        field: str,
        as_of: pd.Timestamp,
        tickers: list[str] | None = None,
        start: pd.Timestamp | str | None = None,
    ) -> pd.DataFrame:
        """Wide (date x ticker) panel of one price field."""
        df = self.prices(market, as_of, tickers, start)
        return df.pivot(index="date", columns="ticker", values=field).sort_index()

    def corporate_actions(self, market: str, as_of: pd.Timestamp) -> pd.DataFrame:
        as_of = self._check(as_of)
        sql = f"SELECT * FROM {self._src('corporate_actions', market)} WHERE known_at <= $as_of"
        return self._con.execute(sql + " ORDER BY ticker, ex_date", {"as_of": as_of}).df()

    # ------------------------------------------------------------ fundamentals
    def fundamentals(
        self, market: str, as_of: pd.Timestamp, tickers: list[str] | None = None
    ) -> pd.DataFrame:
        """Latest known value of every (ticker, metric, period) as of ``as_of``.

        Restatements filed later than ``as_of`` are invisible; restatements filed earlier replace the
        original value. When several XBRL concepts map to one metric, the highest-priority one wins.
        """
        as_of = self._check(as_of)
        params: dict[str, object] = {"as_of": as_of}
        where = "known_at <= $as_of"
        if tickers is not None:
            where += " AND list_contains($tickers, ticker)"
            params["tickers"] = list(tickers)
        sql = f"""
            SELECT * FROM {self._src("fundamentals", market)}
            WHERE {where}
            QUALIFY row_number() OVER (
                PARTITION BY ticker, metric, period_start, period_end
                ORDER BY priority ASC, known_at DESC
            ) = 1
            ORDER BY ticker, metric, period_end
        """
        return self._con.execute(sql, params).df()

    # ------------------------------------------------------------------ universe
    def universe(self, market: str, as_of: pd.Timestamp) -> pd.DataFrame:
        """Members on the date of ``as_of``: start_date <= day and (no end_date or end_date > day)."""
        day = self._check(as_of).tz_convert(None).normalize()
        sql = f"""
            SELECT * FROM {self._src("universe", market)}
            WHERE start_date <= $day AND (end_date IS NULL OR end_date > $day)
            ORDER BY ticker
        """
        return self._con.execute(sql, {"day": day}).df()

    def symbols(self, market: str, as_of: pd.Timestamp) -> pd.DataFrame:
        """The latest symbol/sector snapshot fetched on or before ``as_of``."""
        as_of = self._check(as_of)
        sql = f"""
            SELECT * FROM {self._src("symbols", market)} WHERE known_at <= $as_of
            QUALIFY dense_rank() OVER (ORDER BY snapshot_date DESC) = 1
        """
        return self._con.execute(sql, {"as_of": as_of}).df()

    def index_constituents(self, market: str, index: str, as_of: pd.Timestamp) -> pd.DataFrame:
        """The latest constituent snapshot of ``index`` fetched on or before ``as_of``."""
        as_of = self._check(as_of)
        sql = f"""
            SELECT * FROM {self._src("index_constituents", market)}
            WHERE "index" = $index AND known_at <= $as_of
            QUALIFY dense_rank() OVER (ORDER BY snapshot_date DESC) = 1
            ORDER BY symbol
        """
        return self._con.execute(sql, {"index": index, "as_of": as_of}).df()

    # ------------------------------------------------------------------ helpers
    def _check(self, as_of: pd.Timestamp) -> pd.Timestamp:
        as_of = pd.Timestamp(as_of)
        if as_of.tzinfo is None:
            raise ValueError("as_of must be timezone-aware (UTC)")
        as_of = as_of.tz_convert("UTC")
        if self._clock is not None and as_of > self._clock.now():
            raise LookAheadError(f"as_of {as_of} is after the clock's now {self._clock.now()}")
        return as_of

    def _src(self, table: str, market: str) -> str:
        path = self._lake.path(table, market)
        if not path.exists():
            raise FileNotFoundError(f"No {table} data for {market} at {path}; run ingestion first")
        return f"read_parquet('{path.as_posix()}')"
