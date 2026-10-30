"""Physical storage: one Parquet file per (table, market).

Only ingestion jobs, quality checks and ``PITStore`` use this module. Research code (features, models,
strategies, backtests) must read through ``PITStore`` so it can never see data that was not yet known.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Primary keys: rows with the same key are replaced on upsert (latest write wins).
TABLE_KEYS: dict[str, list[str]] = {
    "prices_daily": ["ticker", "date"],
    "corporate_actions": ["ticker", "ex_date", "type"],
    "fundamentals": ["ticker", "concept", "period_start", "period_end", "accn"],
    "universe": ["ticker", "start_date"],
    "index_constituents": ["index", "snapshot_date", "symbol"],
    "sessions": ["date"],
    "symbols": ["snapshot_date", "symbol"],
    "announcements": ["announcement_id"],
    "payouts": ["symbol", "announced_at", "details"],
    "corporate_events": ["ticker", "ex_date"],
    "psx_results": ["announcement_id"],
}


class ParquetLake:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def path(self, table: str, market: str) -> Path:
        if table not in TABLE_KEYS:
            raise KeyError(f"Unknown table {table!r}")
        return self.root / "parquet" / table / f"{market}.parquet"

    def exists(self, table: str, market: str) -> bool:
        return self.path(table, market).exists()

    def read(self, table: str, market: str) -> pd.DataFrame:
        return pd.read_parquet(self.path(table, market))

    def upsert(self, table: str, market: str, rows: pd.DataFrame) -> int:
        """Insert or replace rows by primary key. Returns the table's row count after the write."""
        keys = TABLE_KEYS[table]
        missing = [k for k in keys if k not in rows.columns]
        if missing:
            raise ValueError(f"{table}: rows are missing key columns {missing}")
        path = self.path(table, market)
        combined = pd.concat([self.read(table, market), rows], ignore_index=True) if path.exists() else rows
        combined = combined.drop_duplicates(subset=keys, keep="last").sort_values(keys).reset_index(drop=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        combined.to_parquet(tmp, index=False)
        tmp.replace(path)
        return len(combined)
