"""Operational state in SQLite: quality-check log, run registry and scraper watermarks."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS quality_log (
    checked_at TEXT, market TEXT, table_name TEXT, check_name TEXT,
    severity TEXT, ticker TEXT, date TEXT, detail TEXT
);
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, name TEXT, config_hash TEXT, started_at TEXT,
    finished_at TEXT, status TEXT, metrics_json TEXT
);
CREATE TABLE IF NOT EXISTS watermarks (
    dataset TEXT, key TEXT, last_value TEXT, updated_at TEXT, PRIMARY KEY (dataset, key)
);
"""


class StateDB:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._con = sqlite3.connect(path)
        self._con.executescript(SCHEMA)

    def log_quality(self, market: str, table: str, issues: pd.DataFrame) -> None:
        if issues.empty:
            return
        now = pd.Timestamp.now(tz="UTC").isoformat()
        rows = [
            (now, market, table, r.check, r.severity, r.ticker, str(r.date), r.detail)
            for r in issues.itertuples(index=False)
        ]
        self._con.executemany("INSERT INTO quality_log VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
        self._con.commit()

    def record_run(
        self,
        run_id: str,
        name: str,
        config_hash: str,
        started_at: str,
        finished_at: str,
        status: str,
        metrics: dict[str, Any],
    ) -> None:
        self._con.execute(
            "INSERT OR REPLACE INTO runs VALUES (?, ?, ?, ?, ?, ?, ?)",
            (run_id, name, config_hash, started_at, finished_at, status, json.dumps(metrics, default=str)),
        )
        self._con.commit()

    def get_watermark(self, dataset: str, key: str) -> str | None:
        row = self._con.execute(
            "SELECT last_value FROM watermarks WHERE dataset = ? AND key = ?", (dataset, key)
        ).fetchone()
        return row[0] if row else None

    def set_watermark(self, dataset: str, key: str, value: str) -> None:
        self._con.execute(
            "INSERT OR REPLACE INTO watermarks VALUES (?, ?, ?, ?)",
            (dataset, key, value, pd.Timestamp.now(tz="UTC").isoformat()),
        )
        self._con.commit()

    def close(self) -> None:
        self._con.close()
