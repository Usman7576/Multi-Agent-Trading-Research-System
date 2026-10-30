import gzip
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from tra.data.lake import ParquetLake
from tra.data.pit_store import PITStore
from tra.scrapers.common.drift import DriftError
from tra.scrapers.psx.client import PSXClient
from tra.scrapers.psx.fetch_prices import fetch_datewise, fetch_index_history, fetch_symbols, session_known_at
from tra.scrapers.psx.parsers.historical import parse_datewise, parse_symbol_month
from tra.state import StateDB

FIX = Path(__file__).parent / "fixtures" / "psx"


def read(name: str) -> str:
    return (FIX / name).read_text(encoding="utf-8")


def test_parse_datewise():
    df = parse_datewise(read("historical_date_2015-03-02.html"))
    abl = df.set_index("symbol").loc["ABL"]
    assert (abl["ldcp"], abl["open"], abl["high"], abl["low"], abl["close"]) == (
        108.5,
        108.5,
        108.5,
        107.1,
        108.0,
    )
    assert abl["volume"] == 49400
    assert set(df["security_type"]) == {"equity"}


def test_holiday_page_is_empty_not_an_error():
    assert parse_datewise(read("historical_holiday.html")).empty


def test_datewise_drift_is_rejected():
    with pytest.raises(DriftError):
        parse_datewise(read("historical_date_2015-03-02.html").replace(">LDCP<", ">PREV<"))


def test_parse_symbol_month_sorted_oldest_first():
    df = parse_symbol_month(read("historical_KSE100_2015-03.html"))
    assert df["date"].is_monotonic_increasing
    assert df["close"].iloc[-1] == pytest.approx(30233.87)


def test_known_at_is_after_the_close_in_karachi():
    ts = session_known_at(pd.Series([pd.Timestamp("2024-01-05")])).iloc[0]
    assert ts == pd.Timestamp("2024-01-05 12:00", tz="UTC")  # 17:00 PKT = 12:00 UTC


def _client(responses: dict[str, str]) -> MagicMock:
    client = MagicMock(spec=PSXClient)
    client.post.side_effect = lambda path, data, use_cache=False: responses[
        data.get("date") or f"{data['year']}-{data['month']}"
    ]
    return client


def test_fetch_datewise_stores_bars_calendar_and_is_resumable(tmp_path):
    responses = {
        "2015-03-02": read("historical_date_2015-03-02.html"),
        "2015-03-03": read("historical_holiday.html"),
    }
    client = _client(responses)
    lake, state = ParquetLake(tmp_path), StateDB(tmp_path / "state.sqlite")
    days = pd.DatetimeIndex(["2015-03-02", "2015-03-03"])
    fetch_datewise(client, lake, state, tmp_path / "raw", days)
    sessions = lake.read("sessions", "PSX").set_index("date")
    assert bool(sessions.loc["2015-03-02", "is_trading_day"]) and not bool(
        sessions.loc["2015-03-03", "is_trading_day"]
    )
    bars = PITStore(lake).prices("PSX", pd.Timestamp("2015-03-02 12:00", tz="UTC"))
    assert set(bars["ticker"]) >= {"ABL", "LUCK"}
    assert (
        PITStore(lake).prices("PSX", pd.Timestamp("2015-03-02 11:59", tz="UTC")).empty
    )  # not before 17:00 PKT
    with gzip.open(
        tmp_path / "raw" / "historical" / "2015" / "2015-03-02.html.gz", "rt", encoding="utf-8"
    ) as fh:
        assert "ABL" in fh.read()
    calls = client.post.call_count
    fetch_datewise(client, lake, state, tmp_path / "raw", days)  # second run fetches nothing
    assert client.post.call_count == calls


def test_index_history_is_incremental(tmp_path):
    client = _client(
        {"2015-3": read("historical_KSE100_2015-03.html"), "2015-4": read("historical_KSE100_2015-03.html")}
    )
    lake = ParquetLake(tmp_path)
    fetch_index_history(
        client,
        lake,
        tmp_path / "raw",
        "KSE100",
        pd.Timestamp("2015-03-01").date(),
        pd.Timestamp("2015-03-31").date(),
    )
    stored = lake.read("prices_daily", "PSX")
    assert set(stored["ticker"]) == {"KSE100"} and len(stored) == 4
    client.post.reset_mock()
    fetch_index_history(
        client,
        lake,
        tmp_path / "raw",
        "KSE100",
        pd.Timestamp("2015-03-01").date(),
        pd.Timestamp("2015-04-30").date(),
    )
    assert [c.args[1]["month"] for c in client.post.call_args_list] == ["3", "4"]


def test_fetch_symbols(tmp_path):
    import json

    client = MagicMock(spec=PSXClient)
    client.get_json.return_value = json.loads(read("symbols.json"))
    df = fetch_symbols(client, ParquetLake(tmp_path), pd.Timestamp("2026-10-30 12:00", tz="UTC"))
    luck = df.set_index("symbol").loc["LUCK"]
    assert luck["sector"] == "CEMENT" and not luck["is_debt"]
