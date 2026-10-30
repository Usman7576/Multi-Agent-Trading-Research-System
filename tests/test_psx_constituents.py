import math
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from tra.data.lake import ParquetLake
from tra.data.pit_store import PITStore
from tra.scrapers.common.drift import DriftError
from tra.scrapers.common.parse_number import parse_number
from tra.scrapers.psx.fetch_constituents import snapshot_constituents, validate_snapshot
from tra.scrapers.psx.parsers.indices import parse_index_constituents
from tra.state import StateDB

FIXTURES = Path(__file__).parent / "fixtures" / "psx"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1,234.5", 1234.5),
        ("(120)", -120.0),
        ("-0.15", -0.15),
        ("0.42%", 0.42),
        ("Rs. 12", 12.0),
        (" 7 ", 7.0),
    ],
)
def test_parse_number(text, expected):
    assert parse_number(text) == expected


@pytest.mark.parametrize("text", ["", "-", "--", None, "N/A"])
def test_parse_number_missing_is_nan(text):
    assert math.isnan(parse_number(text))


def test_parse_number_rejects_garbage():
    with pytest.raises(ValueError):
        parse_number("abc")


def test_parser_reads_fixture():
    df = parse_index_constituents((FIXTURES / "kse100_constituents.html").read_text(encoding="utf-8"))
    assert df["symbol"].tolist()[:2] == ["ABL", "ABOT"]
    abl = df.set_index("symbol").loc["ABL"]
    assert abl["idx_weight_pct"] == pytest.approx(0.42)
    assert abl["free_float_shares"] == 114507383  # unrounded value from data-order
    assert abl["name"] == "Allied Bank Limited"


def test_drift_detection_rejects_renamed_column():
    with pytest.raises(DriftError):
        parse_index_constituents((FIXTURES / "kse100_constituents_drift.html").read_text(encoding="utf-8"))


def test_snapshot_sanity_checks():
    df = pd.DataFrame({"symbol": [f"S{i}" for i in range(100)], "idx_weight_pct": [1.0] * 100})
    validate_snapshot(df, "KSE100")
    with pytest.raises(ValueError):
        validate_snapshot(df.head(50), "KSE100")
    with pytest.raises(ValueError):
        validate_snapshot(df.assign(idx_weight_pct=0.5), "KSE100")


def test_snapshot_is_stored_point_in_time(tmp_path, monkeypatch):
    html = (FIXTURES / "kse100_constituents.html").read_text(encoding="utf-8")
    client = MagicMock()
    client.get_text.return_value = html
    monkeypatch.setattr("tra.scrapers.psx.fetch_constituents.validate_snapshot", lambda df, index: None)
    lake, state = ParquetLake(tmp_path), StateDB(tmp_path / "state.sqlite")
    fetched = pd.Timestamp("2026-10-30 12:30", tz="UTC")  # 17:30 Karachi
    snapshot_constituents(client, lake, state, tmp_path / "raw", "KSE100", fetched_at=fetched)

    pit = PITStore(lake)
    assert pit.index_constituents("PSX", "KSE100", fetched)["symbol"].tolist() == ["ABL", "ABOT", "AGP"]
    # nothing is known before the first snapshot was fetched
    assert pit.index_constituents("PSX", "KSE100", fetched - pd.Timedelta(hours=1)).empty
    assert state.get_watermark("psx_index_constituents", "KSE100") == "2026-10-30"
    assert (tmp_path / "raw" / "index_constituents" / "2026-10-30" / "KSE100.html").exists()
