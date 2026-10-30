from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from tra.data.lake import ParquetLake
from tra.scrapers.psx.client import PSXClient
from tra.scrapers.psx.fetch_announcements import fetch_announcements
from tra.scrapers.psx.parsers.announcements import (
    parse_announcements,
    parse_company_payouts,
    result_period_end,
)

FIX = Path(__file__).parent / "fixtures" / "psx"


def read(name: str) -> str:
    return (FIX / name).read_text(encoding="utf-8")


def test_parse_announcements_with_timestamp_and_pdf():
    df, total = parse_announcements(read("announcements_LUCK.html"))
    assert total == 634 and len(df) == 3
    res = df[df["title"].str.startswith("Financial Results")].iloc[0]
    assert res["announcement_id"] == "287455" and res["doc_path"] == "/download/document/287455.pdf"
    assert res["known_at"] == pd.Timestamp("2026-10-29 03:00", tz="UTC")  # 8:00 AM Karachi


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("Financial Results for the Quarter Ended 2026-09-30", "2026-09-30"),
        ("Financial Results for the Year Ended 2025-06-30 ", "2025-06-30"),
        ("Financial Results for the Half Year Ended 2024-12-31", "2024-12-31"),
        ("Presentation of Corporate Briefing Session", None),
    ],
)
def test_result_period_end(title, expected):
    got = result_period_end(title)
    assert (got is None and expected is None) or got == pd.Timestamp(expected)


def test_parse_payouts():
    df = parse_company_payouts(read("company_payouts_EFERT.html"), "EFERT")
    first = df.iloc[0]
    assert (first["kind"], first["percent"], first["period_code"]) == ("dividend", 55.0, "IIIQ")
    assert first["period_end"] == pd.Timestamp("2026-09-30")
    assert first["book_closure_from"] == pd.Timestamp("2026-11-02")
    assert first["announced_at"] == pd.Timestamp("2026-10-19 11:13", tz="UTC")


def test_bonus_and_right_payouts_are_classified():
    html = (
        read("company_payouts_EFERT.html")
        .replace("55%(iii) (D)", "10% (B)")
        .replace("17.5%(ii) (D)", "25% (R)")
    )
    kinds = parse_company_payouts(html, "EFERT")["kind"].tolist()
    assert kinds[:2] == ["bonus", "right"]


def test_incremental_paging_stops_at_known_announcements(tmp_path):
    client = MagicMock(spec=PSXClient)
    client.post.return_value = read("announcements_LUCK.html")
    lake = ParquetLake(tmp_path)
    assert fetch_announcements(client, lake, ["LUCK"]) == 3
    client.post.reset_mock()
    assert fetch_announcements(client, lake, ["LUCK"]) == 0
    assert client.post.call_count == 1  # first page already known: stop


def test_invalid_dates_typed_by_companies_become_missing_not_errors():
    html = read("company_payouts_EFERT.html").replace("30/09/2026(IIIQ)", "31/09/2026(IIIQ)")
    df = parse_company_payouts(html, "EFERT")
    assert pd.isna(df["period_end"].iloc[0]) and df["percent"].iloc[0] == 55.0
