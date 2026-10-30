import pandas as pd
import pytest

from tra.data.psx_financials import (
    ResultsExtract,
    accounting_checks,
    agreement,
    implied_share_stability,
    rule_extract,
    to_record,
)

# Typical layout of a PSX results announcement (current period first, comparative period second)
LETTER = """
LUCKY CEMENT LIMITED
FINANCIAL RESULTS FOR THE QUARTER ENDED SEPTEMBER 30, 2026
                                          Quarter ended
                                     September 30,  September 30,
                                          2026          2025
                                        (Rupees in '000)
Net sales                             45,123,456    40,100,200
Cost of sales                        (30,000,000)  (28,000,000)
Gross profit                          15,123,456    12,100,200
Profit before taxation                12,000,000     9,500,000
Taxation                              (3,600,000)   (2,850,000)
Profit after taxation                  8,400,000     6,650,000
Earnings per share - basic and diluted (Rupees)  28.75   22.76
"""


def test_rule_extract_reads_current_period_and_units():
    x = rule_extract(LETTER, "Financial Results for the Quarter Ended 2026-09-30")
    assert x.unit == "PKR_000" and x.period_months == 3
    assert x.revenue == 45_123_456 and x.gross_profit == 15_123_456
    assert x.profit_before_tax == 12_000_000 and x.taxation == 3_600_000
    assert x.profit_after_tax == 8_400_000 and x.eps == pytest.approx(28.75)


def test_units_are_applied_in_records():
    rec = to_record(rule_extract(LETTER, "quarter ended"), "rules")
    assert rec["profit_after_tax_pkr"] == pytest.approx(8.4e9)
    assert rec["check_pbt_tax_pat"] is True and rec["check_eps_sign"] is True


@pytest.mark.parametrize(
    ("title", "months"), [("Half Year Ended", 6), ("Nine Months ended", 9), ("Year Ended", 12)]
)
def test_period_length_from_title(title, months):
    assert (
        rule_extract(
            "Profit after taxation 10 9", f"Financial Results for the {title} 2026-06-30"
        ).period_months
        == months
    )


def test_losses_in_brackets_are_negative():
    x = rule_extract(
        "Rupees in million\nLoss after taxation (1,250.5) 300\nLoss per share (2.10) 0.5", "Year Ended"
    )
    assert (
        x.profit_after_tax == pytest.approx(-1250.5) and x.unit == "PKR_mn" and x.eps == pytest.approx(-2.10)
    )


def test_accounting_check_catches_inconsistent_figures():
    bad = ResultsExtract(profit_before_tax=100, taxation=30, profit_after_tax=90, eps=-1.0)
    checks = accounting_checks(bad)
    assert checks.pbt_tax_pat is False and checks.eps_sign is False


def test_unit_error_is_flagged_by_implied_shares():
    df = pd.DataFrame(
        {
            "symbol": ["A"] * 4,
            "eps": [2.0, 2.2, 2.1, 2.0],
            "profit_after_tax_pkr": [200e6, 220e6, 210e9, 200e6],
        }
    )  # third row read as millions
    assert implied_share_stability(df).tolist() == [True, True, False, True]


def test_rules_vs_llm_agreement():
    a = ResultsExtract(revenue=45_123_456, profit_after_tax=8_400_000, eps=28.75, unit="PKR_000")
    b = ResultsExtract(revenue=45_123.456, profit_after_tax=8_400.0, eps=28.75, unit="PKR_mn")
    assert agreement(a, b) == {"revenue": True, "profit_after_tax": True, "eps": True}


def test_ocr_style_text_without_spaces_and_dot_separators():
    ocr = (
        "PKRin'000'\nNet Revenue 131.863.008 125,207.898\nProfitbefore taxation 28,484,339 29,771,555\n"
        "Taxation (5,926,872) (6,210,065)\nProfit after taxation 22,557,467 23,561,490\n"
        "Earningspershare-basicanddiluted 15.01 15.68"
    )
    x = rule_extract(ocr, "Financial Results for the Quarter Ended 2026-09-30")
    assert x.revenue == 131_863_008 and x.profit_after_tax == 22_557_467 and x.eps == pytest.approx(15.01)
    assert x.unit == "PKR_000"


def test_process_reports_falls_back_and_marks_failures(tmp_path, monkeypatch):
    from unittest.mock import MagicMock

    from tra.data.lake import ParquetLake
    from tra.scrapers.psx import fetch_reports

    texts = {"1": LETTER, "2": "Board meeting notice without figures"}
    monkeypatch.setattr(fetch_reports, "pdf_text", lambda b: (texts[b.decode()], "text"))
    client = MagicMock()
    client.download.side_effect = lambda path: path.split("/")[-1].split(".")[0].encode()
    ann = pd.DataFrame(
        {
            "announcement_id": ["1", "2"],
            "symbol": ["LUCK", "LUCK"],
            "title": ["Financial Results for the Quarter Ended 2026-09-30"] * 2,
            "period_end": [pd.Timestamp("2026-09-30")] * 2,
            "known_at": [pd.Timestamp("2026-10-29 03:00", tz="UTC")] * 2,
            "doc_path": ["/download/document/1.pdf", "/download/document/2.pdf"],
        }
    )
    out = fetch_reports.process_reports(client, ParquetLake(tmp_path), tmp_path / "raw", ann).set_index(
        "announcement_id"
    )
    assert out.loc["1", "method"] == "rules" and out.loc["1", "profit_after_tax_pkr"] == pytest.approx(8.4e9)
    assert out.loc["2", "method"] == "failed"
