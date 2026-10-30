"""SEC EDGAR companyfacts: fundamentals with the date each value was filed.

``filed`` (not the fiscal period end) decides when a number becomes visible. One company may report the
same metric under different XBRL concepts over the years, so each metric lists concepts in priority order.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from tra.data.calendar import end_of_local_day_utc
from tra.scrapers.common.http import HttpClient

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
FORMS = {"10-K", "10-K/A", "10-Q", "10-Q/A"}

# metric -> [(taxonomy, concept, unit)], highest priority first
METRICS: dict[str, list[tuple[str, str, str]]] = {
    "revenue": [
        ("us-gaap", "Revenues", "USD"),
        ("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax", "USD"),
        ("us-gaap", "RevenueFromContractWithCustomerIncludingAssessedTax", "USD"),
        ("us-gaap", "SalesRevenueNet", "USD"),
    ],
    "net_income": [("us-gaap", "NetIncomeLoss", "USD")],
    "equity": [
        ("us-gaap", "StockholdersEquity", "USD"),
        ("us-gaap", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest", "USD"),
    ],
    "liabilities": [("us-gaap", "Liabilities", "USD")],
    "liabilities_and_equity": [("us-gaap", "LiabilitiesAndStockholdersEquity", "USD")],
    "assets": [("us-gaap", "Assets", "USD")],
    "shares_outstanding": [
        ("dei", "EntityCommonStockSharesOutstanding", "shares"),
        ("us-gaap", "WeightedAverageNumberOfDilutedSharesOutstanding", "shares"),
    ],
}


def ticker_to_cik(client: HttpClient) -> dict[str, int]:
    data = json.loads(client.get_text(TICKERS_URL))
    return {row["ticker"].upper(): int(row["cik_str"]) for row in data.values()}


def fetch_company_facts(client: HttpClient, cik: int) -> dict[str, Any]:
    return json.loads(client.get_text(FACTS_URL.format(cik=cik), use_cache=False))


def parse_company_facts(facts: dict[str, Any], ticker: str) -> pd.DataFrame:
    """Flatten a companyfacts document into one row per reported fact for the metrics we use."""
    rows: list[dict[str, Any]] = []
    for metric, concepts in METRICS.items():
        for priority, (taxonomy, concept, unit) in enumerate(concepts):
            entries = facts.get("facts", {}).get(taxonomy, {}).get(concept, {}).get("units", {}).get(unit, [])
            for e in entries:
                if e.get("form") not in FORMS or "filed" not in e or "end" not in e:
                    continue
                rows.append(
                    {
                        "ticker": ticker,
                        "metric": metric,
                        "concept": concept,
                        "priority": priority,
                        "value": float(e["val"]),
                        "period_start": e.get("start"),
                        "period_end": e["end"],
                        "form": e["form"],
                        "fy": e.get("fy"),
                        "fp": e.get("fp"),
                        "accn": e.get("accn", ""),
                        "filed": e["filed"],
                    }
                )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for col in ["period_start", "period_end", "filed"]:
        df[col] = pd.to_datetime(df[col])
    df["known_at"] = end_of_local_day_utc(df["filed"], "America/New_York")
    return df
