import json
from unittest.mock import MagicMock

import pandas as pd
import pytest

from tra.data.lake import ParquetLake
from tra.data.pit_store import PITStore
from tra.scrapers.common.drift import DriftError
from tra.scrapers.common.http import HttpError
from tra.scrapers.news.mapper import SymbolMapper
from tra.scrapers.news.run import link_symbols
from tra.scrapers.news.sources import company_search_name, doc_id, fetch_gdelt, parse_gdelt, parse_rss

NOW = pd.Timestamp("2026-11-04 10:00", tz="UTC")
RSS = b"""<?xml version="1.0"?><rss version="2.0" xmlns:content="http://purl.org/rss/1.0/modules/content/"><channel>
<item><title>Lucky Cement posts higher profit</title><link>https://x.pk/a</link>
<pubDate>Tue, 03 Nov 26 14:17:48 +0500</pubDate><content:encoded><![CDATA[<p>Profit <b>rose</b> 20%.</p>]]></content:encoded></item>
<item><title>No date here</title><link>https://x.pk/b</link></item>
<item><title>Bad date</title><link>https://x.pk/c</link><pubDate>soon</pubDate></item>
</channel></rss>"""
NAMES = {
    "PSX": "Pakistan Stock Exchange Limited",
    "LUCK": "Lucky Cement Limited",
    "POWER": "Power Cement Limited",
    "HBL": "Habib Bank Limited",
    "ABL": "Allied Bank Limited",
    "ENGROH": "Engro Holdings Limited",
    "EFERT": "Engro Fertilizers Limited",
    "ATRL": "Attock Refinery Limited",
    "XYZ": "Attock Refinery Limited",
}


def test_parse_rss_keeps_time_and_text_and_drops_undated_items():
    df = parse_rss(RSS, "dawn", NOW)
    assert df["title"].tolist() == ["Lucky Cement posts higher profit"]
    assert df["known_at"].iloc[0] == pd.Timestamp("2026-11-03 09:17:48", tz="UTC")
    assert df["text"].iloc[0] == "Profit rose 20%."


def test_rss_drift_is_rejected():
    with pytest.raises(DriftError):
        parse_rss(b"<html><body>Not found</body></html>", "dawn", NOW)


def test_parse_gdelt_uses_seen_date():
    body = json.dumps({"articles": [
        {"url": "https://x.pk/a", "title": "Hub  Power gains", "seendate": "20220118T171500Z", "domain": "x.pk",
         "language": "English"},
        {"url": "https://x.pk/b", "title": "no date", "seendate": ""},
    ]})  # fmt: skip
    df = parse_gdelt(body, NOW)
    assert len(df) == 1 and df["title"].iloc[0] == "Hub Power gains" and df["text"].iloc[0] == ""
    assert df["known_at"].iloc[0] == pd.Timestamp("2022-01-18 17:15", tz="UTC")


def test_gdelt_throttle_message_is_retried_then_fails():
    http = MagicMock()
    http.get_text.side_effect = ["Please limit requests to one every 5 seconds", '{"articles": []}']
    start, end = pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31")
    assert fetch_gdelt(http, "q", start, end, wait=0) == '{"articles": []}'
    http.get_text.side_effect = ["Please limit"] * 2
    with pytest.raises(HttpError):
        fetch_gdelt(http, "q", start, end, attempts=2, wait=0)


@pytest.mark.parametrize(
    ("name", "short"),
    [("Lucky Cement Limited", "Lucky Cement"), ("Abbott Laboratories (Pakistan) Limited", "Abbott Laboratories"),
     ("Fauji Fertilizer Company Limited", "Fauji Fertilizer")],
)  # fmt: skip
def test_company_search_name(name, short):
    assert company_search_name(name) == short


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Lucky Cement posts higher profit", {"LUCK"}),
        ("lucky cement announces dividend", {"LUCK"}),
        ("Habib Bank and Allied Bank report results", {"HBL", "ABL"}),
        ("EFERT rises 3pc while Engro Holdings falls", {"EFERT", "ENGROH"}),
        ("Luck ran out for the market", set()),  # a ticker that is an ordinary word
        ("Power shortages hit cement makers", set()),
        ("Pakistan Stock Exchange gains 1,400 points", set()),  # market news, not the listed company
        ("Attock Refinery shuts a unit", set()),  # the name belongs to two symbols, so it identifies neither
    ],
)
def test_symbol_mapper_prefers_missing_links_to_wrong_ones(text, expected):
    assert SymbolMapper(NAMES).symbols_in(text) == expected


def test_link_symbols_marks_title_links():
    articles = pd.DataFrame([{
        "doc_id": "1", "source": "s", "title": "Lucky Cement profit", "text": "Peers include Habib Bank.", "url": "u",
        "known_at": NOW, "fetched_at": NOW, "language": "English",
    }])  # fmt: skip
    linked = link_symbols(articles, SymbolMapper(NAMES)).set_index("symbol")
    assert bool(linked.loc["LUCK", "in_title"]) and not bool(linked.loc["HBL", "in_title"])


def test_news_is_point_in_time(tmp_path):
    lake = ParquetLake(tmp_path)
    rows = pd.DataFrame([{
        "doc_id": doc_id("u"), "symbol": "LUCK", "in_title": True, "source": "s", "title": "t", "text": "", "url": "u",
        "known_at": NOW, "fetched_at": NOW, "language": "English",
    }])  # fmt: skip
    lake.upsert("news", "PSX", rows)
    pit = PITStore(lake)
    assert pit.news("PSX", NOW - pd.Timedelta(minutes=1)).empty
    assert pit.news("PSX", NOW)["symbol"].tolist() == ["LUCK"]
    assert pit.news("PSX", NOW, symbols=["HBL"]).empty
