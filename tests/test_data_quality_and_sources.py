from unittest.mock import MagicMock

import pandas as pd
import pytest

from tests.conftest import make_prices
from tra.data.calendar import session_closes
from tra.data.edgar import parse_company_facts
from tra.data.quality import DataQualityError, check_prices, raise_on_errors
from tra.scrapers.common.http import HttpClient, HttpError


def _sessions():
    return session_closes("XNYS", "2016-01-04", "2020-12-31").index


def test_clean_fixture_has_no_errors():
    issues = check_prices(make_prices(), _sessions())
    assert issues[issues.severity == "error"].empty


def test_quality_catches_bad_rows():
    prices = make_prices()
    prices.loc[0, "close"] = -1.0
    prices.loc[1, "high"] = prices.loc[1, "low"] * 0.5
    prices = pd.concat([prices, prices.iloc[[5]]], ignore_index=True)
    issues = check_prices(prices, _sessions())
    assert {"non_positive_price", "high_below_body", "duplicate_bar"} <= set(issues["check"])
    with pytest.raises(DataQualityError):
        raise_on_errors(issues)


def test_quality_flags_unadjusted_split_and_missing_session():
    prices = make_prices()
    aaa = prices.index[prices.ticker == "AAA"]
    prices.loc[aaa[100] : aaa[-1], ["open", "high", "low", "close", "adj_close"]] /= 4.0
    prices = prices.drop(index=aaa[200])
    issues = check_prices(prices, _sessions())
    assert "extreme_return" in set(issues.loc[issues.ticker == "AAA", "check"])
    assert "missing_session" in set(issues.loc[issues.ticker == "AAA", "check"])


def test_edgar_parser_keeps_filed_date_and_priority():
    doc = {
        "facts": {
            "us-gaap": {
                "NetIncomeLoss": {
                    "units": {
                        "USD": [
                            {
                                "start": "2019-01-01",
                                "end": "2019-12-31",
                                "val": 10,
                                "form": "10-K",
                                "fy": 2019,
                                "fp": "FY",
                                "filed": "2020-02-20",
                                "accn": "a1",
                            },
                            {
                                "start": "2019-01-01",
                                "end": "2019-12-31",
                                "val": 10,
                                "form": "8-K",
                                "filed": "2020-01-30",
                            },
                        ]
                    }
                },
                "Revenues": {
                    "units": {
                        "USD": [
                            {
                                "start": "2019-01-01",
                                "end": "2019-12-31",
                                "val": 100,
                                "form": "10-K",
                                "filed": "2020-02-20",
                            }
                        ]
                    }
                },
                "SalesRevenueNet": {
                    "units": {
                        "USD": [
                            {
                                "start": "2019-01-01",
                                "end": "2019-12-31",
                                "val": 99,
                                "form": "10-K",
                                "filed": "2020-02-20",
                            }
                        ]
                    }
                },
            }
        }
    }
    df = parse_company_facts(doc, "XYZ")
    assert set(df["form"]) == {"10-K"}  # 8-K facts are ignored
    rev = df[df.metric == "revenue"].set_index("concept")["priority"]
    assert rev["Revenues"] < rev["SalesRevenueNet"]
    # filed 2020-02-20 (New York) -> known at the end of that local day
    assert df["known_at"].iloc[0] == pd.Timestamp("2020-02-21 04:59:59", tz="UTC")


def _response(status: int, body: bytes = b"ok"):
    resp = MagicMock()
    resp.status_code = status
    resp.content = body
    return resp


def test_http_client_retries_then_succeeds(tmp_path):
    session = MagicMock()
    session.headers = {}
    session.get.side_effect = [_response(503), _response(200, b"data")]
    client = HttpClient(
        "test agent", requests_per_second=1000, backoff_seconds=0, cache_dir=tmp_path, session=session
    )
    assert client.get_bytes("https://example.com/x") == b"data"
    assert client.get_bytes("https://example.com/x") == b"data"  # served from cache
    assert session.get.call_count == 2


def test_http_client_does_not_retry_client_errors():
    session = MagicMock()
    session.headers = {}
    session.get.return_value = _response(404)
    client = HttpClient("test agent", requests_per_second=1000, backoff_seconds=0, session=session)
    with pytest.raises(HttpError):
        client.get_bytes("https://example.com/missing")
    assert session.get.call_count == 1
