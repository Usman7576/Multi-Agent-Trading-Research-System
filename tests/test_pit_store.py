import ast
from pathlib import Path

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tra.clock import SimClock
from tra.data.lake import ParquetLake
from tra.data.pit_store import LookAheadError, PITStore

UTC_MOMENTS = st.datetimes(
    min_value=pd.Timestamp("2016-01-01").to_pydatetime(), max_value=pd.Timestamp("2021-01-31").to_pydatetime()
).map(lambda d: pd.Timestamp(d, tz="UTC"))


@settings(max_examples=60, deadline=None)
@given(as_of=UTC_MOMENTS)
def test_pit_no_lookahead_prices(lake, as_of):
    pit = PITStore(lake)
    got = pit.prices("US", as_of)
    raw = lake.read("prices_daily", "US")
    assert (got["known_at"] <= as_of).all()
    # completeness: everything known by as_of is returned
    assert len(got) == int((raw["known_at"] <= as_of).sum())


@settings(max_examples=40, deadline=None)
@given(as_of=UTC_MOMENTS)
def test_pit_no_lookahead_fundamentals(lake, as_of):
    got = PITStore(lake).fundamentals("US", as_of)
    assert (got["known_at"] <= as_of).all()


def test_bar_is_invisible_until_its_session_closes(lake):
    pit = PITStore(lake)
    day = pd.Timestamp("2019-03-15")
    before_close = pd.Timestamp("2019-03-15 19:59", tz="UTC")  # 15:59 New York
    after_close = pd.Timestamp("2019-03-15 20:00", tz="UTC")
    assert day not in set(pit.prices("US", before_close, tickers=["AAA"])["date"])
    assert day in set(pit.prices("US", after_close, tickers=["AAA"])["date"])


def test_fundamentals_visible_only_after_filing_day(lake):
    pit = PITStore(lake)
    filed_day_evening = pd.Timestamp("2019-02-15 23:00", tz="America/New_York")
    next_day = pd.Timestamp("2019-02-16 00:30", tz="America/New_York")
    fy2018 = pd.Timestamp("2018-12-31")
    assert fy2018 not in set(pit.fundamentals("US", filed_day_evening, ["AAA"])["period_end"])
    assert fy2018 in set(pit.fundamentals("US", next_day, ["AAA"])["period_end"])


def test_restatement_replaces_value_only_after_it_is_filed(tmp_path, lake):
    facts = lake.read("fundamentals", "US")
    original = facts[(facts.ticker == "AAA") & (facts.metric == "net_income") & (facts.fy == 2018)]
    restated = original.assign(
        value=-1.0,
        accn="AAA-2018-A",
        filed=pd.Timestamp("2019-06-01"),
        known_at=pd.Timestamp("2019-06-02 03:59:59", tz="UTC"),
    )
    local = ParquetLake(tmp_path)
    local.upsert("fundamentals", "US", pd.concat([original, restated]))
    pit = PITStore(local)
    assert (
        pit.fundamentals("US", pd.Timestamp("2019-05-01", tz="UTC"))["value"].item()
        == original["value"].item()
    )
    assert pit.fundamentals("US", pd.Timestamp("2019-07-01", tz="UTC"))["value"].item() == -1.0


def test_clock_guard_blocks_future_queries(lake):
    clock = SimClock("2019-01-02 21:00")
    pit = PITStore(lake, clock)
    pit.prices("US", pd.Timestamp("2019-01-02 21:00", tz="UTC"))
    with pytest.raises(LookAheadError):
        pit.prices("US", pd.Timestamp("2019-01-03 21:00", tz="UTC"))


def test_naive_as_of_is_rejected(lake):
    with pytest.raises(ValueError):
        PITStore(lake).prices("US", pd.Timestamp("2019-01-02"))


def test_universe_membership_by_date(tmp_path):
    local = ParquetLake(tmp_path)
    local.upsert(
        "universe",
        "US",
        pd.DataFrame(
            {
                "ticker": ["OLD", "NEW"],
                "sector": ["X", "X"],
                "start_date": [pd.Timestamp("2015-01-01"), pd.Timestamp("2019-01-01")],
                "end_date": [pd.Timestamp("2018-06-30"), pd.NaT],
                "source": "test",
            }
        ),
    )
    pit = PITStore(local)
    assert pit.universe("US", pd.Timestamp("2017-01-01", tz="UTC"))["ticker"].tolist() == ["OLD"]
    assert pit.universe("US", pd.Timestamp("2020-01-01", tz="UTC"))["ticker"].tolist() == ["NEW"]


RESEARCH_PACKAGES = ["features", "models", "strategies", "backtest", "portfolio", "risk", "eval"]


def test_no_direct_table_access():
    """Research code must read data only through PITStore, never from the lake or Parquet directly."""
    src = Path(__file__).resolve().parents[1] / "src" / "tra"
    offenders = []
    for pkg in RESEARCH_PACKAGES:
        for path in (src / pkg).rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            tree = ast.parse(text)
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module in {"tra.data.lake", "duckdb"}:
                    offenders.append(f"{path.name}: imports {node.module}")
                if isinstance(node, ast.Import) and any(a.name == "duckdb" for a in node.names):
                    offenders.append(f"{path.name}: imports duckdb")
            if "read_parquet" in text:
                offenders.append(f"{path.name}: calls read_parquet")
    assert not offenders, offenders
