import numpy as np
import pandas as pd
import pytest

from tests.conftest import BENCH, SECTORS, TICKERS
from tra.config import Instrument, MarketConfig
from tra.data.pit_store import PITStore
from tra.eval.signal import daily_ic, ic_summary
from tra.features.fundamental import ttm
from tra.features.labels import forward_excess_returns
from tra.features.pipeline import build_research_data
from tra.models.cv import PurgedKFold
from tra.models.regime import RegimeModel, regime_observations
from tra.models.return_model import ReturnModel

MARKET = MarketConfig(
    name="US",
    timezone="America/New_York",
    calendar="XNYS",
    close_time="16:00",
    benchmark=BENCH,
    universe=[Instrument(ticker=t, sector=SECTORS[t]) for t in TICKERS],
)


def test_feature_shift(lake):
    """Features on date D are identical whether or not data after D exists."""
    pit = PITStore(lake)
    full = build_research_data(pit, MARKET, pd.Timestamp("2021-01-01", tz="UTC"), horizon=10)
    cut_day = pd.Timestamp("2018-06-29")
    cut = build_research_data(pit, MARKET, pd.Timestamp("2018-06-29 21:00", tz="UTC"), horizon=10)
    cols = [c for c in full.raw.columns if c in cut.raw.columns]
    a = full.raw.loc[full.raw.index.get_level_values("date") <= cut_day, cols]
    b = cut.raw.loc[a.index, cols]
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-10)
    # fundamentals were present and are part of the check
    assert "earnings_yield" in cols and a["earnings_yield"].notna().any()


def test_fundamentals_use_filing_date_not_period_end(lake):
    data = build_research_data(PITStore(lake), MARKET, pd.Timestamp("2021-01-01", tz="UTC"), horizon=10)
    aaa = data.raw.xs("AAA", level="ticker")
    # FY2017 results are filed on 2018-02-15 and known after that day ends: first usable on 2018-02-16
    assert aaa.loc["2018-02-15", "days_since_filing"] > 300
    assert aaa.loc["2018-02-16", "days_since_filing"] == 0
    roe_old, roe_new = aaa.loc["2018-02-15", "roe"], aaa.loc["2018-02-16", "roe"]
    assert roe_old == pytest.approx((100 + 1) / 1000)  # FY2016 net income / equity
    assert roe_new == pytest.approx((100 + 2) / 1000)  # FY2017


def test_ttm_from_ytd_and_annual():
    rows = [
        ("2019-01-01", "2019-12-31", 400.0),  # FY2019
        ("2019-01-01", "2019-06-30", 180.0),  # H1 2019
        ("2020-01-01", "2020-06-30", 220.0),  # H1 2020
    ]
    facts = pd.DataFrame(
        [{"period_start": pd.Timestamp(s), "period_end": pd.Timestamp(e), "value": v} for s, e, v in rows]
    )
    assert ttm(facts) == pytest.approx(400 + 220 - 180)
    assert ttm(facts, cutoff=pd.Timestamp("2019-12-31")) == pytest.approx(400)


def test_labels_are_forward_and_dated():
    idx = pd.bdate_range("2020-01-01", periods=30)
    adj = pd.DataFrame({"A": np.linspace(100, 129, 30), "B": 100.0}, index=idx)
    lab = forward_excess_returns(adj, horizon=5)
    first = lab.loc[(idx[0], "A")]
    assert first["fwd_ret"] == pytest.approx(adj["A"].iloc[5] / adj["A"].iloc[0] - 1)
    assert first["label_date"] == idx[5]
    assert lab.index.get_level_values("date").max() == idx[-6]  # no label without its future


def test_purged_kfold_never_overlaps():
    dates = pd.DatetimeIndex(np.repeat(pd.bdate_range("2020-01-01", periods=200), 3))
    pos = np.searchsorted(np.sort(dates.unique()), dates)
    for train, test in PurgedKFold(4, horizon=10, embargo=5).split(dates):
        lo, hi = pos[test].min(), pos[test].max()
        tr = pos[train]
        assert not ((tr + 10 >= lo) & (tr <= hi + 15)).any()
        assert len(set(train) & set(test)) == 0


def test_regime_filter_is_causal():
    rng = np.random.default_rng(1)
    calm, wild = rng.normal(0.0005, 0.007, 400), rng.normal(-0.001, 0.03, 150)
    close = pd.Series(
        100 * np.exp(np.cumsum(np.r_[calm, wild, calm])), index=pd.bdate_range("2015-01-01", periods=950)
    )
    obs = regime_observations(close)
    model = RegimeModel(seed=3).fit(obs.iloc[:700])
    p_full = model.p_high_vol(obs)
    p_trunc = model.p_high_vol(obs.iloc[:500])
    pd.testing.assert_series_equal(p_full.iloc[:500], p_trunc)  # appending the future changes nothing
    assert p_full.iloc[430:520].mean() > 0.8 and p_full.iloc[100:350].mean() < 0.2


def test_return_model_finds_planted_signal():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2018-01-01", periods=300)
    idx = pd.MultiIndex.from_product([dates, [f"T{i}" for i in range(30)]], names=["date", "ticker"])
    X = pd.DataFrame({"signal": rng.random(len(idx)), "noise": rng.random(len(idx))}, index=idx)
    y = (X["signal"] + rng.normal(0, 0.5, len(idx))).groupby(level="date").rank(pct=True)
    grid = [
        {"num_leaves": 7, "max_depth": 3, "n_estimators": 60, "learning_rate": 0.05, "min_child_samples": 50},
        {"num_leaves": 3, "max_depth": 2, "n_estimators": 30, "learning_rate": 0.05, "min_child_samples": 50},
    ]
    train = X.index.get_level_values("date") < dates[200]
    model = ReturnModel(grid, horizon=5, embargo=2, cv_folds=3).fit(X[train], y[train])
    ic = daily_ic(model.predict(X[~train]), y[~train])
    assert ic_summary(ic)["mean_ic"] > 0.3
    assert model.feature_importance().index[0] == "signal"
