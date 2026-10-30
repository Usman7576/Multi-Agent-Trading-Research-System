import numpy as np
import pandas as pd
import pytest

from tests.conftest import BENCH, SECTORS, TICKERS
from tra.backtest.broker_sim import Order, SimBroker
from tra.backtest.costs import CostModel
from tra.backtest.engine import is_rebalance_day, run_backtest
from tra.config import (
    CostConfig,
    DatesConfig,
    ExperimentConfig,
    Instrument,
    MarketConfig,
    ModelConfig,
    PortfolioConfig,
    RiskConfig,
)
from tra.data.pit_store import PITStore
from tra.eval.metrics import max_drawdown, performance_summary
from tra.eval.significance import bootstrap_sharpe_ci, deflated_sharpe_ratio, probabilistic_sharpe_ratio
from tra.features.pipeline import build_research_data
from tra.portfolio.construct import construct
from tra.risk.limits import RiskContext, RiskEngine, historical_var_cvar
from tra.strategies.alphas import make_alpha

MARKET = MarketConfig(
    name="US",
    timezone="America/New_York",
    calendar="XNYS",
    close_time="16:00",
    benchmark=BENCH,
    universe=[Instrument(ticker=t, sector=SECTORS[t]) for t in TICKERS],
)
DAY = pd.Timestamp("2020-01-02")


def _cfg(
    model: str = "factor_composite", method: str = "mean_variance", risk: bool = True, **kw
) -> ExperimentConfig:
    return ExperimentConfig(
        name=f"test_{model}",
        rebalance=kw.get("rebalance", "weekly"),
        market=MARKET,
        dates=DatesConfig(start="2016-01-04", test_start="2018-01-02", end="2020-06-30"),
        model=ModelConfig(
            type=model,
            min_train_days=250,
            retrain_every=126,
            param_grid=[
                {
                    "num_leaves": 7,
                    "max_depth": 3,
                    "n_estimators": 50,
                    "learning_rate": 0.05,
                    "min_child_samples": 50,
                }
            ],
        ),
        portfolio=PortfolioConfig(method=method, top_n=kw.get("top_n", 4)),
        risk=RiskConfig(enabled=risk, max_pos=0.30, max_sector=0.60, top5_max=1.0, adv_cap=1.0),
        costs=CostConfig(),
    )


@pytest.fixture(scope="module")
def data(lake):
    return build_research_data(PITStore(lake), MARKET, pd.Timestamp("2021-01-01", tz="UTC"), horizon=10)


# ---------------------------------------------------------------- broker and costs
def test_costs_known_trade():
    cm = CostModel(commission_bps=1.0, slippage_bps=5.0, impact_k=0.1)
    # 6 bps fixed + 0.1 * 2% * sqrt(1e6 / 1e8) = 6 bps + 2 bps
    assert cm.cost_rate(1e6, adv=1e8, daily_vol=0.02) == pytest.approx(0.0008)
    assert cm.cost_rate(1e6, adv=np.nan, daily_vol=0.02) == pytest.approx(0.0006)


def test_fill_next_open_only():
    broker = SimBroker(1_000.0, CostModel(0, 0, 0))
    broker.submit([Order("AAA", 10, created=DAY)])
    with pytest.raises(RuntimeError):
        broker.fill_pending_orders(DAY, pd.Series({"AAA": 10.0}), pd.Series(), pd.Series())
    broker = SimBroker(1_000.0, CostModel(0, 0, 0))
    broker.submit([Order("AAA", 10, created=DAY)])
    fills = broker.fill_pending_orders(
        DAY + pd.Timedelta(days=1), pd.Series({"AAA": 10.0}), pd.Series(), pd.Series()
    )
    assert fills[0].price == 10.0 and broker.positions["AAA"] == 10 and broker.cash == pytest.approx(900.0)


def test_buys_are_scaled_to_available_cash_and_sells_run_first():
    broker = SimBroker(100.0, CostModel(0, 10.0, 0), positions={"BBB": 5.0})
    nxt = DAY + pd.Timedelta(days=1)
    broker.submit([Order("AAA", 100, DAY), Order("BBB", -5, DAY)])
    broker.fill_pending_orders(nxt, pd.Series({"AAA": 10.0, "BBB": 20.0}), pd.Series(), pd.Series())
    assert "BBB" not in broker.positions
    assert broker.cash == pytest.approx(0.0, abs=1e-6)  # 100 + 100 sale - costs, all spent on AAA
    assert broker.positions["AAA"] < 20


# ---------------------------------------------------------------- portfolio and risk
def _returns(n=300, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(0, 0.01, (n, len(TICKERS))), columns=TICKERS, index=pd.bdate_range("2019-01-01", periods=n)
    )


@pytest.mark.parametrize("method", ["equal", "inverse_vol", "mean_variance"])
def test_construction_respects_limits(method):
    scores = pd.Series(np.arange(len(TICKERS), dtype=float), index=TICKERS)
    risk = RiskConfig(max_pos=0.25, max_sector=0.40)
    cfg = PortfolioConfig(method=method, top_n=None, target_vol=None)
    w = construct(scores, _returns(), pd.Series(dtype=float), SECTORS, cfg, risk, BENCH)
    assert w.sum() <= 1 + 1e-6 and (w >= 0).all()
    if method == "mean_variance":
        assert w.max() <= 0.25 + 1e-6
        assert w.groupby(w.index.map(SECTORS)).sum().max() <= 0.40 + 1e-6


def test_vol_target_scales_down():
    cfg = PortfolioConfig(method="equal", top_n=None, target_vol=0.05)
    w = construct(
        pd.Series(1.0, index=TICKERS),
        _returns() * 3,
        pd.Series(dtype=float),
        SECTORS,
        cfg,
        RiskConfig(),
        BENCH,
    )
    assert w.sum() < 1.0


def test_risk_limits_and_veto_log():
    cfg = RiskConfig(max_pos=0.2, max_sector=0.3, top5_max=1.0, adv_cap=0.01, cvar_max=1.0)
    engine = RiskEngine(cfg, SECTORS)
    target = pd.Series({"AAA": 0.4, "BBB": 0.3, "CCC": 0.2, "EEE": 0.1})
    ctx = RiskContext(nav=1e6, adv=pd.Series(1e9, index=TICKERS), returns=_returns(), p_high_vol=0.0)
    w = engine.apply(DAY, target, ctx)
    assert w.max() <= 0.2 + 1e-9
    assert w[["AAA", "BBB"]].sum() <= 0.3 + 1e-9  # Tech sector cap
    assert {v.rule for _, v in engine.vetoes} >= {"max_position", "max_sector"}


def test_regime_and_liquidity_caps():
    cfg = RiskConfig(max_pos=1.0, max_sector=1.0, top5_max=1.0, adv_cap=0.01, cvar_max=1.0)
    engine = RiskEngine(cfg, SECTORS)
    target = pd.Series(0.2, index=["AAA", "CCC", "EEE", "BBB", "DDD"])
    adv = pd.Series(1e9, index=TICKERS).replace({1e9: 1e9}).astype(float)
    adv["AAA"] = 5e6  # 1% of ADV = 50k = 5% of a 1m NAV
    w = engine.apply(DAY, target, RiskContext(1e6, adv, _returns(), p_high_vol=0.9))
    assert w["AAA"] <= 0.05 + 1e-9
    assert w.sum() <= cfg.gross_high_vol + 1e-9


def test_drawdown_kill_switch_turns_on_and_off():
    cfg = RiskConfig(dd_kill=0.15, dd_recovery_window=5)
    engine = RiskEngine(cfg, SECTORS)
    navs = [100, 95, 84] + [84 + i for i in range(1, 8)]
    flips = [engine.update(DAY + pd.Timedelta(days=i), n) for i, n in enumerate(navs)]
    assert flips[2] is True and engine.events[0][1].startswith("kill-switch ON")
    assert any(flips[3:]) and not engine.kill_switch


def test_var_cvar():
    r = pd.DataFrame({"A": np.r_[np.full(95, 0.01), np.full(5, -0.05)]})
    var, cvar = historical_var_cvar(pd.Series({"A": 1.0}), r)
    assert var == pytest.approx(0.05) and cvar == pytest.approx(0.05)


# ---------------------------------------------------------------- metrics and significance
def test_metrics_known_values():
    nav = pd.Series([100, 110, 99, 120.0], index=pd.bdate_range("2020-01-01", periods=4))
    mdd, length = max_drawdown(nav)
    assert mdd == pytest.approx(0.1) and length == 1
    s = performance_summary(nav, nav)
    assert s["total_return"] == pytest.approx(0.2) and s["beta"] == pytest.approx(1.0)


def test_significance_sanity():
    rng = np.random.default_rng(0)
    good = pd.Series(rng.normal(0.001, 0.01, 1500))
    noise = pd.Series(rng.normal(0.0, 0.01, 1500))
    lo, hi = bootstrap_sharpe_ci(good, n_boot=300)
    assert lo < 1.59 < hi  # true annualised Sharpe = 0.1 * sqrt(252)
    assert probabilistic_sharpe_ratio(good) > 0.95
    assert deflated_sharpe_ratio(good, [1.6, 0.2, -0.3, 0.5, 0.1]) <= probabilistic_sharpe_ratio(good)
    assert probabilistic_sharpe_ratio(noise) < 0.95


# ---------------------------------------------------------------- end to end
def test_rebalance_calendar():
    fri, mon = pd.Timestamp("2020-01-03"), pd.Timestamp("2020-01-06")
    assert is_rebalance_day(fri, mon, "weekly") and not is_rebalance_day(
        mon, mon + pd.Timedelta(days=1), "weekly"
    )
    assert is_rebalance_day(pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-03"), "monthly")


@pytest.mark.parametrize(
    ("model", "method"),
    [("benchmark", "hold_benchmark"), ("factor_composite", "mean_variance"), ("lightgbm", "equal")],
)
def test_backtest_runs_and_is_deterministic(data, model, method):
    cfg = _cfg(model, method, risk=model != "benchmark")
    a = run_backtest(cfg, data, make_alpha(cfg.model, cfg.seed))
    b = run_backtest(cfg, data, make_alpha(cfg.model, cfg.seed))
    pd.testing.assert_series_equal(a.nav, b.nav)
    assert a.nav.iloc[0] == pytest.approx(cfg.initial_nav)
    assert (a.weights.sum(axis=1) <= 1 + 1e-6).all()
    for f in a.fills:
        assert f.date > f.order_created
    if model == "benchmark":
        ratio = a.nav / a.benchmark_nav
        assert ratio.iloc[5:].std() < 1e-3  # holds SPY after the first fill


def test_costs_reduce_returns(data):
    cfg = _cfg("factor_composite", "equal", rebalance="daily")
    base = run_backtest(cfg, data, make_alpha(cfg.model, cfg.seed))
    stressed = run_backtest(cfg, data, make_alpha(cfg.model, cfg.seed), cost_multiplier=3.0)
    assert stressed.nav.iloc[-1] < base.nav.iloc[-1]
