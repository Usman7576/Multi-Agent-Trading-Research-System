"""Event-driven daily backtester.

Each session:  fill yesterday's orders at today's open -> mark to market at the close -> update the risk
engine -> on rebalance days score stocks, build the target portfolio, apply risk limits and submit orders
(which fill at the next open).

Look-ahead protection: features are built once from PITStore and are causal (proven by the feature-shift
test); the walk-forward model only trains on labels revealed by the decision day's close; the broker raises
if an order would fill on the bar that created it; and on day D the loop only indexes data at dates <= D.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from tra.backtest.broker_sim import Fill, SimBroker
from tra.backtest.costs import CostModel
from tra.config import ExperimentConfig
from tra.features.pipeline import ResearchData
from tra.models.regime import RegimeModel, regime_observations
from tra.portfolio.construct import construct
from tra.risk.limits import RiskContext, RiskEngine
from tra.strategies.alphas import AlphaModel

log = logging.getLogger(__name__)


@dataclass
class Decision:
    date: pd.Timestamp
    proposed: pd.Series
    approved: pd.Series
    p_high_vol: float
    kill_switch: bool


@dataclass
class BacktestResult:
    name: str
    config_hash: str
    nav: pd.Series
    benchmark_nav: pd.Series
    weights: pd.DataFrame
    fills: list[Fill]
    decisions: list[Decision]
    risk: RiskEngine
    regime: pd.Series
    turnover: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    @property
    def returns(self) -> pd.Series:
        return self.nav.pct_change().dropna()

    def trades_frame(self) -> pd.DataFrame:
        return pd.DataFrame([f.__dict__ for f in self.fills])


def is_rebalance_day(day: pd.Timestamp, next_day: pd.Timestamp | None, freq: str) -> bool:
    """Rebalance after the last session of each period (orders fill at the next period's first open)."""
    if freq == "daily":
        return True
    if next_day is None or freq == "never":
        return False
    if freq == "weekly":
        return day.isocalendar()[:2] != next_day.isocalendar()[:2]
    if freq == "monthly":
        return (day.year, day.month) != (next_day.year, next_day.month)
    raise ValueError(freq)


class RegimeTracker:
    """Refits the HMM every ``refit_every`` sessions on index data up to that day.

    Only filtered probabilities are used, so the regime on day D depends on data up to D.
    """

    def __init__(self, index_close: pd.Series, refit_every: int, min_obs: int, seed: int) -> None:
        self.obs = regime_observations(index_close)
        self.refit_every, self.min_obs, self.seed = refit_every, min_obs, seed
        self.model: RegimeModel | None = None
        self.since_fit = 0
        self.probs = pd.Series(dtype=float)

    def p_high_vol(self, day: pd.Timestamp) -> float:
        n_hist = int(self.obs.index.searchsorted(day, side="right"))
        if n_hist < self.min_obs:
            return 0.0
        self.since_fit += 1
        if self.model is None or self.since_fit >= self.refit_every:
            self.model = RegimeModel(seed=self.seed).fit(self.obs.iloc[:n_hist])
            # Filtered probabilities are causal: row t depends only on observations up to t,
            # so they can be computed for the whole series once per refit and read at each day.
            self.probs = self.model.p_high_vol(self.obs)
            self.since_fit = 0
        return float(self.probs.iloc[n_hist - 1])


def run_backtest(
    cfg: ExperimentConfig, data: ResearchData, alpha: AlphaModel, cost_multiplier: float = 1.0
) -> BacktestResult:
    market = cfg.market
    px = data.prices
    factor = px["adj_close"] / px["close"]
    open_adj = px.assign(v=px["open"] * factor).pivot(index="date", columns="ticker", values="v")
    close_adj = px.pivot(index="date", columns="ticker", values="adj_close")
    returns = close_adj.pct_change()
    adv = data.raw["adv_20"].unstack("ticker")
    daily_vol = data.raw["vol_21_daily"].unstack("ticker")

    end = cfg.dates.end or close_adj.index.max()
    sessions = close_adj.loc[pd.Timestamp(cfg.dates.test_start) : pd.Timestamp(end)].index

    broker = SimBroker(cfg.initial_nav, CostModel.from_config(cfg.costs, cost_multiplier))
    risk = RiskEngine(cfg.risk, market.sectors)
    regime = RegimeTracker(close_adj[market.benchmark].dropna(), cfg.model.retrain_every, 504, cfg.seed)
    navs, weights, decisions, regime_p, turnover = {}, {}, [], {}, {}

    for i, day in enumerate(sessions):
        next_day = sessions[i + 1] if i + 1 < len(sessions) else None
        empty = pd.Series(dtype=float)
        fills = broker.fill_pending_orders(
            day,
            open_adj.loc[day],
            adv.loc[day] if day in adv.index else empty,
            daily_vol.loc[day] if day in daily_vol.index else empty,
        )
        close_px = close_adj.loc[day]
        nav = broker.nav(close_px)
        navs[day] = nav
        weights[day] = broker.weights(close_px)
        turnover[day] = sum(abs(f.value) for f in fills) / nav / 2
        forced = risk.update(day, nav)
        p_hv = regime.p_high_vol(day) if cfg.risk.enabled else 0.0
        regime_p[day] = p_hv

        if i == 0 or forced or is_rebalance_day(day, next_day, cfg.rebalance):
            if next_day is None:
                continue
            scores = alpha.scores(day, data)
            w_prev = weights[day]
            proposed = construct(
                scores, returns.loc[:day], w_prev, market.sectors, cfg.portfolio, cfg.risk, market.benchmark
            )
            ctx = RiskContext(nav, adv.loc[day] if day in adv.index else empty, returns.loc[:day], p_hv)
            approved = risk.apply(day, proposed, ctx)
            decisions.append(Decision(day, proposed, approved, p_hv, risk.kill_switch))
            broker.submit(broker.orders_for_target(approved, close_px, day, cfg.costs.min_trade_weight))

    nav_s = pd.Series(navs, name=cfg.name)
    bench = close_adj[market.benchmark].reindex(nav_s.index)
    bench_nav = (bench / bench.iloc[0] * cfg.initial_nav).rename(market.benchmark)
    w_df = pd.DataFrame(weights).T.fillna(0.0)
    if broker.unfilled:
        log.info("%d orders could not fill (no open price)", len(broker.unfilled))
    return BacktestResult(
        cfg.name,
        cfg.config_hash(),
        nav_s,
        bench_nav,
        w_df,
        broker.fills,
        decisions,
        risk,
        pd.Series(regime_p, name="p_high_vol"),
        pd.Series(turnover, name="turnover"),
    )
