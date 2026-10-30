"""Risk engine with veto power over every proposed portfolio.

Rules run in a fixed order and each one can only reduce exposure; weight removed from a position goes to
cash (it is never redistributed to other names). Every change is logged with the rule that caused it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from tra.config import RiskConfig


@dataclass(frozen=True)
class RiskContext:
    nav: float
    adv: pd.Series  # 20-day average dollar volume per ticker
    returns: pd.DataFrame  # trailing daily returns (date x ticker) for VaR/CVaR
    p_high_vol: float  # filtered HMM probability of the high-volatility regime


@dataclass(frozen=True)
class Veto:
    rule: str
    ticker: str
    before: float
    after: float


def historical_var_cvar(
    weights: pd.Series, returns: pd.DataFrame, level: float = 0.95
) -> tuple[float, float]:
    """One-day historical VaR and CVaR of a portfolio, as positive fractions of NAV."""
    cols = [t for t in weights.index if t in returns.columns]
    if not cols:
        return 0.0, 0.0
    pnl = returns[cols].fillna(0.0).to_numpy() @ weights[cols].to_numpy()
    q = np.quantile(pnl, 1 - level, method="lower")  # conservative: an observed loss
    return float(-q), float(-pnl[pnl <= q].mean())


@dataclass
class RiskEngine:
    cfg: RiskConfig
    sectors: dict[str, str]
    peak_nav: float = 0.0
    kill_switch: bool = False
    nav_history: list[float] = field(default_factory=list)
    vetoes: list[tuple[pd.Timestamp, Veto]] = field(default_factory=list)
    events: list[tuple[pd.Timestamp, str]] = field(default_factory=list)

    def update(self, day: pd.Timestamp, nav: float) -> bool:
        """Track drawdown; return True when the kill-switch turns on or off (forces a rebalance)."""
        self.nav_history.append(nav)
        self.peak_nav = max(self.peak_nav, nav)
        if not self.cfg.enabled:
            return False
        drawdown = 1 - nav / self.peak_nav
        if not self.kill_switch and drawdown > self.cfg.dd_kill:
            self.kill_switch = True
            self.events.append((day, f"kill-switch ON (drawdown {drawdown:.1%})"))
            return True
        recent = self.nav_history[-self.cfg.dd_recovery_window :]
        if self.kill_switch and len(recent) == self.cfg.dd_recovery_window and nav >= max(recent):
            self.kill_switch = False
            # Measure future drawdowns from here, otherwise the switch would re-trigger the next day
            self.peak_nav = nav
            self.events.append((day, f"kill-switch OFF (new {self.cfg.dd_recovery_window}-day high)"))
            return True
        return False

    def apply(self, day: pd.Timestamp, target: pd.Series, ctx: RiskContext) -> pd.Series:
        if not self.cfg.enabled or target.empty:
            return target
        w = target.astype(float).copy()

        def log(rule: str, before: pd.Series, after: pd.Series) -> None:
            for t in before.index:
                if after.get(t, 0.0) < before[t] - 1e-9:
                    self.vetoes.append((day, Veto(rule, t, float(before[t]), float(after.get(t, 0.0)))))

        # 1. position limit
        capped = w.clip(upper=self.cfg.max_pos)
        log("max_position", w, capped)
        w = capped
        # 2. sector limit: scale names in an over-weight sector down proportionally
        sec = w.groupby(w.index.map(lambda t: self.sectors.get(t, "Unknown"))).transform("sum")
        scaled = w * np.minimum(1.0, self.cfg.max_sector / sec.where(sec > 0, 1.0))
        log("max_sector", w, scaled)
        w = scaled
        # 3. concentration: top-5 names
        top5 = w.nlargest(5).sum()
        if top5 > self.cfg.top5_max:
            scaled = w * self.cfg.top5_max / top5
            log("top5_concentration", w, scaled)
            w = scaled
        # 4. liquidity: position value below adv_cap x 20-day average dollar volume
        cap = (self.cfg.adv_cap * ctx.adv.reindex(w.index) / ctx.nav).fillna(0.0)
        limited = np.minimum(w, cap)
        log("liquidity", w, limited)
        w = limited
        # 5. gross exposure by regime and drawdown kill-switch
        gross_cap = self.cfg.gross_normal
        rule = "gross_exposure"
        if ctx.p_high_vol > self.cfg.high_vol_prob:
            gross_cap, rule = min(gross_cap, self.cfg.gross_high_vol), "high_vol_regime"
        if self.kill_switch:
            gross_cap, rule = min(gross_cap, self.cfg.dd_kill_gross), "drawdown_kill_switch"
        if w.sum() > gross_cap:
            scaled = w * gross_cap / w.sum()
            log(rule, w, scaled)
            w = scaled
        # 6. ex-ante CVaR: shrink gross until the 95% one-day CVaR is within the limit
        window = ctx.returns.iloc[-self.cfg.cvar_window :]
        _, cvar = historical_var_cvar(w, window)
        if cvar > self.cfg.cvar_max:
            scaled = w * self.cfg.cvar_max / cvar
            log("cvar_limit", w, scaled)
            w = scaled
        return w[w > 1e-6]
