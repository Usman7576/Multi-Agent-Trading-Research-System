"""Transaction cost model: commission + spread/slippage + square-root market impact."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tra.config import CostConfig


@dataclass(frozen=True)
class CostModel:
    commission_bps: float
    slippage_bps: float
    impact_k: float

    @classmethod
    def from_config(cls, cfg: CostConfig, multiplier: float = 1.0) -> CostModel:
        """``multiplier`` scales every component (cost stress tests at 2x and 3x)."""
        return cls(cfg.commission_bps * multiplier, cfg.slippage_bps * multiplier, cfg.impact_k * multiplier)

    def cost_rate(self, trade_value: float, adv: float, daily_vol: float) -> float:
        """Cost as a fraction of traded value.

        Impact follows the square-root law: k * sigma_daily * sqrt(Q / ADV). Missing ADV or volatility
        (e.g. a newly listed stock) falls back to commission + slippage only.
        """
        rate = (self.commission_bps + self.slippage_bps) / 1e4
        if adv > 0 and np.isfinite(adv) and np.isfinite(daily_vol):
            rate += self.impact_k * daily_vol * np.sqrt(abs(trade_value) / adv)
        return float(rate)
