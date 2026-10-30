"""Performance metrics. Sharpe-type ratios use a zero risk-free rate unless one is passed in."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def max_drawdown(nav: pd.Series) -> tuple[float, int]:
    """Maximum drawdown (positive fraction) and the longest time under water, in sessions."""
    peak = nav.cummax()
    dd = 1 - nav / peak
    under = dd > 0
    longest = run = 0
    for flag in under:
        run = run + 1 if flag else 0
        longest = max(longest, run)
    return float(dd.max()), int(longest)


def performance_summary(
    nav: pd.Series,
    benchmark_nav: pd.Series,
    turnover: pd.Series | None = None,
    total_costs: float = 0.0,
    rf_daily: float = 0.0,
) -> dict[str, float]:
    r = nav.pct_change().dropna()
    b = benchmark_nav.pct_change().reindex(r.index).fillna(0.0)
    years = len(r) / TRADING_DAYS
    excess = r - rf_daily
    downside = excess[excess < 0].std()
    mdd, dd_len = max_drawdown(nav)
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan
    active = r - b
    beta = float(np.cov(r, b)[0, 1] / b.var()) if b.var() > 0 else np.nan
    q = r.quantile(0.05)
    out = {
        "total_return": float(nav.iloc[-1] / nav.iloc[0] - 1),
        "cagr": float(cagr),
        "ann_vol": float(r.std() * np.sqrt(TRADING_DAYS)),
        "sharpe": float(excess.mean() / r.std() * np.sqrt(TRADING_DAYS)) if r.std() > 0 else np.nan,
        "sortino": float(excess.mean() / downside * np.sqrt(TRADING_DAYS)) if downside > 0 else np.nan,
        "max_drawdown": mdd,
        "max_dd_sessions": dd_len,
        "calmar": float(cagr / mdd) if mdd > 0 else np.nan,
        "beta": beta,
        "ann_excess_vs_bench": float(active.mean() * TRADING_DAYS),
        "information_ratio": float(active.mean() / active.std() * np.sqrt(TRADING_DAYS))
        if active.std() > 0
        else np.nan,
        "var_95_daily": float(-q),
        "cvar_95_daily": float(-r[r <= q].mean()),
        "hit_rate_daily": float((r > 0).mean()),
        "n_sessions": int(len(r)),
    }
    if turnover is not None:
        out["ann_turnover"] = float(turnover.sum() / years) if years > 0 else np.nan
    out["cost_drag_ann"] = float(total_costs / nav.mean() / years) if years > 0 else np.nan
    return out
