"""Portfolio construction: turn alpha scores into target weights.

Methods: hold the benchmark, equal weight, inverse volatility, or a mean-variance optimiser (cvxpy) with
a turnover penalty, position and sector caps. Optionally the result is scaled so its ex-ante volatility
does not exceed a target (never levered above 100% gross).
"""

from __future__ import annotations

import logging

import cvxpy as cp
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from tra.config import PortfolioConfig, RiskConfig

log = logging.getLogger(__name__)
TRADING_DAYS = 252
ASSUMED_IC = 0.03  # Grinold: expected excess return ~ IC * volatility * z-score


def shrunk_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    """Annualised Ledoit-Wolf covariance from daily returns (columns with gaps are dropped)."""
    clean = returns.dropna(axis=1, how="any")
    lw = LedoitWolf().fit(clean.to_numpy())
    return pd.DataFrame(lw.covariance_ * TRADING_DAYS, index=clean.columns, columns=clean.columns)


def portfolio_vol(w: pd.Series, cov: pd.DataFrame) -> float:
    w = w.reindex(cov.index).fillna(0.0)
    return float(np.sqrt(max(w @ cov @ w, 0.0)))


def construct(
    scores: pd.Series,
    returns: pd.DataFrame,
    w_prev: pd.Series,
    sectors: dict[str, str],
    cfg: PortfolioConfig,
    risk: RiskConfig,
    benchmark: str,
) -> pd.Series:
    """Target weights (fractions of NAV, summing to <= 1)."""
    if cfg.method == "hold_benchmark":
        return pd.Series({benchmark: 1.0})
    scores = scores.dropna()
    candidates = scores.sort_values(ascending=False)
    if cfg.top_n is not None:
        candidates = candidates.head(cfg.top_n)
    if candidates.empty:
        return pd.Series(dtype=float)

    window = returns[candidates.index].iloc[-cfg.cov_window :]
    if cfg.method == "equal":
        w = pd.Series(1.0 / len(candidates), index=candidates.index)
    elif cfg.method == "inverse_vol":
        inv = 1.0 / window.std().replace(0, np.nan)
        w = (inv / inv.sum()).fillna(0.0)
    else:
        w = _mean_variance(candidates, window, w_prev, sectors, cfg, risk)

    if cfg.target_vol is not None and len(window) >= 20:
        vol = portfolio_vol(w, shrunk_covariance(window))
        if vol > cfg.target_vol:
            w = w * cfg.target_vol / vol
    return w[w > 1e-6]


def _mean_variance(
    candidates: pd.Series,
    window: pd.DataFrame,
    w_prev: pd.Series,
    sectors: dict[str, str],
    cfg: PortfolioConfig,
    risk: RiskConfig,
) -> pd.Series:
    cov = shrunk_covariance(window)
    names = list(cov.index)
    if not names:
        return pd.Series(dtype=float)
    s = candidates[names]
    z = (s - s.mean()) / s.std() if len(s) > 1 and s.std() > 0 else s * 0.0
    vol = np.sqrt(np.diag(cov.to_numpy()))
    alpha = ASSUMED_IC * vol * z.to_numpy()
    prev = w_prev.reindex(names).fillna(0.0).to_numpy()

    w = cp.Variable(len(names))
    sector_names = sorted({sectors.get(n, "Unknown") for n in names})
    sector_matrix = np.array(
        [[sectors.get(n, "Unknown") == sec for n in names] for sec in sector_names], float
    )
    objective = cp.Maximize(
        alpha @ w
        - cfg.risk_aversion * cp.quad_form(w, cp.psd_wrap(cov.to_numpy()))
        - cfg.turnover_penalty * cp.norm1(w - prev)
    )
    constraints = [cp.sum(w) <= 1, w >= 0, w <= risk.max_pos, sector_matrix @ w <= risk.max_sector]
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=cp.CLARABEL)
    except cp.SolverError:
        problem.status = "solver_error"
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or w.value is None:
        log.warning("Optimiser status %s; falling back to inverse volatility", problem.status)
        inv = pd.Series(1.0 / vol, index=names)
        return (inv / inv.sum()).clip(upper=risk.max_pos)
    return pd.Series(np.clip(w.value, 0.0, None), index=names)
