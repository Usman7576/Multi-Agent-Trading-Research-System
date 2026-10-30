"""Statistical honesty: bootstrap confidence intervals and the deflated Sharpe ratio."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

TRADING_DAYS = 252
EULER_GAMMA = 0.5772156649015329


def stationary_bootstrap_indices(n: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    """Politis-Romano stationary bootstrap: blocks of geometric length that wrap around the sample."""
    new_block = rng.random(n) < 1.0 / mean_block
    new_block[0] = True
    starts = rng.integers(n, size=n)
    t = np.arange(n)
    block_start = np.maximum.accumulate(np.where(new_block, t, 0))
    return (starts[block_start] + t - block_start) % n


def sharpe(r: np.ndarray) -> float:
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else float("nan")


def bootstrap_sharpe_ci(
    returns: pd.Series, n_boot: int = 2000, mean_block: float = 20.0, level: float = 0.95, seed: int = 7
) -> tuple[float, float]:
    """Confidence interval for the annualised Sharpe ratio that respects autocorrelation in returns."""
    r = returns.dropna().to_numpy()
    rng = np.random.default_rng(seed)
    stats = np.array(
        [sharpe(r[stationary_bootstrap_indices(len(r), mean_block, rng)]) for _ in range(n_boot)]
    )
    lo, hi = np.nanquantile(stats, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def bootstrap_sharpe_diff_pvalue(
    a: pd.Series, b: pd.Series, n_boot: int = 2000, mean_block: float = 20.0, seed: int = 7
) -> float:
    """One-sided p-value that strategy ``a`` has no higher Sharpe than ``b`` (paired block bootstrap)."""
    df = pd.concat([a, b], axis=1).dropna().to_numpy()
    observed = sharpe(df[:, 0]) - sharpe(df[:, 1])
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        s = df[stationary_bootstrap_indices(len(df), mean_block, rng)]
        diffs.append(sharpe(s[:, 0]) - sharpe(s[:, 1]))
    centred = np.array(diffs) - observed  # distribution under the null of no difference
    return float((centred >= observed).mean())


def probabilistic_sharpe_ratio(returns: pd.Series, benchmark_sr_annual: float = 0.0) -> float:
    """P(true Sharpe > benchmark), accounting for sample length, skewness and fat tails.

    Bailey & Lopez de Prado (2012), "The Sharpe Ratio Efficient Frontier".
    """
    r = returns.dropna().to_numpy()
    n = len(r)
    sr = r.mean() / r.std(ddof=1)  # per-period
    sr_star = benchmark_sr_annual / np.sqrt(TRADING_DAYS)
    g3, g4 = skew(r), kurtosis(r, fisher=False)
    denom = np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr**2)
    return float(norm.cdf((sr - sr_star) * np.sqrt(n - 1) / denom))


def deflated_sharpe_ratio(returns: pd.Series, trial_sharpes_annual: list[float]) -> float:
    """PSR against the Sharpe expected from the best of N unskilled trials (Bailey & Lopez de Prado, 2014).

    ``trial_sharpes_annual`` holds the annualised Sharpe of every configuration tried, this one included.
    """
    n_trials = len(trial_sharpes_annual)
    if n_trials < 2:
        return probabilistic_sharpe_ratio(returns, 0.0)
    var = float(np.var(np.asarray(trial_sharpes_annual) / np.sqrt(TRADING_DAYS), ddof=1))
    expected_max = np.sqrt(var) * (
        (1 - EULER_GAMMA) * norm.ppf(1 - 1 / n_trials) + EULER_GAMMA * norm.ppf(1 - 1 / (n_trials * np.e))
    )
    return probabilistic_sharpe_ratio(returns, expected_max * np.sqrt(TRADING_DAYS))
