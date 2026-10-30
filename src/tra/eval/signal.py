"""Signal evaluation: information coefficient and quantile spreads."""

from __future__ import annotations

import numpy as np
import pandas as pd


def daily_ic(pred: pd.Series, realised: pd.Series, min_names: int = 5) -> pd.Series:
    """Spearman correlation between prediction and realised return across stocks, per date."""
    df = pd.DataFrame({"p": pred, "r": realised}).dropna()
    out = {}
    for d, g in df.groupby(level="date"):
        if len(g) >= min_names and g["p"].nunique() > 1 and g["r"].nunique() > 1:
            out[d] = g["p"].rank().corr(g["r"].rank())
    return pd.Series(out, name="ic", dtype=float)


def ic_summary(ic: pd.Series, horizon: int = 1) -> dict[str, float]:
    """Mean IC and its t-stat. Overlapping labels make daily ICs autocorrelated, so the t-stat uses
    non-overlapping observations (every ``horizon``-th date)."""
    ic = ic.dropna()
    sub = ic.iloc[:: max(horizon, 1)]
    t = sub.mean() / (sub.std(ddof=1) / np.sqrt(len(sub))) if len(sub) > 2 else np.nan
    return {
        "mean_ic": float(ic.mean()),
        "ic_std": float(ic.std()),
        "ic_t_stat": float(t),
        "hit_rate": float((ic > 0).mean()),
        "n_dates": int(len(ic)),
    }


def quantile_returns(pred: pd.Series, realised: pd.Series, n_quantiles: int = 5) -> pd.Series:
    """Mean realised return per prediction quantile (1 = lowest prediction)."""
    df = pd.DataFrame({"p": pred, "r": realised}).dropna()
    df["q"] = df.groupby(level="date")["p"].transform(
        lambda s: pd.qcut(s.rank(method="first"), n_quantiles, labels=False) + 1
    )
    return df.groupby("q")["r"].mean()
