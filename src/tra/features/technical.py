"""Price and volume features, computed per ticker from data up to each date (causal rolling windows only).

Input is a long prices frame (ticker, date, open, high, low, close, adj_close, volume). Returns use
``adj_close``; range-based measures use OHLC scaled by the same adjustment factor so dividends and splits
do not create jumps. Output is indexed by (date, ticker).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _per_ticker(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date").set_index("date")
    factor = g["adj_close"] / g["close"]
    o, h, lo, c = (g[col] * factor for col in ["open", "high", "low", "close"])
    logp = np.log(c)
    r1 = c.pct_change()
    dollar_vol = g["close"] * g["volume"]

    f = pd.DataFrame(index=g.index)
    for n in (5, 21, 63, 126, 252):
        f[f"ret_{n}"] = c / c.shift(n) - 1
    f["mom_12_1"] = c.shift(21) / c.shift(252) - 1

    f["vol_21"] = r1.rolling(21).std() * np.sqrt(TRADING_DAYS)
    f["vol_63"] = r1.rolling(63).std() * np.sqrt(TRADING_DAYS)
    f["park_vol_21"] = np.sqrt((np.log(h / lo) ** 2).rolling(21).mean() / (4 * np.log(2)) * TRADING_DAYS)
    f["vol_of_vol"] = f["vol_21"].rolling(63).std()
    f["zret_5"] = f["ret_5"] / (f["vol_21"] / np.sqrt(TRADING_DAYS) * np.sqrt(5))

    prev_c = c.shift(1)
    true_range = pd.concat([h - lo, (h - prev_c).abs(), (lo - prev_c).abs()], axis=1).max(axis=1)
    atr_14 = true_range.rolling(14).mean()
    ma_20, ma_50, ma_200 = c.rolling(20).mean(), c.rolling(50).mean(), c.rolling(200).mean()
    f["dist_ma20_atr"] = (c - ma_20) / atr_14
    f["ma_20_50"] = ma_20 / ma_50 - 1
    f["ma_50_200"] = ma_50 / ma_200 - 1
    f["rsi_14"] = _rsi(c, 14)
    f["trend_slope_63"] = _rolling_slope(logp, 63) * TRADING_DAYS
    f["gap_open"] = o / prev_c - 1

    log_vol = np.log(g["volume"].replace(0, np.nan))
    f["volume_z_20"] = (log_vol - log_vol.rolling(20).mean()) / log_vol.rolling(20).std()
    f["amihud_21"] = (r1.abs() / dollar_vol).rolling(21).mean() * 1e9
    f["adv_20"] = dollar_vol.rolling(20).mean()
    f["log_adv_20"] = np.log(f["adv_20"])
    f["vol_21_daily"] = r1.rolling(21).std()
    return f


def _rsi(c: pd.Series, n: int) -> pd.Series:
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + gain / loss)


def _rolling_slope(y: pd.Series, n: int) -> pd.Series:
    """OLS slope of y on time over a trailing window of n observations."""
    x = np.arange(n, dtype=float)
    x_c = x - x.mean()
    denom = (x_c**2).sum()
    return y.rolling(n).apply(lambda w: float(np.dot(x_c, w - w.mean()) / denom), raw=True)


def technical_features(prices: pd.DataFrame) -> pd.DataFrame:
    frames = {t: _per_ticker(g) for t, g in prices.groupby("ticker", sort=True)}
    out = pd.concat(frames, names=["ticker", "date"]).swaplevel().sort_index()
    return out.replace([np.inf, -np.inf], np.nan)


# Features used by models (adv_20 and vol_21_daily are kept for sizing and risk, not as signals)
SIGNAL_COLUMNS = [
    "ret_5",
    "ret_21",
    "ret_63",
    "ret_126",
    "ret_252",
    "mom_12_1",
    "vol_21",
    "vol_63",
    "park_vol_21",
    "vol_of_vol",
    "zret_5",
    "dist_ma20_atr",
    "ma_20_50",
    "ma_50_200",
    "rsi_14",
    "trend_slope_63",
    "gap_open",
    "volume_z_20",
    "amihud_21",
    "log_adv_20",
]
