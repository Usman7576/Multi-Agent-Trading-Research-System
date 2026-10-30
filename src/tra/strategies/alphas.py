"""Alpha models: score stocks on a date using only information known at that date's close."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

import pandas as pd

from tra.config import ModelConfig
from tra.features.pipeline import ResearchData
from tra.models.return_model import ReturnModel

log = logging.getLogger(__name__)


class AlphaModel(Protocol):
    def scores(self, day: pd.Timestamp, data: ResearchData) -> pd.Series:
        """Ticker -> score (higher is better). Tickers that cannot be scored are absent or NaN."""
        ...


def _day_slice(frame: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    try:
        return frame.xs(day, level="date")
    except KeyError:
        return frame.iloc[0:0].droplevel("date")


class BenchmarkAlpha:
    def scores(self, day: pd.Timestamp, data: ResearchData) -> pd.Series:
        return pd.Series(dtype=float)  # construction holds the benchmark directly


class EqualWeightAlpha:
    def scores(self, day: pd.Timestamp, data: ResearchData) -> pd.Series:
        row = _day_slice(data.raw, day)
        return pd.Series(1.0, index=row.index[row["ret_21"].notna()])


class MomentumAlpha:
    """Classic 12-1 month momentum (Jegadeesh-Titman): skip the latest month to avoid reversal."""

    def scores(self, day: pd.Timestamp, data: ResearchData) -> pd.Series:
        return _day_slice(data.raw, day)["mom_12_1"].dropna()


# factor -> direction; equal-weight average of percentile ranks (no fitting at all)
COMPOSITE_FACTORS: dict[str, int] = {
    "mom_12_1": +1,  # momentum
    "vol_63": -1,  # low volatility
    "ret_5": -1,  # short-term reversal
    "earnings_yield": +1,  # value
    "book_to_market": +1,  # value
    "roe": +1,  # quality
    "debt_to_equity": -1,  # quality / leverage
}


class FactorCompositeAlpha:
    def scores(self, day: pd.Timestamp, data: ResearchData) -> pd.Series:
        ranked = _day_slice(data.ranked, day)
        parts = [
            ranked[f] if sign > 0 else 1 - ranked[f] for f, sign in COMPOSITE_FACTORS.items() if f in ranked
        ]
        if not parts:
            return pd.Series(dtype=float)
        combo = pd.concat(parts, axis=1)
        enough = combo.notna().sum(axis=1) >= max(2, combo.shape[1] // 2)
        return combo.mean(axis=1)[enough]


@dataclass
class LightGBMAlpha:
    """Walk-forward LightGBM: retrains every ``retrain_every`` sessions on labels already known."""

    cfg: ModelConfig
    seed: int = 7
    model: ReturnModel | None = None
    last_train: pd.Timestamp | None = None
    train_log: list[dict[str, Any]] = field(default_factory=list)
    predictions: list[pd.Series] = field(default_factory=list)

    def feature_columns(self, data: ResearchData) -> list[str]:
        return [c for c in data.ranked.columns if c in data.signal_columns or c in data.context.columns]

    def scores(self, day: pd.Timestamp, data: ResearchData) -> pd.Series:
        if self.model is None or self._sessions_since_train(day, data) >= self.cfg.retrain_every:
            self._train(day, data)
        if self.model is None:
            return pd.Series(dtype=float)
        X = _day_slice(data.ranked, day)[self.feature_columns(data)]
        if X.empty:
            return pd.Series(dtype=float)
        pred = self.model.predict(X)
        self.predictions.append(pred.rename(day))
        return pred

    def _sessions_since_train(self, day: pd.Timestamp, data: ResearchData) -> int:
        """Trading sessions elapsed since the last retrain (scores() is only called on rebalance days)."""
        sessions = data.context.index
        return int(sessions.searchsorted(day) - sessions.searchsorted(self.last_train))

    def _train(self, day: pd.Timestamp, data: ResearchData) -> None:
        labels = data.labels[data.labels["label_date"] <= day]  # only labels revealed by today's close
        cols = self.feature_columns(data)
        X = data.ranked[cols].join(labels[["fwd_rank"]], how="inner")
        X = X[X.index.get_level_values("date") <= day]
        n_days = X.index.get_level_values("date").nunique()
        if n_days < self.cfg.min_train_days:
            return
        y = X.pop("fwd_rank")
        model = ReturnModel(
            self.cfg.param_grid, self.cfg.horizon, self.cfg.embargo_days, self.cfg.cv_folds, self.seed
        )
        self.model = model.fit(X, y)
        self.last_train = day
        self.train_log.append(
            {
                "date": day,
                "rows": len(X),
                "train_days": n_days,
                "params": model.chosen_params,
                "cv_ic": [round(s, 4) for s in model.cv_scores],
                "top_features": model.feature_importance().head(5).to_dict(),
            }
        )
        log.info("Retrained on %s: %d rows, CV IC %s", day.date(), len(X), model.cv_scores)

    def prediction_panel(self) -> pd.Series:
        if not self.predictions:
            return pd.Series(dtype=float)
        return pd.concat({p.name: p for p in self.predictions}, names=["date", "ticker"]).astype(float)


def make_alpha(cfg: ModelConfig, seed: int) -> AlphaModel:
    kinds: dict[str, Any] = {
        "benchmark": BenchmarkAlpha,
        "equal_weight": EqualWeightAlpha,
        "momentum": MomentumAlpha,
        "factor_composite": FactorCompositeAlpha,
    }
    if cfg.type == "lightgbm":
        return LightGBMAlpha(cfg, seed)
    return kinds[cfg.type]()
