"""LightGBM cross-sectional return model.

The target is the percentile rank of each stock's forward excess return on its date, centred at zero:
robust to outliers and directly aligned with how the signal is used (ranking stocks). Hyperparameters are
chosen inside the training window only, by mean daily IC across purged CV folds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from tra.eval.signal import daily_ic
from tra.models.cv import PurgedKFold

BASE_PARAMS: dict[str, Any] = {
    "objective": "regression",
    "subsample": 0.7,
    "subsample_freq": 1,
    "colsample_bytree": 0.7,
    "reg_lambda": 5.0,
    "verbose": -1,
    "n_jobs": 1,
    "deterministic": True,
    "force_row_wise": True,
}


@dataclass
class ReturnModel:
    param_grid: list[dict[str, Any]]
    horizon: int
    embargo: int
    cv_folds: int = 4
    seed: int = 7
    model: lgb.LGBMRegressor | None = None
    chosen_params: dict[str, Any] = field(default_factory=dict)
    cv_scores: list[float] = field(default_factory=list)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> ReturnModel:
        """``X`` and ``y`` are indexed by (date, ticker); ``y`` is the forward-return rank in [0, 1]."""
        target = y - 0.5
        grid = self.param_grid or [{}]
        if len(grid) > 1:
            self.cv_scores = [self._cv_score(X, target, p) for p in grid]
            self.chosen_params = grid[int(np.nanargmax(self.cv_scores))]
        else:
            self.chosen_params = grid[0]
        self.model = self._new(self.chosen_params).fit(X, target)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.model is None:
            raise RuntimeError("Model is not fitted")
        return pd.Series(self.model.predict(X), index=X.index, name="score")

    def _new(self, params: dict[str, Any]) -> lgb.LGBMRegressor:
        return lgb.LGBMRegressor(**{**BASE_PARAMS, **params, "random_state": self.seed})

    def _cv_score(self, X: pd.DataFrame, target: pd.Series, params: dict[str, Any]) -> float:
        dates = X.index.get_level_values("date")
        scores = []
        for train, test in PurgedKFold(self.cv_folds, self.horizon, self.embargo).split(dates):
            if len(train) == 0 or len(test) == 0:
                continue
            m = self._new(params).fit(X.iloc[train], target.iloc[train])
            pred = pd.Series(m.predict(X.iloc[test]), index=X.index[test])
            scores.append(daily_ic(pred, target.iloc[test]).mean())
        return float(np.nanmean(scores)) if scores else float("nan")

    def feature_importance(self) -> pd.Series:
        if self.model is None:
            raise RuntimeError("Model is not fitted")
        return pd.Series(
            self.model.booster_.feature_importance("gain"), index=self.model.booster_.feature_name()
        ).sort_values(ascending=False)
