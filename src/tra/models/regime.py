"""Market regime detection with a Gaussian HMM.

The model is fitted on index data up to a retrain date. Probabilities are *filtered*
P(state_t | obs_1..t) from the forward algorithm, not the smoothed P(state_t | all obs) that hmmlearn's
``predict_proba`` returns: smoothing uses future observations and would leak into a backtest.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from scipy.stats import multivariate_normal


def regime_observations(index_close: pd.Series) -> pd.DataFrame:
    """Daily log return and 21-day realised volatility of the index."""
    r = np.log(index_close).diff()
    return pd.DataFrame({"ret": r, "vol": r.rolling(21).std() * np.sqrt(252)}).dropna()


@dataclass
class RegimeModel:
    n_states: int = 2
    seed: int = 7
    hmm: GaussianHMM | None = None
    high_vol_state: int = 0
    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None

    def fit(self, obs: pd.DataFrame) -> RegimeModel:
        x = obs.to_numpy()
        self.mean_, self.std_ = x.mean(axis=0), x.std(axis=0)
        z = (x - self.mean_) / self.std_
        self.hmm = GaussianHMM(
            n_components=self.n_states, covariance_type="full", n_iter=200, random_state=self.seed
        ).fit(z)
        self.high_vol_state = int(np.argmax(self.hmm.means_[:, 1]))  # column 1 = volatility
        return self

    def filtered_probs(self, obs: pd.DataFrame) -> pd.DataFrame:
        """P(state_t | observations up to t) for every row of ``obs``."""
        if self.hmm is None or self.mean_ is None or self.std_ is None:
            raise RuntimeError("Model is not fitted")
        z = (obs.to_numpy() - self.mean_) / self.std_
        log_emission = np.column_stack(
            [
                multivariate_normal(self.hmm.means_[k], self.hmm.covars_[k]).logpdf(z)
                for k in range(self.n_states)
            ]
        )
        log_trans = np.log(self.hmm.transmat_)
        log_alpha = np.empty_like(log_emission)
        log_alpha[0] = np.log(self.hmm.startprob_) + log_emission[0]
        log_alpha[0] -= logsumexp(log_alpha[0])
        for t in range(1, len(z)):
            log_alpha[t] = logsumexp(log_alpha[t - 1][:, None] + log_trans, axis=0) + log_emission[t]
            log_alpha[t] -= logsumexp(log_alpha[t])
        return pd.DataFrame(
            np.exp(log_alpha), index=obs.index, columns=[f"p_state_{k}" for k in range(self.n_states)]
        )

    def p_high_vol(self, obs: pd.DataFrame) -> pd.Series:
        return self.filtered_probs(obs)[f"p_state_{self.high_vol_state}"].rename("p_high_vol")
