"""Purged and embargoed time-series cross-validation (Lopez de Prado, Advances in Financial ML, ch. 7).

Labels span ``horizon`` days, so a training row whose label window overlaps the test fold leaks test
information. Such rows are purged, and rows right after the test fold are embargoed as well, because
serial correlation carries test-period information into them.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd


class PurgedKFold:
    def __init__(self, n_splits: int, horizon: int, embargo: int) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        self.n_splits = n_splits
        self.horizon = horizon
        self.embargo = embargo

    def split(self, dates: pd.Series | pd.Index) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield (train_idx, test_idx) row positions. ``dates`` is the sample date of each row."""
        dates = pd.DatetimeIndex(dates)
        unique = np.sort(dates.unique())
        pos = np.searchsorted(unique, dates.to_numpy())  # trading-day index of each row
        for fold in np.array_split(np.arange(len(unique)), self.n_splits):
            lo, hi = fold[0], fold[-1]
            test = (pos >= lo) & (pos <= hi)
            # purge: label window [p, p + horizon] must end before the test fold starts
            before = pos + self.horizon < lo
            after = pos > hi + self.horizon + self.embargo
            yield np.flatnonzero(before | after), np.flatnonzero(test)
