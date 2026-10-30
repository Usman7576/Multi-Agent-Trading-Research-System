"""Training labels. Labels use future prices by definition, so they are kept apart from features and carry
the date on which they become known; the walk-forward trainer only uses labels already known."""

from __future__ import annotations

import pandas as pd


def forward_excess_returns(
    adj_close: pd.DataFrame, horizon: int, members: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Forward ``horizon``-day return minus the cross-sectional mean, plus its percentile rank.

    ``adj_close`` is a (date x ticker) panel. Returns a frame indexed by (date, ticker) with columns
    ``fwd_ret``, ``fwd_excess``, ``fwd_rank`` and ``label_date`` (the session whose close reveals it).
    """
    fwd = adj_close.shift(-horizon) / adj_close - 1
    if members is not None:  # excess and rank are measured against that day's universe only
        fwd = fwd.where(members.reindex_like(fwd).fillna(False).astype(bool))
    excess = fwd.sub(fwd.mean(axis=1), axis=0)
    rank = fwd.rank(axis=1, pct=True)
    dates = pd.Series(adj_close.index, index=adj_close.index)
    label_date = dates.shift(-horizon)
    out = pd.DataFrame(
        {
            "fwd_ret": fwd.stack(),
            "fwd_excess": excess.stack(),
            "fwd_rank": rank.stack(),
        }
    )
    out.index = out.index.set_names(["date", "ticker"])
    out["label_date"] = out.index.get_level_values("date").map(label_date)
    return out.dropna(subset=["fwd_ret", "label_date"])
