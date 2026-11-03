"""Daily text features from scored headlines.

A headline counts on the first session whose close is at or after its ``known_at``, so an article published
after the close is first usable the next day. Sentiment decays with a 5-session half-life.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HALF_LIFE = 5
COUNT_WINDOW = 21
SCORE_COLUMNS = {"lm": "lm_polarity", "llm": "llm_sentiment"}


def text_features(signals: pd.DataFrame, closes: pd.Series, method: str) -> pd.DataFrame:
    """Features indexed by (date, ticker). ``closes`` maps each session date to its close time in UTC."""
    prefix = f"news_{method}"
    columns = [f"{prefix}_sentiment", "news_count_21"]
    score = SCORE_COLUMNS[method]
    usable = signals.dropna(subset=[score])
    if usable.empty:
        return pd.DataFrame(
            columns=columns, index=pd.MultiIndex.from_arrays([[], []], names=["date", "ticker"])
        )
    sessions = closes.sort_values()
    position = sessions.searchsorted(pd.to_datetime(usable["known_at"], utc=True).to_numpy(), side="left")
    inside = position < len(sessions)
    frame = pd.DataFrame(
        {
            "date": sessions.index[position[inside]],
            "ticker": usable["symbol"].to_numpy()[inside],
            "score": usable[score].to_numpy()[inside],
        }
    )
    daily = frame.groupby(["date", "ticker"])["score"].agg(["mean", "count"]).unstack("ticker")
    mean = daily["mean"].reindex(sessions.index)
    count = daily["count"].reindex(sessions.index).fillna(0.0)
    # Weighted by number of headlines through the decayed sum, and undefined until a company has any news
    weights = count.ewm(halflife=HALF_LIFE, adjust=True).mean()
    values = (mean.fillna(0.0) * count).ewm(halflife=HALF_LIFE, adjust=True).mean()
    sentiment = (values / weights.where(weights > 0)).where(count.cumsum() > 0)
    out = pd.DataFrame(
        {
            columns[0]: sentiment.stack(future_stack=True),
            columns[1]: np.log1p(count.rolling(COUNT_WINDOW, min_periods=1).sum()).stack(future_stack=True),
        }
    )
    out.index = out.index.set_names(["date", "ticker"])
    return out[out[columns[0]].notna()]
