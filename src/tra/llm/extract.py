"""Turn headlines into numbers: a word-list baseline and a schema-validated LLM score.

The LLM sees each headline with the company name and all dates masked, so it is less able to recall what
later happened to that company. It must quote the words behind its score: a quote that is not an exact
substring of the masked headline means the score is discarded, never guessed.

    uv run --all-extras python -m tra.llm.extract --limit 200
"""

from __future__ import annotations

import argparse
import logging
import re
import time
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from tra.config import Settings
from tra.data.lake import ParquetLake
from tra.llm.local import DEFAULT_MODEL, LLMError, LocalLLM
from tra.scrapers.news.run import company_names
from tra.scrapers.news.sources import company_search_name

log = logging.getLogger(__name__)
MARKET = "PSX"
PROMPT_VERSION = "headline-v1"
SYSTEM = (
    "You score one business news headline about a company listed on the Pakistan Stock Exchange for what it "
    "means for that company's share price. sentiment runs from -1 (clearly bad) through 0 (neutral or "
    "unclear) to 1 (clearly good). Routine notices and market round-ups are 0. key_quote must be words "
    "copied exactly from the headline. Judge only the words given."
)
_DATE = re.compile(
    r"\b(?:19|20)\d\d\b|\b\d{1,2}(?:st|nd|rd|th)?\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b"
    r"|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}\b",
    re.I,
)


class HeadlineSignal(BaseModel):
    sentiment: float = Field(ge=-1, le=1)
    event: Literal["earnings", "dividend", "expansion", "legal", "management", "market", "other"]
    key_quote: str


def mask(title: str, symbol: str, name: str) -> str:
    """Hide what identifies the company and the time, so the score rests on the news itself."""
    text = title
    for form in sorted({name, company_search_name(name)}, key=len, reverse=True):
        text = re.sub(re.escape(form), "the company", text, flags=re.I)
    text = re.sub(rf"(?<![A-Za-z]){re.escape(symbol)}(?![A-Za-z])", "the company", text)
    return " ".join(_DATE.sub("[date]", text).split())


def quote_verified(signal: HeadlineSignal, masked: str) -> bool:
    quote = signal.key_quote.strip()
    return bool(quote) and quote.lower() in masked.lower()


def lm_polarity(titles: list[str]) -> list[float]:
    """Loughran-McDonald word-list polarity: (positive - negative) / (positive + negative)."""
    import pysentiment2 as ps

    lm = ps.LM()
    return [float(lm.get_score(lm.tokenize(t))["Polarity"]) for t in titles]


def score_headlines(lake: ParquetLake, llm: LocalLLM | None, limit: int | None = None) -> pd.DataFrame:
    """Score title-linked headlines not yet scored by the requested methods. Resumable."""
    news = lake.read("news", MARKET)
    news = news[news["in_title"]].drop_duplicates(["doc_id", "symbol"])
    done = lake.read("news_signals", MARKET) if lake.exists("news_signals", MARKET) else pd.DataFrame()
    if done.empty:
        base = news[["doc_id", "symbol", "known_at", "title"]].assign(
            lm_polarity=lm_polarity(news["title"].tolist())
        )
        lake.upsert("news_signals", MARKET, base)
        done = base
    else:
        new = news[~(news["doc_id"] + news["symbol"]).isin(done["doc_id"] + done["symbol"])]
        if not new.empty:
            fresh = new[["doc_id", "symbol", "known_at", "title"]].assign(
                lm_polarity=lm_polarity(new["title"].tolist())
            )
            lake.upsert("news_signals", MARKET, fresh)
            done = lake.read("news_signals", MARKET)
    if llm is None:
        return done
    names = company_names(lake)
    scored = done["llm_sentiment"].notna() if "llm_sentiment" in done else pd.Series(False, index=done.index)
    tried = done["llm_status"].notna() if "llm_status" in done else pd.Series(False, index=done.index)
    todo = done[~scored & ~tried].sort_values("known_at", ascending=False)
    todo = todo.head(limit) if limit else todo
    rows, started = [], time.monotonic()
    for n, a in enumerate(todo.itertuples(index=False), start=1):
        masked = mask(a.title, a.symbol, names.get(a.symbol, a.symbol))
        rec = {"doc_id": a.doc_id, "symbol": a.symbol, "llm_sentiment": None, "llm_event": None}
        try:
            signal = llm.extract(HeadlineSignal, SYSTEM, f"Headline: {masked}", PROMPT_VERSION)
            if quote_verified(signal, masked):
                rec.update(llm_sentiment=signal.sentiment, llm_event=signal.event, llm_status="ok")
            else:
                rec["llm_status"] = "quote_failed"
        except LLMError:
            rec["llm_status"] = "invalid"
        rows.append(rec)
        if n % 100 == 0 or n == len(todo):
            _merge(lake, rows)
            rows = []
            log.info(
                "Scored %d of %d headlines (%.1f s each)", n, len(todo), (time.monotonic() - started) / n
            )
    return lake.read("news_signals", MARKET)


def _merge(lake: ParquetLake, rows: list[dict[str, object]]) -> None:
    table = lake.read("news_signals", MARKET).set_index(["doc_id", "symbol"])
    update = pd.DataFrame(rows).set_index(["doc_id", "symbol"])
    for col in update.columns:
        if col not in table:
            table[col] = pd.Series(index=table.index, dtype="float64" if col == "llm_sentiment" else "object")
        table.loc[update.index, col] = update[col]
    lake.replace("news_signals", MARKET, table.reset_index())


def main() -> None:
    parser = argparse.ArgumentParser(description="Score news headlines")
    parser.add_argument("--no-llm", action="store_true", help="Word-list scores only")
    parser.add_argument(
        "--limit", type=int, default=None, help="Score at most this many headlines with the LLM"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = Settings()
    lake = ParquetLake(settings.data_dir)
    llm = None
    if not args.no_llm:
        llm = LocalLLM(
            settings.data_dir / "models" / DEFAULT_MODEL,
            settings.data_dir / "cache" / "llm.sqlite",
            n_ctx=2048,
        )
    score_headlines(lake, llm, args.limit)


if __name__ == "__main__":
    main()
