"""Turn headlines and company filings into numbers: a word-list baseline and a schema-validated LLM score.

The LLM sees each text with the company name and all dates masked, so it is less able to recall what
later happened to that company. It must quote the words behind its score: a quote that is not an exact
substring of the masked text means the score is discarded, never guessed.

    uv run --all-extras python -m tra.llm.extract --limit 200
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from tra.config import Settings
from tra.data.lake import ParquetLake
from tra.llm.local import DEFAULT_MODEL, LLMError, LocalLLM
from tra.rag.corpus import unglue
from tra.scrapers.news.run import company_names
from tra.scrapers.news.sources import company_search_name

log = logging.getLogger(__name__)
MARKET = "PSX"
PROMPT_VERSION = "text-v2"
FILING_WORDS = 220
SYSTEM = (
    "You score one news headline or company filing from the Pakistan Stock Exchange for what it "
    "means for that company's share price. sentiment runs from -1 (clearly bad) through 0 (neutral or "
    "unclear) to 1 (clearly good). Routine notices and market round-ups are 0. key_quote must be words "
    "copied exactly from the text. Judge only the words given."
)
_DATE = re.compile(
    r"\b(?:19|20)\d\d\b|\b\d{1,2}(?:st|nd|rd|th)?\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b"
    r"|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}\b",
    re.I,
)


# Where an announcement starts, most reliable marker first
_BODY = [
    re.compile(r"following\s*(?:material\s*)?information\s*:?", re.I),
    re.compile(r"hereby\s+\w+", re.I),
    re.compile(r"dear\s+sir\S*\s*[,:]?", re.I),
]
MATERIAL = re.compile(r"material\s*information", re.I)


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
    return bool(quote) and _squash(quote) in _squash(masked)


def _squash(text: str) -> str:
    """Letters and digits only, since PDF text gains and loses spaces at random."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def filing_body(text: str) -> str:
    """The announcement itself, without the letterhead and addressees before it."""
    text = unglue(text)
    starts = [m for m in (pattern.search(text) for pattern in _BODY) if m]
    body = text[starts[0].end() :] if starts else text
    return " ".join(body.split()[:FILING_WORDS])


def filing_rows(lake: ParquetLake, raw_dir: Path) -> pd.DataFrame:
    """Material-information filings with cached text, in the same shape as headlines."""
    ann = lake.read("announcements", MARKET)
    ann = ann[ann["title"].str.contains(MATERIAL)]
    rows = []
    for a in ann.itertuples(index=False):
        cache = raw_dir / "reports" / a.symbol / f"{a.announcement_id}.text.json"
        if cache.exists():
            body = filing_body(json.loads(cache.read_text(encoding="utf-8"))["text"])
            if body:
                rows.append((str(a.announcement_id), a.symbol, a.known_at, body))
    return pd.DataFrame(rows, columns=["doc_id", "symbol", "known_at", "title"])


def lm_polarity(titles: list[str]) -> list[float]:
    """Loughran-McDonald word-list polarity: (positive - negative) / (positive + negative)."""
    import pysentiment2 as ps

    lm = ps.LM()
    return [float(lm.get_score(lm.tokenize(t))["Polarity"]) for t in titles]


def score_texts(
    lake: ParquetLake, raw_dir: Path, llm: LocalLLM | None, limit: int | None = None
) -> pd.DataFrame:
    """Score title-linked headlines and filings not yet scored. Resumable."""
    news = lake.read("news", MARKET)
    news = news[news["in_title"] & ~news["title"].str.contains("sponsored", case=False)]
    news = news.drop_duplicates(["doc_id", "symbol"])[["doc_id", "symbol", "known_at", "title"]]
    news = pd.concat([news.assign(source="news"), filing_rows(lake, raw_dir).assign(source="filing")])
    done = lake.read("news_signals", MARKET) if lake.exists("news_signals", MARKET) else pd.DataFrame()
    if done.empty:
        base = news.assign(lm_polarity=lm_polarity(news["title"].tolist()))
        lake.upsert("news_signals", MARKET, base)
        done = base
    else:
        new = news[~(news["doc_id"] + news["symbol"]).isin(done["doc_id"] + done["symbol"])]
        if not new.empty:
            fresh = new.assign(lm_polarity=lm_polarity(new["title"].tolist()))
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
            signal = llm.extract(HeadlineSignal, SYSTEM, f"Text: {masked}", PROMPT_VERSION)
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
            log.info("Scored %d of %d texts (%.1f s each)", n, len(todo), (time.monotonic() - started) / n)
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
    parser = argparse.ArgumentParser(description="Score news headlines and filings")
    parser.add_argument("--no-llm", action="store_true", help="Word-list scores only")
    parser.add_argument("--limit", type=int, default=None, help="Score at most this many texts with the LLM")
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
    score_texts(lake, settings.data_dir / "raw" / "psx", llm, args.limit)


if __name__ == "__main__":
    main()
