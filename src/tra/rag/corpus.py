"""Build searchable chunks from PSX announcements, read reports and news.

Every chunk keeps ``known_at``, the moment its document became public, so retrieval can filter by date.
Reports with cached text are split into overlapping word windows; other announcements and headlines are
one chunk.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd

from tra.data.lake import ParquetLake

MARKET = "PSX"
WINDOW_WORDS = 220
OVERLAP_WORDS = 40
# Routine notices that would crowd search results without saying anything about the business
NOISE_TITLES = re.compile(r"disclosure of interest|transmission of|notice of|credit of|book closure", re.I)
COLUMNS = ["chunk_id", "doc_id", "symbol", "doc_type", "title", "text", "known_at", "url"]
_GLUED = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[a-z])(?=\d)|(?<=\d)(?=[a-z]{2})")


def unglue(text: str) -> str:
    """PDF text often loses spaces ('Sections96and131ofthe'). Restores those visible from case and digits."""
    return _GLUED.sub(" ", text)


def split_words(text: str, window: int = WINDOW_WORDS, overlap: int = OVERLAP_WORDS) -> list[str]:
    words = text.split()
    if len(words) <= window:
        return [" ".join(words)] if words else []
    step = window - overlap
    return [" ".join(words[i : i + window]) for i in range(0, max(len(words) - overlap, 1), step)]


def _chunk_id(doc_id: str, symbol: str, n: int) -> str:
    return hashlib.sha256(f"{doc_id}|{symbol}|{n}".encode()).hexdigest()[:16]


def announcement_chunks(announcements: pd.DataFrame, raw_dir: Path, since: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    recent = announcements[announcements["announced_local"] >= pd.Timestamp(since)]
    for a in recent.itertuples(index=False):
        if NOISE_TITLES.search(a.title):
            continue
        doc = str(a.announcement_id)
        cache = raw_dir / "reports" / a.symbol / f"{doc}.text.json"
        text = json.loads(cache.read_text(encoding="utf-8"))["text"] if cache.exists() else ""
        parts = split_words(unglue(text)) or [a.title]
        url = f"https://dps.psx.com.pk{a.doc_path}" if isinstance(a.doc_path, str) else ""
        for n, part in enumerate(parts):
            rows.append(
                {
                    "chunk_id": _chunk_id(doc, a.symbol, n),
                    "doc_id": doc,
                    "symbol": a.symbol,
                    "doc_type": "psx_report" if text else "psx_announcement",
                    "title": a.title,
                    "text": part,
                    "known_at": a.known_at,
                    "url": url,
                }
            )
    return rows


def news_chunks(news: pd.DataFrame, since: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    recent = news[news["known_at"] >= pd.Timestamp(since, tz="UTC")]
    for a in recent.itertuples(index=False):
        if not a.in_title and not a.text:
            continue
        for n, part in enumerate(split_words(f"{a.title}. {a.text}".strip(" .")) or [a.title]):
            rows.append(
                {
                    "chunk_id": _chunk_id(a.doc_id, a.symbol, n),
                    "doc_id": a.doc_id,
                    "symbol": a.symbol,
                    "doc_type": "news",
                    "title": a.title,
                    "text": part,
                    "known_at": a.known_at,
                    "url": a.url,
                }
            )
    return rows


def build_chunks(lake: ParquetLake, raw_dir: Path, since: str = "2018-01-01") -> pd.DataFrame:
    rows = announcement_chunks(lake.read("announcements", MARKET), raw_dir, since)
    if lake.exists("news", MARKET):
        rows += news_chunks(lake.read("news", MARKET), since)
    df = pd.DataFrame(rows, columns=COLUMNS).drop_duplicates("chunk_id")
    lake.replace("chunks", MARKET, df)
    return df
