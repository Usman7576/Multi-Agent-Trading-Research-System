"""Time-aware hybrid retrieval: BM25 + embeddings, fused by reciprocal rank, optionally reranked.

A query carries the simulated moment ``as_of``. Both search paths only see chunks with
``known_at <= as_of``, and the result is checked again before returning, so a question asked "on" a date
cannot retrieve anything published later.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi

COLLECTION = "chunks"
RRF_K = 60
CANDIDATES = 40
_TOKEN = re.compile(r"[a-z0-9]+")


class LookAheadLeak(RuntimeError):
    pass


class Embedder(Protocol):
    size: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


class Reranker(Protocol):
    def score(self, query: str, texts: list[str]) -> list[float]: ...


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = RRF_K) -> list[int]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda i: (-scores[i], i))


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    doc_id: str
    symbol: str
    doc_type: str
    title: str
    text: str
    known_at: pd.Timestamp
    url: str
    score: float

    @property
    def citation(self) -> str:
        return f"[{self.doc_type} {self.symbol} {self.known_at.date()}] {self.title}"


class HybridRetriever:
    def __init__(
        self,
        chunks: pd.DataFrame,
        embedder: Embedder,
        reranker: Reranker | None = None,
        path: Path | None = None,
    ) -> None:
        self.chunks = chunks.reset_index(drop=True)
        self.embedder = embedder
        self.reranker = reranker
        self._known = pd.to_datetime(self.chunks["known_at"], utc=True)
        self._seconds = (self._known.astype("int64") // self._unit_divisor()).to_numpy()
        self._symbols = self.chunks["symbol"].to_numpy()
        self._types = self.chunks["doc_type"].to_numpy()
        self._bm25 = BM25Okapi(
            [tokenize(f"{t} {x}") for t, x in zip(self.chunks["title"], self.chunks["text"], strict=True)]
        )
        self._client = QdrantClient(path=str(path)) if path else QdrantClient(location=":memory:")
        self._ensure_index()

    def _unit_divisor(self) -> int:
        unit = np.datetime_data(self._known.dt.tz_localize(None).to_numpy().dtype)[0]
        return {"s": 1, "ms": 1_000, "us": 1_000_000, "ns": 1_000_000_000}[unit]

    def _ensure_index(self, batch: int = 256) -> None:
        if self._client.collection_exists(COLLECTION):
            if self._client.count(COLLECTION).count == len(self.chunks):
                return
            self._client.delete_collection(COLLECTION)
        self._client.create_collection(
            COLLECTION,
            vectors_config=models.VectorParams(size=self.embedder.size, distance=models.Distance.COSINE),
        )
        for start in range(0, len(self.chunks), batch):
            part = self.chunks.iloc[start : start + batch]
            vectors = self.embedder.embed((part["title"] + ". " + part["text"]).tolist())
            self._client.upsert(
                COLLECTION,
                points=[
                    models.PointStruct(
                        id=int(i),
                        vector=vectors[n].tolist(),
                        payload={
                            "known_at": int(self._seconds[i]),
                            "symbol": str(self._symbols[i]),
                            "doc_type": str(self._types[i]),
                        },
                    )
                    for n, i in enumerate(part.index)
                ],
            )

    def _mask(self, cutoff: int, symbol: str | None, doc_types: list[str] | None) -> np.ndarray:
        mask = self._seconds <= cutoff
        if symbol:
            mask &= self._symbols == symbol
        if doc_types:
            mask &= np.isin(self._types, doc_types)
        return mask

    def _bm25_ids(self, query: str, mask: np.ndarray, limit: int) -> list[int]:
        scores = self._bm25.get_scores(tokenize(query))
        scores = np.where(mask & (scores > 0), scores, -np.inf)
        top = np.argsort(-scores, kind="stable")[:limit]
        return [int(i) for i in top if np.isfinite(scores[i])]

    def _dense_ids(
        self, query: str, cutoff: int, symbol: str | None, doc_types: list[str] | None, limit: int
    ) -> list[int]:
        must: list[models.Condition] = [models.FieldCondition(key="known_at", range=models.Range(lte=cutoff))]
        if symbol:
            must.append(models.FieldCondition(key="symbol", match=models.MatchValue(value=symbol)))
        if doc_types:
            must.append(models.FieldCondition(key="doc_type", match=models.MatchAny(any=doc_types)))
        vector = self.embedder.embed([query])[0].tolist()
        found = self._client.query_points(
            COLLECTION, query=vector, query_filter=models.Filter(must=must), limit=limit
        )
        return [int(p.id) for p in found.points]

    def search(
        self,
        query: str,
        as_of: pd.Timestamp,
        symbol: str | None = None,
        doc_types: list[str] | None = None,
        k: int = 5,
        mode: str = "rerank",
    ) -> list[Hit]:
        """``mode`` is one of 'bm25', 'dense', 'hybrid' or 'rerank' (hybrid + cross-encoder)."""
        if as_of.tzinfo is None:
            raise ValueError("as_of must be timezone-aware")
        cutoff = int(as_of.tz_convert("UTC").timestamp())
        mask = self._mask(cutoff, symbol, doc_types)
        if not mask.any():
            return []
        bm25 = self._bm25_ids(query, mask, CANDIDATES) if mode != "dense" else []
        dense = self._dense_ids(query, cutoff, symbol, doc_types, CANDIDATES) if mode != "bm25" else []
        ids = bm25 if mode == "bm25" else dense if mode == "dense" else reciprocal_rank_fusion([bm25, dense])
        scores = {i: 1.0 / (rank + 1) for rank, i in enumerate(ids)}
        if mode == "rerank" and self.reranker is not None and ids:
            pool = ids[:CANDIDATES]
            texts = [f"{self.chunks.at[i, 'title']}. {self.chunks.at[i, 'text']}" for i in pool]
            scores = dict(zip(pool, self.reranker.score(query, texts), strict=True))
            ids = sorted(pool, key=lambda i: -scores[i])
        hits = []
        for i in ids[:k]:
            row = self.chunks.iloc[i]
            if self._seconds[i] > cutoff:
                raise LookAheadLeak(f"chunk {row['chunk_id']} is dated after {as_of}")
            hits.append(
                Hit(
                    str(row["chunk_id"]),
                    str(row["doc_id"]),
                    str(row["symbol"]),
                    str(row["doc_type"]),
                    str(row["title"]),
                    str(row["text"]),
                    self._known.iloc[i],
                    str(row["url"]),
                    float(scores[i]),
                )  # fmt: skip
            )
        return hits

    def close(self) -> None:
        self._client.close()
