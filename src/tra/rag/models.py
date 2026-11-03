"""Local embedding and reranking models (ONNX through fastembed, CPU only, no PyTorch)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"


class LocalEmbedder:
    size = 384

    def __init__(self, cache_dir: Path, threads: int | None = None) -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding(EMBEDDING_MODEL, cache_dir=str(cache_dir), threads=threads)

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.array(list(self._model.embed(texts, batch_size=32)))


class LocalReranker:
    def __init__(self, cache_dir: Path, threads: int | None = None) -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self._model = TextCrossEncoder(RERANK_MODEL, cache_dir=str(cache_dir), threads=threads)

    def score(self, query: str, texts: list[str]) -> list[float]:
        return [float(s) for s in self._model.rerank(query, texts, batch_size=16)]
