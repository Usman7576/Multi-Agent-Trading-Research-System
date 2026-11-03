"""Retrieval metrics at document level: several chunks of one PDF count as one result."""

from __future__ import annotations

import math


def unique_docs(doc_ids: list[str]) -> list[str]:
    return list(dict.fromkeys(doc_ids))


def recall_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    return len(set(ranked[:k]) & gold) / len(gold)


def reciprocal_rank(ranked: list[str], gold: set[str]) -> float:
    return next((1.0 / (i + 1) for i, d in enumerate(ranked) if d in gold), 0.0)


def ndcg_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(ranked[:k]) if d in gold)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return dcg / ideal
