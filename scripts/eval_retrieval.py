"""Compare retrieval modes on the labelled question set and write the report.

uv run --all-extras python scripts/eval_retrieval.py
"""

from __future__ import annotations

import argparse
import json
import logging
import time

import pandas as pd
import yaml

from tra.config import REPO_ROOT, Settings
from tra.data.lake import ParquetLake
from tra.rag.evaluate import ndcg_at_k, recall_at_k, reciprocal_rank, unique_docs
from tra.rag.models import LocalEmbedder, LocalReranker
from tra.rag.retrieve import HybridRetriever

log = logging.getLogger("eval_retrieval")
MODES = ["bm25", "dense", "hybrid", "rerank"]
QUESTIONS = REPO_ROOT / "configs" / "retrieval_questions.yaml"
REPORT = REPO_ROOT / "reports" / "psx_retrieval_eval.md"
POOL = 40


def evaluate(
    retriever: HybridRetriever, questions: list[dict], mode: str
) -> tuple[dict[str, float], list[dict]]:
    rows, started = [], time.monotonic()
    for q in questions:
        as_of = pd.Timestamp(q["as_of"], tz="UTC")
        hits = retriever.search(q["question"], as_of, symbol=q.get("symbol"), k=POOL, mode=mode)
        ranked = unique_docs([h.doc_id for h in hits])
        gold = {str(d) for d in q["gold"]}
        rows.append(
            {
                "id": q["id"],
                "kind": q["kind"],
                "recall_5": recall_at_k(ranked, gold, 5),
                "recall_10": recall_at_k(ranked, gold, 10),
                "mrr": reciprocal_rank(ranked, gold),
                "ndcg_10": ndcg_at_k(ranked, gold, 10),
            }
        )
    frame = pd.DataFrame(rows)
    summary = frame[["recall_5", "recall_10", "mrr", "ndcg_10"]].mean().to_dict()
    summary["seconds_per_query"] = (time.monotonic() - started) / len(questions)
    return summary, rows


def kind_table(kinds: pd.DataFrame) -> list[str]:
    out = ["| type | " + " | ".join(kinds.columns) + " |", "|---" * (len(kinds.columns) + 1) + "|"]
    return out + [f"| {k} | " + " | ".join(f"{v:.3f}" for v in r) + " |" for k, r in kinds.iterrows()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = Settings()
    chunks = ParquetLake(settings.data_dir).read("chunks", "PSX")
    questions = yaml.safe_load(QUESTIONS.read_text(encoding="utf-8"))["questions"]
    missing = {str(d) for q in questions for d in q["gold"]} - set(chunks["doc_id"])
    if missing:
        raise SystemExit(f"Gold documents not in the corpus: {sorted(missing)}")
    cache = settings.data_dir / "models" / "fastembed"
    started = time.monotonic()
    retriever = HybridRetriever(
        chunks, LocalEmbedder(cache), LocalReranker(cache), path=settings.data_dir / "vectors"
    )
    log.info("Index ready in %.0f s for %d chunks", time.monotonic() - started, len(chunks))
    summaries, details = {}, {}
    for mode in MODES:
        summaries[mode], details[mode] = evaluate(retriever, questions, mode)
        log.info("%s: %s", mode, summaries[mode])
    retriever.close()
    table = pd.DataFrame(summaries).T
    kinds = pd.DataFrame(
        {m: pd.DataFrame(details[m]).groupby("kind")["recall_5"].mean() for m in MODES}
    ).round(3)
    counts = chunks["doc_type"].value_counts().to_dict()
    lines = [
        "# PSX retrieval evaluation",
        "",
        f"{len(questions)} questions over {len(chunks):,} chunks ({counts}). Scored per document, "
        "not per chunk, and each question is asked as of a date, so later documents cannot be returned.",
        "",
        "| mode | Recall@5 | Recall@10 | MRR | nDCG@10 | s/query |",
        "|---|---|---|---|---|---|",
    ]
    for mode, r in table.iterrows():
        lines.append(
            f"| {mode} | {r['recall_5']:.3f} | {r['recall_10']:.3f} | {r['mrr']:.3f} | {r['ndcg_10']:.3f} "
            f"| {r['seconds_per_query']:.2f} |"
        )
    lines += ["", "Recall@5 by question type:", "", *kind_table(kinds), ""]
    lines += [
        "**Limits.** Claude wrote and labelled these questions after reading the documents; no independent "
        "person checked them. A question written from a document tends to share its words, favouring "
        "BM25. Gold labels list the documents found to answer each question, not all that might, so "
        "Recall can understate. Choosing the best mode on this set makes its number optimistic.",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    REPORT.with_suffix(".json").write_text(json.dumps({"summary": summaries, "detail": details}, indent=1))


if __name__ == "__main__":
    main()
