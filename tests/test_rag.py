import hashlib

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tra.rag.corpus import announcement_chunks, news_chunks, split_words, unglue
from tra.rag.evaluate import ndcg_at_k, recall_at_k, reciprocal_rank, unique_docs
from tra.rag.retrieve import HybridRetriever, reciprocal_rank_fusion, tokenize


class HashEmbedder:
    """Deterministic bag-of-words vectors, so tests need no model files."""

    size = 64

    def embed(self, texts):
        out = np.zeros((len(texts), self.size))
        for i, text in enumerate(texts):
            for tok in tokenize(text):
                out[i, int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.size] += 1.0
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


class ReverseReranker:
    def score(self, query, texts):
        return [float(len(t)) for t in texts]


def _chunks(n=200):
    days = pd.date_range("2019-01-01", periods=n, freq="7D", tz="UTC")
    rows = []
    for i, day in enumerate(days):
        symbol = "LUCK" if i % 2 else "HBL"
        topic = "dividend announced" if i % 3 else "profit after tax rose"
        rows.append({
            "chunk_id": f"c{i}", "doc_id": f"d{i}", "symbol": symbol, "doc_type": "news" if i % 5 else "psx_report",
            "title": f"{symbol} {topic}", "text": f"Item {i} {topic} for {symbol}", "known_at": day, "url": f"u{i}",
        })  # fmt: skip
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def retriever():
    r = HybridRetriever(_chunks(), HashEmbedder(), ReverseReranker())
    yield r
    r.close()


MOMENTS = st.datetimes(
    min_value=pd.Timestamp("2018-06-01").to_pydatetime(), max_value=pd.Timestamp("2023-12-31").to_pydatetime()
).map(lambda d: pd.Timestamp(d, tz="UTC"))


@settings(max_examples=200, deadline=None)
@given(as_of=MOMENTS, mode=st.sampled_from(["bm25", "dense", "hybrid", "rerank"]))
def test_rag_no_future_chunks(retriever, as_of, mode):
    """The leakage test: whatever the moment or search mode, nothing published after as_of comes back."""
    for hit in retriever.search("dividend profit LUCK", as_of, k=10, mode=mode):
        assert hit.known_at <= as_of


def test_search_sees_a_chunk_only_from_its_publication_moment(retriever):
    chunks = retriever.chunks
    target = chunks.iloc[50]
    query = target["text"]
    at = pd.Timestamp(target["known_at"])
    assert target["chunk_id"] in {h.chunk_id for h in retriever.search(query, at, k=40, mode="bm25")}
    before = at - pd.Timedelta(seconds=1)
    for mode in ["bm25", "dense", "hybrid", "rerank"]:
        assert target["chunk_id"] not in {
            h.chunk_id for h in retriever.search(query, before, k=40, mode=mode)
        }


def test_naive_as_of_is_rejected(retriever):
    with pytest.raises(ValueError):
        retriever.search("profit", pd.Timestamp("2020-01-01"))


def test_before_all_documents_returns_nothing(retriever):
    assert retriever.search("profit", pd.Timestamp("2010-01-01", tz="UTC")) == []


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid", "rerank"])
def test_filters_by_company_and_type(retriever, mode):
    as_of = pd.Timestamp("2024-01-01", tz="UTC")
    hits = retriever.search("dividend", as_of, symbol="LUCK", doc_types=["news"], k=10, mode=mode)
    assert hits and {h.symbol for h in hits} == {"LUCK"} and {h.doc_type for h in hits} == {"news"}


def test_reranker_decides_final_order(retriever):
    hits = retriever.search("dividend", pd.Timestamp("2024-01-01", tz="UTC"), k=5, mode="rerank")
    assert [len(f"{h.title}. {h.text}") for h in hits] == sorted(
        (len(f"{h.title}. {h.text}") for h in hits), reverse=True
    )


def test_citation_names_type_company_date_and_title(retriever):
    hit = retriever.search("dividend", pd.Timestamp("2024-01-01", tz="UTC"), symbol="LUCK", k=1, mode="bm25")[
        0
    ]
    assert (
        hit.citation.startswith(f"[{hit.doc_type} LUCK {hit.known_at.date()}]") and hit.title in hit.citation
    )


def test_rrf_rewards_items_found_by_both():
    assert reciprocal_rank_fusion([[1, 2, 3], [3, 4, 5]])[0] == 3


def test_split_words_overlaps_and_keeps_short_text_whole():
    assert split_words("a b c", window=5, overlap=1) == ["a b c"]
    assert split_words("") == []
    parts = split_words(" ".join(str(i) for i in range(10)), window=4, overlap=1)
    assert parts[0] == "0 1 2 3" and parts[1].startswith("3 ") and parts[-1].endswith("9")


def test_corpus_skips_routine_notices_and_old_items(tmp_path):
    t = pd.Timestamp("2024-05-01 05:00", tz="UTC")
    ann = pd.DataFrame({
        "announcement_id": ["1", "2", "3"], "symbol": ["LUCK"] * 3, "doc_path": ["/d/1.pdf", None, None],
        "title": ["Material Information", "Disclosure of Interest by a Director", "Material Information"],
        "announced_local": pd.to_datetime(["2024-05-01", "2024-05-01", "2010-01-01"]), "known_at": [t, t, t],
    })  # fmt: skip
    rows = announcement_chunks(ann, tmp_path, "2018-01-01")
    assert (
        [r["doc_id"] for r in rows] == ["1"]
        and rows[0]["known_at"] == t
        and rows[0]["doc_type"] == "psx_announcement"
    )


def test_body_only_news_links_without_text_are_skipped():
    t = pd.Timestamp("2024-05-01", tz="UTC")
    news = pd.DataFrame({
        "doc_id": ["a", "b"], "symbol": ["LUCK", "LUCK"], "in_title": [True, False], "title": ["Lucky profit", "Market"],
        "text": ["", ""], "known_at": [t, t], "url": ["u", "v"],
    })  # fmt: skip
    assert [r["doc_id"] for r in news_chunks(news, "2018-01-01")] == ["a"]


def test_retrieval_metrics_by_hand():
    ranked = unique_docs(["a", "a", "b", "c", "d"])
    assert ranked == ["a", "b", "c", "d"]
    gold = {"b", "z"}
    assert recall_at_k(ranked, gold, 2) == 0.5 and recall_at_k(ranked, gold, 1) == 0.0
    assert reciprocal_rank(ranked, gold) == 0.5 and reciprocal_rank(ranked, {"z"}) == 0.0
    assert ndcg_at_k(["b", "x"], {"b"}, 10) == 1.0
    assert 0 < ndcg_at_k(["x", "b"], {"b"}, 10) < 1.0


def test_unglue_restores_spaces_lost_in_pdfs():
    assert unglue("Sections96and131 ofTheAct") == "Sections 96 and 131 of The Act"
    assert unglue("Rs.5 Q3 2026 30th PSX") == "Rs.5 Q3 2026 30th PSX"
