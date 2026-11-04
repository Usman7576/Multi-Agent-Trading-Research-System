# PSX retrieval evaluation

60 questions over 20,211 chunks ({'psx_announcement': 18126, 'psx_report': 1944, 'news': 141}). Scored per document, not per chunk, and each question is asked as of a date, so later documents cannot be returned.

| mode | Recall@5 | Recall@10 | MRR | nDCG@10 | s/query |
|---|---|---|---|---|---|
| bm25 | 0.706 | 0.814 | 0.653 | 0.689 | 0.05 |
| dense | 0.753 | 0.839 | 0.573 | 0.635 | 0.62 |
| hybrid | 0.761 | 0.847 | 0.652 | 0.691 | 0.69 |
| rerank | 0.869 | 0.897 | 0.798 | 0.825 | 5.93 |

Recall@5 by question type:

| type | bm25 | dense | hybrid | rerank |
|---|---|---|---|---|
| company | 0.825 | 0.850 | 0.850 | 0.925 |
| open | 0.463 | 0.565 | 0.593 | 0.731 |
| time | 0.500 | 0.500 | 0.500 | 1.000 |

**Limits.** Claude wrote and labelled these questions after reading the documents; no independent person checked them. A question written from a document tends to share its words, favouring BM25. Gold labels list the documents found to answer each question, not all that might, so Recall can understate. Choosing the best mode on this set makes its number optimistic.
