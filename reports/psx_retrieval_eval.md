# PSX retrieval evaluation

60 questions over 21,227 chunks ({'psx_announcement': 17028, 'psx_report': 4026, 'news': 173}). Scored per document, not per chunk, and each question is asked as of a date, so later documents cannot be returned.

| mode | Recall@5 | Recall@10 | MRR | nDCG@10 | s/query |
|---|---|---|---|---|---|
| bm25 | 0.750 | 0.835 | 0.787 | 0.761 | 0.12 |
| dense | 0.687 | 0.794 | 0.658 | 0.657 | 1.36 |
| hybrid | 0.778 | 0.838 | 0.780 | 0.760 | 1.39 |
| rerank | 0.876 | 0.933 | 0.893 | 0.883 | 9.72 |

Recall@5 by question type:

| type | bm25 | dense | hybrid | rerank |
|---|---|---|---|---|
| company | 0.831 | 0.762 | 0.869 | 0.912 |
| open | 0.560 | 0.505 | 0.551 | 0.782 |
| time | 0.833 | 0.833 | 1.000 | 1.000 |

**Limits.** Claude wrote and labelled these questions after reading the documents; no independent person checked them. A question written from a document tends to share its words, favouring BM25. Gold labels list the documents found to answer each question, not all that might, and a correct document nobody judged counts as wrong, so Recall can understate. Choosing the best mode on this set makes its number optimistic.
