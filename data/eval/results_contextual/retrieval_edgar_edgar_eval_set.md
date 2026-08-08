# Retrieval matrix — edgar (isolated)

- Eval set: `data/eval/edgar_eval_set.json` (174 samples)
- Collection: `rag_corpus__edgar`

| variant | hit | recall | prec | MRR | NDCG | Δ NDCG | s |
|---|---|---|---|---|---|---|---|
| `baseline` | 0.891 | 0.891 | 0.194 | 0.719 | 0.806 | — | 212 |
| | | | | | | | |
| `mode=dense` | 0.782 | 0.782 | 0.174 | 0.640 | 0.721 | -0.084 | 226 |
| `mode=hybrid` | 0.891 | 0.891 | 0.194 | 0.719 | 0.806 | +0.000 | 207 |
