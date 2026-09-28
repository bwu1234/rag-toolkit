# Retrieval matrix — edgar (isolated)

- Eval set: `data/eval/edgar_period_set.json` (55 samples)
- Collection: `rag_corpus__edgar`

| variant | hit | hit 95% CI | recall | prec | MRR | NDCG | Δ hit [95% CI] | Δ NDCG [95% CI] | unmatch. | s |
|---|---|---|---|---|---|---|---|---|---|---|
| `baseline` | 0.582 | [0.450, 0.703] | 0.582 | 0.116 | 0.331 | 0.393 | — | — | 0 | 55 |

`hit 95% CI` is a Wilson interval on that rate alone, the noise floor of one run on this many questions. Δ is variant minus `baseline`, paired by sample. `*` marks a 95% interval that excludes zero; `W/L` counts the questions the variant gained / lost and `p` is McNemar's exact test on them -- trust it over the CI when W+L is small. `(no CI)` rows predate per-sample scores and need a re-run to be tested. `unmatch.` counts expected spans no chunk contains under that variant's chunking (`—` predates the count).
