# Retrieval matrix — edgar (isolated)

- Eval set: `data/eval/edgar_eval_set.json` (174 samples)
- Collection: `rag_corpus__edgar`

| variant | hit | recall | prec | MRR | NDCG | Δ hit [95% CI] | Δ NDCG [95% CI] | unmatch. | s |
|---|---|---|---|---|---|---|---|---|---|
| `baseline` | 0.914 | 0.914 | 0.200 | 0.732 | 0.824 | — | — | 0 | 181 |
| | | | | | | | | | |
| `mode=dense` | 0.799 | 0.799 | 0.177 | 0.654 | 0.735 | -0.115 [-0.170, -0.060]* 3W/23L p=8.8e-05 | -0.089 [-0.138, -0.039]* | 0 | 180 |
| `mode=hybrid` | 0.914 | 0.914 | 0.200 | 0.732 | 0.824 | +0.000 [+0.000, +0.000] 0W/0L p=1 | +0.000 [+0.000, +0.000] | 0 | 174 |

Δ is variant minus `baseline`, paired by sample. `*` marks a 95% interval that excludes zero; `W/L` counts the questions the variant gained / lost and `p` is McNemar's exact test on them -- trust it over the CI when W+L is small. `(no CI)` rows predate per-sample scores and need a re-run to be tested. `unmatch.` counts expected spans no chunk contains under that variant's chunking (`—` predates the count).
