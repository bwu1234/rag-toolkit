# Retrieval matrix — edgar (isolated)

- Eval set: `data/eval/edgar_period_set.json` (55 samples)
- Collection: `rag_corpus__edgar`

| variant | hit | hit 95% CI | recall | prec | MRR | NDCG | Δ hit [95% CI] | Δ NDCG [95% CI] | unmatch. | s |
|---|---|---|---|---|---|---|---|---|---|---|
| `baseline` | 0.582 | [0.450, 0.703] | 0.582 | 0.116 | 0.331 | 0.393 | — | — | 0 | 55 |
| | | | | | | | | | | |
| `mode=dense` | 0.509 | [0.381, 0.636] | 0.509 | 0.102 | 0.307 | 0.357 | -0.073 [-0.173, +0.027] 2W/6L p=0.29 | -0.036 [-0.094, +0.022] | 0 | 56 |
| | | | | | | | | | | |
| `stage1_top_k=20` | 0.709 | [0.579, 0.812] | 0.709 | 0.036 | 0.348 | 0.440 | +0.127 [+0.038, +0.216]* 7W/0L p=0.016 | +0.047 [+0.017, +0.078]* | 0 | 52 |
| | | | | | | | | | | |
| `query_instruction=retrieval` | 0.545 | [0.415, 0.670] | 0.545 | 0.109 | 0.310 | 0.369 | -0.036 [-0.086, +0.014] 0W/2L p=0.5 | -0.024 [-0.062, +0.014] | 0 | 52 |
| `query_instruction=retrieval stage1_top_k=20` | 0.691 | [0.560, 0.797] | 0.691 | 0.035 | 0.331 | 0.423 | +0.109 [+0.012, +0.207]* 7W/1L p=0.07 | +0.030 [-0.018, +0.078] | 0 | 52 |
| `query_instruction=retrieval mode=dense` | 0.455 | [0.330, 0.585] | 0.455 | 0.091 | 0.291 | 0.333 | -0.127 [-0.230, -0.025]* 1W/8L p=0.039 | -0.061 [-0.135, +0.014] | 0 | 54 |

`hit 95% CI` is a Wilson interval on that rate alone, the noise floor of one run on this many questions. Δ is variant minus `baseline`, paired by sample. `*` marks a 95% interval that excludes zero; `W/L` counts the questions the variant gained / lost and `p` is McNemar's exact test on them -- trust it over the CI when W+L is small. `(no CI)` rows predate per-sample scores and need a re-run to be tested. `unmatch.` counts expected spans no chunk contains under that variant's chunking (`—` predates the count).
