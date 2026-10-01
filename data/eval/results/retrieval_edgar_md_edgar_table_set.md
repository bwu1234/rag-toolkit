# Retrieval matrix — edgar_md (isolated)

- Eval set: `data/eval/edgar_table_set.json` (95 samples)
- Collection: `rag_corpus__edgar_md`

| variant | hit | hit 95% CI | recall | prec | MRR | NDCG | Δ hit [95% CI] | Δ NDCG [95% CI] | unmatch. | s |
|---|---|---|---|---|---|---|---|---|---|---|
| `baseline` | 0.863 | [0.780, 0.918] | 0.863 | 0.244 | 0.713 | 0.751 | — | — | 0 | 440 |
| | | | | | | | | | | |
| `mode=dense` | 0.832 | [0.744, 0.894] | 0.832 | 0.234 | 0.687 | 0.724 | -0.032 [-0.086, +0.023] 2W/5L p=0.45 | -0.027 [-0.072, +0.017] | 0 | 397 |
| | | | | | | | | | | |
| `stage1_top_k=20` | 0.937 | [0.869, 0.971] | 0.937 | 0.067 | 0.724 | 0.776 | +0.074 [+0.021, +0.126]* 7W/0L p=0.016 | +0.024 [+0.007, +0.042]* | 0 | 424 |
| | | | | | | | | | | |
| `chunker=structured` | 0.979 | [0.926, 0.994] | 0.979 | 0.253 | 0.818 | 0.859 | +0.116 [+0.045, +0.187]* 12W/1L p=0.0034 | +0.107 [+0.031, +0.183]* | 0 | 350 |
| `chunker=structured stage1_top_k=20` | 1.000 | [0.961, 1.000] | 1.000 | 0.065 | 0.821 | 0.866 | +0.137 [+0.067, +0.206]* 13W/0L p=0.00024 | +0.114 [+0.039, +0.190]* | 0 | 222 |
| `chunker=structured mode=dense` | 0.916 | [0.843, 0.957] | 0.916 | 0.240 | 0.758 | 0.798 | +0.053 [-0.032, +0.137] 11W/6L p=0.33 | +0.046 [-0.041, +0.134] | 0 | 218 |
| `chunker=structured prose_overlap` | 0.968 | [0.911, 0.989] | 0.968 | 0.248 | 0.799 | 0.842 | +0.105 [+0.031, +0.180]* 12W/2L p=0.013 | +0.090 [+0.014, +0.167]* | 0 | 217 |

`hit 95% CI` is a Wilson interval on that rate alone, the noise floor of one run on this many questions. Δ is variant minus `baseline`, paired by sample. `*` marks a 95% interval that excludes zero; `W/L` counts the questions the variant gained / lost and `p` is McNemar's exact test on them -- trust it over the CI when W+L is small. `(no CI)` rows predate per-sample scores and need a re-run to be tested. `unmatch.` counts expected spans no chunk contains under that variant's chunking (`—` predates the count).
