# Retrieval matrix — edgar (isolated)

- Eval set: `data/eval/edgar_underspecified_set.json` (118 samples)
- Collection: `rag_corpus__edgar`

| variant | hit | recall | prec | MRR | NDCG | Δ hit [95% CI] | Δ NDCG [95% CI] | unmatch. | s |
|---|---|---|---|---|---|---|---|---|---|
| `baseline` | 0.652 | 0.652 | 0.141 | 0.493 | 0.562 | — | — | 0 | 118 |
| | | | | | | | | | |
| `mode=dense` | 0.619 | 0.619 | 0.134 | 0.469 | 0.535 | -0.034 [-0.108, +0.040] 8W/12L p=0.5 | -0.027 [-0.087, +0.034] | 0 | 118 |
| | | | | | | | | | |
| `stage1_top_k=20` | 0.686 | 0.686 | 0.038 | 0.497 | 0.578 | +0.034 [+0.001, +0.067]* 4W/0L p=0.12 | +0.016 [+0.003, +0.029]* | 0 | 117 |
| | | | | | | | | | |
| `query_instruction=retrieval` | 0.619 | 0.619 | 0.136 | 0.484 | 0.551 | -0.034 [-0.074, +0.006] 1W/5L p=0.22 | -0.011 [-0.041, +0.020] | 0 | 118 |
| `query_instruction=retrieval stage1_top_k=20` | 0.644 | 0.644 | 0.036 | 0.487 | 0.564 | -0.008 [-0.058, +0.042] 4W/5L p=1 | +0.002 [-0.031, +0.035] | 0 | 118 |
| `query_instruction=retrieval mode=dense` | 0.576 | 0.576 | 0.129 | 0.444 | 0.513 | -0.076 [-0.159, +0.006] 8W/17L p=0.11 | -0.049 [-0.113, +0.015] | 0 | 117 |

Δ is variant minus `baseline`, paired by sample. `*` marks a 95% interval that excludes zero; `W/L` counts the questions the variant gained / lost and `p` is McNemar's exact test on them -- trust it over the CI when W+L is small. `(no CI)` rows predate per-sample scores and need a re-run to be tested. `unmatch.` counts expected spans no chunk contains under that variant's chunking (`—` predates the count).
