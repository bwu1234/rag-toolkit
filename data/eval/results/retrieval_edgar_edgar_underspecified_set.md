# Retrieval matrix — edgar (isolated)

- Eval set: `data/eval/edgar_underspecified_set.json` (118 samples)
- Collection: `rag_corpus__edgar`

| variant | hit | hit 95% CI | recall | prec | MRR | NDCG | Δ hit [95% CI] | Δ NDCG [95% CI] | unmatch. | s |
|---|---|---|---|---|---|---|---|---|---|---|
| `baseline` | 0.652 | [0.563, 0.732] | 0.652 | 0.141 | 0.493 | 0.562 | — | — | 0 | 118 |
| | | | | | | | | | | |
| `mode=dense` | 0.619 | [0.529, 0.701] | 0.619 | 0.134 | 0.469 | 0.535 | -0.034 [-0.108, +0.040] 8W/12L p=0.5 | -0.027 [-0.087, +0.034] | 0 | 119 |
| | | | | | | | | | | |
| `stage1_top_k=20` | 0.686 | [0.598, 0.763] | 0.686 | 0.038 | 0.497 | 0.578 | +0.034 [+0.001, +0.067]* 4W/0L p=0.12 | +0.016 [+0.003, +0.029]* | 0 | 114 |
| | | | | | | | | | | |
| `query_instruction=retrieval` | 0.619 | [0.529, 0.701] | 0.619 | 0.136 | 0.484 | 0.551 | -0.034 [-0.074, +0.006] 1W/5L p=0.22 | -0.011 [-0.041, +0.020] | 0 | 115 |
| `query_instruction=retrieval stage1_top_k=20` | 0.644 | [0.554, 0.725] | 0.644 | 0.036 | 0.487 | 0.564 | -0.008 [-0.058, +0.042] 4W/5L p=1 | +0.002 [-0.031, +0.035] | 0 | 115 |
| `query_instruction=retrieval mode=dense` | 0.576 | [0.486, 0.662] | 0.576 | 0.129 | 0.444 | 0.513 | -0.076 [-0.159, +0.006] 8W/17L p=0.11 | -0.049 [-0.113, +0.015] | 0 | 117 |

By kind, each reported on its own (the rows above mix them):

| variant | kind | n | hit | hit 95% CI | NDCG | Δ hit [95% CI] |
|---|---|---|---|---|---|---|
| `baseline` | implicit | 54 | 0.833 | [0.713, 0.910] | 0.741 | — |
| `baseline` | paraphrase | 64 | 0.500 | [0.381, 0.619] | 0.410 | — |
| `mode=dense` | implicit | 54 | 0.704 | [0.572, 0.809] | 0.640 | -0.130 [-0.234, -0.025]* 1W/8L p=0.039 |
| `mode=dense` | paraphrase | 64 | 0.547 | [0.426, 0.663] | 0.447 | +0.047 [-0.055, +0.149] 7W/4L p=0.55 |
| `stage1_top_k=20` | implicit | 54 | 0.870 | [0.756, 0.936] | 0.765 | +0.037 [-0.014, +0.088] 2W/0L p=0.5 |
| `stage1_top_k=20` | paraphrase | 64 | 0.531 | [0.411, 0.648] | 0.419 | +0.031 [-0.012, +0.074] 2W/0L p=0.5 |
| `query_instruction=retrieval` | implicit | 54 | 0.833 | [0.713, 0.910] | 0.754 | +0.000 [+0.000, +0.000] 0W/0L p=1 |
| `query_instruction=retrieval` | paraphrase | 64 | 0.438 | [0.323, 0.559] | 0.380 | -0.062 [-0.137, +0.012] 1W/5L p=0.22 |
| `query_instruction=retrieval stage1_top_k=20` | implicit | 54 | 0.870 | [0.756, 0.936] | 0.777 | +0.037 [-0.014, +0.088] 2W/0L p=0.5 |
| `query_instruction=retrieval stage1_top_k=20` | paraphrase | 64 | 0.453 | [0.337, 0.574] | 0.384 | -0.047 [-0.128, +0.034] 2W/5L p=0.45 |
| `query_instruction=retrieval mode=dense` | implicit | 54 | 0.667 | [0.534, 0.778] | 0.631 | -0.167 [-0.280, -0.054]* 1W/10L p=0.012 |
| `query_instruction=retrieval mode=dense` | paraphrase | 64 | 0.500 | [0.381, 0.619] | 0.413 | +0.000 [-0.115, +0.115] 7W/7L p=1 |

`hit 95% CI` is a Wilson interval on that rate alone, the noise floor of one run on this many questions. Δ is variant minus `baseline`, paired by sample. `*` marks a 95% interval that excludes zero; `W/L` counts the questions the variant gained / lost and `p` is McNemar's exact test on them -- trust it over the CI when W+L is small. `(no CI)` rows predate per-sample scores and need a re-run to be tested. `unmatch.` counts expected spans no chunk contains under that variant's chunking (`—` predates the count).
