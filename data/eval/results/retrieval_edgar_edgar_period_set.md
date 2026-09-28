# Retrieval matrix — edgar (isolated)

- Eval set: `data/eval/edgar_period_set.json` (55 samples)
- Collection: `rag_corpus__edgar`

| variant | hit | hit 95% CI | recall | prec | MRR | NDCG | Δ hit [95% CI] | Δ NDCG [95% CI] | unmatch. | s |
|---|---|---|---|---|---|---|---|---|---|---|
| `baseline` | 0.582 | [0.450, 0.703] | 0.582 | 0.116 | 0.331 | 0.393 | — | — | 0 | 55 |
| | | | | | | | | | | |
| `mode=dense` | 0.509 | [0.381, 0.636] | 0.509 | 0.102 | 0.307 | 0.357 | -0.073 [-0.173, +0.027] 2W/6L p=0.29 | -0.036 [-0.094, +0.022] | 0 | 52 |
| | | | | | | | | | | |
| `stage1_top_k=20` | 0.709 | [0.579, 0.812] | 0.709 | 0.036 | 0.348 | 0.440 | +0.127 [+0.038, +0.216]* 7W/0L p=0.016 | +0.047 [+0.017, +0.078]* | 0 | 51 |
| | | | | | | | | | | |
| `query_instruction=retrieval` | 0.545 | [0.415, 0.670] | 0.545 | 0.109 | 0.310 | 0.369 | -0.036 [-0.086, +0.014] 0W/2L p=0.5 | -0.024 [-0.062, +0.014] | 0 | 52 |
| `query_instruction=retrieval stage1_top_k=20` | 0.691 | [0.560, 0.797] | 0.691 | 0.035 | 0.331 | 0.423 | +0.109 [+0.012, +0.207]* 7W/1L p=0.07 | +0.030 [-0.018, +0.078] | 0 | 52 |
| `query_instruction=retrieval mode=dense` | 0.455 | [0.330, 0.585] | 0.455 | 0.091 | 0.291 | 0.333 | -0.127 [-0.230, -0.025]* 1W/8L p=0.039 | -0.061 [-0.135, +0.014] | 0 | 54 |
| | | | | | | | | | | |
| `embedder=4b` | 0.600 | [0.468, 0.719] | 0.600 | 0.120 | 0.356 | 0.417 | +0.018 [-0.017, +0.054] 1W/0L p=1 | +0.024 [-0.012, +0.061] | 0 | 59 |
| `embedder=4b stage1_top_k=20` | 0.709 | [0.579, 0.812] | 0.709 | 0.036 | 0.371 | 0.459 | +0.127 [+0.038, +0.216]* 7W/0L p=0.016 | +0.066 [+0.021, +0.111]* | 0 | 56 |
| `embedder=4b mode=dense` | 0.545 | [0.415, 0.670] | 0.545 | 0.109 | 0.356 | 0.403 | -0.036 [-0.124, +0.051] 2W/4L p=0.69 | +0.010 [-0.048, +0.068] | 0 | 54 |
| `embedder=8b` | 0.582 | [0.450, 0.703] | 0.582 | 0.116 | 0.351 | 0.409 | +0.000 [-0.051, +0.051] 1W/1L p=1 | +0.016 [-0.023, +0.055] | 0 | 64 |
| `embedder=8b stage1_top_k=20` | 0.727 | [0.598, 0.827] | 0.727 | 0.037 | 0.372 | 0.464 | +0.145 [+0.051, +0.239]* 8W/0L p=0.0078 | +0.071 [+0.025, +0.117]* | 0 | 61 |
| `embedder=8b mode=dense` | 0.564 | [0.433, 0.686] | 0.564 | 0.116 | 0.338 | 0.402 | -0.018 [-0.113, +0.077] 3W/4L p=1 | +0.008 [-0.067, +0.084] | 0 | 60 |
| `embedder=4b query_instruction=retrieval` | 0.564 | [0.433, 0.686] | 0.564 | 0.116 | 0.334 | 0.403 | -0.018 [-0.080, +0.044] 1W/2L p=1 | +0.010 [-0.048, +0.068] | 0 | 61 |
| `embedder=4b query_instruction=retrieval mode=dense` | 0.527 | [0.398, 0.653] | 0.527 | 0.113 | 0.326 | 0.395 | -0.055 [-0.161, +0.052] 3W/6L p=0.51 | +0.002 [-0.080, +0.083] | 0 | 56 |
| | | | | | | | | | | |
| `header=on` | 0.654 | [0.523, 0.766] | 0.654 | 0.138 | 0.412 | 0.492 | +0.073 [-0.013, +0.159] 5W/1L p=0.22 | +0.098 [+0.020, +0.177]* | 0 | 50 |
| `header=on stage1_top_k=20` | 0.946 | [0.851, 0.981] | 0.946 | 0.051 | 0.452 | 0.599 | +0.364 [+0.235, +0.492]* 20W/0L p=1.9e-06 | +0.206 [+0.126, +0.285]* | 0 | 50 |
| `header=on mode=dense` | 0.654 | [0.523, 0.766] | 0.654 | 0.138 | 0.415 | 0.493 | +0.073 [-0.013, +0.159] 5W/1L p=0.22 | +0.100 [+0.026, +0.175]* | 0 | 52 |
| `header=on rerank_header` | 0.836 | [0.717, 0.911] | 0.836 | 0.182 | 0.735 | 0.803 | +0.255 [+0.128, +0.381]* 15W/1L p=0.00052 | +0.410 [+0.292, +0.528]* | 0 | 52 |

`hit 95% CI` is a Wilson interval on that rate alone, the noise floor of one run on this many questions. Δ is variant minus `baseline`, paired by sample. `*` marks a 95% interval that excludes zero; `W/L` counts the questions the variant gained / lost and `p` is McNemar's exact test on them -- trust it over the CI when W+L is small. `(no CI)` rows predate per-sample scores and need a re-run to be tested. `unmatch.` counts expected spans no chunk contains under that variant's chunking (`—` predates the count).
