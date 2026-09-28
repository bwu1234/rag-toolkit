# Retrieval matrix — edgar (isolated)

- Eval set: `data/eval/edgar_underspecified_set.json` (118 samples)
- Collection: `rag_corpus__edgar`

| variant | hit | hit 95% CI | recall | prec | MRR | NDCG | Δ hit [95% CI] | Δ NDCG [95% CI] | unmatch. | s |
|---|---|---|---|---|---|---|---|---|---|---|
| `baseline` | 0.652 | [0.563, 0.732] | 0.652 | 0.141 | 0.493 | 0.562 | — | — | 0 | 117 |
| | | | | | | | | | | |
| `mode=dense` | 0.619 | [0.529, 0.701] | 0.619 | 0.134 | 0.469 | 0.535 | -0.034 [-0.108, +0.040] 8W/12L p=0.5 | -0.027 [-0.087, +0.034] | 0 | 115 |
| | | | | | | | | | | |
| `stage1_top_k=20` | 0.686 | [0.598, 0.763] | 0.686 | 0.038 | 0.497 | 0.578 | +0.034 [+0.001, +0.067]* 4W/0L p=0.12 | +0.016 [+0.003, +0.029]* | 0 | 114 |
| | | | | | | | | | | |
| `query_instruction=retrieval` | 0.619 | [0.529, 0.701] | 0.619 | 0.136 | 0.484 | 0.551 | -0.034 [-0.074, +0.006] 1W/5L p=0.22 | -0.011 [-0.041, +0.020] | 0 | 115 |
| `query_instruction=retrieval stage1_top_k=20` | 0.644 | [0.554, 0.725] | 0.644 | 0.036 | 0.487 | 0.564 | -0.008 [-0.058, +0.042] 4W/5L p=1 | +0.002 [-0.031, +0.035] | 0 | 115 |
| `query_instruction=retrieval mode=dense` | 0.576 | [0.486, 0.662] | 0.576 | 0.129 | 0.444 | 0.513 | -0.076 [-0.159, +0.006] 8W/17L p=0.11 | -0.049 [-0.113, +0.015] | 0 | 117 |
| | | | | | | | | | | |
| `embedder=4b` | 0.695 | [0.607, 0.771] | 0.695 | 0.152 | 0.516 | 0.598 | +0.042 [-0.007, +0.092] 7W/2L p=0.18 | +0.036 [-0.002, +0.073] | 0 | 119 |
| `embedder=4b stage1_top_k=20` | 0.729 | [0.642, 0.801] | 0.729 | 0.040 | 0.520 | 0.609 | +0.076 [+0.018, +0.135]* 11W/2L p=0.022 | +0.048 [+0.009, +0.086]* | 0 | 117 |
| `embedder=4b mode=dense` | 0.686 | [0.598, 0.763] | 0.686 | 0.151 | 0.496 | 0.580 | +0.034 [-0.037, +0.104] 11W/7L p=0.48 | +0.018 [-0.037, +0.074] | 0 | 118 |
| `embedder=8b` | 0.686 | [0.598, 0.763] | 0.686 | 0.151 | 0.513 | 0.592 | +0.034 [-0.024, +0.091] 8W/4L p=0.39 | +0.031 [-0.018, +0.079] | 0 | 130 |
| `embedder=8b stage1_top_k=20` | 0.720 | [0.633, 0.793] | 0.720 | 0.040 | 0.517 | 0.608 | +0.068 [+0.007, +0.129]* 11W/3L p=0.057 | +0.047 [-0.002, +0.095] | 0 | 127 |
| `embedder=8b mode=dense` | 0.678 | [0.589, 0.756] | 0.678 | 0.147 | 0.509 | 0.584 | +0.025 [-0.047, +0.098] 11W/8L p=0.65 | +0.022 [-0.033, +0.077] | 0 | 129 |
| `embedder=4b query_instruction=retrieval` | 0.686 | [0.598, 0.763] | 0.686 | 0.151 | 0.504 | 0.585 | +0.034 [-0.024, +0.091] 8W/4L p=0.39 | +0.024 [-0.020, +0.067] | 0 | 123 |
| `embedder=4b query_instruction=retrieval mode=dense` | 0.661 | [0.572, 0.740] | 0.661 | 0.147 | 0.483 | 0.568 | +0.008 [-0.068, +0.085] 11W/10L p=1 | +0.006 [-0.053, +0.066] | 0 | 120 |
| | | | | | | | | | | |
| `header=on` | 0.797 | [0.715, 0.859] | 0.797 | 0.180 | 0.571 | 0.679 | +0.144 [+0.072, +0.216]* 19W/2L p=0.00022 | +0.118 [+0.065, +0.171]* | 0 | 114 |
| `header=on stage1_top_k=20` | 0.873 | [0.801, 0.921] | 0.873 | 0.050 | 0.580 | 0.714 | +0.220 [+0.142, +0.299]* 27W/1L p=2.2e-07 | +0.152 [+0.098, +0.206]* | 0 | 113 |
| `header=on mode=dense` | 0.746 | [0.660, 0.816] | 0.746 | 0.170 | 0.555 | 0.656 | +0.093 [+0.012, +0.175]* 18W/7L p=0.043 | +0.095 [+0.025, +0.164]* | 0 | 118 |
| `header=on rerank_header` | 0.822 | [0.743, 0.881] | 0.822 | 0.188 | 0.657 | 0.767 | +0.169 [+0.090, +0.249]* 23W/3L p=8.8e-05 | +0.205 [+0.125, +0.285]* | 0 | 119 |

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
| `embedder=4b` | implicit | 54 | 0.833 | [0.713, 0.910] | 0.742 | +0.000 [-0.052, +0.052] 1W/1L p=1 |
| `embedder=4b` | paraphrase | 64 | 0.578 | [0.456, 0.691] | 0.475 | +0.078 [-0.001, +0.157] 6W/1L p=0.12 |
| `embedder=4b stage1_top_k=20` | implicit | 54 | 0.889 | [0.778, 0.948] | 0.764 | +0.056 [-0.025, +0.136] 4W/1L p=0.38 |
| `embedder=4b stage1_top_k=20` | paraphrase | 64 | 0.594 | [0.471, 0.705] | 0.479 | +0.094 [+0.010, +0.178]* 7W/1L p=0.07 |
| `embedder=4b mode=dense` | implicit | 54 | 0.759 | [0.631, 0.854] | 0.681 | -0.074 [-0.162, +0.013] 1W/5L p=0.22 |
| `embedder=4b mode=dense` | paraphrase | 64 | 0.625 | [0.503, 0.733] | 0.495 | +0.125 [+0.023, +0.227]* 10W/2L p=0.039 |
| `embedder=8b` | implicit | 54 | 0.815 | [0.692, 0.896] | 0.727 | -0.019 [-0.082, +0.045] 1W/2L p=1 |
| `embedder=8b` | paraphrase | 64 | 0.578 | [0.456, 0.691] | 0.479 | +0.078 [-0.012, +0.169] 7W/2L p=0.18 |
| `embedder=8b stage1_top_k=20` | implicit | 54 | 0.870 | [0.756, 0.936] | 0.750 | +0.037 [-0.036, +0.110] 3W/1L p=0.62 |
| `embedder=8b stage1_top_k=20` | paraphrase | 64 | 0.594 | [0.471, 0.705] | 0.488 | +0.094 [-0.001, +0.189] 8W/2L p=0.11 |
| `embedder=8b mode=dense` | implicit | 54 | 0.741 | [0.611, 0.839] | 0.686 | -0.093 [-0.186, +0.001] 1W/6L p=0.12 |
| `embedder=8b mode=dense` | paraphrase | 64 | 0.625 | [0.503, 0.733] | 0.497 | +0.125 [+0.023, +0.227]* 10W/2L p=0.039 |
| `embedder=4b query_instruction=retrieval` | implicit | 54 | 0.833 | [0.713, 0.910] | 0.736 | +0.000 [-0.052, +0.052] 1W/1L p=1 |
| `embedder=4b query_instruction=retrieval` | paraphrase | 64 | 0.562 | [0.441, 0.677] | 0.458 | +0.062 [-0.034, +0.159] 7W/3L p=0.34 |
| `embedder=4b query_instruction=retrieval mode=dense` | implicit | 54 | 0.741 | [0.611, 0.839] | 0.672 | -0.093 [-0.200, +0.014] 2W/7L p=0.18 |
| `embedder=4b query_instruction=retrieval mode=dense` | paraphrase | 64 | 0.594 | [0.471, 0.705] | 0.481 | +0.094 [-0.011, +0.198] 9W/3L p=0.15 |
| `header=on` | implicit | 54 | 0.833 | [0.713, 0.910] | 0.760 | +0.000 [-0.073, +0.073] 2W/2L p=1 |
| `header=on` | paraphrase | 64 | 0.766 | [0.649, 0.853] | 0.612 | +0.266 [+0.157, +0.375]* 17W/0L p=1.5e-05 |
| `header=on stage1_top_k=20` | implicit | 54 | 0.889 | [0.778, 0.948] | 0.790 | +0.056 [-0.025, +0.136] 4W/1L p=0.38 |
| `header=on stage1_top_k=20` | paraphrase | 64 | 0.859 | [0.754, 0.924] | 0.649 | +0.359 [+0.241, +0.478]* 23W/0L p=2.4e-07 |
| `header=on mode=dense` | implicit | 54 | 0.722 | [0.591, 0.824] | 0.681 | -0.111 [-0.210, -0.012]* 1W/7L p=0.07 |
| `header=on mode=dense` | paraphrase | 64 | 0.766 | [0.649, 0.853] | 0.636 | +0.266 [+0.157, +0.375]* 17W/0L p=1.5e-05 |
| `header=on rerank_header` | implicit | 54 | 0.852 | [0.734, 0.923] | 0.810 | +0.019 [-0.078, +0.115] 4W/3L p=1 |
| `header=on rerank_header` | paraphrase | 64 | 0.797 | [0.683, 0.877] | 0.730 | +0.297 [+0.184, +0.410]* 19W/0L p=3.8e-06 |

`hit 95% CI` is a Wilson interval on that rate alone, the noise floor of one run on this many questions. Δ is variant minus `baseline`, paired by sample. `*` marks a 95% interval that excludes zero; `W/L` counts the questions the variant gained / lost and `p` is McNemar's exact test on them -- trust it over the CI when W+L is small. `(no CI)` rows predate per-sample scores and need a re-run to be tested. `unmatch.` counts expected spans no chunk contains under that variant's chunking (`—` predates the count).
