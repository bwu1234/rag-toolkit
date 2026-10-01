# Retrieval matrix — edgar_md (isolated)

- Eval set: `data/eval/edgar_underspecified_set.json` (118 samples)
- Collection: `rag_corpus__edgar_md`

| variant | hit | hit 95% CI | recall | prec | MRR | NDCG | Δ hit [95% CI] | Δ NDCG [95% CI] | unmatch. | s |
|---|---|---|---|---|---|---|---|---|---|---|
| `baseline` | 0.822 | [0.743, 0.881] | 0.822 | 0.175 | 0.666 | 0.706 | — | — | 0 | 109 |
| | | | | | | | | | | |
| `mode=dense` | 0.763 | [0.678, 0.830] | 0.763 | 0.164 | 0.627 | 0.661 | -0.059 [-0.114, -0.005]* 2W/9L p=0.065 | -0.044 [-0.082, -0.007]* | 0 | 132 |
| | | | | | | | | | | |
| `stage1_top_k=20` | 0.873 | [0.801, 0.921] | 0.873 | 0.047 | 0.673 | 0.721 | +0.051 [+0.011, +0.091]* 6W/0L p=0.031 | +0.015 [+0.003, +0.027]* | 0 | 249 |
| | | | | | | | | | | |
| `chunker=structured` | 0.831 | [0.753, 0.888] | 0.831 | 0.173 | 0.711 | 0.738 | +0.008 [-0.047, +0.064] 6W/5L p=1 | +0.033 [-0.023, +0.088] | 0 | 559 |
| `chunker=structured stage1_top_k=20` | 0.873 | [0.801, 0.921] | 0.873 | 0.045 | 0.715 | 0.750 | +0.051 [-0.006, +0.108] 9W/3L p=0.15 | +0.044 [-0.011, +0.099] | 0 | 596 |
| `chunker=structured mode=dense` | 0.797 | [0.715, 0.859] | 0.797 | 0.166 | 0.704 | 0.724 | -0.025 [-0.094, +0.043] 7W/10L p=0.63 | +0.018 [-0.046, +0.083] | 0 | 500 |
| `chunker=structured prose_overlap` | 0.805 | [0.724, 0.866] | 0.805 | 0.171 | 0.675 | 0.708 | -0.017 [-0.075, +0.041] 5W/7L p=0.77 | +0.002 [-0.055, +0.059] | 0 | 476 |

By kind, each reported on its own (the rows above mix them):

| variant | kind | n | hit | hit 95% CI | NDCG | Δ hit [95% CI] |
|---|---|---|---|---|---|---|
| `baseline` | implicit | 54 | 0.796 | [0.671, 0.882] | 0.689 | — |
| `baseline` | paraphrase | 64 | 0.844 | [0.736, 0.913] | 0.719 | — |
| `mode=dense` | implicit | 54 | 0.722 | [0.591, 0.824] | 0.631 | -0.074 [-0.162, +0.013] 1W/5L p=0.22 |
| `mode=dense` | paraphrase | 64 | 0.797 | [0.683, 0.877] | 0.687 | -0.047 [-0.115, +0.021] 1W/4L p=0.38 |
| `stage1_top_k=20` | implicit | 54 | 0.852 | [0.734, 0.923] | 0.707 | +0.056 [-0.006, +0.117] 3W/0L p=0.25 |
| `stage1_top_k=20` | paraphrase | 64 | 0.891 | [0.791, 0.946] | 0.732 | +0.047 [-0.005, +0.099] 3W/0L p=0.25 |
| `chunker=structured` | implicit | 54 | 0.833 | [0.713, 0.910] | 0.741 | +0.037 [-0.036, +0.110] 3W/1L p=0.62 |
| `chunker=structured` | paraphrase | 64 | 0.828 | [0.718, 0.901] | 0.736 | -0.016 [-0.097, +0.066] 3W/4L p=1 |
| `chunker=structured stage1_top_k=20` | implicit | 54 | 0.889 | [0.778, 0.948] | 0.756 | +0.093 [+0.015, +0.171]* 5W/0L p=0.062 |
| `chunker=structured stage1_top_k=20` | paraphrase | 64 | 0.859 | [0.754, 0.924] | 0.745 | +0.016 [-0.066, +0.097] 4W/3L p=1 |
| `chunker=structured mode=dense` | implicit | 54 | 0.722 | [0.591, 0.824] | 0.675 | -0.074 [-0.188, +0.040] 3W/7L p=0.34 |
| `chunker=structured mode=dense` | paraphrase | 64 | 0.859 | [0.754, 0.924] | 0.765 | +0.016 [-0.066, +0.097] 4W/3L p=1 |
| `chunker=structured prose_overlap` | implicit | 54 | 0.796 | [0.671, 0.882] | 0.728 | +0.000 [-0.073, +0.073] 2W/2L p=1 |
| `chunker=structured prose_overlap` | paraphrase | 64 | 0.812 | [0.700, 0.889] | 0.690 | -0.031 [-0.118, +0.056] 3W/5L p=0.73 |

`hit 95% CI` is a Wilson interval on that rate alone, the noise floor of one run on this many questions. Δ is variant minus `baseline`, paired by sample. `*` marks a 95% interval that excludes zero; `W/L` counts the questions the variant gained / lost and `p` is McNemar's exact test on them -- trust it over the CI when W+L is small. `(no CI)` rows predate per-sample scores and need a re-run to be tested. `unmatch.` counts expected spans no chunk contains under that variant's chunking (`—` predates the count).
