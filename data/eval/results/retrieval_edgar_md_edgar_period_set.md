# Retrieval matrix — edgar_md (isolated)

- Eval set: `data/eval/edgar_period_set.json` (55 samples)
- Collection: `rag_corpus__edgar_md`

| variant | hit | hit 95% CI | recall | prec | MRR | NDCG | Δ hit [95% CI] | Δ NDCG [95% CI] | unmatch. | s |
|---|---|---|---|---|---|---|---|---|---|---|
| `baseline` | 0.891 | [0.782, 0.949] | 0.891 | 0.178 | 0.694 | 0.744 | — | — | 2 | 52 |
| | | | | | | | | | | |
| `mode=dense` | 0.855 | [0.738, 0.924] | 0.855 | 0.171 | 0.647 | 0.699 | -0.036 [-0.086, +0.014] 0W/2L p=0.5 | -0.045 [-0.088, -0.001]* | 2 | 49 |
| | | | | | | | | | | |
| `stage1_top_k=20` | 0.927 | [0.827, 0.971] | 0.927 | 0.046 | 0.699 | 0.756 | +0.036 [-0.014, +0.086] 2W/0L p=0.5 | +0.012 [-0.005, +0.029] | 2 | 49 |
| | | | | | | | | | | |
| `chunker=structured` | 0.964 | [0.877, 0.990] | 0.964 | 0.196 | 0.867 | 0.891 | +0.073 [-0.013, +0.159] 5W/1L p=0.22 | +0.148 [+0.049, +0.246]* | 0 | 45 |
| `chunker=structured stage1_top_k=20` | 0.964 | [0.877, 0.990] | 0.964 | 0.050 | 0.867 | 0.891 | +0.073 [-0.013, +0.159] 5W/1L p=0.22 | +0.148 [+0.049, +0.246]* | 0 | 44 |
| `chunker=structured mode=dense` | 0.946 | [0.851, 0.981] | 0.946 | 0.193 | 0.849 | 0.873 | +0.055 [-0.039, +0.149] 5W/2L p=0.45 | +0.129 [+0.023, +0.236]* | 0 | 44 |
| `chunker=structured prose_overlap` | 0.964 | [0.877, 0.990] | 0.964 | 0.204 | 0.864 | 0.889 | +0.073 [-0.027, +0.173] 6W/2L p=0.29 | +0.146 [+0.040, +0.252]* | 0 | 44 |

`hit 95% CI` is a Wilson interval on that rate alone, the noise floor of one run on this many questions. Δ is variant minus `baseline`, paired by sample. `*` marks a 95% interval that excludes zero; `W/L` counts the questions the variant gained / lost and `p` is McNemar's exact test on them -- trust it over the CI when W+L is small. `(no CI)` rows predate per-sample scores and need a re-run to be tested. `unmatch.` counts expected spans no chunk contains under that variant's chunking (`—` predates the count).
