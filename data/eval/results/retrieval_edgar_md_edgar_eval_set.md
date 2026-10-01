# Retrieval matrix — edgar_md (isolated)

- Eval set: `data/eval/edgar_eval_set.json` (174 samples)
- Collection: `rag_corpus__edgar_md`

| variant | hit | hit 95% CI | recall | prec | MRR | NDCG | Δ hit [95% CI] | Δ NDCG [95% CI] | unmatch. | s |
|---|---|---|---|---|---|---|---|---|---|---|
| `baseline` | 0.966 | [0.927, 0.984] | 0.966 | 0.213 | 0.860 | 0.887 | — | — | 0 | 158 |
| | | | | | | | | | | |
| `mode=dense` | 0.931 | [0.883, 0.960] | 0.931 | 0.205 | 0.841 | 0.864 | -0.034 [-0.066, -0.003]* 1W/7L p=0.07 | -0.023 [-0.044, -0.002]* | 0 | 160 |
| | | | | | | | | | | |
| `stage1_top_k=20` | 0.989 | [0.959, 0.997] | 0.989 | 0.056 | 0.863 | 0.894 | +0.023 [+0.001, +0.045]* 4W/0L p=0.12 | +0.007 [+0.000, +0.015]* | 0 | 153 |
| | | | | | | | | | | |
| `chunker=structured` | 0.977 | [0.942, 0.991] | 0.977 | 0.205 | 0.917 | 0.932 | +0.011 [-0.011, +0.034] 3W/1L p=0.62 | +0.045 [+0.015, +0.076]* | 0 | 146 |
| `chunker=structured stage1_top_k=20` | 0.989 | [0.959, 0.997] | 0.989 | 0.052 | 0.918 | 0.935 | +0.023 [-0.004, +0.050] 5W/1L p=0.22 | +0.049 [+0.018, +0.079]* | 0 | 145 |
| `chunker=structured mode=dense` | 0.954 | [0.912, 0.977] | 0.954 | 0.199 | 0.904 | 0.914 | -0.011 [-0.043, +0.020] 3W/5L p=0.73 | +0.028 [-0.009, +0.064] | 0 | 149 |
| `chunker=structured prose_overlap` | 0.971 | [0.935, 0.988] | 0.971 | 0.208 | 0.891 | 0.912 | +0.006 [-0.019, +0.031] 3W/2L p=1 | +0.025 [-0.007, +0.056] | 0 | 146 |

`hit 95% CI` is a Wilson interval on that rate alone, the noise floor of one run on this many questions. Δ is variant minus `baseline`, paired by sample. `*` marks a 95% interval that excludes zero; `W/L` counts the questions the variant gained / lost and `p` is McNemar's exact test on them -- trust it over the CI when W+L is small. `(no CI)` rows predate per-sample scores and need a re-run to be tested. `unmatch.` counts expected spans no chunk contains under that variant's chunking (`—` predates the count).
