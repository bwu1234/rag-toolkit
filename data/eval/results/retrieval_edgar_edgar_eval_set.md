# Retrieval matrix — edgar (isolated)

- Eval set: `data/eval/edgar_eval_set.json` (174 samples)
- Collection: `rag_corpus__edgar`

| variant | hit | recall | prec | MRR | NDCG | Δ hit [95% CI] | Δ NDCG [95% CI] | s |
|---|---|---|---|---|---|---|---|---|
| `baseline` | 0.874 | 0.874 | 0.194 | 0.698 | 0.796 | — | — | 177 |
| | | | | | | | | |
| `rr=minilm-L6` | 0.770 | 0.770 | 0.168 | 0.576 | 0.658 | -0.103 [-0.154, -0.053]* 2W/20L p=0.00012 | -0.138 [-0.181, -0.094]* | 29 |
| `rr=bge-base` | 0.828 | 0.828 | 0.180 | 0.659 | 0.742 | -0.046 [-0.081, -0.011]* 1W/9L p=0.021 | -0.054 [-0.098, -0.009]* | 66 |
| `rr=bge-v2-m3` | 0.874 | 0.874 | 0.194 | 0.698 | 0.796 | +0.000 [+0.000, +0.000] 0W/0L p=1 | +0.000 [+0.000, +0.000] | 166 |
| `rr=qwen3-0.6b` | 0.276 | 0.276 | 0.055 | 0.116 | 0.155 | -0.598 (no CI) | -0.640 (no CI) | 419 |
| `rr=qwen3-0.6b +prompt` | 0.322 | 0.322 | 0.067 | 0.163 | 0.207 | -0.552 (no CI) | -0.588 (no CI) | 449 |
| | | | | | | | | |
| `rr=minilm-L6 pool=100` | 0.759 | 0.759 | 0.166 | 0.566 | 0.647 | -0.115 (no CI) | -0.149 (no CI) | 77 |
| `rr=bge-base pool=100` | 0.845 | 0.845 | 0.180 | 0.658 | 0.739 | -0.029 (no CI) | -0.057 (no CI) | 280 |
| `rr=bge-v2-m3 pool=100` | 0.908 | 0.908 | 0.202 | 0.719 | 0.823 | +0.034 (no CI) | +0.027 (no CI) | 848 |

Δ is variant minus `baseline`, paired by sample. `*` marks a 95% interval that excludes zero; `W/L` counts the questions the variant gained / lost and `p` is McNemar's exact test on them -- trust it over the CI when W+L is small. `(no CI)` rows predate per-sample scores and need a re-run to be tested.
