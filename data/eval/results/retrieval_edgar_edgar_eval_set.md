# Retrieval matrix — edgar (isolated)

- Eval set: `data/eval/edgar_eval_set.json` (174 samples)
- Collection: `rag_corpus__edgar`

| variant | hit | recall | prec | MRR | NDCG | Δ hit [95% CI] | Δ NDCG [95% CI] | unmatch. | s |
|---|---|---|---|---|---|---|---|---|---|
| `baseline` | 0.908 | 0.908 | 0.201 | 0.722 | 0.821 | — | — | 0 | 199 |
| | | | | | | | | | |
| `mode=dense` | 0.741 | 0.741 | 0.169 | 0.632 | 0.715 | -0.167 [-0.222, -0.111]* 0W/29L p=3.7e-09 | -0.106 [-0.150, -0.062]* | 0 | 172 |
| | | | | | | | | | |
| `stage1_top_k=20` | 0.931 | 0.931 | 0.052 | 0.725 | 0.831 | +0.023 [+0.001, +0.045]* 4W/0L p=0.12 | +0.009 [+0.001, +0.018]* | 0 | 177 |
| | | | | | | | | | |
| `query_instruction=retrieval` | 0.902 | 0.902 | 0.200 | 0.706 | 0.807 | -0.006 [-0.025, +0.014] 1W/2L p=1 | -0.015 [-0.032, +0.003] | 0 | 170 |
| `query_instruction=retrieval stage1_top_k=20` | 0.919 | 0.919 | 0.051 | 0.708 | 0.814 | +0.011 [-0.016, +0.039] 4W/2L p=0.69 | -0.007 [-0.026, +0.012] | 0 | 169 |
| `query_instruction=retrieval mode=dense` | 0.718 | 0.718 | 0.164 | 0.611 | 0.694 | -0.190 [-0.248, -0.131]* 0W/33L p=2.3e-10 | -0.127 [-0.175, -0.079]* | 0 | 168 |
| | | | | | | | | | |
| `rr=minilm-L6` | 0.805 | 0.805 | 0.175 | 0.599 | 0.683 | -0.103 [-0.154, -0.053]* 2W/20L p=0.00012 | -0.138 [-0.183, -0.094]* | 0 | 33 |
| `rr=bge-base` | 0.862 | 0.862 | 0.187 | 0.680 | 0.766 | -0.046 [-0.081, -0.011]* 1W/9L p=0.021 | -0.055 [-0.100, -0.011]* | 0 | 66 |
| `rr=bge-v2-m3` | 0.908 | 0.908 | 0.201 | 0.722 | 0.821 | +0.000 [+0.000, +0.000] 0W/0L p=1 | +0.000 [+0.000, +0.000] | 0 | 174 |
| `rr=qwen3-0.6b` | 0.293 | 0.293 | 0.059 | 0.128 | 0.166 | -0.615 [-0.687, -0.542]* 0W/107L p=1.2e-32 | -0.655 [-0.717, -0.594]* | 0 | 412 |
| `rr=qwen3-0.6b +prompt` | 0.339 | 0.339 | 0.070 | 0.174 | 0.217 | -0.569 [-0.643, -0.495]* 0W/99L p=3.2e-30 | -0.604 [-0.668, -0.540]* | 0 | 431 |
| | | | | | | | | | |
| `rr=minilm-L6 pool=100` | 0.787 | 0.787 | 0.171 | 0.587 | 0.669 | -0.121 [-0.179, -0.063]* 4W/25L p=0.0001 | -0.152 [-0.201, -0.104]* | 0 | 70 |
| `rr=bge-base pool=100` | 0.851 | 0.851 | 0.183 | 0.661 | 0.746 | -0.057 [-0.110, -0.005]* 6W/16L p=0.052 | -0.075 [-0.131, -0.019]* | 0 | 271 |
| `rr=bge-v2-m3 pool=100` | 0.919 | 0.919 | 0.207 | 0.728 | 0.836 | +0.011 [-0.024, +0.047] 6W/4L p=0.75 | +0.014 [-0.012, +0.041] | 0 | 974 |

Δ is variant minus `baseline`, paired by sample. `*` marks a 95% interval that excludes zero; `W/L` counts the questions the variant gained / lost and `p` is McNemar's exact test on them -- trust it over the CI when W+L is small. `(no CI)` rows predate per-sample scores and need a re-run to be tested. `unmatch.` counts expected spans no chunk contains under that variant's chunking (`—` predates the count).
