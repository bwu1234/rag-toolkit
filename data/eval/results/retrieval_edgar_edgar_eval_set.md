# Retrieval matrix — edgar (isolated)

- Eval set: `data/eval/edgar_eval_set.json` (174 samples)
- Collection: `rag_corpus__edgar`

| variant | hit | recall | prec | MRR | NDCG | Δ NDCG | s |
|---|---|---|---|---|---|---|---|
| `rr=minilm-L6` | 0.770 | 0.770 | 0.168 | 0.576 | 0.658 | — | 44 |
| `rr=bge-base` | 0.828 | 0.828 | 0.180 | 0.659 | 0.742 | — | 83 |
| `rr=bge-v2-m3` | 0.874 | 0.874 | 0.194 | 0.698 | 0.796 | — | 186 |
| `rr=qwen3-0.6b` | 0.276 | 0.276 | 0.055 | 0.116 | 0.155 | — | 419 |
| `rr=qwen3-0.6b +prompt` | 0.322 | 0.322 | 0.067 | 0.163 | 0.207 | — | 449 |
| | | | | | | | |
| `rr=minilm-L6 pool=100` | 0.759 | 0.759 | 0.166 | 0.566 | 0.647 | — | 77 |
| `rr=bge-base pool=100` | 0.845 | 0.845 | 0.180 | 0.658 | 0.739 | — | 280 |
| `rr=bge-v2-m3 pool=100` | 0.908 | 0.908 | 0.202 | 0.719 | 0.823 | — | 848 |
