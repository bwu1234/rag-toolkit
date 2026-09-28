# Retrieval matrix — edgar (isolated)

- Eval set: `data/eval/edgar_underspecified_set.json` (118 samples)
- Collection: `rag_corpus__edgar`

| variant | hit | hit 95% CI | recall | prec | MRR | NDCG | Δ hit [95% CI] | Δ NDCG [95% CI] | unmatch. | s |
|---|---|---|---|---|---|---|---|---|---|---|
| `baseline` | 0.652 | [0.563, 0.732] | 0.652 | 0.141 | 0.493 | 0.562 | — | — | 0 | 118 |

By kind, each reported on its own (the rows above mix them):

| variant | kind | n | hit | hit 95% CI | NDCG | Δ hit [95% CI] |
|---|---|---|---|---|---|---|
| `baseline` | implicit | 54 | 0.833 | [0.713, 0.910] | 0.741 | — |
| `baseline` | paraphrase | 64 | 0.500 | [0.381, 0.619] | 0.410 | — |

`hit 95% CI` is a Wilson interval on that rate alone, the noise floor of one run on this many questions. Δ is variant minus `baseline`, paired by sample. `*` marks a 95% interval that excludes zero; `W/L` counts the questions the variant gained / lost and `p` is McNemar's exact test on them -- trust it over the CI when W+L is small. `(no CI)` rows predate per-sample scores and need a re-run to be tested. `unmatch.` counts expected spans no chunk contains under that variant's chunking (`—` predates the count).
