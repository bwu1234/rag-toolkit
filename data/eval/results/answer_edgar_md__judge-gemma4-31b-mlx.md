# Answer eval — edgar_md (isolated), judge `gemma4:31b-mlx`

| variant | answerable pass | Δ vs `crag=off` [95% CI] | fails: retrieval / generation | n | s | refusal pass | n | s |
|---|---|---|---|---|---|---|---|---|
| `crag=off` | 0.948 | — | 3 / 6 | 174 | 2010 | — | — | — |
| `chunker=structured` | 0.948 | +0.000 [-0.028, +0.028] 3W/3L p=1 | 3 / 6 | 174 | 2364 | — | — | — |

Chunking-plan tiers, each reported on its own (CI: 95% Wilson interval on that rate alone):

| variant | set | pass | 95% CI | fails: retrieval / generation | n | s |
|---|---|---|---|---|---|---|
| `crag=off` | table | 0.811 | [0.720, 0.877] | 13 / 5 | 95 | 1328 |
| `chunker=structured` | table | 0.947 | [0.883, 0.977] | 2 / 3 | 95 | 1664 |
