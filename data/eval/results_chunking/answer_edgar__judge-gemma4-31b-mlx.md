# Answer eval — edgar (isolated), judge `gemma4:31b-mlx`

| variant | answerable pass | Δ vs `crag=off` [95% CI] | fails: retrieval / generation | n | s | refusal pass | n | s |
|---|---|---|---|---|---|---|---|---|
| `crag=off` | 0.862 | — | 15 / 9 | 174 | 2753 | — | — | — |
| `header=on` | 0.937 | +0.075 [+0.029, +0.120]* 15W/2L p=0.0023 | 5 / 6 | 174 | 2692 | — | — | — |
| `header=on rerank_header` | 0.943 | +0.080 [+0.026, +0.134]* 19W/5L p=0.0066 | 3 / 7 | 174 | 3066 | — | — | — |

Chunking-plan tiers, each reported on its own (CI: 95% Wilson interval on that rate alone):

| variant | set | pass | 95% CI | fails: retrieval / generation | n | s |
|---|---|---|---|---|---|---|
| `crag=off` | period | 0.782 | [0.656, 0.871] | 12 / 0 | 55 | 862 |
| `crag=off` | underspecified | 0.610 | [0.520, 0.693] | 37 / 9 | 118 | 2093 |
| `crag=off` | underspecified: implicit | 0.722 | [0.591, 0.824] | — | 54 | — |
| `crag=off` | underspecified: paraphrase | 0.516 | [0.396, 0.634] | — | 64 | — |
| `header=on` | period | 0.800 | [0.676, 0.884] | 11 / 0 | 55 | 924 |
| `header=on` | underspecified | 0.754 | [0.669, 0.823] | 15 / 14 | 118 | 2225 |
| `header=on` | underspecified: implicit | 0.741 | [0.611, 0.839] | — | 54 | — |
| `header=on` | underspecified: paraphrase | 0.766 | [0.649, 0.853] | — | 64 | — |
| `header=on rerank_header` | period | 0.927 | [0.827, 0.971] | 4 / 0 | 55 | 930 |
| `header=on rerank_header` | underspecified | 0.763 | [0.678, 0.830] | 17 / 11 | 118 | 2243 |
| `header=on rerank_header` | underspecified: implicit | 0.722 | [0.591, 0.824] | — | 54 | — |
| `header=on rerank_header` | underspecified: paraphrase | 0.797 | [0.683, 0.877] | — | 64 | — |
