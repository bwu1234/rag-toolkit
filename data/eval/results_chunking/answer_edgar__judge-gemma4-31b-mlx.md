# Answer eval — edgar (isolated), judge `gemma4:31b-mlx`

| variant | answerable pass | Δ vs `crag=off` [95% CI] | fails: retrieval / generation | n | s | refusal pass | n | s |
|---|---|---|---|---|---|---|---|---|
| `crag=off` | 0.862 | — | 15 / 9 | 174 | 2753 | — | — | — |

Chunking-plan tiers, each reported on its own (CI: 95% Wilson interval on that rate alone):

| variant | set | pass | 95% CI | fails: retrieval / generation | n | s |
|---|---|---|---|---|---|---|
| `crag=off` | period | 0.782 | [0.656, 0.871] | 12 / 0 | 55 | 862 |
| `crag=off` | underspecified | 0.610 | [0.520, 0.693] | 37 / 9 | 118 | 2093 |
| `crag=off` | underspecified: implicit | 0.722 | [0.591, 0.824] | — | 54 | — |
| `crag=off` | underspecified: paraphrase | 0.516 | [0.396, 0.634] | — | 64 | — |
