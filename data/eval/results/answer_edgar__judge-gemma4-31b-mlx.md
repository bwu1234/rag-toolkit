# Answer eval — edgar (isolated), judge `gemma4:31b-mlx`

| variant | answerable pass | Δ vs `crag=off` [95% CI] | fails: retrieval / generation | n | s | refusal pass | n | s |
|---|---|---|---|---|---|---|---|---|
| `crag=off` | 0.875 | — | — | 40 | 640 | 0.933 | 15 | 303 |

| variant | multi-hop complete | completeness | evidence recall | n | s/turn | LLM calls/turn | LLM s/turn | prompt / gen tokens/turn |
|---|---|---|---|---|---|---|---|---|
| `crag=off` | 0.412 | 0.566 | 0.598 | 34 | 11.2 | 1.0 | 9.3 | 1,680 / 184 |
