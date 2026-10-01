# Answer eval — edgar (isolated), judge `gemma4:31b-mlx`

| variant | answerable pass | Δ vs `pipeline / 9b #1` [95% CI] | fails: retrieval / generation | n | s | refusal pass | n | s |
|---|---|---|---|---|---|---|---|---|
| `pipeline / 9b #1` | 0.975 | — | 0 / 1 | 40 | 571 | 0.933 | 15 | 251 |
| `pipeline / 9b #2` | 0.950 | -0.025 [-0.074, +0.024] 0W/1L p=1 | 0 / 2 | 40 | 580 | 0.933 | 15 | 261 |
| `pipeline / 9b #3` | 0.975 | +0.000 [-0.070, +0.070] 1W/1L p=1 | 0 / 1 | 40 | 557 | 0.933 | 15 | 252 |
| `oracle / 9b #1` | 1.000 | +0.025 [-0.024, +0.074] 1W/0L p=1 | 0 / 0 | 40 | 152 | — | — | — |
| `oracle / 9b #2` | 1.000 | +0.025 [-0.024, +0.074] 1W/0L p=1 | 0 / 0 | 40 | 134 | — | — | — |
| `oracle / 9b #3` | 1.000 | +0.025 [-0.024, +0.074] 1W/0L p=1 | 0 / 0 | 40 | 133 | — | — | — |
| `agentic react / 9b #1` | 0.925 | -0.050 [-0.118, +0.018] 0W/2L p=0.5 | 1 / 2 | 40 | 750 | 1.000 | 15 | 324 |
| `agentic react / 9b #2` | 0.950 | -0.025 [-0.074, +0.024] 0W/1L p=1 | 2 / 0 | 40 | 679 | 1.000 | 15 | 280 |
| `agentic react / 9b #3` | 0.900 | -0.075 [-0.158, +0.008] 0W/3L p=0.25 | 1 / 3 | 40 | 695 | 1.000 | 15 | 289 |
| `agentic planned / 9b #1` | 0.925 | -0.050 [-0.118, +0.018] 0W/2L p=0.5 | 1 / 2 | 40 | 777 | 1.000 | 15 | 286 |
| `agentic planned / 9b #2` | 0.925 | -0.050 [-0.118, +0.018] 0W/2L p=0.5 | 1 / 2 | 40 | 725 | 1.000 | 15 | 307 |
| `agentic planned / 9b #3` | 0.925 | -0.050 [-0.118, +0.018] 0W/2L p=0.5 | 1 / 2 | 40 | 776 | 1.000 | 15 | 303 |

| variant | multi-hop complete | completeness | evidence recall | n | s/turn | LLM calls/turn | LLM s/turn | prompt / gen tokens/turn |
|---|---|---|---|---|---|---|---|---|
| `pipeline / 9b #1` | 0.686 | 0.798 | 0.791 | 35 | 9.9 | 1.0 | 8.1 | 1,649 / 130 |
| `pipeline / 9b #2` | 0.714 | 0.805 | 0.791 | 35 | 10.5 | 1.0 | 8.5 | 1,649 / 135 |
| `pipeline / 9b #3` | 0.714 | 0.805 | 0.791 | 35 | 10.6 | 1.0 | 8.7 | 1,649 / 143 |
| `oracle / 9b #1` | 0.971 | 0.986 | 1.000 | 35 | 4.6 | 1.0 | 4.6 | 791 / 121 |
| `oracle / 9b #2` | 0.971 | 0.986 | 1.000 | 35 | 4.8 | 1.0 | 4.8 | 791 / 142 |
| `oracle / 9b #3` | 0.943 | 0.979 | 1.000 | 35 | 4.9 | 1.0 | 4.9 | 791 / 140 |
| `agentic react / 9b #1` | 0.714 | 0.833 | 0.839 | 35 | 16.3 | 2.1 | 13.2 | 4,883 / 209 |
| `agentic react / 9b #2` | 0.686 | 0.802 | 0.825 | 35 | 16.8 | 2.1 | 13.6 | 4,926 / 219 |
| `agentic react / 9b #3` | 0.771 | 0.845 | 0.868 | 35 | 17.1 | 2.1 | 13.9 | 5,042 / 218 |
| `agentic planned / 9b #1` | 0.657 | 0.812 | 0.868 | 35 | 18.4 | 2.0 | 15.0 | 4,028 / 243 |
| `agentic planned / 9b #2` | 0.800 | 0.879 | 0.882 | 35 | 17.8 | 2.0 | 14.6 | 4,130 / 243 |
| `agentic planned / 9b #3` | 0.714 | 0.843 | 0.896 | 35 | 18.8 | 2.0 | 15.3 | 4,172 / 245 |

Spread across repeats (counts per run; a difference between variants smaller than the range is within run-to-run noise):

| variant | set | per run | mean | range |
|---|---|---|---|---|
| `pipeline / 9b` | answerable pass | 39/40, 38/40, 39/40 | 38.7 | 1 |
| `pipeline / 9b` | refusals pass | 14/15, 14/15, 14/15 | 14.0 | 0 |
| `pipeline / 9b` | multihop complete | 24/35, 25/35, 25/35 | 24.7 | 1 |
| `oracle / 9b` | answerable pass | 40/40, 40/40, 40/40 | 40.0 | 0 |
| `oracle / 9b` | multihop complete | 34/35, 34/35, 33/35 | 33.7 | 1 |
| `agentic react / 9b` | answerable pass | 37/40, 38/40, 36/40 | 37.0 | 2 |
| `agentic react / 9b` | refusals pass | 15/15, 15/15, 15/15 | 15.0 | 0 |
| `agentic react / 9b` | multihop complete | 25/35, 24/35, 27/35 | 25.3 | 3 |
| `agentic planned / 9b` | answerable pass | 37/40, 37/40, 37/40 | 37.0 | 0 |
| `agentic planned / 9b` | refusals pass | 15/15, 15/15, 15/15 | 15.0 | 0 |
| `agentic planned / 9b` | multihop complete | 23/35, 28/35, 25/35 | 25.3 | 5 |
