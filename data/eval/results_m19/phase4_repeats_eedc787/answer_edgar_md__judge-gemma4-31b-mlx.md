# Answer eval — edgar_md (isolated), judge `gemma4:31b-mlx`

| variant | answerable pass | Δ vs `agentic react / 27b, think=low #1` [95% CI] | fails: retrieval / generation | n | s | refusal pass | n | s |
|---|---|---|---|---|---|---|---|---|
| `agentic react / 27b, think=low #1` | 0.950 | — | 0 / 2 | 40 | 1987 | 0.933 | 15 | 1363 |
| `agentic react / 27b, think=low #2` | 0.925 | -0.025 [-0.074, +0.024] 0W/1L p=1 | 1 / 2 | 40 | 1241 | 1.000 | 15 | 1425 |
| `agentic react / 27b, think=low #3` | 0.925 | -0.025 [-0.074, +0.024] 0W/1L p=1 | 1 / 2 | 40 | 1300 | 1.000 | 15 | 1306 |

| variant | multi-hop complete | completeness | evidence recall | n | s/turn | LLM calls/turn | LLM s/turn | prompt / gen tokens/turn |
|---|---|---|---|---|---|---|---|---|
| `agentic react / 27b, think=low #1` | 0.914 | 0.964 | 0.991 | 35 | 51.8 | 2.2 | 50.0 | 4,935 / 616 |
| `agentic react / 27b, think=low #2` | 0.914 | 0.964 | 0.991 | 35 | 48.5 | 2.1 | 46.4 | 4,574 / 589 |
| `agentic react / 27b, think=low #3` | 0.971 | 0.981 | 1.000 | 35 | 49.2 | 2.2 | 47.5 | 4,816 / 611 |

| variant | adaptive complete | completeness | evidence recall | n | s/turn | LLM calls/turn | LLM s/turn | prompt / gen tokens/turn | complete by kind |
|---|---|---|---|---|---|---|---|---|---|
| `agentic react / 27b, think=low #1` | 0.867 | 0.922 | 0.933 | 15 | 72.3 | 3.5 | 69.2 | 10,468 / 697 | bridge 0.900, discovery 0.800 |
| `agentic react / 27b, think=low #2` | 0.800 | 0.889 | 0.911 | 15 | 76.2 | 3.3 | 71.8 | 9,747 / 666 | bridge 0.900, discovery 0.600 |
| `agentic react / 27b, think=low #3` | 0.933 | 0.956 | 0.922 | 15 | 76.3 | 3.4 | 73.2 | 10,877 / 694 | bridge 0.900, discovery 1.000 |

Spread across repeats (counts per run; a difference between variants smaller than the range is within run-to-run noise). Δ pairs repeat k with `agentic react / 27b, think=low`'s repeat k, on the same questions; repeats are not extra questions, so this is not a CI:

| variant | set | per run | mean | SD | range | Δ vs `agentic react / 27b, think=low` per repeat |
|---|---|---|---|---|---|---|
| `agentic react / 27b, think=low` | answerable pass | 38/40, 37/40, 37/40 | 37.3 | 0.6 | 1 | — |
| `agentic react / 27b, think=low` | refusals pass | 14/15, 15/15, 15/15 | 14.7 | 0.6 | 1 | — |
| `agentic react / 27b, think=low` | multihop complete | 32/35, 32/35, 34/35 | 32.7 | 1.2 | 2 | — |
| `agentic react / 27b, think=low` | adaptive complete | 13/15, 12/15, 14/15 | 13.0 | 1.0 | 2 | — |
