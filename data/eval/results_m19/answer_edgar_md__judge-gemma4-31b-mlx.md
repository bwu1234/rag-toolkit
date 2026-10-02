# Answer eval — edgar_md (isolated), judge `gemma4:31b-mlx`

| variant | answerable pass | Δ vs `pipeline / 9b #1` [95% CI] | fails: retrieval / generation | n | s | refusal pass | n | s |
|---|---|---|---|---|---|---|---|---|
| `pipeline / 9b #1` | 0.925 | — | 1 / 2 | 40 | 432 | 1.000 | 15 | 167 |
| `pipeline / 9b #2` | 0.950 | +0.025 [-0.024, +0.074] 1W/0L p=1 | 1 / 1 | 40 | 444 | 1.000 | 15 | 198 |
| `pipeline / 9b #3` | 0.925 | +0.000 [-0.070, +0.070] 1W/1L p=1 | 1 / 2 | 40 | 415 | 1.000 | 15 | 174 |

Chunking-plan tiers, each reported on its own (CI: 95% Wilson interval on that rate alone):

| variant | set | pass | 95% CI | fails: retrieval / generation | n | s |
|---|---|---|---|---|---|---|
| `pipeline / 9b #1` | period | 0.982 | [0.904, 0.997] | 1 / 0 | 55 | 542 |
| `pipeline / 9b #1` | underspecified | 0.763 | [0.678, 0.830] | 14 / 14 | 118 | 1404 |
| `pipeline / 9b #1` | underspecified: implicit | 0.741 | [0.611, 0.839] | — | 54 | — |
| `pipeline / 9b #1` | underspecified: paraphrase | 0.781 | [0.666, 0.865] | — | 64 | — |
| `pipeline / 9b #1` | table | 0.947 | [0.883, 0.977] | 2 / 3 | 95 | 1414 |
| `pipeline / 9b #2` | period | 0.982 | [0.904, 0.997] | 1 / 0 | 55 | 601 |
| `pipeline / 9b #2` | underspecified | 0.763 | [0.678, 0.830] | 13 / 15 | 118 | 1454 |
| `pipeline / 9b #2` | underspecified: implicit | 0.722 | [0.591, 0.824] | — | 54 | — |
| `pipeline / 9b #2` | underspecified: paraphrase | 0.797 | [0.683, 0.877] | — | 64 | — |
| `pipeline / 9b #2` | table | 0.937 | [0.869, 0.971] | 2 / 4 | 95 | 1309 |
| `pipeline / 9b #3` | period | 0.982 | [0.904, 0.997] | 1 / 0 | 55 | 608 |
| `pipeline / 9b #3` | underspecified | 0.754 | [0.669, 0.823] | 13 / 16 | 118 | 1420 |
| `pipeline / 9b #3` | underspecified: implicit | 0.759 | [0.630, 0.854] | — | 54 | — |
| `pipeline / 9b #3` | underspecified: paraphrase | 0.750 | [0.632, 0.840] | — | 64 | — |
| `pipeline / 9b #3` | table | 0.947 | [0.883, 0.977] | 2 / 3 | 95 | 1307 |

| variant | multi-hop complete | completeness | evidence recall | n | s/turn | LLM calls/turn | LLM s/turn | prompt / gen tokens/turn |
|---|---|---|---|---|---|---|---|---|
| `pipeline / 9b #1` | 0.714 | 0.826 | 0.853 | 35 | 8.3 | 1.0 | 7.0 | 1,347 / 138 |
| `pipeline / 9b #2` | 0.714 | 0.826 | 0.853 | 35 | 8.3 | 1.0 | 7.0 | 1,347 / 132 |
| `pipeline / 9b #3` | 0.686 | 0.807 | 0.853 | 35 | 8.5 | 1.0 | 7.2 | 1,347 / 140 |

| variant | adaptive complete | completeness | evidence recall | n | s/turn | LLM calls/turn | LLM s/turn | prompt / gen tokens/turn | complete by kind |
|---|---|---|---|---|---|---|---|---|---|
| `pipeline / 9b #1` | 0.467 | 0.583 | 0.689 | 15 | 10.9 | 1.0 | 9.7 | 1,333 / 235 | bridge 0.500, discovery 0.400 |
| `pipeline / 9b #2` | 0.400 | 0.552 | 0.689 | 15 | 10.5 | 1.0 | 9.0 | 1,333 / 212 | bridge 0.400, discovery 0.400 |
| `pipeline / 9b #3` | 0.467 | 0.580 | 0.689 | 15 | 9.6 | 1.0 | 8.4 | 1,333 / 175 | bridge 0.500, discovery 0.400 |

Spread across repeats (counts per run; a difference between variants smaller than the range is within run-to-run noise):

| variant | set | per run | mean | range |
|---|---|---|---|---|
| `pipeline / 9b` | answerable pass | 37/40, 38/40, 37/40 | 37.3 | 1 |
| `pipeline / 9b` | refusals pass | 15/15, 15/15, 15/15 | 15.0 | 0 |
| `pipeline / 9b` | multihop complete | 25/35, 25/35, 24/35 | 24.7 | 1 |
| `pipeline / 9b` | adaptive complete | 7/15, 6/15, 7/15 | 6.7 | 1 |
| `pipeline / 9b` | period pass | 54/55, 54/55, 54/55 | 54.0 | 0 |
| `pipeline / 9b` | underspecified pass | 90/118, 90/118, 89/118 | 89.7 | 1 |
| `pipeline / 9b` | table pass | 90/95, 89/95, 90/95 | 89.7 | 1 |
