# Answer eval — edgar_md (isolated), judge `gemma4:31b-mlx`

| variant | answerable pass | Δ vs `pipeline / 9b #1` [95% CI] | fails: retrieval / generation | n | s | refusal pass | n | s |
|---|---|---|---|---|---|---|---|---|
| `pipeline / 9b #1` | 0.925 | — | 1 / 2 | 40 | 432 | 1.000 | 15 | 167 |
| `pipeline / 9b #2` | 0.950 | +0.025 [-0.024, +0.074] 1W/0L p=1 | 1 / 1 | 40 | 444 | 1.000 | 15 | 198 |
| `pipeline / 9b #3` | 0.925 | +0.000 [-0.070, +0.070] 1W/1L p=1 | 1 / 2 | 40 | 415 | 1.000 | 15 | 174 |
| `oracle / 9b #1` | 0.975 | +0.050 [-0.018, +0.118] 2W/0L p=0.5 | 0 / 1 | 40 | 144 | — | — | — |
| `oracle / 9b #2` | 0.975 | +0.050 [-0.018, +0.118] 2W/0L p=0.5 | 0 / 1 | 40 | 142 | — | — | — |
| `oracle / 9b #3` | 0.975 | +0.050 [-0.018, +0.118] 2W/0L p=0.5 | 0 / 1 | 40 | 151 | — | — | — |
| `agentic react / 9b #1` | 0.925 | +0.000 [-0.070, +0.070] 1W/1L p=1 | 2 / 1 | 40 | 565 | 1.000 | 15 | 232 |
| `agentic react / 9b #2` | 0.950 | +0.025 [-0.061, +0.111] 2W/1L p=1 | 2 / 0 | 40 | 566 | 1.000 | 15 | 224 |
| `agentic react / 9b #3` | 0.925 | +0.000 [-0.070, +0.070] 1W/1L p=1 | 3 / 0 | 40 | 555 | 1.000 | 15 | 270 |
| `agentic planned / 9b #1` | 0.900 | -0.025 [-0.111, +0.061] 1W/2L p=1 | 3 / 1 | 40 | 632 | 0.933 | 15 | 262 |
| `agentic planned / 9b #2` | 0.900 | -0.025 [-0.111, +0.061] 1W/2L p=1 | 3 / 1 | 40 | 586 | 0.933 | 15 | 250 |
| `agentic planned / 9b #3` | 0.925 | +0.000 [-0.070, +0.070] 1W/1L p=1 | 3 / 0 | 40 | 587 | 1.000 | 15 | 231 |
| `pipeline / 27b #1` | 0.925 | +0.000 [-0.070, +0.070] 1W/1L p=1 | 1 / 2 | 40 | 1100 | 1.000 | 15 | 476 |
| `pipeline / 27b #2` | 0.950 | +0.025 [-0.024, +0.074] 1W/0L p=1 | 1 / 1 | 40 | 1188 | 1.000 | 15 | 493 |
| `pipeline / 27b #3` | 0.950 | +0.025 [-0.024, +0.074] 1W/0L p=1 | 1 / 1 | 40 | 1296 | 1.000 | 15 | 488 |
| `agentic react / 27b #1` | 0.925 | +0.000 [-0.099, +0.099] 2W/2L p=1 | 0 / 3 | 40 | 1784 | 0.933 | 15 | 2299 |
| `agentic react / 27b #2` | 0.925 | +0.000 [-0.099, +0.099] 2W/2L p=1 | 0 / 3 | 40 | 1716 | 0.867 | 15 | 1885 |
| `agentic react / 27b #3` | 0.950 | +0.025 [-0.061, +0.111] 2W/1L p=1 | 0 / 2 | 40 | 1718 | 0.933 | 15 | 2280 |
| `agentic react / 27b, think=low #1` | 0.975 | +0.050 [-0.018, +0.118] 2W/0L p=0.5 | 1 / 0 | 40 | 1756 | 1.000 | 15 | 1706 |
| `agentic react / 27b, think=low #2` | 0.950 | +0.025 [-0.061, +0.111] 2W/1L p=1 | 1 / 1 | 40 | 1695 | 0.933 | 15 | 1802 |
| `agentic react / 27b + groundedness` | 0.950 | +0.025 [-0.061, +0.111] 2W/1L p=1 | 0 / 2 | 40 | 2742 | 0.933 | 15 | 2606 |

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
| `oracle / 9b #1` | 0.914 | 0.960 | 1.000 | 35 | 4.5 | 1.0 | 4.5 | 674 / 131 |
| `oracle / 9b #2` | 0.943 | 0.976 | 1.000 | 35 | 4.3 | 1.0 | 4.3 | 674 / 128 |
| `oracle / 9b #3` | 0.914 | 0.969 | 1.000 | 35 | 4.0 | 1.0 | 4.0 | 674 / 113 |
| `agentic react / 9b #1` | 0.800 | 0.850 | 0.902 | 35 | 13.7 | 2.0 | 11.4 | 4,474 / 206 |
| `agentic react / 9b #2` | 0.771 | 0.874 | 0.902 | 35 | 14.1 | 2.0 | 11.7 | 4,461 / 208 |
| `agentic react / 9b #3` | 0.771 | 0.860 | 0.930 | 35 | 13.8 | 2.0 | 11.6 | 4,452 / 202 |
| `agentic planned / 9b #1` | 0.743 | 0.864 | 0.916 | 35 | 14.7 | 2.0 | 12.5 | 3,635 / 240 |
| `agentic planned / 9b #2` | 0.743 | 0.860 | 0.930 | 35 | 14.3 | 2.0 | 12.2 | 3,633 / 228 |
| `agentic planned / 9b #3` | 0.771 | 0.883 | 0.916 | 35 | 14.3 | 2.0 | 12.4 | 3,613 / 240 |
| `pipeline / 27b #1` | 0.771 | 0.845 | 0.853 | 35 | 27.7 | 1.0 | 22.8 | 1,347 / 132 |
| `pipeline / 27b #2` | 0.743 | 0.836 | 0.853 | 35 | 30.8 | 1.0 | 25.0 | 1,347 / 138 |
| `pipeline / 27b #3` | 0.743 | 0.836 | 0.853 | 35 | 31.2 | 1.0 | 25.4 | 1,347 / 140 |
| `agentic react / 27b #1` | 0.886 | 0.948 | 0.971 | 35 | 62.6 | 2.5 | 53.8 | 7,536 / 396 |
| `agentic react / 27b #2` | 0.914 | 0.962 | 0.976 | 35 | 59.8 | 2.6 | 53.0 | 8,048 / 380 |
| `agentic react / 27b #3` | 0.914 | 0.962 | 0.971 | 35 | 58.2 | 2.4 | 50.7 | 6,948 / 383 |
| `agentic react / 27b, think=low #1` | 0.914 | 0.964 | 0.991 | 35 | 65.2 | 2.2 | 59.0 | 5,898 / 631 |
| `agentic react / 27b, think=low #2` | 0.943 | 0.971 | 0.991 | 35 | 63.6 | 2.2 | 57.4 | 5,878 / 623 |
| `agentic react / 27b + groundedness` | 0.886 | 0.948 | 0.985 | 35 | 86.3 | 3.4 | 75.0 | 9,655 / 384 |

| variant | adaptive complete | completeness | evidence recall | n | s/turn | LLM calls/turn | LLM s/turn | prompt / gen tokens/turn | complete by kind |
|---|---|---|---|---|---|---|---|---|---|
| `pipeline / 9b #1` | 0.467 | 0.583 | 0.689 | 15 | 10.9 | 1.0 | 9.7 | 1,333 / 235 | bridge 0.500, discovery 0.400 |
| `pipeline / 9b #2` | 0.400 | 0.552 | 0.689 | 15 | 10.5 | 1.0 | 9.0 | 1,333 / 212 | bridge 0.400, discovery 0.400 |
| `pipeline / 9b #3` | 0.467 | 0.580 | 0.689 | 15 | 9.6 | 1.0 | 8.4 | 1,333 / 175 | bridge 0.500, discovery 0.400 |
| `oracle / 9b #1` | 0.800 | 0.897 | 1.000 | 15 | 4.2 | 1.0 | 4.2 | 766 / 113 | bridge 0.900, discovery 0.600 |
| `oracle / 9b #2` | 0.867 | 0.930 | 1.000 | 15 | 4.3 | 1.0 | 4.3 | 766 / 118 | bridge 0.900, discovery 0.800 |
| `oracle / 9b #3` | 0.867 | 0.930 | 1.000 | 15 | 4.3 | 1.0 | 4.3 | 766 / 116 | bridge 0.900, discovery 0.800 |
| `agentic react / 9b #1` | 0.400 | 0.574 | 0.578 | 15 | 12.3 | 2.1 | 10.3 | 4,122 / 204 | bridge 0.500, discovery 0.200 |
| `agentic react / 9b #2` | 0.400 | 0.574 | 0.578 | 15 | 11.6 | 2.1 | 9.5 | 4,124 / 185 | bridge 0.500, discovery 0.200 |
| `agentic react / 9b #3` | 0.400 | 0.574 | 0.578 | 15 | 12.3 | 2.1 | 9.9 | 4,108 / 180 | bridge 0.500, discovery 0.200 |
| `agentic planned / 9b #1` | 0.267 | 0.463 | 0.578 | 15 | 12.7 | 2.0 | 10.7 | 3,112 / 216 | bridge 0.300, discovery 0.200 |
| `agentic planned / 9b #2` | 0.400 | 0.539 | 0.567 | 15 | 12.2 | 2.0 | 10.6 | 2,944 / 225 | bridge 0.400, discovery 0.400 |
| `agentic planned / 9b #3` | 0.400 | 0.539 | 0.589 | 15 | 12.9 | 2.0 | 11.0 | 3,023 / 230 | bridge 0.400, discovery 0.400 |
| `pipeline / 27b #1` | 0.467 | 0.597 | 0.689 | 15 | 28.7 | 1.0 | 23.6 | 1,333 / 163 | bridge 0.500, discovery 0.400 |
| `pipeline / 27b #2` | 0.467 | 0.613 | 0.689 | 15 | 34.5 | 1.0 | 26.7 | 1,333 / 178 | bridge 0.500, discovery 0.400 |
| `pipeline / 27b #3` | 0.467 | 0.613 | 0.689 | 15 | 35.2 | 1.0 | 28.3 | 1,333 / 175 | bridge 0.500, discovery 0.400 |
| `agentic react / 27b #1` | 0.867 | 0.922 | 0.922 | 15 | 79.4 | 3.7 | 69.4 | 12,358 / 396 | bridge 0.900, discovery 0.800 |
| `agentic react / 27b #2` | 0.933 | 0.956 | 0.944 | 15 | 83.2 | 3.7 | 72.4 | 12,420 / 479 | bridge 0.900, discovery 1.000 |
| `agentic react / 27b #3` | 0.800 | 0.882 | 0.922 | 15 | 90.7 | 3.9 | 78.9 | 13,949 / 468 | bridge 0.800, discovery 0.800 |
| `agentic react / 27b, think=low #1` | 0.867 | 0.922 | 0.911 | 15 | 90.8 | 3.4 | 82.9 | 12,115 / 694 | bridge 0.900, discovery 0.800 |
| `agentic react / 27b, think=low #2` | 0.867 | 0.922 | 0.933 | 15 | 90.3 | 3.4 | 80.9 | 12,016 / 653 | bridge 0.900, discovery 0.800 |
| `agentic react / 27b + groundedness` | 0.867 | 0.950 | 0.956 | 15 | 121.2 | 5.3 | 108.4 | 19,877 / 465 | bridge 1.000, discovery 0.600 |

Groundedness verdicts against the judge (a fail is a judged FAIL, or an incomplete multi-hop answer):

| variant | set | checked | flagged ungrounded | flagged & failed | passed check & failed | unchecked |
|---|---|---|---|---|---|---|
| `agentic react / 27b + groundedness` | answerable | 40 | 0 | 0 | 2 | 0 |
| `agentic react / 27b + groundedness` | refusals | 14 | 2 | 0 | 0 | 1 |
| `agentic react / 27b + groundedness` | multihop | 35 | 1 | 1 | 3 | 0 |
| `agentic react / 27b + groundedness` | adaptive | 15 | 1 | 0 | 2 | 0 |

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
| `oracle / 9b` | answerable pass | 39/40, 39/40, 39/40 | 39.0 | 0 |
| `oracle / 9b` | multihop complete | 32/35, 33/35, 32/35 | 32.3 | 1 |
| `oracle / 9b` | adaptive complete | 12/15, 13/15, 13/15 | 12.7 | 1 |
| `agentic react / 9b` | answerable pass | 37/40, 38/40, 37/40 | 37.3 | 1 |
| `agentic react / 9b` | refusals pass | 15/15, 15/15, 15/15 | 15.0 | 0 |
| `agentic react / 9b` | multihop complete | 28/35, 27/35, 27/35 | 27.3 | 1 |
| `agentic react / 9b` | adaptive complete | 6/15, 6/15, 6/15 | 6.0 | 0 |
| `agentic planned / 9b` | answerable pass | 36/40, 36/40, 37/40 | 36.3 | 1 |
| `agentic planned / 9b` | refusals pass | 14/15, 14/15, 15/15 | 14.3 | 1 |
| `agentic planned / 9b` | multihop complete | 26/35, 26/35, 27/35 | 26.3 | 1 |
| `agentic planned / 9b` | adaptive complete | 4/15, 6/15, 6/15 | 5.3 | 2 |
| `pipeline / 27b` | answerable pass | 37/40, 38/40, 38/40 | 37.7 | 1 |
| `pipeline / 27b` | refusals pass | 15/15, 15/15, 15/15 | 15.0 | 0 |
| `pipeline / 27b` | multihop complete | 27/35, 26/35, 26/35 | 26.3 | 1 |
| `pipeline / 27b` | adaptive complete | 7/15, 7/15, 7/15 | 7.0 | 0 |
| `agentic react / 27b` | answerable pass | 37/40, 37/40, 38/40 | 37.3 | 1 |
| `agentic react / 27b` | refusals pass | 14/15, 13/15, 14/15 | 13.7 | 1 |
| `agentic react / 27b` | multihop complete | 31/35, 32/35, 32/35 | 31.7 | 1 |
| `agentic react / 27b` | adaptive complete | 13/15, 14/15, 12/15 | 13.0 | 2 |
| `agentic react / 27b, think=low` | answerable pass | 39/40, 38/40 | 38.5 | 1 |
| `agentic react / 27b, think=low` | refusals pass | 15/15, 14/15 | 14.5 | 1 |
| `agentic react / 27b, think=low` | multihop complete | 32/35, 33/35 | 32.5 | 1 |
| `agentic react / 27b, think=low` | adaptive complete | 13/15, 13/15 | 13.0 | 0 |
