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
| `pipeline / 27b #1` | 0.950 | -0.025 [-0.074, +0.024] 0W/1L p=1 | 0 / 2 | 40 | 1132 | 1.000 | 15 | 500 |
| `pipeline / 27b #2` | 0.950 | -0.025 [-0.074, +0.024] 0W/1L p=1 | 0 / 2 | 40 | 1253 | 1.000 | 15 | 540 |
| `pipeline / 27b #3` | 0.975 | +0.000 [+0.000, +0.000] 0W/0L p=1 | 0 / 1 | 40 | 1248 | 1.000 | 15 | 541 |
| `agentic react / 27b #1` | 0.950 | -0.025 [-0.074, +0.024] 0W/1L p=1 | 0 / 2 | 40 | 1806 | 1.000 | 15 | 2682 |
| `agentic react / 27b #2` | 0.975 | +0.000 [-0.070, +0.070] 1W/1L p=1 | 0 / 1 | 40 | 1802 | 0.933 | 15 | 2550 |
| `agentic react / 27b #3` | 0.950 | -0.025 [-0.074, +0.024] 0W/1L p=1 | 0 / 2 | 40 | 1862 | 1.000 | 15 | 2696 |
| `agentic react / 27b, think=low #1` | 0.950 | -0.025 [-0.074, +0.024] 0W/1L p=1 | 1 / 1 | 40 | 1902 | 0.933 | 15 | 1820 |
| `agentic react / 27b, think=low #2` | 0.950 | -0.025 [-0.074, +0.024] 0W/1L p=1 | 1 / 1 | 40 | 2037 | 1.000 | 15 | 1909 |
| `agentic react / 27b + groundedness` | 0.950 | -0.025 [-0.074, +0.024] 0W/1L p=1 | 0 / 2 | 40 | 2904 | 0.933 | 15 | 3114 |

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
| `pipeline / 27b #1` | 0.743 | 0.819 | 0.791 | 35 | 30.3 | 1.0 | 25.6 | 1,649 / 146 |
| `pipeline / 27b #2` | 0.743 | 0.819 | 0.791 | 35 | 31.1 | 1.0 | 25.7 | 1,649 / 151 |
| `pipeline / 27b #3` | 0.743 | 0.819 | 0.791 | 35 | 31.5 | 1.0 | 26.5 | 1,649 / 161 |
| `agentic react / 27b #1` | 0.943 | 0.962 | 0.985 | 35 | 76.0 | 2.9 | 65.5 | 10,377 / 398 |
| `agentic react / 27b #2` | 0.914 | 0.952 | 0.991 | 35 | 69.2 | 2.6 | 61.3 | 9,008 / 368 |
| `agentic react / 27b #3` | 0.943 | 0.971 | 0.956 | 35 | 76.2 | 2.7 | 65.8 | 9,408 / 381 |
| `agentic react / 27b, think=low #1` | 0.943 | 0.971 | 0.985 | 35 | 72.2 | 2.2 | 63.7 | 6,818 / 626 |
| `agentic react / 27b, think=low #2` | 0.943 | 0.962 | 0.985 | 35 | 79.9 | 2.2 | 68.7 | 6,978 / 621 |
| `agentic react / 27b + groundedness` | 0.943 | 0.962 | 0.962 | 35 | 102.1 | 3.7 | 88.5 | 13,522 / 392 |

| variant | adaptive complete | completeness | evidence recall | n | s/turn | LLM calls/turn | LLM s/turn | prompt / gen tokens/turn | complete by kind |
|---|---|---|---|---|---|---|---|---|---|
| `pipeline / 9b #1` | 0.267 | 0.480 | 0.644 | 15 | 12.0 | 1.0 | 10.0 | 1,632 / 238 | bridge 0.400, discovery 0.000 |
| `pipeline / 9b #2` | 0.267 | 0.458 | 0.644 | 15 | 9.1 | 1.0 | 7.2 | 1,632 / 261 | bridge 0.400, discovery 0.000 |
| `pipeline / 9b #3` | 0.333 | 0.480 | 0.644 | 15 | 8.8 | 1.0 | 6.7 | 1,632 / 261 | bridge 0.500, discovery 0.000 |
| `oracle / 9b #1` | 0.867 | 0.930 | 1.000 | 15 | 5.6 | 1.0 | 5.6 | 913 / 122 | bridge 0.900, discovery 0.800 |
| `oracle / 9b #2` | 0.800 | 0.891 | 1.000 | 15 | 2.9 | 1.0 | 2.9 | 913 / 124 | bridge 0.800, discovery 0.800 |
| `oracle / 9b #3` | 0.800 | 0.897 | 1.000 | 15 | 3.0 | 1.0 | 3.0 | 913 / 133 | bridge 0.900, discovery 0.600 |
| `agentic react / 9b #1` | 0.200 | 0.436 | 0.533 | 15 | 17.3 | 2.1 | 13.2 | 4,605 / 227 | bridge 0.300, discovery 0.000 |
| `agentic react / 9b #2` | 0.133 | 0.468 | 0.567 | 15 | 15.3 | 2.3 | 11.8 | 6,297 / 274 | bridge 0.200, discovery 0.000 |
| `agentic react / 9b #3` | 0.133 | 0.441 | 0.522 | 15 | 11.9 | 2.1 | 8.8 | 4,566 / 244 | bridge 0.200, discovery 0.000 |
| `agentic planned / 9b #1` | 0.200 | 0.452 | 0.556 | 15 | 17.7 | 2.0 | 14.4 | 3,400 / 305 | bridge 0.300, discovery 0.000 |
| `agentic planned / 9b #2` | 0.200 | 0.452 | 0.556 | 15 | 11.6 | 2.0 | 8.5 | 3,428 / 255 | bridge 0.300, discovery 0.000 |
| `agentic planned / 9b #3` | 0.200 | 0.436 | 0.556 | 15 | 12.4 | 2.0 | 9.5 | 3,520 / 275 | bridge 0.300, discovery 0.000 |
| `pipeline / 27b #1` | 0.400 | 0.541 | 0.644 | 15 | 45.6 | 1.0 | 39.6 | 1,632 / 400 | bridge 0.600, discovery 0.000 |
| `pipeline / 27b #2` | 0.400 | 0.541 | 0.644 | 15 | 42.4 | 1.0 | 35.4 | 1,632 / 304 | bridge 0.600, discovery 0.000 |
| `pipeline / 27b #3` | 0.400 | 0.541 | 0.644 | 15 | 36.4 | 1.0 | 29.6 | 1,632 / 212 | bridge 0.600, discovery 0.000 |
| `agentic react / 27b #1` | 0.800 | 0.889 | 0.911 | 15 | 109.3 | 3.9 | 95.8 | 15,903 / 452 | bridge 0.900, discovery 0.600 |
| `agentic react / 27b #2` | 0.733 | 0.856 | 0.922 | 15 | 111.2 | 4.1 | 97.1 | 16,707 / 417 | bridge 0.900, discovery 0.400 |
| `agentic react / 27b #3` | 0.800 | 0.922 | 0.867 | 15 | 103.3 | 3.9 | 88.8 | 15,055 / 404 | bridge 0.900, discovery 0.600 |
| `agentic react / 27b, think=low #1` | 0.867 | 0.922 | 0.956 | 15 | 111.8 | 3.3 | 95.1 | 13,002 / 750 | bridge 0.900, discovery 0.800 |
| `agentic react / 27b, think=low #2` | 0.733 | 0.842 | 0.889 | 15 | 104.8 | 3.3 | 88.3 | 13,187 / 719 | bridge 0.800, discovery 0.600 |
| `agentic react / 27b + groundedness` | 0.800 | 0.883 | 0.856 | 15 | 127.2 | 4.9 | 113.4 | 19,754 / 433 | bridge 1.000, discovery 0.400 |

Groundedness verdicts against the judge (a fail is a judged FAIL, or an incomplete multi-hop answer):

| variant | set | checked | flagged ungrounded | flagged & failed | passed check & failed | unchecked |
|---|---|---|---|---|---|---|
| `agentic react / 27b + groundedness` | answerable | 40 | 2 | 1 | 1 | 0 |
| `agentic react / 27b + groundedness` | refusals | 14 | 1 | 1 | 0 | 1 |
| `agentic react / 27b + groundedness` | multihop | 35 | 4 | 0 | 2 | 0 |
| `agentic react / 27b + groundedness` | adaptive | 15 | 2 | 1 | 2 | 0 |

Spread across repeats (counts per run; a difference between variants smaller than the range is within run-to-run noise):

| variant | set | per run | mean | range |
|---|---|---|---|---|
| `pipeline / 9b` | answerable pass | 39/40, 38/40, 39/40 | 38.7 | 1 |
| `pipeline / 9b` | refusals pass | 14/15, 14/15, 14/15 | 14.0 | 0 |
| `pipeline / 9b` | multihop complete | 24/35, 25/35, 25/35 | 24.7 | 1 |
| `pipeline / 9b` | adaptive complete | 4/15, 4/15, 5/15 | 4.3 | 1 |
| `oracle / 9b` | answerable pass | 40/40, 40/40, 40/40 | 40.0 | 0 |
| `oracle / 9b` | multihop complete | 34/35, 34/35, 33/35 | 33.7 | 1 |
| `oracle / 9b` | adaptive complete | 13/15, 12/15, 12/15 | 12.3 | 1 |
| `agentic react / 9b` | answerable pass | 37/40, 38/40, 36/40 | 37.0 | 2 |
| `agentic react / 9b` | refusals pass | 15/15, 15/15, 15/15 | 15.0 | 0 |
| `agentic react / 9b` | multihop complete | 25/35, 24/35, 27/35 | 25.3 | 3 |
| `agentic react / 9b` | adaptive complete | 3/15, 2/15, 2/15 | 2.3 | 1 |
| `agentic planned / 9b` | answerable pass | 37/40, 37/40, 37/40 | 37.0 | 0 |
| `agentic planned / 9b` | refusals pass | 15/15, 15/15, 15/15 | 15.0 | 0 |
| `agentic planned / 9b` | multihop complete | 23/35, 28/35, 25/35 | 25.3 | 5 |
| `agentic planned / 9b` | adaptive complete | 3/15, 3/15, 3/15 | 3.0 | 0 |
| `pipeline / 27b` | answerable pass | 38/40, 38/40, 39/40 | 38.3 | 1 |
| `pipeline / 27b` | refusals pass | 15/15, 15/15, 15/15 | 15.0 | 0 |
| `pipeline / 27b` | multihop complete | 26/35, 26/35, 26/35 | 26.0 | 0 |
| `pipeline / 27b` | adaptive complete | 6/15, 6/15, 6/15 | 6.0 | 0 |
| `agentic react / 27b` | answerable pass | 38/40, 39/40, 38/40 | 38.3 | 1 |
| `agentic react / 27b` | refusals pass | 15/15, 14/15, 15/15 | 14.7 | 1 |
| `agentic react / 27b` | multihop complete | 33/35, 32/35, 33/35 | 32.7 | 1 |
| `agentic react / 27b` | adaptive complete | 12/15, 11/15, 12/15 | 11.7 | 1 |
| `agentic react / 27b, think=low` | answerable pass | 38/40, 38/40 | 38.0 | 0 |
| `agentic react / 27b, think=low` | refusals pass | 14/15, 15/15 | 14.5 | 1 |
| `agentic react / 27b, think=low` | multihop complete | 33/35, 33/35 | 33.0 | 0 |
| `agentic react / 27b, think=low` | adaptive complete | 13/15, 11/15 | 12.0 | 2 |
