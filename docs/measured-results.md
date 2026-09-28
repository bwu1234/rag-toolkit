# Measured results (Milestone 11, pass 1)

Run with `scripts/run_matrix.py`; raw records in `data/eval/results/`, each row
carrying a fingerprint over the settings that produced it.

**Setup.** Corpus `edgar` (61 SEC filings, 3.59M chars, 4,236 chunks),
non-contextual index, `data/eval/edgar_eval_set.json` (174 span-matched
samples). Defaults: hybrid, cross-encoder, `top_k 20`, `rerank_top_k 5`,
`min_score 0.1`, no expansion. One factor at a time off that baseline.

### Where retrieval loses answers

| stage | recall | lost |
|---|---|---|
| Stage 1 ceiling (`top_k=100`, rerank filters nothing) | **0.977** | — |
| Stage 1 at shipped `top_k=20` | 0.885 | −9.2pp |
| After reranking to `rerank_top_k=5` (shipped) | **0.770** | −11.5pp |

### Main effects

| variant | hit | MRR | NDCG | Δ NDCG | s |
|---|---|---|---|---|---|
| `baseline` | 0.770 | 0.576 | 0.658 | — | 43 |
| `mode=dense` | 0.667 | 0.531 | 0.605 | −0.053 | 37 |
| `reranker=none` | 0.701 | 0.501 | 0.581 | −0.077 | 25 |
| `min_score` 0.0 / 0.1 / 0.3 | 0.770 | 0.576 | 0.658 | ±0.000 | 42 |
| `rerank_top_k=10` | 0.851 | 0.587 | 0.688 | +0.030 | 42 |
| `rerank_top_k=20` | 0.885 | 0.590 | 0.703 | +0.044 | 43 |
| `expansion=hyde` | 0.747 | 0.567 | 0.648 | −0.010 | 516 |
| `expansion=multi_query` | 0.718 | 0.556 | 0.629 | −0.029 | 506 |
| `aggregate=mean` (+multi_query) | 0.632 | 0.448 | 0.521 | −0.137 | 500 |

### Findings

- **The reranker is the bottleneck, not retrieval.** Holding `rerank_top_k=5`
  and enlarging the candidate pool does not help and slightly hurts
  (`top_k` 20 → 50 → 100 gives hit 0.770 → 0.759 → 0.759). Stage 1 hands the
  right chunk to the reranker 97.7% of the time at `top_k=100`; the
  cross-encoder fails to promote it into the top 5, and more candidates only
  add distractors it can't separate. A stronger reranker — or an LLM reranker
  behind the existing `Reranker` interface — is worth more here than any
  retrieval tuning. **This is the actionable result of the whole exercise.**
- **Hybrid earns its place**: dropping BM25 costs 10pp of hit rate, expected on
  text dense with entity names and figures.
- **The reranker still beats none** (+6.9pp hit) — it is worth keeping, it is
  just far below the ceiling stage 1 offers.
- **`min_score` is inert.** 0.0, 0.1 and 0.3 are identical on every metric. It
  was hand-tuned against the old 31-chunk corpus and now neither helps nor
  protects. Now set to 0.0; re-derive it from the new reranker's score
  distribution before turning it back on, and do not assume it is doing work.
- **Loading a reranker is not the same as using it correctly.** `predict()`
  applies a per-model default activation — `Identity` for `ms-marco-*`,
  **`Sigmoid`** for `BAAI/bge-reranker-*` — so swapping in a BGE model without
  forcing raw logits double-sigmoids every score into `[0.5, 0.73]`. Ranking
  order survives (sigmoid is monotonic), so hit rate and NDCG look *fine* while
  `min_score` and `aggregate: mean` are silently broken. `CrossEncoderReranker`
  now forces an identity activation so normalization happens in exactly one
  place, with regression tests asserting a raw logit of 0.0 maps to 0.5 and not
  0.622. This is the kind of interface bug Milestone 18's "swapping is a config
  change" claim exists to surface.
- **Expansion hurts and costs 12x the latency.** Both providers land below
  baseline. Plausible mechanism, stated as hypothesis not fact: generated
  questions already name company and period, and on a corpus of near-duplicate
  filings differing mainly in entity and period, a rewrite that drops those
  identifiers is fatal. Expansion stays off.
- **`aggregate: mean` is much worse than `max`** (−0.137 NDCG), matching the
  documented reasoning — requiring agreement across rewrites suppresses chunks
  only one (correct) rewrite liked.

### Reranker models

`min_score` pinned to 0.0 throughout, so this measures model quality rather than
threshold luck.

| model | params | pool=20 hit | NDCG | s | pool=100 hit | NDCG | s |
|---|---|---|---|---|---|---|---|
| `ms-marco-MiniLM-L-6-v2` | 22M | 0.770 | 0.658 | 44 | 0.759 ↓ | 0.647 | 77 |
| `BAAI/bge-reranker-base` | 278M | 0.828 | 0.742 | 83 | 0.845 ↑ | 0.739 | 280 |
| **`BAAI/bge-reranker-v2-m3`** | 568M | **0.874** | **0.796** | 186 | **0.908** ↑ | **0.823** | 848 |
| `Qwen3-Reranker-0.6B-seq-cls` | 596M | 0.276 | 0.155 | 419 | — | — | — |
| `Qwen3-…-seq-cls` + instruction template | 596M | 0.322 | 0.207 | 449 | — | — | — |

- **The reranker was the bottleneck, and swapping it is the single largest win
  measured in this project**: +10.4pp hit rate and +0.138 NDCG on the shipped
  config, for ~1.1s/query instead of ~0.25s. `bge-reranker-v2-m3` is now the
  default.
- **The earlier "a bigger candidate pool doesn't help" finding was a property of
  MiniLM, not of retrieval.** MiniLM got worse with more candidates
  (0.770 → 0.759); both BGE models got better (v2-m3: 0.874 → 0.908). Revisit
  `retrieval.top_k` now that reranking can use a larger pool — though pool=100
  costs 4.9s/query, which is batch-only territory.
- Against the 0.977 stage-1 ceiling: MiniLM recovers 78.8%, bge-base 86.5%,
  bge-v2-m3 92.9%.
- **Qwen3-Reranker underperforms badly through this integration and I do not
  have a working configuration for it.** Treat the numbers above as a verdict on
  *the integration*, not on the model. What was tried: the official
  `Qwen/Qwen3-Reranker-*` is generative (yes/no token logits behind an
  instruction template) and does not load as a `CrossEncoder` at all; the
  community `-seq-cls` conversion does load, but scores below using no reranker.
  Adding its instruction template helped marginally (0.276 → 0.322) and
  left-padding made no difference at all. Getting it right most likely needs a
  dedicated adapter reproducing the official generative scoring, which is real
  work, not a config change.

### Reranker models, paired re-test

**Setup.** Same corpus, index and 174 samples as above, re-run on 2026-09-26
with per-sample scores stored so each model can be compared to the shipped
config question by question (`rag/eval/paired.py`). `baseline` is the shipped
`bge-reranker-v2-m3`, `min_score 0.0`. Raw rows:
`data/eval/results/retrieval_edgar_edgar_eval_set.json`.

| comparison | Δ hit [95% CI] | W/L | McNemar p | Δ NDCG [95% CI] |
|---|---|---|---|---|
| `minilm-L6` vs shipped | −0.103 [−0.154, −0.053] | 2/20 | 0.0001 | −0.138 [−0.181, −0.094] |
| `bge-base` vs shipped | −0.046 [−0.081, −0.011] | 1/9 | 0.021 | −0.054 [−0.098, −0.009] |
| `bge-v2-m3` vs shipped | 0.000 [0.000, 0.000] | 0/0 | 1.0 | 0.000 |
| `minilm-L6` → `bge-base` | +0.057 [+0.008, +0.107] | 15/5 | 0.041 | — |

- **The earlier numbers reproduce exactly.** Every aggregate matches the table
  above to three decimals. `rr=bge-v2-m3` and `baseline` share a fingerprint
  and differ on no sample, which confirms that retrieval is deterministic and
  that the pairing is sound.
- **Both steps up the reranker ladder are real.** Going from `bge-base` to
  `bge-v2-m3` gains +4.6pp: 9 questions gained, 1 lost. Judged against the
  unpaired standard error (≈3.8pp for a difference of two independent rates,
  so a 95% interval of about ±7.5pp), that gain would have read as noise.
  Paired, it is clearly outside the noise. The default was right, and now it has a test behind it.
- **The sign test and the NDCG interval can disagree, and that is expected.**
  `bge-base`'s Δ NDCG interval excludes zero, but its sign-test p is 0.24
  (30 gains, 41 losses). The sign test only counts direction; the interval also
  weighs how far each question moved. `bge-base` loses by larger margins than it
  wins by. For binary hit rate the two tests agree.
- **Since tested:** the `pool=100` rows were re-run with per-sample scores
  after the 2026-09-26 label fixes. `bge-v2-m3 pool=100` vs. `pool=20` is
  noise: +1.1pp [−2.4, +4.7], 6W/4L, p=0.75. See
  [Label check](#label-check-of-the-generated-edgar-set-chunking-plan-phase-0-step-1).
  The Qwen rows are a verdict on a broken integration either way.

### Reading these numbers safely

- **Precision is not comparable across different `rerank_top_k`.** With ~1
  relevant span per query, precision@k is bounded near 1/k, so the ceiling moves
  with k; the apparent collapse from 0.168 to 0.060 is mostly that artifact.
- **`recall_by_k` averages only over samples that returned at least k results**,
  so when `min_score` prunes lists to different lengths the denominators differ.
  The summary `recall` column is unaffected. (Harness limitation, not a finding.)
- **Expansion is non-deterministic** — these are single runs, so treat them as
  data points. The effect sizes are large enough to act on; a 1-2pp difference
  here would not be.
- **The eval set's lexical bias inflates every absolute number.** These compare
  configurations against each other on identical questions; they are not a claim
  about the system's accuracy in the wild.
- At n=174, small differences are not reliable. The ±0.000 rows are exact ties
  (identical retrieved sets), not rounding.
- **Every result above was read against an unpaired noise floor**, the
  standard error of one hit rate (≈2.5pp at n=174). That is the wrong yardstick
  for two configs run on the same questions: the right one is the paired
  difference, which depends only on the questions the two runs disagree on and
  can be tighter or wider. None of these results stored per-sample scores, so
  none can be re-tested without a re-run. `run_matrix.py` and
  `run_answer_matrix.py` now store per-sample scores and print a paired 95% CI
  and the win/loss count (`rag/eval/paired.py`); use those for new results.

### CRAG (Milestone 10), measured

40 evenly-spaced answerable samples + 15 hard negatives, `bge-reranker-v2-m3`.

| variant | answerable pass | s | refusal pass | s |
|---|---|---|---|---|
| `crag=off` | 0.775 | 403 | **0.875** | 162 |
| `crag=on (all)` | 0.800 | 726 | **0.875** | 372 |
| `crag=grade only` | 0.850 | 491 | **0.875** | 184 |
| `crag=grade+retry` | 0.800 | 503 | **0.875** | 222 |
| `crag=groundedness only` | 0.850 | 604 | **0.875** | 277 |

- **CRAG shows no measurable benefit on this corpus and roughly doubles latency**
  (403s → 726s). It stays off by default.
- **Refusal accuracy is identical across all five variants.** The plain pipeline
  already declines correctly 14/15 times; `build_rag_prompt`'s "answer only from
  these passages" instruction is doing that work, and CRAG's grader adds nothing
  on top. This is the clearest result of the two.
- The answerable spread (0.775–0.850) is **inside the noise floor.** Measured
  directly: `crag=off` scored 0.825 on one run and 0.775 on another with an
  identical config, so run-to-run variance alone is ±2 samples at n=40. Nothing
  in that column is a finding.
- Verified the grader itself works — on a hard negative it graded out all 5
  passages and produced a correct refusal. Its output is fine; it just isn't
  needed here.
- The one genuine failure is `neg-unanswerable-comparison`: asked which company
  had the highest operating margin, the system answered "Target" from a passage
  covering **only Target**. Asserting a superlative from one data point is the
  multi-hop limitation Milestone 19 targets, and no CRAG setting catches it.

**Two measurement bugs found here, both mine, both worth remembering:**

1. **The judge failed every refusal by construction.** `_JUDGE_SYSTEM_PROMPT`
   says to FAIL an answer that "refuses to answer" — right for answerable
   questions, exactly backwards for a refusal set. Two *correct* refusals
   differing only in verbosity were graded FAIL and PASS. The first run's
   refusal column (`crag=off` 0.438, `grade only` 0.125) was measuring how
   closely refusal prose resembled the reference text, and read as "CRAG
   destroys refusals" — which was false. Fixed with `judge_for()`, which selects
   a rubric by tier: refusal samples are asked *did it decline*, with wording and
   detail explicitly irrelevant.
2. **Two hard negatives were answerable.** Both Costco `out_of_scope_section`
   negatives were removed: warehouse counts *are* in Costco's MD&A, and its
   extraction runs past MD&A into Item 9A where the KPMG auditor attestation
   sits. The system answered both correctly and was scored a failure for it.
   Off-corpus *entity* negatives were verified programmatically against corpus
   text; the section-scope ones were not — they assumed section boundaries the
   extractor doesn't guarantee. Verify a negative's answer is absent before
   trusting it.

Also worth recording as a fetcher limitation: **Costco's extraction leaks past
MD&A into Item 9A.** The digit/pipe heuristics accepted it (134k chars, the
largest 10-K MD&A in the corpus) and nothing flagged it.

### Contextual chunking (Milestone 9), measured

Same 174 samples, same `bge-reranker-v2-m3`, same everything except the index:
one built with `chunking.contextual.enabled: false`, one with it on. The
contextual build used a separate `index_dir`, because the collection name is
derived from the corpus selection alone and would otherwise have overwritten the
index being compared against.

| retrieval mode | non-contextual | contextual | Δ hit | Δ NDCG |
|---|---|---|---|---|
| `dense` | 0.718 | **0.782** | **+6.4pp** | +0.026 |
| `hybrid` (shipped) | 0.874 | **0.891** | +1.7pp | +0.010 |

- **It helps dense retrieval and barely moves hybrid.** The mechanism is
  straightforward: contextual chunking works by writing identifying terms
  (company, period) back into a chunk that never named them. BM25 already
  recovers those terms at query time whenever the query states them — and these
  questions always name company and period, by construction of the generator.
  **Contextual chunking and BM25 are largely redundant here**, solving the same
  problem at index time and query time respectively.
- **+1.7pp on the shipped hybrid config is inside the noise floor** (SE ≈ 2.5pp
  at n=174). The dense +6.4pp is more likely real. Do not read the hybrid number
  as a gain. *Caveat added later:* that SE is unpaired (see "Reading these
  numbers safely"), so this verdict is untested rather than negative. A paired
  re-run would settle it.
- **Cost: ~4 hours of index build** (4,236 chunks, one LLM call each) for a
  difference hybrid retrieval cannot reliably distinguish from zero. It stays
  **off by default**. Worth reconsidering for a dense-only deployment, or a
  corpus whose queries do not name their own entities.
- This inverts the expectation in the Milestone 9 notes, which anticipated
  contextual BM25 being half the gain. On this corpus BM25 is less a
  *beneficiary* of contextualization than a *substitute* for it.

**A bug this surfaced.** The interrupted-then-resumed contextual build finished
with 4,236 vectors against 4,172 BM25 chunks. Chroma persists on write while the
BM25 index is flushed once at the end of a run, so the killed run left 64 chunks
in the vector store that never reached the sparse one — and change detection,
keyed on the vector store alone, then called them unchanged forever. Hybrid
retrieval would have quietly searched a smaller keyword index than its vector
count implied. The skip condition now requires a chunk to be current in **both**
stores (`SparseIndex.has_chunk`), with regression tests; re-running afterwards
detected exactly those 64, took their contexts from cache at zero LLM cost, and
restored 4,236/4,236.

### Generator model: `qwen3.5:9b` vs `qwen3.8:27b` (pre-Milestone 19)

**Setup.** `edgar`, non-contextual index, shipped retrieval (hybrid,
`bge-reranker-v2-m3`, `top_k 20`, `rerank_top_k 5`), CRAG off, `think: false`
for both models. Two experiments, run from scratch scripts rather than the
matrix harness (see the judge bug below for why):

1. **Pipeline, generator swapped only** — the same 40 evenly-spaced answerable
   samples + 15 hard negatives as the CRAG run.
2. **Agentic probe** — a throwaway tool loop (one `rag_search` tool wrapping the
   shipped `Retriever`, passages numbered across calls, cap of 8 searches) on 10
   single-hop samples (disjoint from 1), 4 hard negatives, and 6 hand-written
   multi-company / multi-period questions. The multi-hop answers were also run
   through the pipeline for comparison.

The judge in both was **`gemma4:31b-mlx`, fixed**, a different model family from
the generators.

Scripts: `scripts/experiments/2026-09-generator-probe/` (a frozen record, not
maintained). Raw records: `data/eval/results/probe_2026-09_*.json`, including
every agentic query and answer behind the hand count below. The 27b pipeline
answerable run has only its total: a crash lost its per-sample detail before
per-phase checkpointing was added, and it was not rerun.

| pipeline | answerable | refusals |
|---|---|---|
| `qwen3.5:9b` | 36/40 (0.900) | 14/15 |
| `qwen3.8:27b` | 35/40 (0.875) | 15/15 |

| agentic probe | single-hop | refusals | multi-hop (judge) | multi-hop complete & correct† | searches | prompt tokens |
|---|---|---|---|---|---|---|
| `qwen3.5:9b` | 10/10 | 4/4 | 5/6 (pipeline: 3/6) | **1/5** | 25 | 64k |
| `qwen3.8:27b` | 10/10 | 2/4 | 5/6 (pipeline: 6/6) | **5/5** | 65 | 374k |

† Hand-counted over the five multi-hop questions that name their entities:
does the answer give a correct, cited figure for *every* company/period asked
about? The judge rubric passes an answer that honestly says "the passages don't
cover United", so the judge column can't see this difference. n=5 is small, so
this is a direction, not a result. The one surprising figure, Microsoft's
$443,506M lease obligations, was checked against the filing.

- **In the pipeline, the bigger model buys nothing.** 35/40 vs 36/40 and 15/15
  vs 14/15 are inside the noise floor. Three of the 9b's four answerable
  failures were **retrieval misses**: the top 5 held other companies' filings
  and the 9b correctly said so. No generator fixes that in a single pass. The
  fourth was a real generation error (mixed up the UAL and LUV tickers).
- **As an agent, the two models behave differently, not just faster or
  slower.** The 9b decomposes perfectly (one search per company, entity
  coverage 1.0) but **never searches a second time**. Nearly every turn was
  exactly two LLM calls: fan out, then answer. When a search missed, it
  reported the gap and stopped. The 27b **re-searches after a miss** (4–8
  searches to find United's FY2025 revenue, Microsoft's lease table, Pfizer's
  IRA statement) and picks the *most recent* filing when asked for one. The 9b
  cited Apple's FY2024 lease figure where FY2025 was asked for.
- **The 27b's agentic failures are all harness-fixable, and the prototype
  surfaced them:**
  - *Cap exhaustion without an answer.* Twice it spent all 8 searches and
    returned either nothing or "Let me try to find…". The loop needs a forced
    synthesis turn at the cap.
  - *Identical-query loops.* It sent the same Tesla query five times in a row.
    Repeated queries should be refused without spending a search.
  - *Parametric leakage.* On the Apple FY2015 negative it declined correctly,
    then added the real figure from memory "for reference". A bigger model
    knows more, so it has more to leak. The judge rightly failed it.
  - *Guessed figures in queries* ("United 2025 revenue 51 billion 52 billion").
    Harmless to BM25 here, but it shows the model searching for its prior.
- **Cost.** 2.6× the searches, 5.9× the prompt tokens (history plus every
  passage re-sent each turn), ~1.8× slower decode (26 vs 47 tok/s measured solo
  on an M2 Max 64GB). Hard multi-hop questions took 2–6 minutes each with the
  27b. **Wall-clock numbers from this run are inflated and not quoted here**: an
  unrelated Ollama client ran concurrently from partway through the 27b pipeline
  run onward, and forced a model reload that dropped one request. Token counts
  are unaffected.
- **Decision: `llm.model` stays `qwen3.5:9b-mlx`.** The 27b is the model the
  Milestone 19 agent should be designed around, not a drop-in upgrade. See
  [Milestone 19 plan](milestone-19-plan.md).

**The measurement bug this found:** `run_answer_matrix.py` and `answer_eval`
judge with the **generating** model (`get_llm_client(config.llm)`). Any model
comparison through them swaps the judge too, so the numbers are not comparable.
It also colours earlier numbers: the 9b scored 0.775–0.825 judging itself in the
CRAG runs and 0.900 on the same 40 samples under the gemma judge. Treat the CRAG
table as internally comparable only. Fixing it (a separately configured judge)
is step 0 of the Milestone 19 plan.
**Fixed:** `eval.judge` / `--judge-model`, with results filed per judge. See the
next section.

### Pipeline baseline under a fixed judge (Milestone 19, phase 0)

**Setup.** `edgar`, non-contextual index, shipped retrieval (hybrid,
`bge-reranker-v2-m3`, `top_k 20`, `rerank_top_k 5`), CRAG off, generator
`qwen3.5:9b-mlx`. **Judge `gemma4:31b-mlx` at temperature 0**, set through the
new `eval.judge` config rather than a scratch script. Answerable is the same 40
evenly-spaced samples as the CRAG and generator-swap runs. Multi-hop is the new
`data/eval/edgar_multihop_set.json`: 34 questions (21 cross-period, 8
cross-company, 5 aggregation), each built from 2–3 verified single-hop samples
by `scripts/build_multihop_set.py`, and judged part by part. A question is
**complete** only if every company or period and the requested conclusion
pass. Results: `data/eval/results/answer_edgar__judge-gemma4-31b-mlx.json`.

| set | result |
|---|---|
| answerable | 35/40 (0.875) |
| refusals | 14/15 (0.933); the miss is `neg-unanswerable-comparison` |
| multi-hop, complete & correct | **15/34 (0.441)**: cross-period 13/21, cross-company 2/8, aggregation 0/5 |
| multi-hop, mean completeness | 0.588 |
| multi-hop, evidence recall | **0.583** |
| latency | ~10–16 s/turn, always 1 retrieval round |

- **This reproduces the probe.** The answerable and refusal numbers are within
  one sample of the generator-swap run under the same judge (36/40, 14/15). So
  the harness now gives the same answer that experiment's hand-built script did.
- **The pipeline's multi-hop ceiling is retrieval, not generation.** Evidence
  recall is 0.583: one search of 5 passages usually covers one of the entities
  asked about and not the other. It falls with the number of entities, and
  aggregation questions never complete. Most incomplete answers correctly say
  a company's data "is not in the passages". That is honest, but it is not an
  answer. This is the gap Milestone 19 exists to close, and evidence recall is
  the metric that shows whether searching again helps.
- **Among the answers that do find both pieces of evidence, the 9b's usual
  error is the comparison, not the figures.** In both runs it stated both
  figures correctly and then reversed the conclusion ("decreased from $15.6B to
  $16.8B"; "Walmart is larger" followed by "$7.3B is larger than $5.9B"). Those
  errors are why run-to-run results differ.

**Noise.** The multi-hop set was run twice. The first run used a spec that
also scored conclusions two questions never asked for, which has since been
fixed. Evidence recall was identical across runs (retrieval is deterministic).
Complete & correct moved by ±1 on two questions because of 9b sampling at
temperature 0.2. Treat **±2 questions (~6 pp) on complete & correct** as noise.
Phase 4 should repeat each variant rather than read one run.

**Judge check.** All 15 answers judged complete were read by hand. One is a
false pass that the temperature-0 judge made in both runs: `mh-msft-div` passed
the FY2025 part even though the answer says FY2025 isn't in the passages and
attaches $24.7B to the wrong period. So the true rate is ~14/34. The failures
spot-checked were all real errors.

**Cost, from a re-run with per-turn metering (2026-09-27).** Same config and
judge (temperature 0), multi-hop set only, `crag=off`. This is what the agent's
extra searches will be set against:

| per turn | pipeline |
|---|---|
| LLM calls | 1.0 (generation only; the judge's calls are excluded) |
| LLM time | 9.3 s of 11.2 s wall-clock |
| prompt / generated tokens | 1,680 / 184 (all 34 turns reported counts) |

The re-run's scores sit inside the noise band above: 14/34 complete (was 15),
completeness 0.566, evidence recall 0.598. The recall change isn't retrieval
drift. The label fixes recorded [below](#label-check-of-the-generated-edgar-set-chunking-plan-phase-0-step-1)
landed after this baseline and gave `mh-ual-unrealized` an alternative span
for United's 2024 figure (the FY2025 10-K restates it), so a passage the
pipeline already retrieved now counts as evidence: 0.5 → 1.0 on that sample,
0.583 → 0.598 overall. Three samples flipped on completeness: `mh-aapl-lease-yoy`
and `mh-msft-div` to fail, `mh-auth-mrk-wmt` to pass. `mh-msft-div` was the
known false pass, so 14/34 matches the hand-checked rate. The phase-4
comparison should use the re-run's row, which is the one in the results file
now.

Cap-hit rate and searches per turn are defined only for the agent (the
pipeline always does one retrieval round), and arrive with phase 3.

### Label check of the generated EDGAR set (chunking plan, Phase 0 step 1)

Run on 2026-09-26 at the shipped config
(`retrieval_eval -v --eval-set data/eval/edgar_eval_set.json --corpus edgar`).
It reproduces the recorded baseline exactly: hit 0.874, NDCG 0.796,
22 misses of 174. `unmatchable_spans` is 0, as expected: the set was drafted
from these same fixed chunks, so every span fits one by construction.

Each miss was read against the top 5 it retrieved, plus a corpus-wide search
for other chunks stating the span's figures:

| class | n | samples |
|---|---|---|
| **Label defect: a retrieved chunk states the asked fact** | 4 | MSFT Q3 FY26 tax rate (rank 2 states "19%", span picked a weaker sentence); JNJ FY24 interest (rank 2, FY25 10-K restates it); UAL FY24 unrealized losses (rank 1, FY25 10-K restates it); UAL Q3 FY25 cash (rank 1 is the adjacent chunk; the span also truncates "unrestricted cash, cash equivalents and short-term investments" to "unrestricted cash") |
| **Question defect: ambiguous as drafted** | 3 | UAL "period ended Sep 30" (3 or 9 months; rank 1 answers the 3-month reading); WMT "segment operating income" (names no segment; $0.2B is Sam's Club, rank 1 is Walmart U.S.); MRK "certain other items" (source jargon, meaningless without context) |
| Span split across chunks | 0 | |
| Genuine: right company, wrong period on top | 4 | AAPL Q3 buyback, COST Q1 cash flow, DAL FY24 capex, LUV FY25 salaries |
| Genuine: other companies' chunks win | 4 | AAPL Q2 buyback, LUV FY25 interest expense, NVDA FY26 buybacks, NVDA Q3 opex |
| Genuine: right filing, wrong chunk | 7 | COST impairments, LUV Q3 CASM, NVDA gross margin, NVDA supply chain, TGT impairments, WMT tax rate, WMT tariff refunds |

**Label errors aren't rare: 7 of 22 misses (32%), 4% of the set.** Scoring
the four label defects as hits would put the shipped hit rate near 0.897. That
is roughly the size of the effects later phases will be judged on, so the
drafting process has to change before Phase 0 step 2 writes new tiers:

- The reviewer searches the corpus for every other chunk stating the asked
  fact. Paraphrased restatements survive `generate_eval_set.py`'s
  verbatim-duplicate filter, notably a prior year's figure in the next year's
  filing. Add each one as a span, or reject the question.
- The span is the sentence that states the asked fact, quoted whole enough not
  to change its meaning.
- The question pins the entity (segment, not just company) and the period
  granularity (three months vs. year to date), and doesn't quote source
  jargon.

No miss was table-shaped, so this check gives no reason to start Phases 4–5
early. It classifies the misses the generated set can produce; per the plan it
doesn't decide what to build.

**The 7 defects were fixed on 2026-09-26** under the freeze rule. Each fixed
sample carries a `label_fixes` entry in `edgar_eval_set.json` with the old
values and the reason. Sample ids are unchanged, so paired comparisons still
line up. The two restated-fact samples (JNJ, UAL FY24) needed span
`alternatives`: other quotes of the same fact, any one of which satisfies the
span. A second span wouldn't work, because spans on a sample are conjunctive
for recall and answer evidence. Their later filing was added to
`expected_doc_ids`. The other five got a pinned query, a corrected span, or
both. After the fixes, `unmatchable_spans` is still 0, and every quote occurs
only in its expected filings.

**Re-run on the fixed labels** (both retrieval matrices, same indexes;
`index-report` confirmed both in sync first). These supersede the hit and
NDCG figures in the sections above, which stay as the record of what was
measured at the time:

| | before | after |
|---|---|---|
| shipped baseline (hybrid, `bge-v2-m3`, `top_k 20`) | 0.874 / 0.796 | **0.908 / 0.821** |
| `rr=bge-v2-m3 pool=100` | 0.908 (no CI) | 0.919, Δ **+0.011 [−0.024, +0.047]**, 6W/4L, p=0.75: noise |
| contextual index, hybrid | 0.891 | 0.914, Δ vs. plain **+0.006 [−0.032, +0.043]**, 6W/5L, p=1: noise |
| contextual index, dense | 0.782 | 0.799 |

(hit / NDCG; Δ is a paired hit-rate difference with 95% CI.) Exactly the 7
fixed samples changed at the baseline, and 6 flipped to hits. 4 flipped
because of corrected labels, and 2 because their questions became answerable
as asked (UAL Q3, MRK). WMT still misses with the segment named, so it's a
genuine retrieval miss. Every reranker verdict holds: the paired deltas
against the baseline moved by at most 0.02. Raw records:
`data/eval/results/retrieval_edgar_edgar_eval_set.json` and
`data/eval/results_contextual/`.

### Baselines on the three question sets (chunking plan, Phase 0 step 3)

**Setup.** Run on 2026-09-27 at commit `574fb8a`. `edgar`, non-contextual
index (`index-report`: in sync, 4,236 chunks), shipped retrieval (hybrid,
`bge-reranker-v2-m3`, `top_k 20`, `rerank_top_k 5`, `min_score 0.0`), CRAG
off, generator `qwen3.5:9b-mlx`, judge `gemma4:31b-mlx` at temperature 0.
Every set was run in full: the generated set is 174 questions on the fixed
labels, not the 40-sample subset. Retrieval used `run_matrix.py --variant
baseline`, fingerprint `baseline` in each file. Answers used `run_answer_matrix.py
--variant crag=off --sets answerable,period,underspecified --limit 0
--results-dir data/eval/results_chunking`. Phases 1–3 of the
[chunking plan](chunking-indexing-plan.md) are measured against these rows.

| set | n | retrieval hit [95% CI] | NDCG | answer pass [95% CI] | answer fails: retrieval / generation |
|---|---|---|---|---|---|
| generated | 174 | 0.908 [0.856, 0.943] | 0.821 | 0.862 [0.803, 0.906] | 15 / 9 |
| `period` | 55 | **0.582** [0.450, 0.703] | 0.393 | 0.782 [0.656, 0.871] | 12 / 0 |
| `underspecified` | 118 | 0.652 [0.563, 0.732] | 0.562 | 0.610 [0.520, 0.693] | 37 / 9 |
| · `implicit` | 54 | 0.833 [0.713, 0.910] | 0.741 | 0.722 [0.591, 0.824] | |
| · `paraphrase` | 64 | **0.500** [0.381, 0.619] | 0.410 | 0.516 [0.396, 0.634] | |

The bracketed intervals are 95% Wilson intervals on each rate alone, which is
each tier's noise floor. A later variant is judged by its paired Δ against
these rows, not by whether its rate leaves this interval. `unmatchable_spans`
is 0 on all three sets. The tiers are reported apart from the generated set
and never averaged into it. `underspecified`'s two kinds are reported
separately too, because they fail differently.

- **The generated set reproduces exactly.** The retrieval re-run gave
  0.908 / 0.821, identical to the post-label-fix row above. Answer pass is
  150/174, which agrees with the 40-sample run under the same judge (35/40,
  0.875).
- **`period` is the weakest retrieval tier.** The right filing's copy
  reaches the top 5 for 32 of 55 questions. An ad hoc script (not committed)
  re-ran the tier's retrieval and classified each result against the filings
  in `competing_doc_ids`:

  | outcome | n |
  |---|---|
  | hit, and no copy from the other filing ranks above it | 17 |
  | hit, but the other filing's identical copy ranked above it | 15 |
  | miss: only the other filing's copy is in the top 5 | 10 |
  | miss: neither filing's copy is in the top 5 | 13 |

  In 25 of 55, the same-text copy from the wrong period outranks the right
  one. Nothing in the chunk text can break that tie, which is what Phase 3's
  period filter is for. The other 13 misses are a different problem: the
  paragraph didn't reach the top 5 at all.
- **Answer pass on `period` can't judge Phase 3.** The competing paragraph is
  word-for-word the same, so an answer built from the wrong filing is still
  correct: 11 of the 23 evidence misses passed anyway. Pass (0.782) sits
  above hit (0.582) for that reason. Phase 3 is judged on `period`'s
  retrieval hit and NDCG under `span_and_document`, and answer pass is only a
  check that nothing broke.
- **`paraphrase` halves retrieval.** It keeps company and period, and only
  rewords the question away from the span. Hit falls to 0.500 from 0.908 on
  the questions it was drawn from. This is the first measurement of the
  tier that contextual chunking and query expansion were designed for, which
  both measured as noise on the generated set. Re-measure both on this tier
  before treating either verdict as general.
- **`implicit` costs less than `paraphrase`.** Naming a company through a
  product or description ("the iPhone maker") drops hit to 0.833. The
  question keeps the period and most of the span's wording, so BM25 still
  gets purchase on the rest of the question.
- **Failures on the tiers are retrieval, not generation.** Of
  `underspecified`'s 46 failures, 37 had no gold span in the prompt. That is
  the stage Phases 1–3 change.

**Not checked.** The single-hop sets store verdicts but not the answers or
the judge's replies (the multi-hop set stores both), so no verdict here has
been read by hand. The judge's earlier false-pass rate on multi-hop, about 1
in 15, is the only estimate available. Also not measured: run-to-run noise
from the generator's temperature 0.2. The multi-hop re-run put it at about ±2
questions on 34, and a single answer run on the tiers is one draw.

### Query instruction for the embedder (chunking plan, Phase 1)

**Setup.** Run on 2026-09-27, re-run on the merged Phase 0 step 3 harness
(the re-run reproduced every earlier row exactly). `edgar` corpus, the
shipped plain index (4,236 chunks, `index-report` in sync),
`qwen3-embedding:0.6b` via Ollama (Q8_0). All three sets, against the
[step 3 baselines](#baselines-on-the-three-question-sets-chunking-plan-phase-0-step-3):
the generated set (174), `underspecified` (118) and `period` (55). The
variants set `embedding.query_instruction` to the Qwen3-Embedding model
card's default retrieval task, "Given a web search query, retrieve relevant
passages that answer the query", sent as `Instruct: {task}\nQuery:{query}`.
Documents are unchanged, so no reindex. Each pair differs only in the
instruction, at three settings: shipped (hybrid, `bge-v2-m3`, `top_k 20`,
`rerank_top_k 5`), the stage-1 ceiling (`top_k` and `rerank_top_k` both 20),
and dense-only.

| set | pair | hit, no instr. → instr. | Δ hit [95% CI], W/L, p | Δ NDCG [95% CI] |
|---|---|---|---|---|
| generated | shipped | 0.908 → 0.902 | −0.006 [−0.025, +0.014], 1W/2L, p=1 | −0.015 [−0.032, +0.003] |
| generated | stage-1 ceiling | 0.931 → 0.920 | −0.011 [−0.027, +0.004], 0W/2L, p=0.5 | −0.017 [−0.033, +0.000] |
| generated | dense | 0.741 → 0.718 | −0.023 [−0.045, −0.001]\*, 0W/4L, p=0.12 | −0.021 [−0.043, +0.000] |
| `underspecified` | shipped | 0.652 → 0.619 | −0.034 [−0.074, +0.006], 1W/5L, p=0.22 | −0.011 [−0.041, +0.020] |
| `underspecified` | stage-1 ceiling | 0.686 → 0.644 | −0.042 [−0.086, +0.001], 1W/6L, p=0.12 | −0.014 [−0.045, +0.017] |
| `underspecified` | dense | 0.619 → 0.576 | −0.042 [−0.086, +0.001], 1W/6L, p=0.12 | −0.022 [−0.054, +0.009] |
| `period` | shipped | 0.582 → 0.545 | −0.036 [−0.086, +0.014], 0W/2L, p=0.5 | −0.024 [−0.062, +0.014] |
| `period` | stage-1 ceiling | 0.709 → 0.691 | −0.018 [−0.054, +0.017], 0W/1L, p=1 | −0.017 [−0.053, +0.019] |
| `period` | dense | 0.509 → 0.455 | −0.055 [−0.134, +0.025], 1W/4L, p=0.38 | −0.025 [−0.082, +0.033] |

`underspecified` by kind, at the shipped config: `implicit` hit is unchanged
(0.833, 0W/0L; NDCG 0.741 → 0.754). `paraphrase` goes 0.500 → 0.438,
−0.062 [−0.137, +0.012], 1W/5L, p=0.22. The per-kind rows for the other two
pairs are in the results file's by-kind table. Its Δ column is against
`baseline`, not against each variant's own no-instruction pair, so read the
ceiling and dense deltas from the table above.

**Findings**

- **The instruction stays off.** No pair's hit-rate interval excludes zero
  in the instruction's favour, which was the bar for making it the default.
  Every point estimate but one (`implicit` NDCG) is negative.
- **It leans toward hurting, but that isn't shown either.** The one interval
  that excludes zero (dense, generated set) rests on 4 discordant questions,
  all losses, with McNemar p=0.12. Across all nine pairs the flips are 5 wins
  against 32 losses. That's consistent, but the pairs share questions, so the
  tally isn't independent evidence.
- **Hybrid retrieval doesn't hide a dense gain.** The dense-only pair was run
  to check exactly that, and it's the most negative on every set.
- **The loss is in stage 1, not the reranker.** At the stage-1 ceiling, where
  nothing is filtered, the instruction still loses questions. So it moves
  answer chunks *down* the dense ranking, the opposite of what this plan needs
  (answers ranked 21st–100th).
- **`paraphrase`, the headroom step 3 pointed Phases 1–2 at, is where it
  loses most** (−0.062). The instruction doesn't help a reworded question
  find its span.
- **No answer-side run.** Retrieval didn't clear noise, and the instruction
  changes nothing but the query vector, so there's no generation effect to
  measure separately.
- **Why the card's claimed 1–5% gain doesn't show here is open.** Candidates:
  the web-search task doesn't fit questions about financial filings; the
  Q8_0 build or Ollama's pooling (its template is a bare `{{ .Prompt }}` and
  doesn't show whether it appends the EOS token the card's reference code
  adds); or the corpus. None was tested. A task written for this corpus would
  have to be chosen without looking at these sets' misses, or it's tuned to
  them.
- **Phase 1b inherits `null`**, per the plan: the 4b and 8b embedders are
  compared without the instruction. They're different checkpoints, so one
  pair with the instruction there is cheap and worth running once, rather
  than assuming this result carries over.

Raw records: `data/eval/results/retrieval_edgar_edgar_{eval,underspecified,period}_set.json`.

### Not yet measured

- `retrieval.top_k` above 20 with the new reranker: `bge-v2-m3` gains from a
  larger pool where MiniLM lost, and only `top_k` 20 and 100 were measured.
- Cross-corpus interference (`--corpus baseline --corpus edgar` pooled vs
  isolated) — the registry supports it, no numbers taken.
- Anything on the `baseline` corpus, which is retained as a control and has not
  been re-measured since the reranker change.
- **Answer-side re-runs after the 2026-09-26 label fixes, partly done.** The
  full 174-question pipeline baseline on the fixed labels is now recorded
  ([above](#baselines-on-the-three-question-sets-chunking-plan-phase-0-step-3),
  `data/eval/results_chunking/`). Still on the old labels: the CRAG answer
  matrix, and the 40-sample `crag=off` answerable row in
  `data/eval/results/answer_edgar__judge-gemma4-31b-mlx.json`, which
  Milestone 19 phase 4 pairs against. That subsample includes one fixed
  sample (MRK). Re-run them before comparing new answer results against them.
- `retrieval.top_k` between 20 and 100 with `bge-v2-m3`. 100 measured as noise
  against 20 (see the label-check re-run); intermediate values untested.
