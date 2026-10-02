# Measured results

Run with `scripts/run_matrix.py` (retrieval) and `scripts/run_answer_matrix.py`
(answers); raw records in `data/eval/results/` and
`data/eval/results_chunking/`, each row carrying a fingerprint over the
settings that produced it. Sections are in the order they were run, and each
states its own setup. The shipped config has changed several times since the
first one, so read a section's numbers against its setup, not against today's
defaults.

## Scope of the evidence

These are development benchmarks for specific EDGAR tasks and configurations.
They support diagnosis and regression checks; they do not establish general
RAG quality. The main generated set is nearly saturated at the current config
(170/174 retrieval hits), and 173/174 questions have a single expected span.
Questions were drafted from existing chunks with explicit company and period
names. The underspecified and multi-hop sets reuse facts from that set.

Repeated configuration selection on these questions, including the routing
gate, makes the results in-sample. Paired intervals and McNemar tests do not
remove that selection bias or account for dependencies among related facts
and filings. Fresh held-out questions are needed for generalization claims.

Answer pass means agreement with a reference under an LLM judge. The current
judge does not inspect retrieved evidence or validate citations; a correct
answer from the wrong period's identical paragraph can pass. Keep answer
correctness and retrieval provenance separate when interpreting the tables.

Historical shorthand such as "noise" or "measured off" means a benefit was
not demonstrated sufficiently to enable the feature under that experiment's
conditions. It does not prove no effect elsewhere. In particular, CRAG and
expansion have not been re-evaluated on the harder underspecified tier.
The BEIR sections (public benchmarks plan, phases 3 and 4) are a second
evidence source with independent labels, scoped to those three tasks. The [evaluation rigor plan](evaluation-rigor-plan.md) specifies the next
coverage, holdout, and grading work. Existing tables remain the historical
record; no new runs are reported here.

### Current shipped baseline

What later changes pair against, as of 2026-10-02: `edgar_md`, `structured`
chunker (chunking plan Phase 5), header on and `reranker.include_header`,
hybrid, `bge-reranker-v2-m3`, `top_k 20`, `rerank_top_k 5`, `min_score 0.0`,
no expansion, no routing, generator `qwen3.5:9b-mlx`, judge `gemma4:31b-mlx`.

| set | n | retrieval hit | NDCG | answer pass |
|---|---|---|---|---|
| generated | 174 | 0.977 | 0.932 | 0.948 (one run, all 174) |
| answerable, evenly spaced | 40 | — | — | 37.3 / 40 (3 runs) |
| refusals | 15 | — | — | 15.0 / 15 |
| multi-hop complete | 35 | — | — | 24.7 / 35, evidence recall 0.853 |
| adaptive complete | 15 | — | — | 6.7 / 15, evidence recall 0.689 |
| `period` | 55 | 0.964 | 0.891 | 54.0 / 55 |
| `underspecified` | 118 | 0.831 | 0.738 | 89.7 / 118 |
| `table` | 95 | 0.979 | 0.859 | 89.7 / 95 |

Sources:

- *Answer rows:* the [`edgar_md` pipeline baseline](#edgar_md-pipeline-baseline-after-chunking-plan-phase-5).
- *Retrieval and the 174-question row:* the
  [structure-aware chunker](#structure-aware-chunker-chunking-plan-phase-5).

The previous baseline, `edgar` with the `fixed` chunker as of 2026-09-28,
is in the [chunk header](#deterministic-chunk-header-chunking-plan-phase-2)
and [agentic retrieval](#agentic-retrieval-milestone-19-phase-4) sections.
Rows recorded on it are not paired with rows recorded on this one.

### Milestone 11, pass 1

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
- **Expansion scored lower and cost 12x the latency in this run.** Both providers land below
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

- **NDCG recorded before 2026-09-28 is inflated.** Span matching credited
  every chunk containing an expected span, and adjacent chunks overlap, so
  one span could count twice and push a sample's NDCG above 1 (up to 1.63).
  Hit rate, recall, MRR and answer pass were never affected. Rows re-run
  after the [fix](#ndcg-credits-each-span-once) are correct; older NDCG
  figures in this file are ordinal at best.
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

### Embedder size (chunking plan, Phase 1b)

**Setup.** Run on 2026-09-28. `edgar` corpus, three indexes of the same
4,236 chunks, one per embedder, each built with `--reset` from
`data/eval/config_embedder_{0.6b,4b,8b}.yaml` and in sync per `index-report`.
All three are **Q8_0**: `qwen3-embedding:0.6b` already is, but Ollama's
`:4b` and `:8b` tags point at q4_K_M, so the runs use `:4b-q8_0` and
`:8b-q8_0` to vary size and not quantization. `query_instruction: null`,
per Phase 1. The same three pairs as Phase 1: shipped (hybrid,
`bge-v2-m3`, `top_k 20`, `rerank_top_k 5`), the stage-1 ceiling (`top_k`
and `rerank_top_k` both 20), and dense-only. Each larger size is paired with
its matching 0.6b row. The 0.6b rows were re-run in the same session and
reproduced the committed results sample for sample on all three sets. The
generated set has 174 questions, `underspecified` 118 and `period` 55. M2 Max,
64 GB, Ollama serving one model at a time.

| set | pair | 0.6b → 4b hit | Δ hit [95% CI], W/L, p | 0.6b → 8b hit | Δ hit [95% CI], W/L, p |
|---|---|---|---|---|---|
| generated | shipped | 0.908 → 0.897 | −0.011 [−0.034, +0.011], 1W/3L, p=0.62 | 0.908 → 0.902 | −0.006 [−0.017, +0.006], 0W/1L, p=1 |
| generated | stage-1 ceiling | 0.931 → 0.925 | −0.006 [−0.025, +0.014], 1W/2L, p=1 | 0.931 → 0.931 | +0.000, 0W/0L |
| generated | dense | 0.741 → 0.810 | **+0.069 [+0.025, +0.113]\***, 14W/2L, p=0.004 | 0.741 → 0.828 | **+0.086 [+0.039, +0.134]\***, 17W/2L, p=0.0007 |
| `underspecified` | shipped | 0.653 → 0.695 | +0.042 [−0.007, +0.092], 7W/2L, p=0.18 | 0.653 → 0.686 | +0.034 [−0.024, +0.091], 8W/4L, p=0.39 |
| `underspecified` | stage-1 ceiling | 0.686 → 0.729 | +0.042 [−0.017, +0.102], 9W/4L, p=0.27 | 0.686 → 0.720 | +0.034 [−0.024, +0.091], 8W/4L, p=0.39 |
| `underspecified` | dense | 0.619 → 0.686 | +0.068 [+0.002, +0.133]\*, 12W/4L, p=0.077 | 0.619 → 0.678 | +0.059 [+0.000, +0.118]\*, 10W/3L, p=0.092 |
| `period` | shipped | 0.582 → 0.600 | +0.018 [−0.017, +0.054], 1W/0L, p=1 | 0.582 → 0.582 | +0.000 [−0.051, +0.051], 1W/1L, p=1 |
| `period` | stage-1 ceiling | 0.709 → 0.709 | +0.000 [−0.051, +0.051], 1W/1L, p=1 | 0.709 → 0.727 | +0.018 [−0.044, +0.080], 2W/1L, p=1 |
| `period` | dense | 0.509 → 0.545 | +0.036 [−0.035, +0.108], 3W/1L, p=0.62 | 0.509 → 0.564 | +0.055 [−0.039, +0.149], 5W/2L, p=0.45 |

NDCG moves the same way. The intervals that exclude zero are dense on the
generated set (4b +0.042, 8b +0.060), dense 8b on `underspecified` (+0.048),
and 4b's `paraphrase` NDCG at the shipped config and the ceiling (+0.065,
+0.060). `underspecified` by kind, at the shipped config: `implicit` doesn't
move (0.833 → 0.833 for 4b, → 0.815 for 8b). All of the tier's gain is
`paraphrase`, 0.500 → 0.578 for both sizes (4b: +0.078 [−0.001, +0.157],
6W/1L, p=0.12).

Cost, on the same machine:

| | 0.6b | 4b | 8b |
|---|---|---|---|
| index build (`--reset`, 4,236 chunks) | 243 s | 1,247 s (5.1×) | 2,228 s (9.2×) |
| vector dimensions | 1,024 | 2,560 | 4,096 |
| `embed_query`, median / p95 (174 queries, warm) | 33 / 38 ms | 76 / 101 ms | 110 / 163 ms |

The latency row comes from
`scripts/experiments/2026-09-embedder-size/query_latency.py`, which times
`embed_query` alone. The matrix's `elapsed_s` includes the reranker, and at
the shipped config it grew by 0–14%. The added per-query cost, +43 ms for 4b
and +77 ms for 8b, is small next to the reranker's ~1.1 s. Memory isn't small:
8b sat at 15 GB resident in Ollama at its default 40K context.

**The instruction on 4b.** Phase 1 asked for one instruction pair on a
larger checkpoint, and 4b was it (the size leading at the shipped config on
`underspecified`). The result repeats Phase 1: shipped 0.897 → 0.897 (3W/3L)
on the generated set, 0.695 → 0.686 on `underspecified` and 0.600 → 0.564 on
`period`, with dense −0.023, −0.025 and −0.018. No interval excludes zero,
and every hit-rate estimate is flat or negative.

**Findings**

- **0.6b stays the default.** The plan's bar was a paired CI excluding zero,
  and no larger size clears it at the shipped config on any set. Phase 2 is
  judged on 0.6b.
- **The larger embedders are better embedders, and the pipeline doesn't
  need that where the questions name their subject.** Dense-only, 8b gains
  +8.6pp on the generated set (17W/2L). Through hybrid fusion and the
  reranker, that becomes −0.6pp. Of 4b's 14 dense wins there, 12 were
  questions hybrid 0.6b already answered, because BM25 matches the company
  and period the question states. All three sizes put the answer in stage
  1's top 20 for 92.5–93.1% of those questions.
- **`underspecified` is where a bigger embedder shows, and it isn't shown
  yet.** Only 6 of 4b's 12 dense wins there were already hybrid hits, and
  hybrid 4b keeps 9. The net is +4.2pp at the shipped config (7W/2L,
  p=0.18), all of it on `paraphrase`. That's the tier Phases 1–2 are aimed
  at, and both sizes lean the same way on all three pairs. With 118
  questions, a gain this size can't be told from noise.
- **Neither larger size dominates.** 4b leads 8b on all three
  `underspecified` pairs. 8b leads on all three generated-set pairs and two
  of three `period` pairs. Every gap between them is 1–3 questions. 4b costs
  56% of 8b's build time and 69% of its median query latency, so if a later
  measurement makes the case for a larger embedder, start with 4b.
- **`period` barely moves** (+1.8pp at most at the shipped config). That tier's
  misses are the same text in two periods, and a better vector for the
  same text can't separate them. Phase 3's filters are for that.
- **The query instruction stays off at 4b too.** A second checkpoint gives
  the same answer as Phase 1, so the instruction's failure isn't a 0.6b quirk.
- **Not measured:** answer-side effects (retrieval didn't clear noise), the
  q4_K_M builds that Ollama's default tags point to, and `top_k` above 20.
  So whether the larger embedders lift answers from rank 21–100 into the top
  20 is unknown. The ceiling rows say they don't change much within the top 20.

Raw records: the `embedder=*` variants in
`data/eval/results/retrieval_edgar_edgar_{eval,underspecified,period}_set.json`,
and `data/eval/results/probe_2026-09_embedder_query_latency.json`.

### Deterministic chunk header (chunking plan, Phase 2)

**Setup.** Run on 2026-09-28. `edgar` corpus, two indexes of the same 4,236
chunks with `qwen3-embedding:0.6b` (Phase 1b's verdict): the shipped plain
index, and one built from `data/eval/config_header.yaml`. The second indexes
each chunk as `"{company} ({ticker}) {form}, period ended {period_end}"`,
then the chunk, for example "Apple Inc. (AAPL) 10-K, period ended
2024-09-28". The fields come from YAML front matter, backfilled into the 61
fetched filings without changing any document's loaded text (checked by
hash). All 61 got a header. The build took 248 s against 243 s without, with
no LLM calls. Retrieval pairs are as in Phase 1 (shipped, stage-1 ceiling,
dense-only), plus `rerank_header`, where the cross-encoder also scores the
header (`reranker.include_header`). The three plain rows were re-run on this
branch, which also fixes RRF dropping chunk context, and reproduced the
committed results sample for sample. Answers used `run_answer_matrix.py
--judge-model gemma4:31b-mlx --sets answerable,period,underspecified --limit
0`, generator `qwen3.5:9b-mlx`, paired against the Phase 0 step 3 `crag=off`
rows. With the header on, the prompt labels each passage with its header
instead of its file name, so the answer rows measure retrieval and label
together.

Retrieval, hit rate, each pair against its matching row without a header:

| set | pair | hit | Δ hit [95% CI], W/L, p | Δ NDCG [95% CI] |
|---|---|---|---|---|
| generated | shipped | 0.908 → 0.948 | **+0.040 [+0.007, +0.074]\***, 8W/1L, p=0.039 | +0.043 [+0.015, +0.071]\* |
| generated | stage-1 ceiling | 0.931 → 0.989 | **+0.057 [+0.023, +0.092]\***, 10W/0L, p=0.002 | +0.054 [+0.025, +0.082]\* |
| generated | dense | 0.741 → 0.920 | **+0.178 [+0.119, +0.237]\***, 32W/1L, p<0.001 | +0.136 [+0.087, +0.186]\* |
| generated | + reranker sees header | 0.948 → 0.977 | +0.029 [−0.005, +0.062], 7W/2L, p=0.18 | +0.101 [+0.063, +0.139]\* |
| `period` | shipped | 0.582 → 0.655 | +0.073 [−0.013, +0.159], 5W/1L, p=0.22 | +0.098 [+0.020, +0.177]\* |
| `period` | stage-1 ceiling | 0.709 → 0.945 | **+0.236 [+0.123, +0.350]\***, 13W/0L, p<0.001 | +0.159 [+0.078, +0.239]\* |
| `period` | dense | 0.509 → 0.655 | **+0.145 [+0.027, +0.264]\***, 10W/2L, p=0.039 | +0.136 [+0.046, +0.227]\* |
| `period` | + reranker sees header | 0.655 → 0.836 | **+0.182 [+0.079, +0.285]\***, 10W/0L, p=0.002 | +0.312 [+0.215, +0.409]\* |
| `underspecified` | shipped | 0.653 → 0.797 | **+0.144 [+0.072, +0.216]\***, 19W/2L, p<0.001 | +0.118 [+0.065, +0.171]\* |
| `underspecified` | stage-1 ceiling | 0.686 → 0.873 | **+0.186 [+0.112, +0.261]\***, 23W/1L, p<0.001 | +0.136 [+0.082, +0.190]\* |
| `underspecified` | dense | 0.619 → 0.746 | **+0.127 [+0.054, +0.200]\***, 18W/3L, p=0.002 | +0.121 [+0.062, +0.180]\* |
| `underspecified` | + reranker sees header | 0.797 → 0.822 | +0.025 [−0.024, +0.075], 6W/3L, p=0.51 | +0.087 [+0.034, +0.140]\* |

The "+ reranker sees header" rows pair against `header=on` at the shipped
config, so they measure the reranker switch alone. Against the shipped plain
row, header plus reranker is generated 0.908 → 0.977 (+0.069, 13W/1L),
`period` 0.582 → 0.836 (+0.255, 15W/1L), and `underspecified` 0.653 → 0.822
(+0.169, 23W/3L), all with intervals excluding zero. NDCG at the shipped
config goes 0.821 → 0.965, 0.393 → 0.803 and 0.562 → 0.767. `underspecified` by kind, at the
shipped config: `paraphrase` 0.500 → 0.766 (+0.266 [+0.157, +0.375]\*,
17W/0L), `implicit` 0.833 → 0.833 (2W/2L).

Answers, pass rate against `crag=off`:

| set | `header=on` | Δ [95% CI], W/L, p | `+ rerank_header` | Δ [95% CI], W/L, p |
|---|---|---|---|---|
| generated (174) | 0.862 → 0.937 | **+0.075 [+0.029, +0.120]\***, 15W/2L, p=0.002 | 0.943 | **+0.080 [+0.026, +0.134]\***, 19W/5L, p=0.007 |
| `period` (55) | 0.782 → 0.800 | +0.018 [−0.062, +0.098], 3W/2L, p=1 | 0.927 | **+0.145 [+0.017, +0.274]\***, 11W/3L, p=0.057 |
| `underspecified` (118) | 0.610 → 0.754 | **+0.144 [+0.058, +0.230]\***, 23W/6L, p=0.002 | 0.763 | **+0.153 [+0.065, +0.240]\***, 24W/6L, p=0.001 |
| · `implicit` (54) | 0.722 → 0.741 | +0.019 [−0.091, +0.128], 5W/4L | 0.722 | +0.000, 6W/6L |
| · `paraphrase` (64) | 0.516 → 0.766 | **+0.250 [+0.127, +0.373]\***, 18W/2L, p<0.001 | 0.797 | **+0.281 [+0.170, +0.392]\***, 18W/0L, p<0.001 |

The reranker switch alone, `header=on` → `+ rerank_header`: generated +0.006
(6W/5L), `period` **+0.127 [+0.025, +0.230]\*** (8W/1L, p=0.039),
`underspecified` +0.008 (6W/5L). Failure split (retrieval / generation), off
→ header → header + reranker: generated 15/9 → 5/6 → 3/7, `period` 12/0 →
11/0 → 4/0, `underspecified` 37/9 → 15/14 → 17/11.

**Findings**

- **The header is the first change in this plan to clear noise, on retrieval
  and on answers.** Generated set: +4.0pp hit and +7.5pp answer pass at the
  shipped config. `underspecified`: +14.4pp on both. It costs nothing at
  query time and ~2% at index time, with no LLM calls.
- **It does what contextual chunking was for, and more.** Contextual chunking
  measured +6.4pp dense and +0.6pp hybrid (noise) on the generated set at
  ~4 hours of LLM calls. The header gives +17.8pp dense and +4.0pp hybrid.
  Per the plan, it replaces contextual chunking as the recommended way to put
  document identity into chunks, and contextual isn't worth re-running here.
- **The gain is where questions name a subject the chunk doesn't.**
  `paraphrase` questions name the company and period but reword the fact, and
  gain +26.6pp retrieval and +25.0pp answers. `implicit` questions never name
  the company ("the Dallas-based low-cost carrier"), and the header can't
  bridge that: 0.833 → 0.833. That tier is left for query understanding
  (Milestone 20).
- **On `period`, the header gets the right chunk into stage 1, and the
  reranker has to see it to rank that chunk first.** The stage-1 ceiling goes
  0.709 → 0.945, but the shipped hit only 0.582 → 0.655 and the answer
  0.782 → 0.800 (noise). The cross-encoder scores the text alone, and two
  periods' copies of a paragraph have identical text. With
  `include_header`, the `period` hit goes to 0.836, NDCG 0.491 → 0.803, and
  answers to 0.927 (+12.7pp over the header alone, CI excluding zero). This
  answers the plan's open question: the reranker should see the header.
- **The reranker switch costs little elsewhere.** On the generated set and
  `underspecified` it's noise on answers (6W/5L each) and positive on NDCG. The
  retrieval run took 4% longer (longer passages for the cross-encoder).
  `retrieval.min_score` is 0.0 in the shipped config, so no tuned threshold
  is invalidated.
- **Generation failures rose on `underspecified`** (9 → 14 with the header),
  as retrieval failures fell 37 → 15. More questions now reach the prompt with
  the right chunk and still fail. That's more attempts rather than a worse
  prompt: the generated set's generation failures didn't rise (9 → 6). These
  verdicts weren't read by hand.
- **Not measured:** the hosted comparator (`voyage-context-4`), which needs an
  explicit yes before any corpus text leaves the machine; the header with the
  4b embedder; generator run-to-run noise (about ±2 questions per set at
  temperature 0.2, from the multi-hop re-run).

**Now the default.** `config.yaml` turns on both the header and
`reranker.include_header`, so the shipped config matches the `header=on
rerank_header` rows (on `data/index`, rebuilt with `--reset`). Later phases
pair against those rows, not the plain `baseline` rows above, which stay as
the record of what was measured before. `vanilla.yaml` reads its own
header-free index.

*NDCG in this section predates the [span-credit fix](#ndcg-credits-each-span-once),
which lowered every figure somewhat; the corrected deltas all still exclude zero.*

Raw records: the `header=*` variants in
`data/eval/results/retrieval_edgar_edgar_{eval,period,underspecified}_set.json`,
and `header=on` / `header=on rerank_header` in
`data/eval/results_chunking/answer_edgar__judge-gemma4-31b-mlx.json`.

### NDCG credits each span once

**The bug.** `_judge_by_span` gave a gain to every retrieved chunk containing
an expected span, while the ideal ranking behind NDCG's denominator gives each
span one rank. With 150 characters of overlap, adjacent chunks often both
contain a quote, so DCG could exceed the ideal. In the rows recorded through
Phase 2, 17 of 174 generated-set samples scored above 1 at the plain baseline
and 21 at the shipped config (max 1.63). Metric consumers other than NDCG use
the per-chunk gains and never had this problem: a second chunk quoting the
answer is still a relevant result for precision.

**The fix.** `Judgment.ndcg_gains` credits each span at most once, in rank
order: a chunk earns the best span no earlier chunk has claimed. DCG then
can't exceed the ideal. Document-mode judging already allowed one relevant
chunk per rank and is unchanged.

**Re-run** on 2026-09-28 at the shipped config, `edgar`, all three sets.
`header=off` is a new pinned variant: the plain pipeline on
`data/index_vanilla`, the same chunks and embedder without headers. It stands
in for the old `baseline` row, since `baseline` is now the shipped default.
Every row's hits reproduced the pre-fix run sample for sample; only NDCG
moved, and no sample exceeds 1.

| set | variant | NDCG before → after |
|---|---|---|
| generated | `header=off` (plain) | 0.821 → **0.768** |
| generated | `header=on` | 0.865 → **0.804** |
| generated | `header=on rerank_header` (shipped) | 0.966 → **0.892** |
| `period` | `header=off` | 0.393 → **0.393** |
| `period` | `header=on` | 0.492 → **0.472** |
| `period` | `header=on rerank_header` | 0.803 → **0.761** |
| `underspecified` | `header=off` | 0.562 → **0.533** |
| `underspecified` | `header=on` | 0.679 → **0.628** |
| `underspecified` | `header=on rerank_header` | 0.767 → **0.698** |

Phase 2's NDCG deltas, corrected (paired, 95% CI):

| set | header | reranker sees header | both |
|---|---|---|---|
| generated | +0.036 [+0.011, +0.061]\* | +0.088 [+0.053, +0.124]\* | +0.124 [+0.081, +0.168]\* |
| `period` | +0.079 [+0.012, +0.146]\* | +0.289 [+0.194, +0.384]\* | +0.368 [+0.260, +0.476]\* |
| `underspecified` | +0.095 [+0.047, +0.143]\* | +0.071 [+0.018, +0.123]\* | +0.165 [+0.093, +0.237]\* |

**Findings**

- **No decision changes.** Every default in this file rests on hit rate or
  answer pass, which the bug never touched. Every Phase 2 NDCG delta still
  excludes zero after the fix; the earlier figures overstated them by up to
  0.04 (`period`, both switches: +0.410 recorded, +0.368 corrected).
- **The inflation grew with the variant, not uniformly.** The shipped
  config lost 0.074 on the generated set against 0.053 for the plain one,
  because the header and reranker surface more same-document neighbours. So
  pre-fix NDCG *deltas* are biased toward the variant that surfaces more
  overlapping chunks, not only shifted.
- **Not re-run:** the other recorded rows (reranker models, stage-1, dense,
  query instruction, embedder size). Their hit-rate verdicts stand, and their
  NDCG should be read as inflated. The corrected shipped rows above are what
  later phases pair against.

Raw records: `header=off`, `header=on` and `header=on rerank_header` in
`data/eval/results/retrieval_edgar_edgar_{eval,period,underspecified}_set.json`.
### Metadata filtering oracle (chunking plan, Phase 3)

**Setup.** Run on 2026-09-28 at the shipped config after Phase 2 (header on,
reranker sees it, `edgar` index in sync). No interface change:
`scripts/experiments/2026-09-filter-oracle/oracle.py` restricts each sample's
retrieval using its own labels. Chroma is restricted with a native `where` on
`document_id`, and BM25 by scoring everything and keeping only the allowed
documents' chunks. Both filter before their top-k. Three modes: `none`, which
reproduced the shipped `header=on rerank_header` rows sample for sample on
all three sets, so the harness changes nothing by itself; `company`, every
filing of the expected filing's company, 4.7–4.9 filings on average, the
filter a caller who names the company would pass; and `filing`, exactly
`expected_doc_ids`, the ceiling. On this corpus each company-period pair is
one filing, so `filing` is also what a company-plus-period filter would give.

| set | shipped | `company` | Δ hit, W/L, p | `filing` (ceiling) | Δ hit, W/L, p |
|---|---|---|---|---|---|
| generated (174) | 0.977 | 0.966 | −0.011 [−0.027, +0.004], 0W/2L, p=0.5 | 0.983 | +0.006 [−0.014, +0.025], 2W/1L, p=1 |
| `period` (55) | 0.836 | 0.836 | +0.000, 0W/0L | **0.945** | **+0.109 [+0.026, +0.192]\***, 6W/0L, p=0.031 |
| `underspecified` (118) | 0.822 | 0.831 | +0.008 [−0.042, +0.058], 5W/4L, p=1 | **0.898** | **+0.076 [+0.018, +0.135]\***, 11W/2L, p=0.022 |
| · `implicit` (54) | 0.852 | 0.852 | 2W/2L | 0.870 | +0.019, 3W/2L |
| · `paraphrase` (64) | 0.797 | 0.812 | 3W/2L | **0.922** | **+0.125 [+0.043, +0.207]\***, 8W/0L, p=0.008 |

NDCG isn't reported here; the oracle ran before the NDCG fix.

**Findings**

- **A company filter adds nothing on top of Phase 2.** With the header
  indexed and scored, other companies' chunks already stay out of the top 5.
  The step-3 `period` analysis's "other filing's copy ranked above it" cases
  are the same company, which a company filter keeps.
- **The ceiling clears noise where the plan said it should.** Restricting to
  the right filing gains +10.9pp on `period` and +7.6pp on `underspecified`,
  none of it on `implicit`. That headroom is the period, not the company:
  it's realized only by a filter that names the period (`period_end`),
  which a caller has to supply, or Phase 3b's routing has to infer.
- **The generated set has no headroom left** (0.977; the ceiling is 0.983).
- By the plan's rule (stop unless the oracle beats the baseline by more than
  noise), Phase 3 continues, with the period as the filter that matters.

**A metric bug this surfaced.** `mean_ndcg` came out at 1.027 for `filing` on
the generated set, which led to the [span-credit fix](#ndcg-credits-each-span-once).
This section reports hit rate only.

### Metadata filters (chunking plan, Phase 3)

**Setup.** Run on 2026-09-28 at the shipped config (header on, reranker sees
it, `edgar` index in sync), with the NDCG fix. This is the real `QueryFilter`
path end to end: `run_matrix.py`'s `filters=` variants derive each sample's
filter from its `expected_doc_ids`, shaped as a caller would write it.
`company` is `{"any_of": {"ticker": [...]}}`. `company+period` adds
`{"range": {"period_end": {"gte": ..., "lte": ...}}}`. Chroma applies it as a
`where` clause, and BM25 ranks only matching chunks. `baseline` was re-run and
equals the `header=on rerank_header` rows sample for sample. Both filter
variants reproduced the [oracle](#metadata-filtering-oracle-chunking-plan-phase-3)
hits sample for sample (`company` = `company`; `company+period` = `filing`),
so the interface does what the pre-interface hack did.

| set | filter | hit | Δ hit [95% CI], W/L, p | NDCG | Δ NDCG [95% CI] |
|---|---|---|---|---|---|
| generated (174) | company | 0.977 → 0.966 | −0.011 [−0.027, +0.004], 0W/2L, p=0.5 | 0.892 → 0.890 | −0.002 [−0.010, +0.005] |
| generated | company + period | 0.977 → 0.983 | +0.006 [−0.014, +0.025], 2W/1L, p=1 | 0.892 → 0.937 | **+0.045 [+0.025, +0.065]\*** |
| `period` (55) | company | 0.836 → 0.836 | +0.000, 0W/0L | 0.761 → 0.752 | −0.009 [−0.023, +0.005] |
| `period` | company + period | 0.836 → **0.945** | **+0.109 [+0.026, +0.192]\***, 6W/0L, p=0.031 | 0.761 → 0.861 | **+0.100 [+0.043, +0.157]\*** |
| `underspecified` (118) | company | 0.822 → 0.831 | +0.008 [−0.042, +0.058], 5W/4L, p=1 | 0.698 → 0.724 | +0.026 [−0.011, +0.063] |
| `underspecified` | company + period | 0.822 → **0.898** | **+0.076 [+0.018, +0.135]\***, 11W/2L, p=0.022 | 0.698 → 0.807 | **+0.109 [+0.062, +0.155]\*** |
| · `implicit` (54) | company + period | 0.852 → 0.870 | +0.019, 3W/2L | 0.715 → 0.783 | +0.069 [+0.010, +0.128]\* |
| · `paraphrase` (64) | company + period | 0.797 → **0.922** | **+0.125 [+0.043, +0.207]\***, 8W/0L, p=0.008 | 0.685 → 0.827 | +0.142 [+0.073, +0.211]\* |

Cost: the `company+period` runs took 3–10% longer end to end. That wasn't
profiled; the likely cause is BM25 checking the filter against every record
on each query, which isn't worth optimising at 4,236 chunks.

**Findings**

- **A filter that names the period is the next real gain after the header.**
  `period` hit goes to 0.945, `underspecified` to 0.898, and generated-set
  NDCG rises too, as the right filing's chunks stop sharing the top 5 with
  other periods' copies.
- **A company-only filter adds nothing on top of Phase 2**, and it never
  helps a `period` question (0W/0L). Its two generated-set losses are both
  rank-5 flips: removing other companies from each retriever's top 20 let
  more same-company chunks into the pool, and one edged the answer chunk
  from 5th to 6th. The same-company competition this phase targets shows up
  here as noise.
- **Nothing changes by default.** Filters are caller-supplied: `POST /chat`,
  MCP `rag_search`, or (in Milestone 19) an agent. The gain is available only
  when a caller knows the period. Phase 3b is the version that needs no
  caller input.
- **Not measured:** answer eval with filters. Retrieval moved on the same
  tiers where the header's retrieval gain carried through to answers, but
  that's an expectation, not a measurement. Also not measured: filters
  derived from question text (Milestone 20) or chosen by an agent
  (Milestone 19).

Raw records: `baseline`, `filters=company` and `filters=company+period` in
`data/eval/results/retrieval_edgar_edgar_{eval,period,underspecified}_set.json`.

### Document routing (chunking plan, Phase 3b)

**The router alone, first.** Before any interface change,
`scripts/experiments/2026-09-doc-routing/router_probe.py` ranked one record per
filing (61) against every question, by BM25, dense (`qwen3-embedding:0.6b`) and
their RRF fusion, and recorded where the expected filing landed
(`data/eval/results/probe_2026-09_doc_router.json`). Two record texts: the
Phase 2 header as the plan specified, and the header plus the period end
spelled out ("February 15, 2026"), the way questions write it.

| records | set | R@1 (RRF) | R@5 (RRF) | R@1 (BM25) | chunk hit today |
|---|---|---|---|---|---|
| header | generated (174) | 0.661 | 0.971 | 0.632 | 0.977 |
| header | `period` (55) | 0.836 | 1.0 | 0.927 | 0.836 |
| header | `underspecified` (118) | 0.483 | 0.856 | 0.424 | 0.822 |
| header + date | generated | 0.724 | 0.966 | 0.707 | 0.977 |
| header + date | `period` | 0.891 | 1.0 | **1.0** | 0.836 |
| header + date | `underspecified` | 0.525 | 0.881 | 0.568 | 0.822 |

The plan's version, header-only records filtering to the top *M*, could
only lose. On the generated set the router's top 5 holds the right filing
less often than chunk retrieval's top 5 already does. And a top 5 is about one
company's filings, which Phase 3 showed adds nothing. Two changes followed.
First, the spelled date: the header's ISO date shares only digits with a
question. Second, a gate: route to the top filing only when BM25 and dense
agree on it, and otherwise run unfiltered. Replaying Phase 3's saved
per-sample results predicted the numbers below exactly, before the interface
was built. The record text and the gate were both chosen on these three sets,
so the result is in-sample.

**Setup.** Run on 2026-09-28 at the shipped config (header on, reranker sees
it, `edgar` index in sync), through the real `retrieval.document_routing` path:
`run_matrix.py`'s `routing=top1` and `routing=top2`, default record template
`"{header}; period ended {period_end:date}"`. `baseline` was re-run and equals
the previous rows sample for sample. The ceiling is Phase 3's
`filters=company+period`, the right filing given from labels.

| set | variant | hit | Δ hit [95% CI], W/L, p | NDCG | Δ NDCG [95% CI] | routed (wrong) |
|---|---|---|---|---|---|---|
| generated (174) | ceiling | 0.983 | +0.006 [−0.014, +0.025], 2W/1L | 0.937 | +0.045\* | — |
| generated | `routing=top1` | 0.977 → 0.966 | −0.011 [−0.027, +0.004], 0W/2L, p=0.5 | 0.892 → 0.906 | +0.014 [−0.001, +0.029] | 83 (1) |
| generated | `routing=top2` | 0.966 | −0.011, 0W/2L, p=0.5 | 0.897 | +0.005 [−0.008, +0.017] | 83 (1) |
| `period` (55) | ceiling | 0.945 | +0.109\*, 6W/0L, p=0.031 | 0.861 | +0.100\* | — |
| `period` | `routing=top1` | 0.836 → **0.927** | +0.091 [+0.014, +0.168]\*, 5W/0L, p=0.062 | 0.761 → 0.841 | +0.080 [+0.026, +0.135]\* | 38 (0) |
| `period` | `routing=top2` | 0.891 | +0.055 [−0.006, +0.115], 3W/0L, p=0.25 | 0.798 | +0.037 [+0.005, +0.069]\* | 38 (0) |
| `underspecified` (118) | ceiling | 0.898 | +0.076\*, 11W/2L, p=0.022 | 0.807 | +0.109\* | — |
| `underspecified` | `routing=top1` | 0.822 → 0.856 | +0.034 [+0.001, +0.067]\*, 4W/0L, p=0.12 | 0.698 → 0.732 | +0.034 [+0.006, +0.062]\* | 41 (0) |
| · `implicit` (54) | `routing=top1` | 0.852 | +0.000, 0W/0L | 0.717 | | |
| · `paraphrase` (64) | `routing=top1` | 0.797 → 0.859 | +0.062 [+0.003, +0.122]\*, 4W/0L, p=0.12 | 0.745 | | |

`routed (wrong)`: questions the router filtered, and how many of those
excluded the expected filing. Cost: 5–8% longer end to end. That wasn't profiled. Likely causes are the
router's extra query embedding and the per-chunk BM25 filter check Phase 3
already pays.

**Findings**

- **Routing recovers most of the ceiling where the router is sure, and it
  stays off by default.** On `period`, 5 of the ceiling's 6 wins with no
  losses. On `underspecified`, 4 of its 11. The CIs exclude zero on both,
  but with this few flipped questions McNemar's p (0.062, 0.12) is the test
  to trust, and it doesn't clear 0.05. By this repo's rule that is "not
  shown". Add that the gate and the record text were chosen on these same
  questions, and it isn't grounds to change a default.
- **The gate does what it was built for.** Routing fired on 162 of 347
  questions and excluded the answer once. That question asked about "the
  March 2023 quarter", a period the corpus doesn't hold, and "March" matched
  a 2026 filing. The generated set's other loss is one the ceiling shares:
  the right filing, with the answer chunk edged from 5th to 6th, the same
  rank flip Phase 3's company filter showed.
- **`top_m: 2` is dominated.** Same questions routed, fewer wins: the second
  filing is usually the competing period's copy, the one the filter exists
  to remove. If routing is turned on, it's `top_m: 1`.
- **`implicit` questions never route usefully** (0W/0L). They name neither
  the company nor the date in the form a record holds, so the gate falls back,
  which is the intended failure mode. The `underspecified` gain is all
  `paraphrase`, which keeps the company and date.
- **What it really measures.** `period` questions spell out the exact period
  end, so BM25 over spelled-date records gets `period` R@1 1.0. Real questions
  say "Q3" or "last year" more often. That is Milestone 20's query
  understanding, which would fill Phase 3's `QueryFilter` directly. Routing is
  the no-extraction approximation of it, and it's only as good as the
  question's wording is close to a record's.
- **Not measured:** answers with routing on, and routing on questions not
  drafted from these labels.

Raw records: `baseline`, `routing=top1` and `routing=top2` in
`data/eval/results/retrieval_edgar_edgar_{eval,period,underspecified}_set.json`
(each routed sample carries `routed_to`).

### `table` tier baseline, fixed chunker on `edgar_md` (chunking plan, before Phase 5)

**Setup.** Run on 2026-09-29, tier frozen at commit `b13d8c6` before this
run. Corpus `edgar_md` (Phase 4's Markdown render; `index-report`: in sync,
4,159 chunks, 690 starting mid-table). Shipped config otherwise: `fixed`
chunker 1,000/150 with the deterministic header, hybrid retrieval,
`bge-reranker-v2-m3`, `top_k 20`, `rerank_top_k 5`, CRAG off, generator
`qwen3.5:9b-mlx`, judge `gemma4:31b-mlx`. `retrieval_eval -v` and
`answer_eval -v --judge-model gemma4:31b-mlx`, both with `--corpus edgar_md
--eval-set data/eval/edgar_table_set.json`. Their output, every answer and
judge reply included, is in `data/eval/results_chunking/table_baseline_fixed_edgar_md_*.txt`.

| set | n | retrieval hit [95% CI] | MRR | NDCG | answer pass [95% CI] | answer fails: retrieval / generation |
|---|---|---|---|---|---|---|
| `table` | 95 | 0.863 [0.780, 0.918] | 0.713 | 0.751 | 0.811 [0.720, 0.877] | 13 / 5 |

`unmatchable_spans` is 0: table rows are short (median 56 characters), so a
1,000-character window rarely cuts one. The tier was drafted without the
other tiers' "fits in a fixed chunk" check, so that 0 is the chunker's
result, not the drafting's.

- **The failure is losing the header, not splitting the row.** For 17 of 95
  questions, the fixed chunk holding the row doesn't hold the table's header
  row. Split by that:

  | row's fixed chunk | n | retrieval hit [95% CI] | answer pass [95% CI] |
  |---|---|---|---|
  | keeps the header row | 78 | 0.897 [0.810, 0.947] | 0.872 [0.780, 0.929] |
  | loses it | 17 | 0.706 [0.469, 0.867] | **0.529** [0.310, 0.738] |

  The split is observational. Rows that lose their header may also sit in
  longer, harder tables, so it doesn't measure what keeping the header would
  gain. Phase 5's paired comparison on this tier does. It does locate the
  headroom: the 17 hold 8 of the 18 answer failures.
- **Right filing, wrong chunk.** In 11 of the 13 retrieval misses, the
  expected filing is in the top 5 but not the chunk with the row. Only 2
  (NVDA's tax rate as a percentage of revenue, TGT's GAAP operating income)
  never reach the filing.
- **Generation failures read the wrong cell of the right table.** Each of
  the 5 answers whose row reached the prompt picked another column or row:
  COST's 2023 column for 2024 ($202M for $125M), CVX's natural-gas column for
  oil-equivalent, CVX's total-with-affiliates row, and LUV's and UAL's figures
  for another period. A chunk that keeps its header row is what lets the
  model tell columns apart.
- **Not comparable with `edgar`.** The tier's spans are `edgar_md`'s rendered
  rows, which don't exist in `edgar`'s text, so the parse change alone
  (`edgar` vs. `edgar_md` under the fixed chunker) can't be measured here.
  The other tiers can measure it: every span has a quote in both corpora.

### Structure-aware chunker (chunking plan, Phase 5)

**Setup.** Corpus `edgar_md`, isolated. The `fixed` chunker (1,000/150) is
compared with `structured` (`split_level 2`, `min_chars 200`, no prose
overlap, 1,000-character cap), each with its own index. Everything else is
the shipped config: header on, hybrid retrieval, `bge-reranker-v2-m3`,
`top_k 20`, `rerank_top_k 5`, CRAG off, generator `qwen3.5:9b-mlx`, judge
`gemma4:31b-mlx`.

- *Retrieval rows.* Run 2026-09-29 with `scripts/run_matrix.py`
  (`data/eval/results/retrieval_edgar_md_*`), from the chunker code later
  committed unchanged as `ee379b4`.
- *Answer rows.* Run 2026-10-01 with `scripts/run_answer_matrix.py
  --corpus edgar_md --variant crag=off --variant "chunker=structured"
  --sets answerable,table --limit 0`, at `1e8d29c` from a pinned worktree
  (`data/eval/results/answer_edgar_md__judge-gemma4-31b-mlx.*`). One run per
  row, on all 174 answerable questions and the 95-question `table` tier.

**Index** (`index-report`):

| | fixed | structured |
|---|---|---|
| chunks | 4,159 | 4,808 |
| chunk characters, p10 / median / max | 991 / 996 / 999 | 455 / 812 / 1,164 |
| chunks starting mid-table | 690 (16.6%) | **0** |
| over 1,000 characters | 0 | 39, a heading or stub carried with its unit |
| duplicate chunks (distinct texts) | 46 (21) | 805 (325) |

**Retrieval**, paired against `fixed` (Δ hit, then Δ NDCG, 95% intervals;
wins/losses with McNemar's p):

| set | n | Δ hit | Δ NDCG | W/L |
|---|---|---|---|---|
| `table` | 95 | **+0.116** [+0.045, +0.187] | **+0.107** [+0.031, +0.183] | 12/1, p 0.003 |
| `period` | 55 | +0.073 [−0.013, +0.159] | **+0.148** [+0.049, +0.246] | 5/1, p 0.22 |
| generated | 174 | +0.011 [−0.011, +0.034] | **+0.045** [+0.015, +0.076] | 3/1, p 0.62 |
| `underspecified` | 118 | +0.008 [−0.047, +0.064] | +0.033 [−0.023, +0.088] | 6/5, p 1 |

`prose_overlap` (overlap between paragraphs of one section) was no better
than plain `structured` on any set, so it was not run end to end.

**Answers:**

| set | n | fixed | structured | Δ, W/L | failures, retrieval / generation (fixed → structured) |
|---|---|---|---|---|---|
| `table` | 95 | 0.811 | **0.947** [0.883, 0.977] | **+0.137**, 15/2, p 0.002 | 13 / 5 → 2 / 3 |
| answerable | 174 | 0.948 | 0.948 | +0.000 [−0.028, +0.028], 3/3, p 1 | 3 / 6 → 3 / 6 |

| | prompt tokens, answerable / table | completion tokens | retrieval s/query |
|---|---|---|---|
| fixed | 1,668 / 2,030 | 109 / 117 | 1.03 |
| structured | 1,379 / 1,842 | 97 / 99 | 0.98 |

- **It fixes the failure the baseline located.** The `fixed` row reproduced
  the baseline's 0.811 and its 13 / 5 failure split. Split by whether the
  fixed chunk holding the row kept the table's header row (the same 78/17
  grouping as the baseline):

  | row's fixed chunk | n | fixed | structured |
  |---|---|---|---|
  | keeps the header row | 78 | 70 | 76 |
  | loses it | 17 | 7 | **14** |

  The 17 went from 7 to 14 passes. The kept group gained too (70 → 76),
  likely because a structured chunk holds one table under its heading rather
  than the tail of one table and the start of the next. That reading is not
  tested here. The fixed row's split moved by two
  questions from the baseline's 68 and 9 at the same total, which is
  temperature 0.2 sampling.
- **Failures where the row never reached the prompt fell from 13 to 2.**
  The baseline found 11 of its 13 were "right filing, wrong chunk". With a
  table kept whole, or split under a repeated header, the chunk that matches
  the question's column and period words is the one holding the row.
- **Nothing regressed.** Answerable is a tie, 3 wins and 3 losses, with the
  same failure split; at 0.948 it has 9 failures to fix, so it measures
  harm more than gain. `underspecified` retrieval is flat. The two table
  losses are `DAL_10-Q_2026-03-31 t8r0c2` and `XOM_10-Q_2026-06-30 t8r2c4`.
- **Cheaper prompts.** Chunks are shorter at the median (812 vs. 996
  characters), so the prompt is 17% smaller on answerable and 9% on the
  tier, and answers are about 10% shorter.
- **Latency is not attributable to the chunker.** Mean answer latency was
  8.2 → 9.3 s (answerable) and 10.2 → 12.2 s (`table`), but retrieval timed
  alone is the same (1.03 vs. 0.98 s/query over 40 queries, two rounds) and
  the generator got fewer tokens. The structured rows ran second, so machine
  load is the likely cause. A rerun in the opposite order would settle it.
- **The duplicate chunks don't reach the prompt.** The 805 are boilerplate
  paragraphs a company repeats across its quarterly filings (fair-value and
  revenue policies, up to 5 copies). Paragraph-aligned chunks make the
  copies identical, where fixed windows started at different offsets and
  hid them. Retrieving every answerable and `table` question, 1 of 269 had
  a repeated passage in its top 5, so no deduplication is needed.

**Decision.** This meets the plan's default criterion: answer quality
improves, by more than noise on the `table` tier, and retrieval doesn't
regress on any set. Adopted as the default, with `edgar_md` as the EDGAR
corpus evals run on
([chunking plan](chunking-indexing-plan.md#phase-5--structure-aware-chunker-34-days--done-2026-10-01)).
EDGAR rows recorded before this section ran on `edgar` with the `fixed`
chunker; they are not paired with rows recorded after it.

### `edgar_md` pipeline baseline (after chunking plan Phase 5)

**Setup.** This is the shipped pipeline after Phase 5 was adopted: corpus
`edgar_md`, `structured` chunker, header on, `reranker.include_header`,
hybrid, `bge-reranker-v2-m3`, `top_k 20`, `rerank_top_k 5`, CRAG off,
generator `qwen3.5:9b-mlx`, judge `gemma4:31b-mlx`.

- *Command.* `scripts/run_answer_matrix.py --family m19 --corpus edgar_md
  --variant "pipeline / 9b" --repeat 3 --limit 40` on all seven sets.
- *Provenance.* Run 2026-10-01 to 10-02 at `9745738`, from a pinned
  worktree. `index-report` showed the index in sync, with 4,808 chunks and
  none starting mid-table.
- *Raw records.* `data/eval/results_m19/answer_edgar_md__judge-gemma4-31b-mlx.*`.
- *Not pairable with older rows.* This is the row new EDGAR rows pair
  against. It is not paired with any row recorded on `edgar`, because corpus
  and chunker changed together.

Counts per run, out of each set's size:

| set | n | per run | mean | spread | evidence recall |
|---|---|---|---|---|---|
| answerable (evenly spaced) | 40 | 37, 38, 37 | 37.3 | 1 | — |
| refusals | 15 | 15, 15, 15 | 15.0 | 0 | — |
| multi-hop complete | 35 | 25, 25, 24 | 24.7 | 1 | 0.853 |
| adaptive complete (bridge /10, discovery /5) | 15 | 7, 6, 7 (4.7, 2.0) | 6.7 | 1 | 0.689 |
| `period` | 55 | 54, 54, 54 | 54.0 | 0 | — |
| `underspecified` | 118 | 90, 90, 89 | 89.7 | 1 | — |
| `table` | 95 | 90, 89, 90 | 89.7 | 1 | — |

Prompt tokens per turn are 1,140–1,842 by set, and latency 6.8–10.5 s.

- **The `table` tier holds Phase 5's result:** 89.7 of 95 (0.944) over
  three runs, against Phase 5's single `structured` run at 0.947.
- **Latency.** Table latency here is 9.9–10.5 s, against 12.2 s for
  `structured` in the Phase 5 run. That supports reading Phase 5's latency
  rise as machine load rather than the chunker.
- **For orientation only: the move against the old `edgar` baseline.**
  These are not paired comparisons. On the same questions, the old `edgar`
  rows (`pipeline / 9b`, Milestone 19 phase 4 and the Phase 2 header row)
  read:

  | set | old `edgar` | now `edgar_md` |
  |---|---|---|
  | answerable /40 | 38.7 | 37.3 |
  | refusals /15 | 14.0 | 15.0 |
  | multi-hop /35 | 24.7 | 24.7 |
  | adaptive /15 | 4.3 | 6.7 |
  | evidence recall, multi-hop / adaptive | 0.79 / 0.64 | 0.85 / 0.69 |
  | `period` / `underspecified` answer pass, one run | 0.927 / 0.763 | 0.982 / 0.760 avg |

  - *Refusals.* The refusal every old run missed,
    `neg-unanswerable-comparison`, now passes in all three runs.
  - *Agentic gains.* The Milestone 19 agent gains were measured against the
    old pipeline row. This baseline's adaptive score is higher (6.7 vs
    4.3), so the agent's margin on `edgar_md` is unknown until the agent rows
    are re-run here.

### SQLite FTS5 sparse backend

**Full pipeline.** Run on 2026-09-28 with `scripts/run_matrix.py` at the
shipped config (hybrid, RRF, `bge-v2-m3`, header on) on all three EDGAR sets.
The FTS5 index was loaded from the `edgar` BM25 index's records, so both
backends hold the same 4,236 chunks as Chroma (`index-report`: in sync). The
re-run `baseline` reproduced the committed rows sample for sample. Paired
FTS5 − BM25, hit W/L with McNemar's exact p:

| set | n | at `rerank_top_k` 5 (shipped) | at `rerank_top_k` 20 (stage 1 only) |
|---|---|---|---|
| eval | 174 | 0W/1L, p=1; Δ NDCG +0.000 | 0W/0L; Δ NDCG +0.002 |
| period | 55 | 1W/0L, p=1; Δ NDCG −0.002 | 0W/0L; Δ NDCG −0.009 |
| underspecified | 118 | 0W/2L, p=0.5; Δ NDCG −0.011 | 1W/2L, p=1; Δ NDCG −0.009 |
| all three | 347 | 1W/3L, p=0.63 | 1W/2L, p=1 |

Query time was equal (within ±4 s across 55–174 questions); the reranker dominates it.

**Keyword leg alone.** The earlier probe, with no embedder or reranker:
the `edgar` BM25 index's 4,236 records were loaded into `SqliteFts5Index`,
and both backends answered the 174 EDGAR eval questions at top-20. Scale was
simulated by replicating those chunks under new ids (10× and 35×), so term
statistics are unrealistic there -- read those rows for cost, not ranking.

| | `bm25` | `sqlite_fts5` |
|---|---|---|
| gold-document hit@20 (keyword leg) | 173/174 | 173/174 |
| same top-1 chunk | -- | 138/174 |
| mean top-20 overlap with `bm25` | -- | 0.79 |
| median query, 4,236 chunks | 9 ms | 8 ms |
| query, 148k chunks | ~470 ms | ~270 ms |
| open + first query, 148k chunks | 9.7 s | 0.3 s |
| peak RSS, 148k chunks | 3.3 GB | 52 MB |
| on disk, 148k chunks | 247 MB JSON | 369 MB (540 MB before layout 2) |

**Findings.**

- **No difference shown at EDGAR scale.** Through the full pipeline, 4 of
  347 questions flip (1 gained, 3 lost, p=0.63), and NDCG moves by at most
  0.011 on a set, with no set's interval excluding zero. The losses lean
  toward FTS5, but four discordant questions can't separate that from noise.
  This is "not shown", not "proven equal".
- Why they differ at all: FTS5 fixes k1 = 1.2 (rank_bm25's `BM25Okapi` uses
  1.5) and floors a non-positive IDF at 1e-6 where rank_bm25 substitutes a
  quarter of the mean IDF. Tokens are identical. A fifth of each keyword-leg
  top-20 differs, but RRF and the reranker absorb nearly all of it: at most
  10 of 118 questions change NDCG at all.
- **Default stays `bm25`** for corpora it handles, since switching buys
  nothing measurable at this size. Use `sqlite_fts5` when the corpus outgrows
  `bm25`'s memory and startup cost (below). A pooled or larger corpus should
  get its own comparison, since the k1 and IDF differences act on term
  statistics, and those change with the corpus.
- `bm25`'s real cost at scale is state, not query time: every process loads
  and re-tokenizes the whole corpus, holds it in memory, and any change
  rewrites the full JSON and rebuilds the model. FTS5 opens lazily, holds
  almost nothing, and commits each indexing batch.
- FTS5's file is about 1.5× the JSON: the chunk rows are the same data, and
  the inverted index is extra. Layout 1 also kept a copy of every chunk's
  token string in the FTS table, 540 MB at 148k chunks. Layout 2 makes the
  table contentless (`contentless_delete=1`, SQLite >= 3.43), bringing it to
  369 MB. Migrating a layout-1 file in place took 3.8 s at 148k chunks and
  returned identical top-20 chunks and scores on all 174 eval questions, so
  the ranking results above carry over unchanged.
- Query time grows roughly linearly on both, since every chunk sharing any
  query word is scored. At 10^6+ chunks, stopword-heavy questions will need
  pruning (or a dedicated engine); this backend doesn't solve that.

### BEIR reference reproduction and this repo's backends (public benchmarks plan, phase 3)

**Setup.** Run on 2026-09-29, test splits of FiQA-2018 (648 queries),
SciFact (300) and NFCorpus (323), all judged queries, `trec_eval -c`
semantics, `--remove-query` on everywhere. These are **not tuning results**:
no parameter, feature or model was chosen from them (see the
[test-access log](beir-reference-protocol.md#test-access-log)). Runs,
scores and controls are regenerated into `data/benchmarks/` (gitignored), by
`scripts/reproduce_beir_reference.py`, `retrieval_eval --save-run` and the two
scripts in `scripts/experiments/2026-09-beir-reproduction/`. Protocol and pins:
[BEIR reference protocol](beir-reference-protocol.md).

**The gate: reference runs, recreated.** Pyserini 2.4.0 in an isolated
environment (Python 3.12.13, Java 21.0.12.1, `faiss-cpu` 1.15.1, `torch`
2.14.0, `transformers` 5.17.0, M2 Mac, CPU, `OMP_NUM_THREADS=1`), the
manifests' 2CR commands unchanged, over the downloaded prebuilt indexes whose
md5s the manifests pin. The BGE query encoder runs locally.

| Dataset | BM25 flat nDCG@10 / R@100 | BGE Faiss nDCG@10 / R@100 |
|---|---|---|
| FiQA | 0.2361 / 0.5395 | 0.4065 / 0.7415 |
| SciFact | 0.6789 / 0.9253 | 0.7408 / 0.9667 |
| NFCorpus | 0.3218 / 0.2457 | 0.3735 / 0.3368 |

Every cell equals the published score (difference 0.0000; tolerance
0.0005). `scripts/trec_eval_parity.py` scored all six run files with
`rag.eval.qrels` and `trec_eval` 9.0.4 against the reference qrels (sha256
matching the manifests): 0.0000 per query and in aggregate on all six. The
NFCorpus BM25 run returns nothing for 15 queries, which both evaluators
score 0. So the scorer and the reference path are validated on the
complete runs, not only on phase 2's rankings.

**This repo's backends, same data.** Differences, not reproductions: the
repo's backends are different algorithms, run at their shipped parameters.

| Dataset | Lucene BM25 (ref) | `bm25` (rank_bm25) | `sqlite_fts5` | BGE, exact search over the reference vectors | BGE, exact over this repo's vectors | BGE, this repo's Chroma (shipped path) |
|---|---|---|---|---|---|---|
| FiQA | 0.2361 / 0.5395 | 0.2193 / 0.4717 | 0.2351 / 0.5136 | 0.4065 / 0.7415 | 0.4062 / 0.7415 | **0.3952 / 0.7194** |
| SciFact | 0.6789 / 0.9253 | 0.6535 / 0.8731 | 0.6683 / 0.8859 | 0.7408 / 0.9667 | 0.7404 / 0.9667 | 0.7404 / 0.9667 |
| NFCorpus | 0.3218 / 0.2457 | 0.3038 / 0.2326 | 0.3045 / 0.2334 | 0.3735 / 0.3368 | 0.3735 / 0.3367 | **0.3670 / 0.3268** |

nDCG@10 / R@100. Sparse runs to depth 1,000 over the repo's own chunks of
`rag/config/beir.yaml` (one per passage); dense runs through
`rag/config/beir_bge.yaml` (`sentence_transformers`, BGE-base at the pinned
revision, stage 1 = final = 100). "Exact" is a numpy inner product, the
operation Faiss `IndexFlatIP` performs, with this repo's query vectors.

- **The dense integration is right; Chroma's approximate search is what
  costs.** Our query vectors against the reference document vectors
  reproduce the published run exactly on all three sets, so the query side
  (prefix, pooling, truncation) is right. Our document vectors agree with
  theirs to cosine ≥ 0.999996 on SciFact and NFCorpus and on 99.9% of FiQA
  (median 0.9999995; FiQA's exceptions are below). Exact search over
  them lands within 0.0004 of the reference. That is inside the 0.0005
  reproduction bar, though it is a repo number and makes no reproduction
  claim. The remaining gap is Chroma's HNSW index, which returns only part
  of the exact top k:

  | Dataset | share of exact top 10 returned | share of exact top 100 | nDCG@10 lost | R@100 lost |
  |---|---|---|---|---|
  | FiQA (57,600 passages) | 0.972 | 0.886 | 0.0110 | 0.0221 |
  | NFCorpus (3,633) | 0.975 | 0.914 | 0.0065 | 0.0099 |
  | SciFact (5,183) | 0.998 | 0.966 | 0.0000 | 0.0000 |

  The collections use Chroma 1.5.9's defaults (`hnsw:space: cosine`,
  nothing else set): 16 neighbours per node, `ef_construction` 100,
  `ef_search` 100. At k = 100 the search beam is therefore no wider than the
  result list. See [known limitations](known-limitations.md) for what it means for
  the shipped pipeline.
- **Device and batching are not the cause.** SciFact re-encoded on the CPU
  gives the same scores as the MPS index. The small vector differences move
  top-10 order only between documents whose reference scores differ by
  ≤ 1.1e-4: 17 of 300 SciFact queries, 26 of 323 NFCorpus, 57 of 648 FiQA.
- **One residual is unexplained, and bounded.** Six FiQA document vectors
  sit at cosine 0.9989–0.99986 against the reference, well below the rest.
  Their text is identical to the copy in Pyserini's own FiQA BM25 index.
  Ours is deterministic across CPU, MPS and batch padding, and none has
  unusual characters or passes 512 tokens. What the 2024 reference build
  fed the encoder for them is not recorded anywhere found. The effect,
  bounded by the exact-search control, is −0.0002 nDCG@10 and 0 R@100 on
  FiQA, all six documents included. The same row also carries FiQA's 38
  empty passages, which this repo doesn't index (Faiss does). That costs
  nothing measurable, although one of them is the relevant document for
  test query `5206`.
- **Neither BM25 backend is Lucene's BM25.** They trail the reference by
  0.001–0.025 nDCG@10 and 0.012–0.068 R@100. `sqlite_fts5` is the closer on
  all three sets and much closer on FiQA's recall (0.5136 against
  `bm25`'s 0.4717). Documented causes, not separated by experiment: Lucene
  stems (Porter), drops English stopwords and uses k1 = 0.9, b = 0.4;
  both repo backends index plain lowercase alphanumeric tokens. rank_bm25
  uses k1 = 1.5, b = 0.75 and replaces non-positive IDF with a quarter of
  the mean; FTS5 uses k1 = 1.2, b = 0.75 and floors IDF at 1e-6. The parameters were not tuned on test to close the
  gap, and per the plan, tuning them belongs on dev splits.
- **Cost.** Recreating the six reference runs takes 8–37 s each once the
  indexes are downloaded (277 MB for all six). The Pyserini environment is
  1.4 GB, and none of it is a repo dependency. Indexing with BGE through
  `rag.cli index` took 78 s for SciFact, 58 s for NFCorpus and 600 s for FiQA,
  peaking at 1.2, 1.0 and 2.9 GB resident. The three collections plus
  their BM25 files take 781 MB. On FiQA, `bm25` answered the 648 queries
  at depth 1,000 in 96 s and `sqlite_fts5` in 54 s.

### HNSW `ef_search`

**Setup.** Run on 2026-09-29 to set `vector_store.hnsw_ef_search`, after
[phase 3](#beir-reference-reproduction-and-this-repos-backends-public-benchmarks-plan-phase-3)
traced most of the dense gap to Chroma's approximate search.
`scripts/experiments/2026-09-hnsw-ef-search/ann_recall.py` measures the
share of the exact top k (numpy inner product over the collection's own
vectors) that Chroma returns, with **no relevance labels**, one process per
value (Chroma fixes `ef_search` when a process first loads the collection).
Query texts: the BEIR test queries against the `beir_bge.yaml` indexes at
k = 100, and the 347 questions of the three EDGAR sets against the shipped
`edgar` index (Qwen 0.6b, 4,236 chunks). Latency is Chroma's median query time
on this machine, with the embedding excluded.

| Index | k | 100 (Chroma default) | 200 | 400 | 800 | 1600 |
|---|---|---|---|---|---|---|
| EDGAR | 20 | 0.9994 (3 q < 1) · 0.9 ms | 0.9999 · 1.3 ms | **1.0000** · 2.0 ms | 1.0000 · 3.4 ms | 1.0000 · 4.8 ms |
| EDGAR | 100 | 0.9896 · 1.0 ms | 0.9982 · 1.4 ms | 0.9997 · 2.4 ms | 0.9999 · 3.5 ms | 1.0000 · 4.9 ms |
| FiQA (57,600) | 100 | 0.8857 · 1.5 ms | 0.9484 · 2.4 ms | 0.9801 · 4.0 ms | 0.9936 · 6.3 ms | **0.9978** · 10.1 ms |
| NFCorpus (3,633) | 100 | 0.9143 · 1.1 ms | 0.9723 · 1.6 ms | 0.9933 · 2.4 ms | 0.9985 · 2.9 ms | **0.9998** · 3.6 ms |
| SciFact (5,183) | 100 | 0.9659 · 1.3 ms | 0.9928 · 1.8 ms | 0.9991 · 2.6 ms | 0.9998 · 3.5 ms | **1.0000** · 4.5 ms |

Each cell is the share of the exact top k returned, then the median query
time. The default-column BEIR values reproduce phase 3's (0.886, 0.914,
0.966).

**Shipped pipeline, 100 vs 400.** `retrieval_eval -v` at the shipped config
on all three EDGAR sets (174 + 55 + 118 questions), each value in its own
process. Every per-sample line is identical, and so is every aggregate (hit
0.977 / 0.836 / 0.822, NDCG 0.892 / 0.761 / 0.698). The dense leg's three
partial misses at 100 never survived RRF and the reranker.

**BEIR through Chroma at 1600**, reported only (the value was chosen above,
from recall). `retrieval_eval` with `beir_bge.yaml`, nDCG@10 / R@100:
SciFact 0.7404 / 0.9667 and NFCorpus 0.3735 / 0.3367, both equal to exact
search; FiQA 0.4045 / 0.7396 (exact 0.4062 / 0.7415). Against the reference,
the FiQA gap falls from 0.0113 to 0.0020 nDCG@10.

**Findings.**

- **Shipped default raised to 400.** On EDGAR it makes the dense top 20
  exact for about 1 ms a query, which is small next to the reranker. This
  is an approximation fix, not a quality gain: the evals didn't move. The
  margin matters for what EDGAR doesn't yet test, pooled or larger corpora,
  where recall at 100 fell furthest (FiQA, 57,600 passages).
- **`beir.yaml` uses 1600** (≥ 0.998 on all three sets), so phase-4 dense
  rows measure the retriever rather than HNSW. FiQA's remaining 0.2% is
  0.0017 nDCG@10; state it beside FiQA dense rows.
- **One process per value.** A `run_matrix` variant that changed
  `hnsw_ef_search` would share its process with the baseline, and Chroma
  would silently keep the first value, so `ChromaVectorStore` refuses a
  second value for a collection in one process.

### BEIR query-time stack (public benchmarks plan, phase 4)

**Setup.** Run on 2026-09-29, test splits of FiQA-2018 (648 queries),
NFCorpus (323) and SciFact (300), all judged queries, `trec_eval -c`
semantics, `--remove-query` on. `scripts/run_beir_stack.py` runs seven
variants at the **benchmark depths**: 100 candidates per retriever, 100 after
RRF (`rrf_k` 60), which is also the reranker's pool, and 10 final results.
Every other setting is at its shipped value (`bm25`, `bge-reranker-v2-m3`,
`min_score` 0) except two BEIR settings: `hnsw_ef_search` 1600 and a
512-token reranker cap. The shipped pipeline fuses 20 and keeps 5; that operating
point is not measured here. BGE is `beir_bge.yaml` (the reference encoder);
Qwen is `beir.yaml` (`qwen3-embedding:0.6b` via Ollama, the shipped
embedder). The comparison family, reading rules, reranker cap and grouping
were frozen and committed (`0cda7fd`, PR #59) before any test run:
[phase-4 protocol](beir-phase4-protocol.md). Each variant ran once. Runs,
per-query scores, latency and provenance are in `data/benchmarks/phase4/`
(gitignored).

**Provenance caveat.** Another session switched this checkout's branch
while the runs were in progress. As a result, 18 of the 21 `provenance.json` files record
HEAD `aaa0432` or `883f0d1` rather than `0cda7fd`. All 21 ran in one process
that imported `0cda7fd` at start. The later commits change no BEIR, vanilla or
base config and add only a typing protocol and Gemini-only LLM fields. No
run's saved config contains those fields, and each variant's resolved config
hash is identical on all three datasets. So every run executed `0cda7fd`. The
runner now records the loaded commit, and warns when HEAD moves.

| Variant | FiQA nDCG@10 / R@100 | NFCorpus | SciFact |
|---|---|---|---|
| `bge-dense` | 0.4045 / 0.7396 | 0.3735 / 0.3367 | 0.7404 / 0.9667 |
| `bge-hybrid` | 0.3352 / 0.7057 | 0.3652 / 0.3346 | 0.7149 / 0.9693 |
| `bge-hybrid-rerank` | 0.4298 / 0.7057 | 0.3436 / 0.3346 | 0.7351 / 0.9693 |
| `qwen-dense` | 0.3919 / 0.7432 | 0.2981 / 0.3076 | 0.6789 / 0.9367 |
| `qwen-hybrid` (descriptive) | 0.3545 / 0.7232 | 0.3469 / 0.3195 | 0.7006 / 0.9567 |
| `qwen-hybrid-rerank` (shipped stack) | 0.4389 / 0.7232 | 0.3454 / 0.3195 | 0.7354 / 0.9567 |
| `qwen-dense-instruct` | **0.4691 / 0.7985** | 0.3554 / 0.3286 | 0.6979 / 0.9567 |

The dense rows reproduce phase 3's figures at `ef_search` 1600:
BGE matches exact search on NFCorpus and SciFact, and FiQA sits 0.0020 below
the reference.

**The family: ΔnDCG@10**, candidate minus baseline. First the 95%
interval, then the Bonferroni interval over the 15 comparisons (99.67%);
the reading comes from the second. SciFact intervals are cluster-robust over
247 groups of claims that share a relevant abstract. NFCorpus queries can't
be grouped (318 of 323 connect through shared documents), so **its intervals
understate their width by an unknown amount**.

| # | Comparison | FiQA | NFCorpus | SciFact |
|---|---|---|---|---|
| 1 | hybrid vs dense (BGE) | −0.0692 [−0.089, −0.050]; [−0.099, −0.040] **worse** | −0.0084 [−0.022, +0.005]; [−0.029, +0.012] not shown | −0.0255 [−0.057, +0.006]; [−0.072, +0.021] not shown |
| 2 | reranker over hybrid (BGE) | +0.0946 [+0.074, +0.115]; [+0.064, +0.126] **improved** | −0.0216 [−0.038, −0.005]; [−0.046, +0.003] not shown | +0.0202 [−0.013, +0.054]; [−0.030, +0.070] not shown |
| 3 | Qwen vs BGE, dense | −0.0126 [−0.034, +0.009]; [−0.045, +0.020] not shown | −0.0754 [−0.099, −0.052]; [−0.111, −0.040] **worse** | −0.0614 [−0.095, −0.028]; [−0.111, −0.012] **worse** |
| 4 | shipped stack vs Qwen dense | +0.0470 [+0.026, +0.068]; [+0.015, +0.079] **improved** | +0.0473 [+0.023, +0.072]; [+0.011, +0.084] **improved** | +0.0564 [+0.016, +0.097]; [−0.005, +0.118] not shown |
| 5 | Qwen query instruction | +0.0772 [+0.063, +0.092]; [+0.056, +0.099] **improved** | +0.0573 [+0.037, +0.078]; [+0.027, +0.088] **improved** | +0.0189 [+0.001, +0.037]; [−0.008, +0.046] not shown |

Wins/losses per query, nDCG@10 (#1–5): FiQA 139/254, 280/113, 201/211,
239/174, 266/90. NFCorpus 99/99, 93/122, 73/159, 134/101, 140/70. SciFact
52/62, 50/48, 44/76, 69/53, 51/30.

**R@100 (stage 1, 95%, secondary).** #1, hybrid vs dense: FiQA −0.034
[−0.050, −0.018]. That is beyond the protocol's 0.01 reporting bar, so fusion
also loses recall there. NFCorpus −0.002 and SciFact +0.003 are not shown. #4:
FiQA −0.020 [−0.034, −0.006], NFCorpus +0.012 [+0.001, +0.023], SciFact
+0.020 (not shown). #2 is zero by construction.

**Cost.** Median query latency on an M2 (MPS): dense 17–73 ms, and hybrid
20–230 ms, where FiQA's `bm25` over 57,600 passages is the slow part.
Reranking 100 passages takes 2.8–4.4 s, so a reranked variant takes 21–36 min per test set.
Peak memory for the whole run was 4.9 GB. Indexing FiQA with Qwen through
Ollama took 6,028 s for 57,600 passages, part of it sharing the machine with
dev runs, at 1.6 GB resident for the indexer. The BGE index costs are in
phase 3.

**Findings.** All are scoped to these three test sets at benchmark depths.

- **The shipped stack improves on its own dense leg on FiQA (+0.047) and
  NFCorpus (+0.047). On SciFact, +0.056 is not shown** at the family level.
  No dataset reads *worse*, so under the frozen rule the stack transfers. On
  FiQA it does so while losing 0.020 R@100, because the gain comes from the
  reranker, not from fusion.
- **Fusion alone hurt on FiQA (−0.069 nDCG@10, −0.034 R@100) and showed
  nothing on the other two.** Equal-weight RRF with a much weaker BM25 list
  (0.22 on FiQA in phase 3, against BGE's 0.40) pulls good dense results
  down. That differs from EDGAR, where BM25's exact terms (tickers, periods)
  earned hybrid its place. These queries are natural-language questions
  with little exact-term signal.
- **The reranker is what recovers FiQA** (+0.095 over hybrid). It lands
  above dense alone (0.430 against 0.405), but that comparison is outside
  the family. On NFCorpus it leans negative (−0.022; the 95% interval
  excludes zero, the adjusted one doesn't, and the interval is too narrow
  anyway). On SciFact nothing is shown. Its gain is dataset-dependent, not
  general.
- **Without its instruction, Qwen 0.6b trails BGE-base on NFCorpus (−0.075)
  and SciFact (−0.061).** On FiQA nothing is shown (−0.013).
- **With the instruction, Qwen gains on FiQA (+0.077) and NFCorpus (+0.057).**
  SciFact shows nothing (+0.019). On FiQA instructed Qwen is the best
  row in the table (0.469, above every reranked row), but that comparison is
  outside the family. This is the feature the EDGAR measurement
  ([query instruction](#query-instruction-for-the-embedder-chunking-plan-phase-1))
  left off, where it leaned worse on every set. The two results use the same
  Ollama build, template and instruction text, which makes that section's
  "Q8_0 build or Ollama's pooling" explanation unlikely. Task fit or the
  corpus is the better candidate. **It stays off.** The shipped default is
  decided on EDGAR, and the web-search task text fits BEIR's queries better than
  questions about filings. What this justifies is re-measuring on EDGAR with a
  task written for filings, chosen without reading EDGAR's misses. Whether
  the gain survives fusion and reranking was not measured: no family
  comparison has instructed Qwen in the hybrid.
- **Nothing here changes a shipped default.** The results say how the
  shipped stages behave on outside data at a depth the product doesn't use.
  The product's own defaults rest on EDGAR.
- **Not measured:** query expansion and CRAG (LLM per query, stochastic),
  chunking-time features (BEIR bypasses chunking), and the shipped 20/5
  depths.

Raw records: `data/benchmarks/phase4/{fiqa,nfcorpus,scifact}/test/*/` and
the rendered table `data/benchmarks/phase4/test.md`. Dev runs, exploratory:
`data/benchmarks/phase4/{fiqa,nfcorpus}/dev/`.

### Agentic retrieval (Milestone 19, phase 4)

**Setup.** `edgar`, isolated, run at the shipped config (header on, hybrid,
`bge-reranker-v2-m3`, `top_k 20`, `rerank_top_k 5`, CRAG off), judged by
`gemma4:31b-mlx` at temperature 0. Every row ran on four sets: 40 answerable
questions (evenly spaced from the 174), the 15-question refusal set, the
35-question multi-hop set, and the 15-question adaptive set
(`edgar_adaptive_set.json`: 10 bridge and 5 discovery questions, added after
stage 1 because the multi-hop set names every company it asks about; see the
[plan](milestone-19-plan.md#4--measure-the-milestones-actual-deliverable)). Rows are those of
`scripts/run_answer_matrix.py --family m19`. Each differs from the one it is
compared with in one factor. The agent rows pin `agent.llm` to 4096 tokens
and a 600 s timeout, and `pipeline / 27b` uses those same 27b settings, so
the `pipeline / 27b` vs `agentic react / 27b` pair isolates the loop. Most
rows ran three times (`--repeat 3`), `think=low` twice and `+ groundedness`
once. Stage 1's four 9b rows ran the answerable, refusal and multi-hop
sets at `429c391`, everything else at `c7c6ef9` or `051edd6`. No file under
`rag/` changed between the three, so they are comparable. Raw records:
`data/eval/results_m19/`. The main file's stage 1 rows show `c7c6ef9`
because adding the adaptive set merged into them; `stage1_429c391/` is the
copy taken before that.

Counts are per run, out of each set's size:

| row | answerable /40 | refusals /15 | multi-hop /35 | adaptive /15 (bridge /10, discovery /5) | evidence recall, multi-hop / adaptive |
|---|---|---|---|---|---|
| `pipeline / 9b` (shipped) | 39, 38, 39 | 14, 14, 14 | 24, 25, 25 | 4, 4, 5 (4.3, 0) | 0.79 / 0.64 |
| `oracle / 9b` (gold chunks, ceiling) | 40, 40, 40 | — | 34, 34, 33 | 13, 12, 12 (8.7, 3.7) | 1.00 / 1.00 |
| `agentic react / 9b` | 37, 38, 36 | 15, 15, 15 | 25, 24, 27 | 3, 2, 2 (2.3, 0) | 0.84 / 0.54 |
| `agentic planned / 9b` | 37, 37, 37 | 15, 15, 15 | 23, 28, 25 | 3, 3, 3 (3.0, 0) | 0.88 / 0.56 |
| `pipeline / 27b` | 38, 38, 39 | 15, 15, 15 | 26, 26, 26 | 6, 6, 6 (6.0, 0) | 0.79 / 0.64 |
| **`agentic react / 27b`** | 38, 39, 38 | 15, 14, 15 | **33, 32, 33** | **12, 11, 12** (9.0, 2.7) | **0.98 / 0.90** |
| `agentic react / 27b, think=low` | 38, 38 | 14, 15 | 33, 33 | 13, 11 (8.5, 3.5) | 0.98 / 0.92 |
| `agentic react / 27b + groundedness` | 38 | 14 | 33 | 12 (10.0, 2.0) | 0.96 / 0.86 |

Paired by question, each question's score being its pass (or complete) rate
over the row's runs, so sampling noise averages out without counting a
question three times. Δ in questions, 95% interval, wins/losses over
questions whose rates differ, exact sign-test p:

| comparison | answerable | refusals | multi-hop | adaptive |
|---|---|---|---|---|
| react / 9b vs pipeline / 9b (loop, 9b) | −1.7 [−3.8, +0.5], 0/3 | +1.0 [−1.0, +3.0], 1/0 | +0.7 [−5.5, +6.8], 7/6, p 1.0 | −2.0 [−4.5, +0.5], 1/4, p 0.38 |
| planned / 9b vs pipeline / 9b | −1.7 [−3.8, +0.5], 0/3 | +1.0 [−1.0, +3.0], 1/0 | +0.7 [−4.6, +6.0], 7/5, p 0.77 | −1.3 [−3.6, +0.9], 1/3, p 0.63 |
| pipeline / 27b vs pipeline / 9b (model) | −0.3 [−1.0, +0.3], 0/1 | +1.0 [−1.0, +3.0], 1/0 | +1.3 [−0.7, +3.4], 2/0, p 0.5 | +1.7 [−0.2, +3.5], 3/0, p 0.25 |
| **react / 27b vs pipeline / 27b (loop, 27b)** | +0.0 [−0.9, +0.9], 1/1 | −0.3 [−1.0, +0.3], 0/1 | **+6.7 [+2.1, +11.2], 7/0, p 0.016** | **+5.7 [+2.1, +9.2], 7/0, p 0.016** |
| react / 27b vs pipeline / 9b (both) | −0.3 [−1.0, +0.3], 0/1 | +0.7 [−0.6, +2.0], 1/0 | +8.0 [+3.2, +12.8], 9/0, p 0.004 | +7.3 [+4.0, +10.6], 10/0, p 0.002 |
| think=low vs react / 27b | −0.3 [−1.0, +0.3], 0/1 | −0.2 [−0.5, +0.2], 0/1 | +0.3 [−0.3, +1.0], 1/0 | +0.3 [−2.6, +3.2], 2/3 |

Cost per answering turn (the judge's calls excluded), mean over runs:

| row | LLM calls: ans / ref / multi-hop / adaptive | s/turn: same order | prompt tokens: multi-hop / adaptive / refusals |
|---|---|---|---|
| `pipeline / 9b` | 1.0 / 1.0 / 1.0 / 1.0 | 10 / 11 / 10 / 10 | 1,649 / 1,632 / 1,808 |
| `agentic react / 9b` | 2.0 / 2.0 / 2.1 / 2.2 | 12 / 13 / 17 / 15 | 4,950 / 5,156 / 4,220 |
| `agentic planned / 9b` | 2.0 / 2.0 / 2.0 / 2.0 | 13 / 13 / 18 / 14 | 4,110 / 3,449 / 3,258 |
| `pipeline / 27b` | 1.0 / 1.0 / 1.0 / 1.0 | 24 / 27 / 31 / 41 | 1,649 / 1,632 / 1,808 |
| `agentic react / 27b` | 2.4 / 6.6 / 2.7 / 4.0 | 36 / 158 / 74 / 108 | 9,598 / 15,888 / 34,945 |
| `agentic react / 27b, think=low` | 2.1 / 4.1 / 2.2 / 3.3 | 40 / 109 / 76 / 108 | 6,898 / 13,095 / 17,578 |
| `agentic react / 27b + groundedness` | 3.4 / 7.2 / 3.7 / 4.9 | 61 / 196 / 102 / 127 | 13,522 / 19,754 / 39,416 |

**Findings.**

- **The loop is what closes the multi-hop gap, and only with a model that
  searches again.** With the 27b, adding the loop gains 6.7 multi-hop and 5.7
  adaptive questions over the same model answering once, with 7 wins and no
  losses on each set. Evidence recall goes from 0.79 to 0.98 and from 0.64 to
  0.90: the agent finds the passages a single search misses. That brings it to
  within one question of the oracle on both sets (32.7 vs 33.7; 11.7 vs 12.3).
  The larger model alone, answering once, gains 1.3 and 1.7 questions, not
  shown at 95%, with identical evidence recall, because retrieval is
  unchanged. Both loop comparisons are p = 0.016. Against a Bonferroni
  threshold across all 24 tests in the table (0.002) they don't clear on
  their own; the 7/0 splits on two separately built sets, and the matching
  recall gain, are what make the result credible. React / 27b against the
  shipped `pipeline / 9b` clears it on the adaptive set (p = 0.002).
- **The 9b agent is no better than the pipeline, and worse on bridge and
  discovery questions.** Both 9b strategies make about two LLM calls a
  question, one round of searching and the answer, so the loop never runs a
  second round. That matches the [generator probe](#generator-model-qwen359b-vs-qwen3827b-pre-milestone-19).
  On the multi-hop set they are within noise (+0.7, wins and losses nearly
  even). On the adaptive set they retrieve less than the pipeline's single
  search on the whole question (0.54–0.56 vs 0.64 evidence recall), and none
  of the 9b rows completes a discovery question. `planned / 9b`, which the
  plan would have preferred for a default because it runs at 9b speed, didn't
  win.
- **The bottleneck was retrieval, not the 9b's reasoning.** Given gold chunks,
  the 9b completes 33.7 of 35 multi-hop and 12.3 of 15 adaptive questions.
  Phase 0's worry, from the multi-hop literature, that the generator would
  mis-combine evidence it already has, didn't hold on this corpus. When the
  pipeline retrieves all of a question's evidence it completes 94% of those
  questions, and 25% when evidence is partial. Planned's 83% completion with
  full evidence is the one exception, unexplained.
- **The multi-hop set alone could not have shown this.** 34 of its 35
  questions name every company and period they ask about, so one search
  already reaches most of the evidence (0.79). The adaptive set, where the
  question names an identifying fact or a class rather than the company,
  separates the rows more sharply: 4.3 for the shipped pipeline against 11.7
  for the agent.
- **Nothing is lost on single-hop and refusal questions.** The 27b agent holds
  answerable within 0.3 questions and refusals within one. The 9b agents'
  −1.7 on answerable (0 wins, 3 losses) is not shown at 95% but points the
  wrong way. The only refusal any row missed is the corrected
  `neg-unanswerable-comparison`: every `pipeline / 9b` run missed it, and no
  9b agent or `pipeline / 27b` run did. The 27b agent missed it in 3 of its 6
  runs, so declining "rank all 14 companies" is not something the loop
  reliably does.
- **The cost is what keeps it opt-in.** The 27b agent takes 36 s on a
  single-hop question, 74 s on a multi-hop one and 108 s on a bridge or
  discovery one, against 10 s, with 6–10× the prompt tokens. A refusal is the
  most expensive turn: 6.6 calls, 158 s and 35k prompt tokens, because the
  agent keeps searching before it declines.
- **`think=low` matches the 27b's quality at lower cost.** Within half a
  question on every set, with fewer calls and prompt tokens (6.9k vs 9.6k on
  multi-hop, 17.6k vs 34.9k on refusals) and the same latency. Two runs, so
  this is the setting to ship, not a measured improvement.
- **The groundedness check doesn't earn its cost here.** Over 104 checked
  answers it flagged 9, and the judge had passed 6 of those. It caught 3
  failures and missed 5. It adds 20–70% to latency, so it stays off,
  consistent with [CRAG's measurement](#crag-milestone-10-measured).
- **Judge check.** Read by hand: six of `react / 27b #1`'s adaptive answers.
  Four passes are correct. One pass is lenient: `ad-airline-margin-capex`
  gave Southwest's adjusted 6.7% margin rather than the reported 3.4%, which
  the rubric lists but which doesn't change the ranking. The two failures are
  real: `ad-pharma-ocf` read J&J's fiscal six-month figure as one quarter's,
  and `ad-tariff-refunds` missed that refunds raised Apple's product margin.

**Decision.** This meets the plan's
[default-flip criterion](milestone-19-plan.md#4--measure-the-milestones-actual-deliverable)
on quality: multi-hop gained by more than noise while single-hop and
refusals held. It meets it only with the 27b, at one to two minutes per hard
question, which is the case the plan named in advance: **agentic is an opt-in
mode for hard questions** (`chat.mode: agentic` with the 27b at
`think: low`), not the default. `chat.mode` stays `pipeline`. The scope note
at the top of this file applies: one corpus, questions written from its own
chunks, an LLM judge. The plan's MuSiQue follow-on is the outside check.

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
  Both also predate the chunk header becoming the default. Milestone 19
  phase 4 re-ran its own `pipeline / 9b` row at the current config
  ([above](#agentic-retrieval-milestone-19-phase-4)), so it no longer pairs
  against them. On 2026-09-29 the multi-hop set gained a 35th question and the
  refusal set's `neg-unanswerable-comparison` got a new rubric
  ([why](milestone-19-plan.md#the-superlative-question-stays-a-refusal)), so
  multi-hop totals out of 34 and earlier refusal results don't carry over.
- The Milestone 19 hosted reference pair (`--family m19-hosted`, Gemini
  Flash-Lite): built, not run, because it spends free-tier quota.
- Answers with metadata filters or document routing on; both phases measured
  retrieval only.
- `retrieval.top_k` between 20 and 100 with `bge-v2-m3`. 100 measured as noise
  against 20 (see the label-check re-run); intermediate values untested.
