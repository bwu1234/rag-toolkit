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
  as a gain.
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

### Not yet measured

- `retrieval.top_k` above 20 with the new reranker: `bge-v2-m3` gains from a
  larger pool where MiniLM lost, and only `top_k` 20 and 100 were measured.
- Cross-corpus interference (`--corpus baseline --corpus edgar` pooled vs
  isolated) — the registry supports it, no numbers taken.
- Anything on the `baseline` corpus, which is retained as a control and has not
  been re-measured since the reranker change.
