---
name: measure-change
description: Measure whether a retrieval or generation change actually helps before enabling it, using scripts/run_matrix.py and the eval runners, then record the numbers in docs/measured-results.md. Use when asked to evaluate/benchmark a config change, compare variants or models, decide whether to turn a feature on by default, tune a knob (top_k, min_score, rerank_top_k, chunk size), or interpret an eval run.
---

# Measuring a change

The governing rule in this repo: **a feature stays off until it beats noise on a
real corpus.** Contextual chunking, CRAG, query expansion and `min_score` all
shipped functionally correct and all measured as no better than noise — they are
off by default because of numbers, not doubt. Read `docs/measured-results.md`
before proposing any of them again.

## Before running anything

1. **Read `docs/measured-results.md`.** The change may already be measured.
2. **Pick the corpus deliberately.** `baseline` is 8 documents, ~26k chars —
   `top_k 20` retrieves ~65% of it, so retrieval metrics saturate and every
   variant looks the same. It is a control, not a measurement. Measure on
   `edgar` (61 filings, ~4,200 chunks) with
   `data/eval/edgar_eval_set.json` (174 samples).
3. **Confirm the index matches the config.** Collection names carry the corpus
   selection slug (`rag_corpus__edgar`). Anything under `chunking.*` requires a
   rebuild (`python -m rag.cli index --corpus edgar --reset`) — a config flip
   alone measures nothing. Everything else is a pure query-time knob.

## Running the matrix

`scripts/run_matrix.py` is the harness. It is retrieval-only (no LLM), so the
whole matrix runs in minutes against one index.

```bash
python scripts/run_matrix.py --list                    # variants, run nothing
python scripts/run_matrix.py --corpus edgar --eval-set data/eval/edgar_eval_set.json
python scripts/run_matrix.py --only reranker,min_score --corpus edgar
python scripts/run_matrix.py --variant "rr=bge-base" --merge --corpus edgar
```

Key flags: `--only <axes>` and `--variant <name>` narrow the run; `--merge` adds
to the existing results file instead of re-running hours of already-measured
variants; `--skip-llm` drops the expansion variants (12x the latency);
`--label` stores free text with the results.

Results checkpoint after **every** variant to `data/eval/results/` as
`retrieval_<slug>_<evalset>.json` plus a rendered `.md` table.

To add an axis, append a `Variant` to `VARIANTS` in the script. Two rules the
existing entries follow:

- **One factor at a time.** Every variant differs from `baseline` in exactly one
  setting, so a difference is attributable. A full factorial is dozens of runs
  that answer nothing until the main effects are known.
- **Pin confounders explicitly.** The reranker-model variants all pin
  `retrieval.min_score: 0.0`, because a floor tuned for one model's score
  distribution would confound "is this reranker better" with "does it happen to
  score above an arbitrary threshold".

If the change touches a knob not in `FINGERPRINTED`, add it there too — a metric
whose config isn't captured is unreproducible, and two runs are only comparable
when their fingerprints match.

Answer-side changes (CRAG, prompts, groundedness) use
`scripts/run_answer_matrix.py` instead — it costs LLM calls per sample and takes
hours, so run it only after the cheap retrieval measurements are in.

## Reading the results

Metrics: `hit_rate`, `recall`, `precision`, `mrr`, `ndcg`, plus `recall_by_k`.
The rendered table shows Δ hit and Δ NDCG against `baseline`, **paired by
sample**, with a 95% CI (`*` when it excludes zero) and, for hit, the win/loss
count of questions that flipped. The answer matrix does the same for pass rate
against its first variant, and splits failures into retrieval (gold span never
reached the prompt) and generation (it did; the answer still failed).

- **Read the paired CI, not the unpaired SE.** Both runs answer the same
  questions, so the noise is in the questions they disagree on. Don't judge a
  difference against `sqrt(p(1-p)/n)`. With few discordant pairs, trust the
  sign-test p-value (McNemar's, for hit) over the interval. Report the CI and
  the W/L alongside the delta.
- **A CI that includes zero means "not shown", not "no effect".** State that
  plainly rather than reporting it as a win or as proof of no effect.
- Rows marked `(no CI)` predate per-sample storage; re-run them to test them.
- **Byte-identical rows mean the knob is inert**, not that it is safe. `min_score`
  0.0 / 0.1 / 0.3 gave identical results — the floor was doing nothing at all.
- **Expansion is non-deterministic.** One run is a data point, not a result.
- **Separate the two failure modes.** A variant with `rerank_top_k == top_k`
  reranks nothing, so its recall is the stage-1 ceiling. The gap to a filtering
  variant is the cost of reranking; the gap to 1.0 is what retrieval never found.
  Those need different fixes and the current answer is that reranking is the
  bottleneck (stage 1 finds the right chunk 97.7% of the time).

## Recording the result

Append to `docs/measured-results.md`, matching the existing structure: a
**Setup** paragraph (corpus, index type, eval set, sample count, the defaults the
run varied from), the results table, then **Findings** as prose bullets that say
what the number *means*, not just what it is.

Record negative results too, with the reasoning. Half that document is features
that did not work, and that is what stops them being re-proposed.

Then:

- Update the `config.yaml` comment for any knob whose value or justification
  changed — a default set on evidence should say so inline, as `reranker.model`
  and `retrieval.min_score` already do.
- Update the Milestone 11 line in `CLAUDE.md` only if the headline changes.
- **Only now** change a default. If the numbers do not clear noise, leave the
  feature off and say so.
