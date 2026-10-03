# 0011 — A feature ships off until a paired measurement shows it helps

- **Status:** Accepted
- **Recorded:** 2026-10-03, retroactively, from `CLAUDE.md` and
  [measured results](../measured-results.md)

## Context

Several well-known techniques were built behind config flags: contextual
chunking, CRAG, query expansion (HyDE, multi-query), a relevance floor, an
embedder query instruction, and document routing. Published gains come from
other corpora and other models.

## Decision

Each one ships disabled until a paired comparison on this repo's eval sets
shows a benefit, using McNemar's exact test on per-sample hits where it
applies. Enabling one requires re-measuring (the `measure-change` skill), and
the results go into `measured-results.md` with their caveats.

Current state: contextual chunking, CRAG, expansion, `min_score` and
`query_instruction` showed no sufficient benefit. Document routing gained on
the period tiers without clearing McNemar's test. All stay off.

## Alternatives considered

- **Enable features based on published results.** Rejected. Expansion
  measured below baseline at 12× the latency. The likely cause is rewrites
  dropping the company and period on a corpus of near-duplicate filings.
- **Delete features that didn't help.** Rejected. They are scoped results on
  one workload, not evidence the technique can't help elsewhere, and the code
  is cheap to keep behind a flag.

## Consequences

- The default config is plainer than a typical RAG demo, and every
  non-default can be justified with a number.
- The off-by-default code still needs maintenance and tests.
- Conclusions are scoped to the evaluated EDGAR tasks. Generalizing them
  needs the held-out and public-benchmark work in the
  [evaluation rigor plan](../evaluation-rigor-plan.md).
