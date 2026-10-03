# 0008 — LLM judgments fail open at runtime and fail closed when making labels

- **Status:** Accepted
- **Recorded:** 2026-10-03, retroactively, from
  [CRAG notes](../milestone-notes.md#corrective-rag-notes-milestone-10) and
  [evaluation notes](../milestone-notes.md#evaluation-pipeline-notes-milestone-7)

## Context

Many components ask a small LLM for a judgment: the query condenser, the
expanders, the contextualizer, CRAG's grader, retry rewriter and groundedness
checker, and the eval-set generator's verifier. Each can error or return
something unparseable. The right failure direction depends on what a wrong
outcome costs.

## Decision

- **Runtime components fail open.** An error or unparseable reply means
  "proceed as if this check hadn't run". An unreachable grader keeps the
  passages. A failed condense uses the original query. A broken checker must
  degrade the pipeline to plain RAG, never turn a working turn into a refusal.
- **Eval-label generation fails closed.** Any failed check rejects the sample,
  and every rejection reason is counted and printed. A bad label silently
  corrupts every measurement taken against the set afterwards.

`GroundednessChecker.check` returns `bool | None`: "unsupported" and
"couldn't tell" are different, and only the first justifies regenerating. An
answer still ungrounded after the regeneration budget is returned anyway,
flagged on `ChatAnswer.grounded`.

## Alternatives considered

- **Fail closed everywhere.** A flaky local model would refuse answerable
  questions.
- **Fail open everywhere.** The generator was observed producing spans that
  contradicted its own answers. Admitting those would poison the eval set.

## Consequences

- A broken judge is invisible in answers, so failures are logged and surfaced
  in pipeline events, and the asymmetry is asserted in tests.
- Eval-set yield is low (about 16–20% of attempts), which is the cost of
  trustworthy labels. A collapse in yield is treated as a signal.
