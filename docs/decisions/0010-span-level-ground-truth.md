# 0010 — Retrieval ground truth is verbatim quotes, not document ids or offsets

- **Status:** Accepted
- **Recorded:** 2026-10-03, retroactively, from
  [evaluation notes](../milestone-notes.md#evaluation-pipeline-notes-milestone-7)

## Context

Document-level labels say "the answer is somewhere in this filing". On a
60k-character filing (about 70 chunks), precision@k reads 1.0 while every
result is boilerplate from the wrong part of the document. The granularity
also depended on file format (a PDF page vs. a whole Markdown file).

## Decision

A retrieved chunk is relevant if it contains one of the sample's
`expected_spans`: verbatim quotes, matched against normalized text. Spans are
kept no longer than `chunk_overlap`, so some chunk must contain them. Optional
grades feed NDCG. Metric functions take per-rank gains, so the same metrics
serve span and document labels, which are reported separately. The EDGAR set
is generated from sampled chunks, then verified by a second LLM pass that
fails closed ([0008](0008-fail-open-runtime-fail-closed-labels.md)).

## Alternatives considered

- **Character offsets.** Rejected: any change to cleaning or chunking
  invalidates them, and comparing chunking configs is the reason the eval
  exists.
- **Chunk ids.** Same problem: ids are positional and change with chunking.
- **Hand-written labels only.** 3.6M characters can't be labelled by hand,
  and a set covering 1% of the corpus mostly measures which 1% was picked.

## Consequences

- Generated questions are lexically closer to their passages than real ones,
  which inflates absolute scores. The set is for comparing configurations on
  identical questions, not for headline quality claims
  ([evaluation rigor plan](../evaluation-rigor-plan.md)).
- Spans must be unique across the corpus, since boilerplate repeated across
  filings can't identify a passage.
- Pooling refuses duplicate document ids rather than namespacing them,
  because namespacing would invalidate every recorded `expected_doc_ids`
  ([named corpora notes](../milestone-notes.md#named-corpora-notes-shipped-with-milestone-11)).
