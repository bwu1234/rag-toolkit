# 0004 — Two-stage retrieval: hybrid candidates fused by rank, then a cross-encoder

- **Status:** Accepted
- **Recorded:** 2026-10-03, retroactively, from
  [Milestone 5 notes](../milestone-notes.md#retrieval--reranking-notes-milestone-5)
  and [Milestone 11 findings](../measured-results.md#findings)

## Context

The EDGAR corpus is dense with entity names, tickers, periods and figures.
That is the text where pure embedding search is weakest. It also holds many
near-duplicate paragraphs that differ only by company or period.

## Decision

1. **Stage 1, candidates.** Every (query, source) pair produces one ranked
   list: dense always, BM25 when `retrieval.mode: hybrid` (the default), web
   search if enabled. `top_k` applies per list.
2. **Fuse by rank, never by score.** Reciprocal Rank Fusion compares only
   positions, so lists from different scoring functions and different query
   rewrites can be combined safely. A single list skips fusion, so plain
   dense retrieval is byte-for-byte unchanged.
3. **Stage 2, rerank.** A cross-encoder (`bge-reranker-v2-m3` today) narrows
   to `rerank_top_k`. Its output score is its own judgment normalized to
   `[0, 1]`, so `ScoredChunk.score` always means "what the last stage thought".
4. **Floor last.** `retrieval.min_score` applies after reranking, inside
   `Retriever`, so it works with any reranker, including none.

## Alternatives considered

- **Dense only.** Measured: dropping BM25 cost about 10 points of hit rate.
- **Score-weighted fusion.** Cosine similarity and BM25 scores aren't on
  comparable scales. RRF needs no calibration.
- **No reranker.** Measured: about 7 points of hit rate lower.
- **Filtering after top-k** (for metadata filters). Rejected: it silently
  returns fewer results than exist, most often exactly when the filter
  matters. Filters are applied inside both indexes before top-k
  ([filter notes](../milestone-notes.md#metadata-filter-notes-chunking-plan-phase-3)).

## Consequences

- The reranker is the bottleneck. Stage 1 hands it the right chunk about 98%
  of the time at `top_k=100`, and it fails to promote it. Reranker strength
  is worth more than retrieval tuning ([findings](../measured-results.md#findings)).
- The `min_score` value depends on the reranker's scale and has to be
  re-derived on every reranker change. It currently ships inert (0.0).
- Query expansion fit in with no new code path (more lists to fuse), but
  measured worse, so it is off ([0011](0011-measured-off-stays-off.md)).
