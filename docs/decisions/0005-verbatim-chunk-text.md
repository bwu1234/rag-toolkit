# 0005 — `Chunk.text` stays verbatim; enrichment lives in separate fields

- **Status:** Accepted
- **Recorded:** 2026-10-03, retroactively, from
  [contextual chunking notes](../milestone-notes.md#contextual-chunking-notes-milestone-9)
  and [chunk header notes](../milestone-notes.md#chunk-header-notes-chunking-plan-phase-2)

## Context

Chunks often lose the terms that made them findable. A chunk reading "The
limit is 1,000 requests per minute" doesn't name its API. A table row doesn't
name its company or period. The fix is to add text at index time. But
citations, previews and eval spans all point at character offsets in the
source document, and nothing a user sees should be model-generated.

## Decision

Enrichment is stored in its own fields: `context` (an LLM-written situating
sentence, optional) and `header` (rendered from document metadata by a
template, deterministic). `Chunk.index_text` joins header, context and text
in one helper. Chunk, ScoredChunk and the BM25 records all use that helper,
so the embedded string and the keyword-indexed string cannot drift apart.
`Chunk.text` and `char_start`/`char_end` keep pointing at the source.

The one exception is the structured chunker
([0006](0006-structured-chunker.md)): a continuation piece of a split table
repeats the table's header rows inside `text`, because the model has to see
the column headers when answering.

## Alternatives considered

- **Rewriting `text` in place.** Simpler, but citations would show generated
  text and every span and offset would shift.
- **Regex-parsing titles for metadata.** Rejected in favour of YAML front
  matter written by the fetcher and stripped by the loader. It doesn't
  generalize beyond EDGAR, and it breaks silently when the title format
  changes.

## Consequences

- Both retrievers see the enrichment. The reranker sees the header only
  with `reranker.include_header`, because changing what the cross-encoder
  scores changes every reranker measurement.
- Index-affecting fields go in the index manifest
  ([0007](0007-index-integrity.md)), since the per-chunk content hash only
  covers `text`.
- Fields copied by rebuilding a dataclass field by field were silently
  dropped (RRF dropped `context`). Copies now use `dataclasses.replace`.
