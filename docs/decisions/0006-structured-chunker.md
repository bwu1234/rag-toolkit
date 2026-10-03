# 0006 — A hand-written structure-aware chunker is the default

- **Status:** Accepted (supersedes fixed-size as the default, 2026-10-01)
- **Recorded:** 2026-10-03, retroactively, from
  [structured chunker notes](../milestone-notes.md#structured-chunker-notes-chunking-plan-phase-5)
  and [chunking plan decisions](../chunking-indexing-plan.md#decisions-and-rejected-alternatives)

## Context

The original `FixedSizeChunker` used character windows snapped to whitespace.
This was a deliberate simplicity tradeoff: no tokenizer dependency. Its one
measured boundary defect was that it split tables away from their header
rows, and `table`-tier failures were answers drawn from rows whose column
headers the model never saw.

## Decision

`StructuredChunker` packs Markdown blocks (headings, pipe tables, paragraphs,
list lines) recognized by three regular expressions, keeping every block's
character offsets. Trailing headings and short stubs move forward with the
unit they introduce. Split tables repeat their header rows. Prose overlap
only fills spare room under the size cap. EDGAR evals moved to `edgar_md`, a
Markdown rendering of the same filings with the same document ids.

## Alternatives considered

- **A Markdown library's AST.** Its nodes would have to be mapped back to
  character offsets. The loaders only emit four block types.
- **Semantic chunking** (cut where adjacent-sentence similarity drops).
  External evidence supports it least, and the measured defect was
  structural, not semantic.
- **LLM-driven chunking.** Rejected on cost: one call per chunk, about 4
  hours for EDGAR.
- **Late chunking.** Deferred. It needs token-level embeddings, which
  Ollama's `/api/embed` doesn't return.
- **Token-based sizing.** Still a possible upgrade behind `Chunker`. Not
  needed yet.

## Consequences

- The size cap can be exceeded by a lead-in heading (39 of 4,808 chunks on
  `edgar_md`, at most 1,164 characters).
- A coverage test checks that chunks cover every non-whitespace character of
  every document on every corpus.
- Results recorded before 2026-10-01 are on `edgar` with the fixed chunker
  and must not be paired against new rows.
- `section_path` is metadata, not header, so routing records didn't change in
  the same step. One variable per measurement.
