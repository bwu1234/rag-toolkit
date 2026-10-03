# 0007 — Incremental indexing: diff ids to purge, refuse on a manifest mismatch

- **Status:** Accepted
- **Recorded:** 2026-10-03, retroactively, from
  [index maintenance notes](../milestone-notes.md#index-maintenance-notes)

## Context

Indexing is incremental: chunks whose content hash hasn't changed are
skipped. That left two gaps. Chunk ids are positional (`<doc>::chunk<n>`), so
deleting or shortening a document left stale chunks behind. And the hash only
covers `chunk.text`, so changing the embedder or the enrichment settings
mixed old and new vectors in one collection. Chroma only errors if the
dimensions differ.

## Decision

- **Purge by diffing ids.** At the end of a run, each store deletes the ids it
  holds that this run didn't produce. Each store is diffed on its own, so an
  interrupted run is repaired too. The purge never runs on an empty load, and
  it is skipped (exit 1) if any file failed to load.
- **A sidecar manifest** (`index_manifest__<slug>.json`) records the embedder
  and the index-text settings. `index` refuses to add to an index whose
  manifest differs, names the changed keys, and points at `--reset`.
  `build_retriever` refuses an embedder mismatch at query time.

## Alternatives considered

- **Recording which files existed last run.** Rejected: a third piece of
  state to keep consistent, when the stores already know what they hold.
- **Folding settings into every chunk's hash** (automatic re-embed). Rejected:
  an embedder with new dimensions would fail partway through and leave the
  index half-converted. A refusal costs the same rebuild and never does that.
- **Storing the manifest in Chroma collection metadata.** Rejected: it
  describes the vector and BM25 indexes together, and a new `VectorStore`
  backend shouldn't have to implement it.
- **Putting chunk size in the manifest.** Rejected: size changes alter text
  and ids, which the hash and purge already handle incrementally.

## Consequences

- A config change that invalidates vectors fails loudly instead of degrading
  retrieval silently.
- Indexes built before manifests existed are adopted (manifest written from
  the current config, with a warning), not rejected. A forced rebuild would
  have cost up to about 4 hours to check something that couldn't be verified.
- Generated contexts are cached by a hash of every input to the call, so they
  survive `--reset`. Re-embedding is cheap and regenerating contexts is not.
