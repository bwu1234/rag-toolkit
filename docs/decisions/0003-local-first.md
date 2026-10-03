# 0003 — Local-first: Ollama and embedded Chroma, hosted providers as a config swap

- **Status:** Accepted
- **Recorded:** 2026-10-03, retroactively, from `CLAUDE.md` and
  [Milestone 4 notes](../milestone-notes.md#embedding--indexing-notes-milestone-4)

## Context

Eval runs make thousands of embedding and LLM calls: matrices, repeats, a
roughly 4-hour contextual build. Paying per call or being rate-limited would
cap how much gets measured. Hosted models are still a real deployment target.

## Decision

Ollama serves both embeddings and chat. Chroma runs in persistent local mode:
one on-disk directory, no server process, cosine similarity. Everything runs
on one machine. Hosted providers (Gemini today) go behind the same `LLMClient`
interface and are selected in config. They are not a separate code path.

Supporting choices:

- HTTP clients use `trust_env=False`, so a loopback call is never routed
  through a system proxy.
- Embedding dimensions and cross-encoder weights are discovered lazily, so
  building the pipeline (in tests, or to load config) never needs a running
  daemon or downloaded model.
- Free-tier hosted use is bounded by `llm.requests_per_day`.

## Alternatives considered

- **Hosted-only** (API embeddings and generation). Simpler to set up, but
  every measurement costs money or quota, and runs are subject to provider
  model changes.
- **A vector database server** (Qdrant, pgvector, Chroma server). Not needed
  at this corpus size (thousands of chunks). It would add a process to run.
  `VectorStore` keeps the option open.
- **A lighter reranker via the existing Ollama client.** Rejected in favour
  of a purpose-built `sentence-transformers` cross-encoder. The cost is
  `torch` (about 2 GB) to get accuracy from a proven model.

## Consequences

- Model quality is bounded by what fits locally (9b/27b generators). Agentic
  mode only pays off with the 27b ([0009](0009-pipeline-default-agentic-opt-in.md)).
- Latency is GPU-bound and shared, which is why the agent has a wall-clock
  guard.
- Results are reproducible offline, and the Gemini reference rows give one
  hosted comparison point.
