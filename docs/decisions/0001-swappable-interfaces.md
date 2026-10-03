# 0001 — Every pipeline stage behind an interface, selected by config

- **Status:** Accepted
- **Recorded:** 2026-10-03, retroactively, from `CLAUDE.md` and
  [architecture](../architecture.md#cross-cutting-pieces)

## Context

The project's purpose is to compare implementations — embedders, rerankers,
generators, chunkers, sparse indexes — on the same eval sets. A comparison is
only clean if switching the implementation changes nothing else in the
pipeline.

## Decision

Each swappable stage is an abstract interface: `EmbeddingModel`,
`VectorStore`, `Reranker`, `LLMClient`, `QueryExpander`, `Chunker` (and later
`SparseIndex`, `TurnSink`). Each has a factory (`get_embedder`,
`get_reranker`, ...) that maps a `provider:` string in `config.yaml` to a
concrete class. Pipeline code names only the interfaces and the two builders
(`build_retriever`, `build_chat_service`). All tunables live in
`rag/config/config.yaml`, validated by pydantic.

Interfaces are kept as small as their callers need. `LLMClient` is one method;
tool calling is a subclass (`ToolCallingLLM`) rather than a wider base, so the
14 `generate()` call sites and every test fake don't have to implement it
([Milestone 19 decision 1](../milestone-19-plan.md#decisions)).

## Alternatives considered

- **Concrete classes wired directly.** Less code, but every comparison would
  be a code edit, and a variant could differ in more than the one thing it was
  meant to test.
- **Wider interfaces up front** (e.g. tool calling on `LLMClient`). Rejected:
  forces every adapter and fake to implement capabilities most callers never use.

## Consequences

- An experiment is a config file. `base:` inheritance (`vanilla.yaml`,
  `beir.yaml`) keeps variants to the lines they change.
- Adding a backend is a recipe, not a refactor (the `add-provider` skill).
  `SqliteFts5Index` and the `sentence_transformers` embedder went in this way.
- Swapping is not automatically *correct*. A BGE reranker swapped in without
  forcing raw logits double-applied a sigmoid; ranking looked fine and
  `min_score` was silently broken ([findings](../measured-results.md#findings)).
  Adapters need tests on output *semantics*, not just shape.
- Capability mismatches fail at build time (`build_agent_llm` refuses a
  provider without tool calling), not mid-request.
