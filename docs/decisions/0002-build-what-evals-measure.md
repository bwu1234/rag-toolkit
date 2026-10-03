# 0002 — Build what the evals measure, adopt the plumbing; no framework at the core

- **Status:** Accepted
- **Recorded:** 2026-10-03, retroactively, from
  [architecture: build vs. adopt](../architecture.md#build-vs-adopt)

## Context

RAG frameworks (LangChain, LlamaIndex, LangGraph) and eval frameworks
(Inspect) offer most of this pipeline ready-made. But the repo's results are
claims about specific retrieval and generation choices. If those choices are
a framework's defaults, the numbers describe the framework.

## Decision

Stages whose behavior shows up in [measured results](../measured-results.md)
are written here: chunking, retrieval and fusion, reranking policy, the chat
loop, CRAG, the agent, eval scoring. Infrastructure is adopted behind the
interfaces ([0001](0001-swappable-interfaces.md)): Ollama and Gemini for
models, Chroma and `rank-bm25` / SQLite FTS5 for indexes, FastAPI, pydantic,
the `mcp` SDK; OpenTelemetry, Terraform and Cloud Run are planned.

## Alternatives considered

- **LangGraph for the CRAG loop.** Rejected: the graph is four nodes and a
  bounded loop. An orchestration dependency would add nothing over the `for`
  loop in `_retrieve_with_correction`
  ([CRAG notes](../milestone-notes.md#corrective-rag-notes-milestone-10)).
- **A framework text splitter.** Rejected: it hides the character offsets
  that citations and span-matched eval labels depend on.
- **Inspect for evals.** Deferred, not rejected. It adds about 40 direct
  dependencies, retrieval isn't a model output, and it has no paired
  comparison between runs. Its strengths (agent transcripts, model × benchmark
  grids) fit Milestone 19, so the question was moved there
  ([eval harness decisions](../eval-harness-plan.md#decisions-and-rejected-alternatives)).
- **MLflow.** Moved from rejected to a local pilot (eval harness Phase 4b),
  as an optional tracking export rather than the canonical store.

## Consequences

- More code to own: fusion, filtering, routing and the agent loop are
  maintained here.
- Every measured difference can be traced to a line in this repo.
- A new dependency must say what it replaces and why existing tools don't
  cover it. "A framework does this" is not enough on its own.
- Frameworks can still be used at the edges, as an eval comparator or as an
  MCP client, without becoming pipeline dependencies.
