# Design decisions

The architectural decisions behind this repo, with the alternatives that were
weighed and what each choice costs. Two layers:

- **Decision records (ADRs)** below cover the decisions that shape the
  system. Each one is short: context, decision, alternatives considered,
  consequences.
- **The [decision map](#decision-map)** points every smaller decision at the
  section where it is already argued in full. Those sections stay the source
  of truth. An ADR summarizes them and links back.

For how the pieces fit together at runtime, read
[architecture](../architecture.md) first. For the numbers behind
"measured" claims, see [measured results](../measured-results.md).

## Decision records

| # | Decision | Area |
|---|---|---|
| [0001](0001-swappable-interfaces.md) | Every pipeline stage behind an interface, selected by config | Architecture |
| [0002](0002-build-what-evals-measure.md) | Build what the evals measure, adopt the plumbing; no framework at the core | Architecture |
| [0003](0003-local-first.md) | Local-first: Ollama and embedded Chroma; hosted providers as a config swap | Infrastructure |
| [0004](0004-hybrid-retrieval-rrf-rerank.md) | Two-stage retrieval: hybrid candidates fused by rank, then a cross-encoder | Retrieval |
| [0005](0005-verbatim-chunk-text.md) | `Chunk.text` stays verbatim; enrichment lives in separate fields | Indexing |
| [0006](0006-structured-chunker.md) | A hand-written structure-aware chunker is the default | Indexing |
| [0007](0007-index-integrity.md) | Incremental indexing: diff ids to purge, refuse on a manifest mismatch | Indexing |
| [0008](0008-fail-open-runtime-fail-closed-labels.md) | LLM judgments fail open at runtime and fail closed when making labels | Reliability |
| [0009](0009-pipeline-default-agentic-opt-in.md) | Always-retrieve pipeline by default; agentic search as an opt-in mode | Generation |
| [0010](0010-span-level-ground-truth.md) | Retrieval ground truth is verbatim quotes, not document ids or offsets | Evaluation |
| [0011](0011-measured-off-stays-off.md) | A feature ships off until a paired measurement shows it helps | Evaluation |
| [0012](0012-stateless-api-visible-interventions.md) | Stateless API; every silent intervention travels with the answer | API |
| [0013](0013-turn-log-observability.md) | One append-only turn log, metered at the client, written only by human-facing entrypoints | Observability |
| [0014](0014-mcp-server-scope.md) | MCP server: read-only tools, one protocol revision | Integration |
| [0015](0015-agentic-default.md) | Agentic search is the default; supersedes 0009 | Generation |

0001–0014 were recorded on 2026-10-03, after the fact, from the docs they
link to. 0015 was decided that day.

### Adding a record

Copy an existing record and take the next number. Write one when a change
picks between real alternatives that a later reader could reasonably
reopen. A record is never edited to reverse its decision. Instead, a new
record supersedes it, and the old one's status line points at the new one.
Detailed rationale can still go in the milestone notes or a plan doc, with
the ADR linking to it.

## Decision map

Smaller decisions, and where each is argued. Links into
[milestone notes](../milestone-notes.md) unless another doc is named.

### Ingestion and chunking

| Decision | Where |
|---|---|
| Document ids are readable corpus-relative paths, not hashes | [Ingestion](../milestone-notes.md#ingestion-notes-milestone-2) |
| Cleaning is a separate pass from loading | [Ingestion](../milestone-notes.md#ingestion-notes-milestone-2) |
| Character-based (not token-based) chunk sizing | [Chunking](../milestone-notes.md#chunking-notes-milestone-3) |
| Metadata via YAML front matter, stripped from `Document.text` | [Chunk header](../milestone-notes.md#chunk-header-notes-chunking-plan-phase-2) |
| Header is all-or-nothing per document | [Chunk header](../milestone-notes.md#chunk-header-notes-chunking-plan-phase-2) |
| Semantic, LLM-driven and late chunking not pursued; XBRL left to the agent | [Chunking plan](../chunking-indexing-plan.md#decisions-and-rejected-alternatives) |
| `section_path` stored as metadata, not in the header | [Structured chunker](../milestone-notes.md#structured-chunker-notes-chunking-plan-phase-5) |

### Indexing

| Decision | Where |
|---|---|
| `embed_documents` / `embed_query` split for asymmetric models | [Embedding & indexing](../milestone-notes.md#embedding--indexing-notes-milestone-4) |
| Contextualization runs after the change check; concurrency capped at 4 | [Contextual chunking](../milestone-notes.md#contextual-chunking-notes-milestone-9) |
| Context cache keyed on every call input, kept across `--reset` | [Contextual chunking](../milestone-notes.md#contextual-chunking-notes-milestone-9) |
| Per-selection storage names so isolated and pooled indexes coexist | [Named corpora](../milestone-notes.md#named-corpora-notes-shipped-with-milestone-11) |
| Unknown corpus raises; duplicate ids refused, not namespaced | [Named corpora](../milestone-notes.md#named-corpora-notes-shipped-with-milestone-11) |

### Retrieval

| Decision | Where |
|---|---|
| `min_score` lives in `Retriever`, after reranking | [Retrieval & reranking](../milestone-notes.md#retrieval--reranking-notes-milestone-5) |
| `ExpandedQuery` has separate dense, sparse and rerank lists | [Retrieval & reranking](../milestone-notes.md#retrieval--reranking-notes-milestone-5) |
| `reranker.aggregate: max` over `mean` | [Retrieval & reranking](../milestone-notes.md#retrieval--reranking-notes-milestone-5), [findings](../measured-results.md#findings) |
| Reranker scores `text`, header only with `include_header` | [Chunk header](../milestone-notes.md#chunk-header-notes-chunking-plan-phase-2) |
| One typed `QueryFilter`; filter before top-k; unknown field is an error | [Metadata filters](../milestone-notes.md#metadata-filter-notes-chunking-plan-phase-3) |
| Web search skipped under any filter | [Metadata filters](../milestone-notes.md#metadata-filter-notes-chunking-plan-phase-3) |
| Document routing gated on BM25/dense agreement, not a score floor | [Document routing](../milestone-notes.md#document-routing-notes-chunking-plan-phase-3b) |
| SQLite FTS5 as an alternative sparse backend (not the default) | [Measured results](../measured-results.md#sqlite-fts5-sparse-backend) |

### Generation

| Decision | Where |
|---|---|
| Multi-turn handled by condensing, not by passing history | [Chat API](../milestone-notes.md#chat-api-notes-milestone-6) |
| Numbered passages are the single source of `[n]` → citation | [Chat API](../milestone-notes.md#chat-api-notes-milestone-6) |
| CRAG owned by `ChatService`, not `Retriever` | [CRAG](../milestone-notes.md#corrective-rag-notes-milestone-10) |
| Grading uses the user's question, never a retry rewrite | [CRAG](../milestone-notes.md#corrective-rag-notes-milestone-10) |
| `ToolCallingLLM` subclass, `agent.llm`, shared tool surface, passage ledger, loop guards | [Milestone 19 plan](../milestone-19-plan.md#decisions) |

### Evaluation

| Decision | Where |
|---|---|
| Generated EDGAR set, with stated lexical bias | [Evaluation](../milestone-notes.md#evaluation-pipeline-notes-milestone-7) |
| Refusal sets kept separate from retrieval metrics | [Evaluation](../milestone-notes.md#evaluation-pipeline-notes-milestone-7) |
| Inspect deferred, MLflow piloted, SQLite for tracking, no nested run folders | [Eval harness plan](../eval-harness-plan.md#decisions-and-rejected-alternatives) |
| Eval runners never write the turn log | [Observability](../milestone-notes.md#observability-notes-milestone-12) |

### Interfaces and operations

| Decision | Where |
|---|---|
| FastAPI builds one service in `lifespan`; `Depends` for test overrides | [Chat API](../milestone-notes.md#chat-api-notes-milestone-6) |
| Wire schemas separate from pipeline types | [Chat API](../milestone-notes.md#chat-api-notes-milestone-6) |
| Streamlit logic split into a testable `helpers.py` | [Streamlit UI](../milestone-notes.md#streamlit-ui-notes-milestone-8) |
| Indexing not exposed as an MCP tool | [MCP server](../mcp-server.md#why-indexing-is-not-a-tool) |
| One MCP protocol revision, older ones refused | [MCP server](../mcp-server.md#protocol-revision) |
