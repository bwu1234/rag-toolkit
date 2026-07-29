# RAG_Project

Retrieval-Augmented Generation system, built up milestone by
milestone. Local-first by default: Ollama serves both embeddings
(`qwen3-embedding:0.6b`) and chat (`qwen3.5:9b-mlx`); Chroma is the vector store.

## Status

- [x] Milestone 1 — Scaffolding (config system, logging, test setup)
- [x] Milestone 2 — Ingestion (PDF + Markdown loaders, cleaning, metadata)
- [x] Milestone 3 — Chunking
- [x] Milestone 4 — Embedding + vector index
- [x] Milestone 5 — Retrieval + reranking
- [x] Milestone 6 — Chat API (FastAPI)
- [x] Milestone 7 — Evaluation pipeline
- [x] Milestone 8 — Streamlit UI

## Architecture

Everything pipeline-relevant sits behind a small set of interfaces so
implementations can be swapped via config alone:

| Interface | Location | Default implementation |
|---|---|---|
| `EmbeddingModel` | `rag/embedding/base.py` | Ollama (`qwen3-embedding:0.6b`) |
| `VectorStore` | `rag/vectorstore/base.py` | Chroma (persistent, local) |
| `Reranker` | `rag/retrieval/reranker.py` | none (pure vector retrieval) initially |
| `LLMClient` | `rag/generation/llm.py` | Ollama (`qwen3.5:9b-mlx`, via `/api/chat`) |

To add a new implementation: subclass the relevant ABC, register it in that
module's factory function, and select it via `provider:` in
`rag/config/config.yaml`. Pipeline code never references concrete classes —
only the interfaces and the config-driven factories.

## Config

All component selection and tunables live in `rag/config/config.yaml`,
validated by pydantic models in `rag/config/settings.py`. Load it via
`load_config()`. Don't hardcode model names, chunk sizes, or paths in pipeline
code — read them from `RagConfig`.

## Folder layout

```
rag/
  config/       settings.py (pydantic models), config.yaml
  ingestion/    document loaders + text cleaning
  chunking/     configurable chunkers
  embedding/    EmbeddingModel interface + adapters
  vectorstore/  VectorStore interface + Chroma adapter
  retrieval/    Retriever + Reranker
  generation/   LLMClient interface + Ollama adapter, prompt templates
  api/          FastAPI app and routes
  eval/         retrieval & answer evaluation scripts + eval set
  ui/           Streamlit app
tests/          pytest suite, mirrors rag/ layout
data/corpus/    input documents (gitignored — user-provided)
data/index/     persisted Chroma index (gitignored — rebuildable)
```

## Running things

(Filled in as each milestone lands.)

- Run tests: `pytest`
- Ingest & inspect the corpus: `python -m rag.cli ingest --show 3`
- Chunk & inspect chunk sizes: `python -m rag.cli chunk --show 3`
- Build the index: `python -m rag.cli index` (add `--reset` to rebuild from scratch)
- Retrieve & rerank for a query: `python -m rag.cli retrieve "your question"`
- Ask a question end to end (retrieve → rerank → generate, with citations): `python -m rag.cli chat "your question"`
- Start the API: `uvicorn rag.api.main:app --reload` (then `POST /chat` with `{"query": "..."}`, or check `/health`)
- Start the UI: `streamlit run rag/ui/app.py`
- Run retrieval eval: `python -m rag.eval.retrieval_eval` (add `-v` for per-sample detail)
- Run answer eval (LLM-as-judge): `python -m rag.eval.answer_eval`

## Conventions

- Strong typing throughout — all public functions/classes are annotated;
  pydantic models for config and API schemas, plain typed dataclasses for
  internal data (`Document`, `Chunk`, etc.).
- `logging.getLogger(__name__)` per module; call `configure_logging()` once at
  each entrypoint (CLI, API, UI) — never at import time.
- Tests live in `tests/`, mirroring the `rag/` package structure.
- Keep dependencies minimal — justify any new dependency against what's
  already available (e.g. don't add a second HTTP client, a second YAML
  parser, etc.).

## Ingestion notes (Milestone 2)

- `Document` granularity is loader-defined: PDF loaders emit one `Document`
  per page (so citations can reference a page number); Markdown/text loaders
  emit one `Document` per file. Chunking treats both uniformly.
- `Document.id` is a human-readable, corpus-relative path
  (`"sub/dir/file.pdf#page=3"`), not a hash — easier to trace a chunk or
  citation back to its exact source file and page while debugging or
  authoring an eval set.
- Cleaning is a separate, explicit pass (`rag.ingestion.cleaners.clean_documents`)
  rather than baked into loaders, so extraction and normalization can evolve
  — or be skipped for debugging — independently.

## Chunking notes (Milestone 3)

- Default strategy is `FixedSizeChunker`: character-based windows
  (`chunking.chunk_size` / `chunk_overlap` in config) with both the start and
  end of each window snapped to the nearest whitespace, so words are never
  split mid-token at either edge — including across overlapping regions.
- `Chunk.id` mirrors the `Document.id` scheme:
  `"<document-id>::chunk<index>"`, e.g. `"handbook.pdf#page=2::chunk0"`.
- Chunks carry forward a small allowlist of document metadata (`title`,
  `page`, `page_count`) plus their own (`chunk_index`, `char_start`,
  `char_end`), so a chunk is self-describing for citations without needing
  its parent `Document`.
- Chunking assumes already-cleaned text — run `clean_documents()` first (the
  CLI's `chunk` command does this for you).
- Character-based (not token-based) chunking is a deliberate simplicity
  tradeoff: no tokenizer dependency, but chunk sizes don't map precisely onto
  an embedding/LLM's token budget. A token-aware chunker is a natural future
  upgrade behind the same `Chunker` interface.

## Embedding & indexing notes (Milestone 4)

- `EmbeddingModel` exposes `embed_documents` / `embed_query` separately (not
  one method) so asymmetric models — ones that need different instruction
  prefixes for indexing vs. querying — can be supported later without
  changing the interface; `OllamaEmbedder` currently treats them identically.
- `OllamaEmbedder` talks to a local Ollama daemon's `/api/embed` endpoint over
  `httpx`, batching client-side (`_DEFAULT_BATCH_SIZE = 32`) so request size
  stays predictable regardless of corpus size. It builds its HTTP client with
  `trust_env=False` — a loopback connection to Ollama should never go through
  a system proxy.
- Embedding dimensionality (`EmbeddingConfig.dimensions`) is optional in
  config: if unset, `OllamaEmbedder` discovers it lazily from the first
  response and caches it, so constructing the embedder never requires a
  running daemon (useful for tests and for just loading config).
- `VectorStore` is keyed by `Chunk.id` and `upsert` is idempotent — re-running
  `index` after small corpus edits replaces existing vectors in place rather
  than duplicating them. Use `index --reset` to wipe the collection after a
  chunking/embedding config change that invalidates old ids or vector
  dimensions.
- `ChromaVectorStore` uses Chroma's persistent local mode (a single on-disk
  directory under `paths.index_dir`, no server process) configured for cosine
  similarity (`hnsw:space: cosine`). Chroma metadata values must be flat
  primitives (`str | int | float | bool`, no `None`/`Path`/nested data), so
  the adapter flattens `Chunk` provenance + metadata on the way in and
  reconstructs a typed `ScoredChunk` (with `score` normalized to
  `[0, 1]`-similarity, converted from Chroma's reported cosine distance) on
  the way out — pipeline code never deals with Chroma's wire format directly.
- The `index` CLI command runs the full load → clean → chunk → embed → upsert
  pipeline and is the first command in the project with real side effects and
  cost (one HTTP round trip per embedding batch) — unlike the read-only
  `ingest`/`chunk` preview commands.
- Embedding/vector-store tests are hermetic: `OllamaEmbedder` is tested
  against a mocked HTTP transport (`httpx.MockTransport`, no daemon required),
  and `ChromaVectorStore` against a real temp-directory-backed Chroma instance
  with deterministic orthogonal-vector fixtures (predictable cosine
  similarity) rather than mocks — the adapter's job is format translation, best
  verified end to end.

## Retrieval & reranking notes (Milestone 5)

- Retrieval is a two-stage retrieve→rerank pipeline: `Retriever.retrieve`
  embeds the query, pulls `retrieval.top_k` candidates from the `VectorStore`
  by vector similarity, then narrows them to `retrieval.rerank_top_k` via a
  `Reranker`. Both stages return `ScoredChunk`, so the API and eval pipeline
  downstream never need to know whether reranking ran.
- `Reranker` is a small ABC (`rerank(query, candidates, top_k) -> list[ScoredChunk]`)
  with two adapters: `NoOpReranker` (pure pass-through/truncation — preserves
  incoming vector-similarity scores, the one documented exception to the rule
  below) and `CrossEncoderReranker` (wraps a `sentence-transformers`
  cross-encoder, selected via `reranker.provider: cross_encoder` and
  `reranker.model` in config, defaulting to `cross-encoder/ms-marco-MiniLM-L-6-v2`).
- Convention: a reranker's output `score` reflects *that reranker's own
  judgment* in `[0, 1]`, not the incoming vector-similarity score — so a
  `ScoredChunk.score` always means "how relevant did the most recent stage
  consider this," whichever stage ran last. `CrossEncoderReranker` squashes
  the model's raw logits through a sigmoid (`normalize_rerank_score`) to land
  in that same `[0, 1]` convention as cosine-similarity scores.
- `CrossEncoderReranker` loads the underlying `sentence_transformers.CrossEncoder`
  model lazily on first `rerank()` call (not at construction), and caches it —
  mirroring `OllamaEmbedder.dimensions`'s lazy-discovery pattern. This means
  building/wiring a `CrossEncoderReranker` (e.g. via the factory, or at CLI
  startup) never requires the model weights to be downloaded; only an actual
  `retrieve` call with `reranker.provider: cross_encoder` does.
- Adding `sentence-transformers` (and its `torch` dependency, ~2GB) was a
  deliberate tradeoff for accurate local cross-encoder reranking over a
  lighter-weight option (e.g. an LLM-based reranker via the existing Ollama
  client) — the user chose accuracy and a proven, purpose-built model over
  minimizing dependencies for this component.
- Retrieval/reranker tests are hermetic and don't require model weights or a
  running daemon: `Retriever` is tested against fakes for `EmbeddingModel`,
  `VectorStore`, and `Reranker` that record what they're called with;
  `CrossEncoderReranker` is tested by swapping its lazily-loaded `_model` for
  a fake that returns deterministic logits (verifying rescoring, sigmoid
  normalization, and re-sorting), plus a separate test that stubs the
  `sentence_transformers` module via `sys.modules` to verify the lazy-loading
  and caching behavior itself without the real package installed.

## Chat API notes (Milestone 6)

- `ChatService` (`rag/generation/chat_service.py`) is the seam between
  retrieval and generation: it embeds/retrieves/reranks via the existing
  `Retriever`, renders a numbered, citeable context block with
  `build_rag_prompt`, asks the configured `LLMClient` to answer grounded in
  it, and projects the result into a flat `ChatAnswer`/`Citation` pair. It's
  deliberately framework-free — reusable by the FastAPI route, the `chat` CLI
  command, and (Milestone 7) the eval pipeline without standing up a server.
- `LLMClient` is a one-method ABC (`generate(prompt, *, system=None) -> str`)
  — the smallest surface the chat service needs. `OllamaLLMClient` talks to
  Ollama's `/api/chat` endpoint (messages array, not raw-prompt `/api/generate`)
  so the system prompt and user prompt stay cleanly separated, and translates
  `LLMConfig.temperature`/`max_tokens` into Ollama's `options.temperature`/
  `options.num_predict` — the chat service never needs to know provider-specific
  option names. Same `trust_env=False` loopback rule as `OllamaEmbedder`.
- `LLMConfig.provider` already accepts `"anthropic"`/`"openai"` as valid config
  *values* (so a user can name an intended future provider without failing
  validation), but `get_llm_client` raises a clear "recognized but not
  implemented" error for them — distinct from "unknown provider" — until
  adapters exist.
- `build_rag_prompt` numbers retrieved passages (`Passage [1]`, `Passage [2]`,
  ...) labeled with their source document id and page (when known); the system
  prompt instructs the model to answer only from those passages and cite them
  inline by number. That numbering is the single source of truth mapping a
  model's `[n]` citation back to a `Citation` in the response — both are
  derived from the same `chunks` list in the same order.
- `ChatService.ask` short-circuits (skipping both retrieval and generation, or
  just generation) for a blank query or empty retrieval results — both are
  cases where a generated answer could only be a hallucination. This keeps the
  no-context path fast, free, and honest about what the corpus does/doesn't
  contain.
- The FastAPI app (`rag/api/main.py`) builds one `ChatService` at startup via
  a `lifespan` handler and stores it on `app.state` — routes never re-read
  config or reconnect to Ollama/Chroma per request. `routes/chat.py` exposes
  it through a `Depends`-based dependency (`get_chat_service`) specifically so
  tests can override it (`app.dependency_overrides[get_chat_service] = ...`)
  with a fake, exercising real routing/validation/(de)serialization without a
  running daemon, index, or lifespan.
- `ChatRequest`/`ChatResponse`/`CitationModel` (`rag/api/schemas.py`) are kept
  separate from `ChatAnswer`/`Citation` (the internal pipeline types) — the
  wire contract can evolve (versioning, extra fields) independently of
  pipeline internals, and routes are the only place that translates between
  them.
- API tests are hermetic: `test_llm.py` mocks Ollama's HTTP transport
  (mirroring `test_embedding.py`), `test_chat_service.py` and `test_prompts.py`
  use fakes/plain assertions with no I/O, and `test_api.py` drives the real
  FastAPI app through `TestClient` with `get_chat_service` overridden — so the
  full request/response cycle (routing, pydantic validation, JSON
  (de)serialization, dependency injection) is verified without ever touching
  Ollama, Chroma, or the lifespan handler.

## Evaluation pipeline notes (Milestone 7)

- The eval set lives in `data/eval/eval_set.json` — a JSON array of
  :class:`~rag.eval.dataset.EvalSample` objects, each with a `query`,
  `expected_doc_ids` (for retrieval eval), and an optional `expected_answer`
  (for answer eval).  Extra fields (e.g. `_note`) are round-tripped
  transparently.  Edit the sample file to add your own question/answer pairs
  before running evals.
- Retrieval eval (`rag/eval/retrieval_eval.py`) runs the configured
  `Retriever` (embed → vector-search → rerank) for each query and reports
  four aggregate metrics: **hit rate** (fraction of queries where any expected
  document was retrieved), **recall@k** (fraction of expected documents
  covered), **precision@k** (fraction of retrieved results that were
  relevant), and **MRR** (mean reciprocal rank of the first relevant result).
  Matching is document-level (`chunk.document_id`), not chunk-level — any
  retrieved chunk from the right document counts as a hit.
- Answer eval (`rag/eval/answer_eval.py`) runs the full
  `ChatService.ask` pipeline then asks the configured `LLMClient` to judge
  each answer as `PASS`/`FAIL` against the sample's `expected_answer`
  (LLM-as-judge).  The same Ollama model is used for both generation and
  judging — no extra dependency.  Samples without an `expected_answer` are
  skipped and counted separately so you can have retrieval-only entries in the
  same eval set.
- Metric functions (`rag/eval/metrics.py`) are pure — no I/O, no
  dependencies beyond the stdlib — so they are trivial to test and easy to
  extend with additional metrics (e.g. NDCG, F1) behind the same interface.
- Eval tests (`tests/test_eval.py`) are hermetic: metric functions are tested
  with plain lists, the dataset round-trip uses `tmp_path`, and the runner
  tests use `_FakeRetriever`/`_FakeChatService`/`_FakeLLMClient` — no real
  index, no Ollama daemon, no model weights.

## Streamlit UI notes (Milestone 8)

- The UI lives in two files: `rag/ui/app.py` (the Streamlit script) and
  `rag/ui/helpers.py` (pure formatting functions).  The split keeps all
  testable logic in `helpers.py` — importable without a running Streamlit
  server — while `app.py` stays thin Streamlit wiring.
- `@st.cache_resource` builds the `ChatService` exactly once per Streamlit
  server process (not per page load / browser tab), mirroring the FastAPI
  `lifespan` pattern: one set of connections to Ollama and Chroma, shared
  across all sessions.
- Chat history is stored in `st.session_state["messages"]` as a list of
  `{"role", "content", "answer"}` dicts — `answer` carries the full
  `ChatAnswer` (including citations) so each assistant bubble can render
  collapsible citation cards on re-render without re-querying.
- Error handling covers two failure modes: startup failure (Ollama/Chroma
  unreachable when `build_chat_service` is called — shown as a full-page
  error with instructions) and per-query failure (exception inside
  `chat_service.ask` — shown inline in the assistant bubble without
  crashing the session).
- UI tests (`tests/test_ui.py`) cover `helpers.py` only — citation label
  formatting (with/without page), preview truncation and newline collapsing,
  and the sidebar config summary.  The Streamlit wiring in `app.py` is not
  unit-tested (it requires a live Streamlit runtime); correctness there is
  verified by running `streamlit run rag/ui/app.py` manually.

## Known limitations / roadmap

- Chunking is character-based fixed-size with overlap; token-aware and
  structure-aware/semantic chunking are deferred until the end-to-end
  pipeline is proven (both fit behind the existing `Chunker` interface).
- Reranking is opt-in via config (`reranker.provider: none` is still the
  default in `config.yaml`); switch to `cross_encoder` to enable it. A
  pure-LLM reranker (reusing the existing `LLMClient`/Ollama setup) remains
  a possible lighter-weight alternative behind the same `Reranker` interface.
- The `index` command always re-embeds every chunk it's given; there's no
  change-detection to skip unchanged documents. Fine at corpus sizes seen so
  far — worth revisiting if re-indexing becomes slow.
- Embedding dimensionality is discovered lazily and cached per `OllamaEmbedder`
  instance, not persisted — switching embedding models still requires
  `index --reset` to avoid mixing incompatible vectors in one collection (the
  adapter doesn't currently detect this for you).
