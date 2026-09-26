# Milestone notes (2–11)

Design rationale for each shipped milestone: why things are built the way
they are, not just what they do. Split out of `CLAUDE.md` to keep that file
to the operational core — see `CLAUDE.md`'s Documentation section for how
the docs fit together.

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
- `retrieval.min_score` is a relevance floor applied to the *final* results,
  after reranking, inside `Retriever` — not inside any one reranker. A vector
  store always returns its nearest `top_k` neighbours however distant they are,
  so without a floor an off-corpus question still reaches the LLM with a full
  set of irrelevant passages to "ground" itself in. Living in `Retriever` means
  it applies to `NoOpReranker` (pure vector retrieval) too, and it's what lets
  `ChatService`'s empty-retrieval branch mean "nothing relevant" rather than
  just "empty index".
- Because the floor reads whatever the last stage's score means (per the score
  convention above), the right *value* differs by reranker: a cross-encoder's
  sigmoid-squashed logit and a raw cosine similarity are not the same scale.
  Retune `retrieval.min_score` when changing `reranker.provider`.
- **Query expansion** (`retrieval.expansion`, `rag/retrieval/expansion.py`) runs
  *before* retrieval and turns one query into several. `QueryExpander` returns an
  `ExpandedQuery` with separate `dense`/`sparse` lists, because a transformation
  that helps embedding search can hurt keyword search:
  - `hyde` — generate a hypothetical answer passage and embed *that*, closing the
    shape mismatch between a short question and the long declarative passage that
    answers it. The invented passage goes to the embedder only; **BM25 keeps the
    real question**, since matching literal invented terms ("1,000 requests per
    minute") retrieves on words that may appear nowhere in the corpus.
  - `multi_query` — generate rephrasings, retrieve for each, fuse. Both retrievers
    get all of them: a rephrasing is still a real question, so BM25 gains
    vocabulary rather than noise.
- Expansion needed no new retrieval code path. Every (query, retriever) pair
  yields one ranked list and `reciprocal_rank_fusion` folds them all together —
  hybrid mode was already fusing two lists, expansion just makes more of them.
  Rank-based fusion is what makes this safe: the lists come from different
  queries *and* different scoring functions, and RRF compares only positions.
  A single ranked list still skips fusion entirely, so unexpanded dense
  retrieval is byte-for-byte what it was, similarity scores included.
- `top_k` is per *ranked list*, not per query — N queries pull N×`top_k`
  candidates before fusion narrows the union back to `top_k`. Expansion buys
  recall to fuse over, not a larger final candidate set.
- Expanders fail open like `QueryCondenser`: a raising client or unparseable
  reply falls back to the original query rather than failing the search. HyDE
  with `num_documents > 1` keeps whichever passages it did get.
- Expansion reaches **stage 2 as well**: `Reranker.rerank` takes a *list* of
  queries (`ExpandedQuery.rerank`), and `CrossEncoderReranker` scores every
  candidate against every query, folding the per-query logits together via
  `reranker.aggregate`. Without this, expansion was self-defeating — a
  vocabulary gap the rewrites closed at retrieval time was reintroduced the
  moment scoring fell back to the user's original wording, so the right chunk
  got retrieved and then scored ~0 and cut by `min_score`.
- `reranker.aggregate` is the knob that decides what expansion *means*:
  - `max` (default) — relevant if *any* phrasing says so. This is what lets a
    rephrasing rescue a chunk the original wording scored near zero.
  - `mean` — averages log-odds, requiring broader agreement; suppresses chunks
    only one (possibly drifting) rewrite liked.
  Combining happens in logit space, before the sigmoid, so there's one
  normalization point. `max` is monotonic through a sigmoid so its placement is
  immaterial; `mean` is log-odds pooling. With one query the two are identical,
  so the unexpanded path is unaffected either way.
- `ExpandedQuery` carries **three** lists, one per consumer, because each accepts
  different text: `dense` (embedded — free-form, a hypothetical passage is fine),
  `sparse` (BM25 — literal terms only), and `rerank` (cross-encoder — must be
  **question-shaped**, since these models are trained on (query, passage) pairs
  and a passage in the query slot is off-distribution). HyDE therefore keeps its
  invention in `dense` alone; multi-query's rephrasings, being real questions,
  populate all three. `queries[0]` is always the user's actual question.
- Cost: reranking N queries means N× the cross-encoder pairs. They go through one
  `predict()` call so the per-call overhead isn't multiplied, but the compute is.
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
  contain. There are **three** distinct short-circuit messages, not one: blank
  query, nothing in the index (`candidate_count == 0`), and candidates found
  but all below `retrieval.min_score`. The last two are different problems with
  different fixes ("run the indexer" vs. "your corpus doesn't cover this"), so
  collapsing them into one string would waste the distinction the floor exists
  to create.
- `Retriever.retrieve` returns a `RetrievalResult` (chunks + `candidate_count`
  + `dropped_below_min_score`), not a bare list, precisely so `ChatService` can
  tell those two cases apart. Only the retriever knows which happened, and
  threading it through mutable retriever state would break under FastAPI's
  concurrent requests — so it's returned explicitly. Callers that only want the
  chunks (`rag/eval/retrieval_eval.py`, the `retrieve` CLI command) take
  `.chunks`.
- `ChatAnswer` carries `rewritten_query` and `dropped_below_min_score` so the
  pipeline's two silent interventions — searching for a different question than
  the user typed, and withholding passages it retrieved — are visible to
  *every* caller, not just the one that passes an `on_event` sink. `PipelineEvent`
  remains the debug trace; these two fields are the user-facing contract.
  `rewritten_query` is `None` when condensing didn't run or didn't change
  anything, so callers never render "rewritten to: \<the same question\>".
- **Retrieval is unconditional** — every non-blank query goes through the full
  pipeline; there's no classifier or router deciding whether the corpus is
  needed. That's the right default for a corpus Q&A tool (every question is
  supposed to be corpus-shaped) and the common shape in production RAG. The
  alternative — exposing search as a tool and letting the LLM decide when to
  call it — is what a general-purpose assistant needs, and would replace both
  the always-retrieve decision and the condense step below.
- Multi-turn queries are handled by condensing, not by passing history to the
  answering model: `QueryCondenser` (`rag/generation/query_rewriter.py`)
  rewrites a follow-up plus recent turns into one standalone question, and that
  rewrite drives *both* retrieval and the generation prompt. Retrieval is
  stateless, so embedding a raw "what about part-time staff?" searches for
  those literal words; and `build_rag_prompt` renders passages plus a single
  question with no conversation of its own, so the answering model needs the
  resolved question too. Enabled via `chat.condense_history` (on by default),
  bounded by `chat.max_history_turns`.
- The condenser reuses the pipeline's existing `LLMClient` rather than
  building its own, and only runs on a turn that actually *has* history — so
  the CLI, the eval pipeline, and every first turn pay nothing for it. It
  fails open: an LLM error or empty rewrite logs and falls back to the original
  query, which is exactly what the pipeline used before condensing existed.
- All three entrypoints surface both facts, in the shape that fits them: the
  UI renders them as captions under the answer bubble (`format_answer_notices`)
  rather than only inside the collapsed pipeline trace, so a user re-reading an
  old turn doesn't have to open a debug panel; the API returns them as
  `ChatResponse.rewritten_query` / `.dropped_below_min_score`; the CLI prints
  them around the answer, and its `retrieve` command distinguishes "no
  candidates" from "all below the floor" with a hint to lower `min_score`.
- `history` is supplied by the caller, never stored server-side — the API takes
  it in `ChatRequest.history` and holds no session state, so it stays
  restartable and horizontally scalable. The Streamlit UI derives it from
  `st.session_state["messages"]` via `history_from_messages`, which skips
  failed turns (whose `content` is an error banner, not a reply).
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
- `QueryCondenser` is tested (`tests/test_query_rewriter.py`) against a fake
  `LLMClient` — prompt assembly, history truncation, quote stripping, and both
  fail-open paths (empty rewrite, raising client) — with no daemon involved.
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
  `expected_spans` and/or `expected_doc_ids` (for retrieval eval), and an
  optional `expected_answer` (for answer eval).  Extra fields (e.g. `_note`) are
  round-tripped transparently.  Edit the sample file to add your own
  question/answer pairs before running evals.
- **Ground truth is span-level**: `expected_spans` holds verbatim quotes, and a
  retrieved chunk is relevant if it *contains* one.  Document-level matching
  (`expected_doc_ids`) still works and is used when a sample declares no spans,
  but it is only a proxy for "found the right passage" and the proxy breaks as
  documents grow: on a 60k-character filing (~70 chunks) `precision@k` reads
  1.0 while every result is boilerplate from the wrong end of the document.
  Worse, its granularity depended on file *format* — the PDF loader emits one
  `Document` per page, the Markdown loader one per file.
- Spans are quotes, **not character offsets**, because offsets are invalidated
  by any change to cleaning or chunking — and comparing across chunking configs
  is the whole reason the eval exists.  A ground truth that moves when you
  change `chunk_size` cannot measure `chunk_size`.  Quotes are located at eval
  time against a normalized copy of both sides (whitespace collapsed, case and
  typographic punctuation folded), so neither `Chunk.text` nor the stored span
  is modified.
- Keep spans at most `chunking.chunk_overlap` characters: consecutive windows
  overlap by exactly that much, so any shorter span is guaranteed to sit inside
  some chunk, while a longer one can straddle every boundary and match nothing.
  `warn_on_unmatchable_spans` flags violations at startup.
- Grades are optional (`{"text": ..., "grade": 3}` vs. a bare string). They feed
  **NDCG@k**, the only reported metric that rewards *ordering* and reads grades
  — so surfacing the passage that states the figure above one that merely
  discusses the topic actually scores better.
- Retrieval eval (`rag/eval/retrieval_eval.py`) runs the configured
  `Retriever` (embed → vector-search → rerank) for each query and reports
  **hit rate**, **recall@k** (as a curve over several k, so the candidate-set
  ceiling is visible — flat from 5 to 20 means the loss is upstream of the
  reranker), **precision@k**, **MRR**, and **NDCG@k**.
- `rag/eval/metrics.py` functions take **gains** (relevance grade per result, in
  rank order) rather than id lists.  That is what lets one set of metrics serve
  span and document ground truth without knowing which is in play:
  `rag/eval/relevance.py` decides what "relevant" means, metrics only
  rank-weight the answer.  The two modes are **not comparable** and are reported
  as separate groups when a set mixes them.
- Answer eval (`rag/eval/answer_eval.py`) runs the full
  `ChatService.ask` pipeline then asks the configured `LLMClient` to judge
  each answer as `PASS`/`FAIL` against the sample's `expected_answer`
  (LLM-as-judge).  The same Ollama model is used for both generation and
  judging — no extra dependency.  Samples without an `expected_answer` are
  skipped and counted separately so you can have retrieval-only entries in the
  same eval set.
- **Eval sets in `data/eval/`:**
  - `eval_set.json` — 43 hand-authored samples against `data/corpora/baseline/documents`. Document-matched.
  - `refusal_set.json` — 6 obviously off-domain questions.
  - `edgar_eval_set.json` — span-matched samples against the EDGAR corpus,
    generated by `scripts/generate_eval_set.py`.
  - `edgar_refusal_set.json` — hand-authored hard negatives for EDGAR.
- The EDGAR set is **generated, not hand-written**, because 3.6M characters
  cannot be labelled by hand and a set touching 1% of the corpus mostly measures
  which 1% you picked. The generator samples a chunk, asks for a question plus
  the **verbatim span** answering it, and records the span — the labour that
  normally makes chunk-level labels expensive is *finding* the passage, and here
  the passage is the input.
- **Stated bias:** questions generated from a passage are lexically closer to it
  than real questions are, which inflates retrieval scores across the board.
  Absolute numbers from this set are therefore **not** comparable to numbers
  from a hand-authored set. It exists to compare *configurations against each
  other* on identical questions, which is exactly what Milestone 11 needs.
- Every generated span is checked to appear verbatim in its source chunk (under
  the same normalization used at eval time), to be a clause rather than a bare
  figure, to fit inside `chunk_overlap`, and to be **unique across the whole
  corpus** — boilerplate appearing in several filings can't identify a passage,
  so a retriever surfacing a different filing would be marked wrong for being
  right. Questions must name company and period, since the corpus holds 14
  companies discussing the same topics.
- A second LLM pass then verifies the span answers the question and supports the
  stated answer. This is not optional: the generating model reliably produced
  spans contradicting their own answers — quoting "increased 22% and 21%" beside
  an answer of "9%", or answering "what was operating income" with the
  year-over-year *increase*.
- **Validation fails closed**, the opposite of every runtime component here (the
  contextualizer, CRAG's three checks, `QueryCondenser` all fail *open*). The
  asymmetry is the point: at runtime a broken judgment must not degrade a live
  turn; here a bad label silently corrupts every measurement taken against the
  set afterwards. Yield is ~16-20% of attempts and every rejection reason is
  counted and printed — a generator whose yield quietly collapses is one whose
  output you should not trust.
- `tests/test_eval_generator.py` covers the validator even though it lives in a
  script, because a bad generation is thrown away while a validator bug is
  silent and permanent.
- `edgar_refusal_set.json` is separate from the main set because averaging
  refusals into retrieval metrics would penalise correct behaviour. Its
  negatives are *hard* in a way `refusal_set.json`'s are not — they name real
  concepts and periods, and the period/scope tiers name companies that **are**
  in the corpus, so retrieval surfaces genuinely on-topic passages that don't
  answer the question. Off-corpus entities were verified absent from all 61
  filings: Amazon, Google, Boeing and JPMorgan were *rejected* as negatives
  despite not being in the manifest, because MD&A mentions them as competitors.
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
- The pipeline trace defaults to **expanded** and stays that way, controlled by
  a sidebar checkbox (`st.session_state["expand_trace"]`, seeded in
  `_init_session` before the widget is created so the widget owns the key
  afterwards). Both render paths read it — the live `st.status` block when a
  turn completes, and `_render_events` when history re-renders — so the trace
  doesn't snap shut on the next turn. Untick it to get the old collapsed
  behavior.
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

## Contextual chunking notes (Milestone 9)

- `chunking.contextual.enabled` turns on **contextual retrieval**: at index
  time, one LLM call per chunk writes a sentence saying where that chunk sits
  in its parent document, and the chunk is indexed as *context + text*. It
  addresses a failure query-time tuning cannot reach — fixed-size splitting
  strips the terms that made a chunk findable, so a chunk reading "The limit is
  1,000 requests per minute" never names the API it belongs to and is
  unreachable by a query that does.
- **`Chunk.text` is never modified.** The blurb lives in its own `context`
  field, and `Chunk.contextual_text` / `ScoredChunk.contextual_text` join the
  two only where indexing happens. Citations, previews, and
  `char_start`/`char_end` keep pointing at the verbatim span, so nothing a user
  sees is model-generated. That split is the whole design; it's why the feature
  needed a field rather than a rewrite of `text`.
- The enrichment reaches **both** retrievers: the embedder embeds
  `contextual_text`, and `BM25Index` tokenizes it too (while still returning
  `record["text"]`). Contextual BM25 is half the reported gain in Anthropic's
  writeup, and skipping it would have left keyword search matching against the
  words the split threw away.
- Persistence rides on the existing adapters: Chroma stores `context` alongside
  provenance in its flat metadata (never in `documents`, which must stay the
  verbatim chunk) and pops it back into `ScoredChunk`; BM25's JSON record gains
  a `context` key read with `.get`, so index files written before this existed
  still load.
- **The reranker deliberately still scores `text`, not `contextual_text`.**
  `retrieval.min_score` is hand-tuned against the cross-encoder's output scale,
  and silently changing what that model sees would invalidate the tuning
  without anything reporting it. CRAG's grader *does* read `contextual_text` —
  it's a fresh judgment with no calibrated threshold behind it.
- Contextualization runs **after** the incremental change check in `index`, not
  before, so chunks already in the index cost nothing. The content hash covers
  `chunk.text` alone, which means toggling `contextual.enabled` does *not*
  invalidate an existing index by itself — re-index with `--reset` after
  changing anything under `chunking.contextual`.
- Measured cost on the small corpus: 31 chunks took ~107s and 31 `/api/chat`
  calls on `qwen3.5:9b-mlx`. A no-change re-index afterwards took 1.7s and zero
  LLM calls, confirming the skip path holds.
- **Generation runs concurrently** (`chunking.contextual.concurrency`, default 4).
  Per-chunk calls are independent by construction, so the original sequential
  loop was only ever a parse-reliability choice, which threading doesn't affect.
  `LLMClient.generate` is blocking I/O, so a `ThreadPoolExecutor` fits without
  touching the interface. Results are written back **by position**, so output
  order is the input order regardless of completion order.
- Measured on 12 real EDGAR chunks against `qwen3.5:9b-mlx`:

  | concurrency | per chunk | speedup |
  |---|---|---|
  | 1 | 4.98s | — |
  | 4 | 1.60s | **3.12x** |
  | 8 | 1.97s | 2.53x |

  Concurrency 8 is *slower than 4* — past the serving backend's own parallelism
  (Ollama's `OLLAMA_NUM_PARALLEL`) requests queue on the far side and add
  contention. Raising this knob past your daemon's setting is not a free win;
  measure before changing it.
- **That 1.60s/chunk figure is optimistic and should not be used for planning.**
  All 12 benchmark chunks came from one document, so every prompt carried an
  identical 8,000-char document prefix and benefited from prompt-prefix caching.
  Measured across the real 61-document corpus the steady-state rate is
  **3.40s/chunk**, making a full EDGAR contextual build ~4.0h rather than the
  ~1.9h originally projected.
- **Contexts are checkpointed** to `data/index/contextual_cache.jsonl`
  (`rag/chunking/context_cache.py`), appended as each result completes, so a run
  killed at 80% keeps 80% of its work. A resumed run makes zero LLM calls for
  chunks it already has.
- The cache key hashes **every input to the call** — system prompt, rendered
  prompt (which contains the truncated document and the chunk), reply cap, and
  model name. That is strictly stronger than the index's own change detection
  (which hashes `chunk.text` alone) and closes a hazard documented in
  `docs/known-limitations.md`: editing `CONTEXT_SYSTEM_PROMPT` or
  `max_document_chars` used to leave stale contexts in the index with nothing
  detecting it. Now they simply miss.
- Because the key is that precise, the cache is **deliberately kept across
  `index --reset`** — that is what makes reset affordable, since re-embedding is
  cheap and regenerating contexts is not. `index --clear-context-cache` forces
  regeneration when you actually want it (e.g. comparing two models on identical
  inputs). Only successes are cached; a transient failure is never memoized as
  "this chunk has no context".
- Tests (`tests/test_contextualizer.py`) drive a fake `LLMClient`, covering
  prompt assembly, both truncations, and every fail-open path; the round trips
  are covered where they live (`test_vectorstore.py`, `test_sparse.py`),
  including a BM25 test that a query matching *only* a chunk's context finds it.

## Corrective RAG notes (Milestone 10)

- `crag.enabled` adds three LLM-backed judgments around the existing pipeline,
  each independently switchable (`rag/generation/crag.py`):
  - `DocumentGrader` — per retrieved passage, does this help answer the
    question? Rejected passages never reach the prompt.
  - `RetryQueryRewriter` — when an attempt leaves nothing, reword the query and
    search again (bounded by `crag.max_retries`).
  - `GroundednessChecker` — read the generated answer back against its
    passages; regenerate under `REGROUND_SYSTEM_PROMPT` if unsupported.
- What CRAG adds over `retrieval.min_score` is a **judgment rather than a
  score**. The floor thresholds similarity, and similarity is a statement about
  how alike two texts are, not about whether one answers a question asked of
  the other — so a topically-adjacent passage clears the floor and arrives as
  though it were evidence. On the live check above, 5 passages cleared the
  floor and the grader kept 1; the answer cited that one and was correct.
- Groundedness is what makes the citation contract mean something. The system
  prompt asks the model to cite `[n]`, but an inline `[2]` is a token the model
  chose to emit, not evidence that passage 2 says what sits next to it.
- **`ChatService` owns the loop**, not `Retriever` and not a graph framework.
  Two of the three checks span the retrieve/generate boundary this class exists
  to own — retrying means returning to retrieval *after* judging its output,
  and groundedness compares a generated answer to the passages that produced
  it. Neither half can see both sides. LangGraph was considered and rejected:
  the graph is four nodes and a bounded loop, and an orchestration dependency
  would buy nothing the `for` loop in `_retrieve_with_correction` doesn't.
- **All three fail open**, like `QueryCondenser` and the expanders: an LLM error
  or unparseable reply means "proceed as though this check hadn't run". The
  asymmetry is deliberate and asserted in tests — an unreachable grader *keeps*
  passages. A broken checker must degrade the pipeline to plain RAG, never turn
  a working turn into a refusal.
- `GroundednessChecker.check` returns `bool | None`, and the `None` is load
  bearing: "unsupported" and "couldn't tell" are different, and only the first
  justifies regenerating. An answer that stays ungrounded after
  `max_regenerations` is **returned anyway**, flagged via `ChatAnswer.grounded`
  — withholding it would rest a third kind of refusal on one small model's
  one-word opinion. All three entrypoints surface the failing verdict (CLI
  banner, `ChatResponse.grounded`, UI caption); the *passing* verdict is shown
  nowhere, since captioning the expected outcome trains users to skim past the
  one state that needs attention.
- Grading always runs against the **user's** question, even on a retry that
  searched for something else — otherwise a rewrite that drifted would validate
  the drifted results it found. For the same reason generation always answers
  `search_query`, never a retry rewrite: a rewrite is a search device, like a
  HyDE passage.
- A rewrite that reproduces an already-searched query ends the loop instead of
  spending a retrieval round trip re-deriving the same empty result.
- Grading out everything is the **fourth** distinct no-context message
  (alongside blank query / empty index / all-below-floor), and it's checked
  first because it's the most specific true statement about such a turn: the
  index wasn't empty and the floor wasn't the obstacle.
- `ChatAnswer` gains `graded_out`, `retry_queries`, `retrieval_attempts`, and
  `grounded`, extending the existing contract that every silent intervention is
  visible to every caller — not just the one that passes an `on_event` sink.
  New trace stages: `crag_grade`, `crag_retry`, `crag_groundedness`, `regenerate`.
- Cost is the reason it's off by default: `grade_documents` alone is one LLM
  call per retrieved passage (`retrieval.rerank_top_k`, so 5 by default) on a
  path that previously had one call total. The live query above spent ~5s in
  grading.
- Tests split by concern: `tests/test_crag.py` drives each component with fake
  clients (reply parsing, every fail-open path), while `tests/test_chat_service.py`
  uses stubs for all three to test the loop itself — when a retry fires, which
  query gets graded and answered, when generation is skipped, and the exact
  event sequence.

## Named corpora notes (shipped with Milestone 11)

Moved here from `CLAUDE.md`, which keeps only the operational rules.

- **Why pooling is the point of the registry.** One name is *isolated*
  (retrieval quality within a corpus); several are *pooled* (robustness to
  plausible-but-wrong neighbours). The gap between the two is the
  **cross-corpus interference cost**, which can't be observed with a single
  corpus. Pooling is also what makes `retrieval.min_score` and CRAG's document
  grader measurable at all: on a small, topically-distinct corpus nothing
  irrelevant is ever nearby, so the grader has no job to do.
- **Why selections derive their own storage names** (`rag_corpus__edgar` vs.
  `rag_corpus__baseline+edgar`, `bm25_index__<slug>.json`). Isolated and pooled
  indexes must coexist rather than silently overwrite each other, because
  comparing them is the exercise. Selections are sorted and deduped so
  argument order doesn't create a second index.
- **Why an unknown corpus name raises.** A typo would otherwise produce an
  empty index and a plausible-looking all-zero eval run.
- **Why duplicate `Document.id`s are refused rather than namespaced.** Ids are
  corpus-relative paths, so two corpora each containing `faq.txt` would collide
  and the store would silently upsert one over the other. Namespacing ids by
  corpus was rejected: it would change every `document_id`, invalidating the
  `expected_doc_ids` already recorded in the eval sets, to solve a problem the
  current corpora don't have.
- **Backward compatibility.** An empty registry falls back to a single implicit
  corpus at `paths.corpus_dir`. Indexes built before the registry need one
  rebuild (`python -m rag.cli index --corpus <name>`) because collection and
  BM25 names now carry the selection slug; the contextual cache is keyed by
  prompt content, not index name, so contexts are not re-paid for.
- **What's committed.** `baseline` is committed despite the general
  `data/corpora/*/documents/` ignore rule (see the exemption in `.gitignore`);
  other corpora are fetched and reproduced from their `manifest.json`.
