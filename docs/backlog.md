# Backlog

Planned work, roughly in dependency order. Shipped milestones are marked
*(shipped)*; the first unmarked one is next. The
existing pipeline is feature-rich on the *retrieval/generation* axis and thin
on everything that surrounds it — measurement, operations, and input quality —
which is what this list is.

Every item keeps the project's existing rules: config-selected behind an
interface, off by default until measured, hermetic tests, no second HTTP
client / YAML parser / etc.

**Deliberately not on this list: a query router.** Retrieval is unconditional
by design (see the Chat API notes in `docs/milestone-notes.md`) — every
question asked of a corpus Q&A tool is supposed to be corpus-shaped, so a
classifier deciding whether to retrieve would add an LLM call and a failure
mode to buy back a case that shouldn't arise. The version of that idea worth
building is a different thing entirely — the model calling search itself —
which is Milestone 19, and it subsumes routing rather than adding it as a
stage. (`retrieval.document_routing`, from the
[chunking plan](chunking-indexing-plan.md)'s Phase 3b, is a different thing:
it picks *which filing* to search, with no LLM call, and never decides
*whether* to search.)

### Milestone 11 — Measure Milestones 9 & 10 *(shipped)*

First, because it gates the value of everything else. Contextual chunking and
CRAG both shipped functionally verified and numerically unmeasured, and the eval
harness to fix that already exists.

- `retrieval_eval` with `chunking.contextual.enabled` off vs. on (`--reset`
  between runs; the content hash doesn't cover `context`).
- `answer_eval` with `crag.enabled` off vs. on, and with each of
  `grade_documents` / `check_groundedness` isolated.
- Same for `retrieval.expansion` (`none` / `hyde` / `multi_query`) and
  `reranker.aggregate`, which are equally unmeasured. Expansion is
  non-deterministic — repeat runs or accept the noise, don't read one run as a
  result.
- Record the numbers somewhere durable (`docs/measured-results.md`, or
  `data/eval/results/`), including corpus size and config, since a metric with
  no config attached is unreproducible.

Expect a defaults change to fall out of this, and possibly a retune of
`retrieval.min_score`.

### Milestone 12 — Observability *(shipped)*

Shipped as planned, with one correction to the premise below: `ChatService`
did **not** already know which passages were cited — `citations` was every
passage shown. It now parses the `[n]` markers (`parse_cited_passages`) and
reports `cited_chunk_ids` separately. What shipped and why:
[milestone notes](milestone-notes.md#observability-notes-milestone-12).
Not done here: stage-1 (pre-rerank) candidate ids aren't recorded, since
`Retriever` doesn't return them, and no OpenTelemetry adapter exists yet.

Today a turn's reasoning is visible only live, via `PipelineEvent` in the UI
trace; nothing is persisted, and no turn reports what it cost or how long it
took. That makes regressions invisible and makes the eval set the only source
of signal about quality.

- Persist per-turn records — query, rewritten/expanded queries, retrieved and
  final chunk ids, scores, citations, the CRAG verdicts, answer.
- Per-stage latency and LLM call/token counts on `ChatAnswer`, surfaced on
  `ChatResponse` like the other pipeline-transparency fields.
- Thumbs up/down in the UI, written to the same store — logged queries with
  feedback are the cheapest source of new eval samples, and the eval set is
  hand-authored today.
- `ChatService` already knows which passages the model cited as `[n]` — log
  `(query, candidates_shown, cited_chunk_ids)` as an implicit relevance
  judgment on every turn, not just when a user clicks thumbs up/down. Cheap to
  add alongside the explicit signal, and it compounds: enough judgments
  eventually train a learning-to-rank model behind the existing `Reranker`
  interface.
- Keep the sink behind an interface (`local JSONL` default) rather than
  reaching for a tracing SaaS; OpenTelemetry export is a later adapter.

### Milestone 13 — Query result caching

Nothing is cached anywhere. Repeat and near-repeat queries re-pay embedding,
retrieval, reranking, and generation in full — most visible in demos and evals,
where the same questions run over and over.

- Exact-match first (normalized query + a config fingerprint → answer). Simple,
  correct, and enough for the eval/demo case.
- Semantic cache (query embedding → nearest cached query above a threshold) as
  a second tier, behind its own config flag. Note that this can serve a wrong
  answer where the exact cache can't, so it needs a conservative default
  threshold and a way to see when it hit.
- **The cache key must include the config that produced the entry** — chunking,
  retrieval, reranker, CRAG settings. Serving a pre-CRAG answer after enabling
  CRAG would silently poison Milestone 11's numbers.
- Cache the *answer*, not just reranked docs: generation is the expensive stage
  here (one local 9b call, more with CRAG).

### Milestone 14 — Richer document parsing

**Planned:** see [Chunking and indexing plan](chunking-indexing-plan.md), Phase 4.

`pypdf` gives page text and nothing else. Tables arrive as collapsed
whitespace, headings are indistinguishable from body text, and a scanned PDF
yields an empty `Document` with no error. This is the highest-leverage quality
work in the list: no amount of retrieval sophistication recovers information
the parser threw away.

- A structure-aware parser behind the existing loader interface (Docling or
  `unstructured`) emitting tables and headings as such. Both are heavy
  dependencies — justify against the "keep dependencies minimal" rule, and keep
  `pypdf` as the default until the upgrade is measured.
- Section headings into `Chunk.metadata`, which improves citations and gives
  Milestone 15 something structural to split on.
- Detect an empty/near-empty extraction and warn rather than indexing nothing.
  OCR fallback is optional and probably an adapter, not a default.
- A page-image / Vision-Language embedding path (e.g. ColPali) is a heavier
  alternative worth naming for layout-dense pages — scanned forms, dense
  tables — where even a structure-aware text parser loses information a page
  screenshot wouldn't. Not the default; a later adapter behind the same
  loader interface if the text-based parser proves insufficient.

### Milestone 15 — Semantic chunking

**Planned:** see [Chunking and indexing plan](chunking-indexing-plan.md), Phase 5. It proposes a
structure-aware chunker in place of the semantic one described below, on
the evidence summarized there.

Fixed-size character windows split mid-argument; contextual chunking patches
the symptom at index time. A `SemanticChunker` slots behind the existing
`Chunker` interface: embed sentences, cut where adjacent-sentence similarity
drops below a threshold.

- Costs embedding calls at index time, on top of contextualization if that's on.
- Depends on Milestone 14 for the structure-aware variant (split on real
  headings first, semantically within a section).
- Token-aware chunking is the other long-standing option behind this interface
  and can share the milestone.
- Compare against `fixed` with Milestone 11's harness before changing the
  default.
- Parent-child retrieval is a complementary technique, not a chunking
  strategy change: embed small child chunks for search precision but return
  the surrounding parent chunk — plus a breadcrumb like
  `Document > Section 4.2`, built from Milestone 14's headings — to the LLM.
  Fits behind `Retriever` as an expansion step after ranking, not behind
  `Chunker`.

### Milestone 16 — Async ingestion

`python -m rag.cli index` is synchronous and single-process. Two pieces of
this already shipped with the Milestone 11 measurements: generated contexts
are checkpointed to `contextual_cache.jsonl`, so an interrupted contextual run
resumes, and the contextualizer runs `chunking.contextual.concurrency` calls
at once. Indexing is also incremental (content-hash skip, stale-chunk purge).
What's left:

- Checkpoint/resume for embedding itself: an interrupted plain build re-embeds
  whatever the hash check can't prove is written to both indexes.
- Then a job/worker split behind an interface, so ingestion can be triggered
  by an API call rather than a terminal. Local default: an in-process or
  file-backed queue — not a hosted queue, which belongs with Milestone 18.
- Concurrency for the CRAG grader on the query side — still one sequential
  call per passage, for parse reliability.

### Milestone 17 — PII detection & redaction

Nothing inspects document content today. A corpus with personal data gets
embedded, persisted to disk, and quoted back verbatim in citations.

- A detection pass in the ingestion pipeline, alongside `clean_documents` and
  explicitly separate from it, so it can be inspected or skipped.
- Regex/heuristic detectors as the local default (emails, phones, national ids,
  card numbers); a cloud DLP adapter behind the same interface if ever needed.
- Decide and document the policy per finding — redact in place, drop the
  document, or warn and index anyway — and keep it configurable. Redaction
  changes `Chunk.text`, which is otherwise sacred (see Milestone 9), so the
  interaction with citations and `char_start`/`char_end` needs thought.
- Off by default; it costs an ingestion pass and can mangle legitimate text.

### Milestone 18 — Deployment

The project is local-first by design and should stay runnable with nothing but
Ollama. Deployment is about proving the interfaces are real, not about moving
off local.

- Containerize the API; a scale-to-zero container host is the natural target
  since traffic is bursty and the app holds no session state (`history` is
  caller-supplied precisely so this works).
- Hosted `LLMClient` / `EmbeddingModel` adapters. `LLMConfig.provider` already
  validates `anthropic`/`openai` and raises "recognized but not implemented" —
  this is where that gets closed. Gemini comes first: the GCP plan below
  runs on it.
- A local, non-Ollama `EmbeddingModel` adapter (`sentence-transformers`,
  already a dependency via `CrossEncoderReranker`) is worth adding alongside
  the hosted ones — same interface, no daemon required. `EmbeddingConfig`
  already accepts `provider: sentence_transformers`, but `get_embedder` has
  no branch for it, so selecting it raises today.
- A contextualized chunk embedding adapter (`voyage-context-4`), wanted by
  [chunking plan](chunking-indexing-plan.md#phase-2--document-metadata-and-a-deterministic-chunk-header-23-days)
  Phase 2 as a comparator. It needs `embed_documents` to receive chunks
  grouped by document, so it's an interface change, not only a provider.
- A hosted `VectorStore` adapter (Chroma's own server mode is the smallest
  step; Qdrant or similar if a managed tier is wanted).
- The claim to earn: swapping any of these is a config change, no pipeline
  code touched. If it isn't, that's an interface bug worth finding.

#### GCP deployment plan

The API on Cloud Run (CPU only, scale to zero), with generation on Gemini
3.1 Flash-Lite's free tier. A GPU running Ollama costs about $0.67/hour per
live instance, and a warm minimum instance is fixed monthly cost. That buys
nothing a hosted model doesn't at demo traffic, and swapping the generator by
config is the claim this milestone exists to earn. Local Ollama stays the
default for development and every eval. Depends on Milestone 28's auth, input
bounds and production container: the service is not exposed before those
land.

- **Gemini `LLMClient` adapter** (use the `add-provider` skill). Call the REST
  `generateContent` endpoint over `httpx`, which is already a dependency, and
  don't add the `google-genai` SDK. The key comes from the environment
  (Secret Manager on Cloud Run), never from `config.yaml` or a committed file.
  Distinguish daily-quota exhaustion from transient rate limiting using provider error details and/or `Retry-After`; only report the daily budget as spent when that condition is confirmed.
  Hermetic tests against a mocked transport, like the Ollama adapter's.
  **Shipped early** (`rag/generation/gemini_llm.py`) to run the Gemma 4 31B
  judge off the local GPU. It paces itself under `requests_per_minute` /
  `tokens_per_minute` and raises `GeminiDailyQuotaExhausted` only when the
  429's `QuotaFailure.quotaId` names a per-day quota. The in-app daily
  counter below is still open.
  **Flash-Lite as generator:** `rag/config/gemini-3.1-flash-lite.yaml` and
  `gemini-3.5-flash-lite.yaml` overlay `config.yaml` with only the generator
  changed (15/min, 250k tokens/min, 500/day on the free tier). Both are
  thinking models, so the adapter sends `llm.thinking_level` (pinned to
  `minimal` in both), counts thought tokens as completion tokens, and raises
  when thinking uses up `max_tokens`. Neither is measured yet (see below).
- **The free tier is the budget guard.** A project's actual limits are shown in
  AI Studio. Third-party sources disagree (500 vs. 1,000 requests/day, about
  15/min), so read the numbers from there and put them in config. One
  `/chat` turn is one generation call, two with `history` (the condenser),
  and more with CRAG or expansion, both of which stay off here. That puts a
  ceiling of a few hundred turns a day on the whole deployment.
  - Keep an in-app daily counter that refuses at a margin below the quota, so
    one caller can't use up everyone's day. Milestone 28's per-key rate limit
    covers bursts, not this.
  - **No unbounded evals and no load tests against the Gemini key.** The one
    or two bounded quality runs described below are the sole permitted Gemini
    evals. Do not run `answer_eval` over the 174-sample EDGAR set: generation
    plus a judge call per sample is most of a day's quota. Load-test the
    deployed service with a fake `LLMClient` behind a test-only config instead:
    what's under test is Cloud Run, the reranker and the index, not Google's
    latency.
- **Measure Gemini as a generator before calling it the deployed default.**
  Every answer-quality number in `docs/measured-results.md` is from the local
  qwen model. Run `answer_eval --limit 40` once with Gemini as the generator
  and the usual local judge, using the `measure-change` skill, and record it.
  Two runs of that fit one day's quota. Keep the judge a different model from
  the generator, as now.
- **Free-tier data terms.** On unpaid Gemini services Google may use prompts
  and responses to improve its products, and human reviewers may read them.
  The corpus is public SEC filings, so passages are fine, but user queries go
  to Google too. Say so on any public page, and don't point this deployment
  at a non-public corpus. Paid-tier terms are the remedy if that changes.
- **Embeddings without Ollama.** Query-time embedding must match the index,
  and there is no Ollama on Cloud Run. Implement the `sentence_transformers`
  `EmbeddingModel` above and serve `Qwen/Qwen3-Embedding-0.6B` through it.
  It isn't guaranteed to produce the same vectors as Ollama's
  `qwen3-embedding:0.6b` build, so rebuild the index with the serving
  embedder and re-run `retrieval_eval` on EDGAR. Treat a change in hit rate
  or NDCG as a regression to explain, not noise.
- **The index is baked into the image.** Serving is read-only and the index is
  rebuilt by hand, so a CI job builds it from `manifest.json` with the serving
  embedder. It writes both the Chroma collection and the BM25 index
  (`retrieval.mode: hybrid`) into the image, along with the reranker and
  embedder weights, so a cold start downloads nothing. Every replica is then
  identical and there is no database to run. This holds only while reindexing
  stays manual. Once it's scheduled, move to Chroma server mode or a managed
  store, and take on the zero-downtime reindex that Milestone 28 lists as out
  of scope. The image contains the corpus, so the registry is private.
- **Size the instance from measurements, not guesses.** `bge-reranker-v2-m3`
  took ~1.1s/query on Apple Silicon and will be slower on Cloud Run's CPUs.
  Measure memory with both models loaded, reranker latency, and cold-start
  time, then set CPU and memory from those numbers and record them next to
  Milestone 28's load test. If the reranker is too slow on CPU, that's a
  measured-tradeoff decision (MiniLM was −10.4pp hit rate), not a quiet swap.
- **Infrastructure as code.** Terraform for Artifact Registry, the Cloud Run
  service (with `max-instances` as a hard cost ceiling), Secret Manager
  entries for the Gemini key and Milestone 28's API keys, a least-privilege
  runtime service account, an uptime check and alert on `/ready`, and a
  billing budget alert. Nothing is created by hand in the console.
- **CI deploys on merge.** GitHub Actions authenticates to GCP with Workload
  Identity Federation, not a downloaded service-account key. It builds an
  image tagged with the commit SHA and deploys it as a new Cloud Run
  revision. To roll back, send traffic to the previous revision. Deploys run
  only after `ruff`, `mypy`, `pytest` and Milestone 23's retrieval gate pass.
- **Scope.** The API (`/chat`, `/feedback`, `/mcp`) only. The Streamlit UI
  builds its own pipeline rather than calling the API, so deploying it would
  mean a second service drawing on the same Gemini quota. It stays local
  until it's a client of the API.

### Milestone 19 — Agentic retrieval

**In progress:** see [Milestone 19 plan](milestone-19-plan.md). Phases 0–3
shipped: a fixed eval judge and the multi-hop set, `ToolCallingLLM` (Ollama
and Gemini), one tool surface shared with MCP (`rag/tools.py`), and the agent
loop behind `chat.mode: agentic` (off by default). Phase 4, the measurement
that decides whether the agent is worth having, is next. The cost bullets
below were written before the interface change; `LLMClient` was kept, and
tool calling was added as a `ToolCallingLLM` subclass instead.

Expose search as a **tool the answering model calls**, rather than a stage that
always runs before it. Unlike the rest of this list, this one *replaces* shipped
behavior — it's the largest item here and the only one that can make the system
worse, so it lands last and behind a flag, with the current path staying the
default until measured.

What it subsumes, and why that's the argument for it:

- **Routing**, without a classifier. "Does this need the corpus?" stops being a
  separate LLM call whose verdict can be wrong in a way nothing detects, and
  becomes the model declining to call the tool.
- **Query condensing.** A model holding the conversation resolves "what about
  part-time staff?" by writing a better tool call. The dedicated condense round
  trip (`chat.condense_history`) exists because retrieval is stateless and the
  answering model never sees history — both premises go away here.
- **Multi-hop questions**, which the current pipeline cannot serve at all: one
  retrieval, one generation, no way to search again on what the first search
  turned up. CRAG's retry loop is the closest thing and it only fires on
  *failure*, re-asking the same question differently rather than asking a new
  one.
- **Conversational turns** ("summarize what you just told me") — a documented
  limitation today, since history reaches only the condenser.

What it costs, stated plainly because this is a real tradeoff:

- **`LLMClient` has to grow.** `generate(prompt, *, system=None) -> str` has no
  tool-calling and no multi-turn message list; a one-method ABC becomes a
  conversation loop with tool definitions, tool results, and a stop condition.
  Every adapter follows. That is the single biggest interface change the
  project has made.
- **Non-determinism and unbounded cost.** Turns become 1–N LLM calls with N
  decided by the model. Needs a hard call cap, like `crag.max_retries` but
  load-bearing rather than a safety net.
- **The local 9b model has to be good at tool calling**, which is a different
  skill from answering, and `docs/known-limitations.md` already notes every
  judgment in this pipeline runs on that one model. If it calls search
  erratically, this is strictly worse than always retrieving — which is the
  outcome to measure for.
- **Evals need rethinking.** `retrieval_eval` assumes exactly one retrieval per
  query with a fixed k; here there may be zero, or four with different queries.
  Expect a new metric shape, not just new numbers.

Design notes for whoever picks this up:

- The tool should wrap the **existing `Retriever`**, hybrid/expansion/floor and
  all — this milestone changes *when* retrieval happens and who decides, not
  how it works.
- CRAG is largely redundant with a competent agent loop (grading and retrying
  are what a model does natively between tool calls) but should stay switchable
  and independently measurable rather than being deleted alongside.
- Citations get harder: passages arrive across several tool calls, so the
  `Passage [n]` numbering that is currently the single source of truth mapping
  `[n]` → `Citation` needs to survive accumulation across calls.
- Keep it behind config (`chat.mode: pipeline | agentic`) so the two paths can
  be compared on the same eval set. Comparing them is the point of the
  milestone; shipping the agent isn't.

### Milestone 20 — Search-quality layer

The pipeline is *RAG*-shaped: one query in, `rerank_top_k` chunks out, straight
into a prompt. A *search-engine* shape adds a query-understanding stage before
retrieval and a result-assembly stage after ranking — worth it once the corpus
is browsed as well as asked questions of (a UI/API consumer paging results),
not just answered against.

```mermaid
graph LR
    Q[Raw query] --> QU[Query Understanding<br/>rewrite · expand · extract filters]
    QU --> CG[Candidate Generation<br/>dense + BM25 · filtered]
    CG --> FU[RRF Fusion]
    FU --> DD[Dedup & Diversity]
    DD --> RR[Cross-Encoder Rerank]
    RR --> SB[Signal Blending<br/>recency · title · authority]
    SB --> AG[Document Aggregation<br/>group chunks → docs]
    AG --> SN[Snippet Generation]
    SN --> OUT[Results]
    OUT -.-> FB[(Feedback log:<br/>which passages were cited)]
    FB -.-> RR
```

- **Structured `Query` object + filter pushdown**, land first — it touches
  `VectorStore.query`, `SparseIndex.query`, and `Retriever.retrieve`, so
  everything below is cheaper once it exists. *Filter pushdown shipped* with
  the [chunking plan](chunking-indexing-plan.md)'s Phase 3: a typed
  `QueryFilter` (`rag/query_filter.py`) on both index query methods and
  `Retriever.retrieve`, pushed into Chroma's `where` clause and a BM25
  pre-filter, and exposed as `filters` on `POST /chat` and MCP `rag_search`.
  Still open: replacing the raw `str` query with
  `text`/`expansions`/`filters`/`intent`, and parsing `type:pdf`,
  `after:2026-01-01`, `path:handbook/` (or "Q3 2025") out of raw query text
  into a `QueryFilter`. On EDGAR a period filter measured +10.9pp on the
  `period` tier, so extraction is where that gain reaches ordinary traffic.
- **Query understanding, distinct from routing.** This is not the retrieve-or-
  not classifier rejected above — it's normalization, spell correction, and
  expansion parsing on a query that's already going to be searched, plus using
  intent (navigational/factual/exploratory) to vary `top_k` and reranking
  depth. Should share one `LLMClient`-backed rewriter with the existing
  condenser/HyDE code rather than duplicating that seam.
- **Document-level aggregation.** The eval suite already matches at document
  level (`chunk.document_id`); the pipeline still returns chunks. Group
  `ScoredChunk`s by `document_id`, score each document (max, or sum of its
  top-*n* chunk scores), and return a `ScoredDocument` with its best passage —
  needed for a UI/API that lists results rather than answers one question.
- **Dedup & diversity (MMR).** `chunk_overlap: 150` guarantees adjacent chunks
  share text, so near-duplicates can consume the whole `rerank_top_k` budget.
  Collapse near-duplicates post-fusion (SimHash or an embedding-cosine
  threshold) and apply MMR or a per-document result cap.
- **Query-independent ranking signals.** Every score today is pure
  query–chunk similarity; nothing says one document is simply *better* than
  another — recency (EDGAR's `period_end` and `filed` now ride on every chunk
  via `chunking.carry_metadata`; other corpora capture no date), structural
  position (title/heading match — `title` already rides in
  `Chunk.metadata`, unused for scoring), and an intra-corpus link-graph
  authority prior for Markdown docs that cross-reference each other.
- **Snippet generation.** Full chunk text (~1000 chars) goes to both the
  prompt and the UI regardless of how long the relevant span is; select the
  best query-biased sentence window and mark matched terms.
- **BM25F.** `tokenize()` in `rag/retrieval/sparse.py` is a bare regex — no
  stemming, no stopwords, no distinction between a title match and a body
  match. Add both, shared between the index and query paths so they stay
  consistent; move to per-field weights (title/headings/body) once Milestone
  14 gives the index real fields to weight.
- **Deletion support** *(shipped, ahead of the rest of this milestone)*.
  `VectorStore`/`SparseIndex` gained `ids()`/`delete(ids)`, and `index`
  purges chunks the corpus no longer produces
  ([notes](milestone-notes.md#index-maintenance-notes)).
- **Scalable sparse backend**, only once corpus scale demands it. `BM25Index`
  holds every chunk's text in a Python dict, persists as one JSON blob, and
  rebuilds the whole index after any upsert — workable to roughly 10⁴ chunks,
  degrades beyond. SQLite FTS5 or Tantivy behind the existing `SparseIndex`
  ABC is a straight substitution; no pipeline code should need to change.
- Query result caching is **not** repeated here — it's Milestone 13, already
  scoped with the config-fingerprint requirement a naive cache would miss.
- Suggested build order: filter pushdown (shared interfaces) → aggregation +
  dedup (largest visible result-quality win) → ranking signals (title match
  and recency first, link graph later) → snippets → BM25F → scalable backend
  (last, scale-gated).

### Milestone 21 — Streaming & citation fidelity

Two answer-delivery gaps, grouped because both sit at the `ChatService` →
API/UI boundary rather than in retrieval:

- **SSE streaming.** `/chat` and the UI wait for the full generation call
  before showing anything, which reads as high latency on long answers. Needs
  `LLMClient.generate_stream(prompt, *, system=None)` yielding token chunks
  (every adapter implements it, mirroring the existing `generate` ABC), a
  `POST /chat/stream` route returning `text/event-stream`, and
  `rag/ui/app.py` rendering via `st.write_stream`. Citations and the
  pipeline-trace fields only make sense once generation finishes, so the
  stream is answer-text-only — the existing non-streaming response stays the
  source of truth for citations.
- **Active citation filtering.** *Mostly shipped with Milestone 12:*
  `ChatService` parses the `[n]` markers, and `ChatAnswer`/`ChatResponse`
  report `cited_chunk_ids` next to the full `citations` list, so a consumer
  can already tell load-bearing passages from ignored ones. What's left is
  cosmetic: a `cited: bool` on each `Citation` instead of a separate id list,
  and having the UI de-emphasize uncited passages rather than drop them.

### Milestone 22 — Reference-free eval metrics

`answer_eval` needs a hand-written `expected_answer` to grade against, so a
corpus question with no reference answer can't be evaluated at all today.
Faithfulness, answer relevance and context relevance can run without one.

The offline scoring integration is now scheduled in
[eval harness Phase 4a](eval-harness-plan.md#phase-4a--grounding-completeness-and-citation-scoring-estimate-pending).
Its first requirements are faithfulness, required-answer-point coverage and
citation support alongside existing correctness. Answer and context relevance
remain companion scorers on that interface. Calibrate each rubric separately,
retain immutable judgments, and report per-metric applicability and denominators.

- **Faithfulness/groundedness** — already exists as a runtime check
  (`GroundednessChecker`, Milestone 10); use it as a starting point for the
  calibrated offline scorer, adding claim-level evidence and explanations.
  Report it per eval run instead of only inferring it from a live `grounded`
  flag, and keep it separate from correctness.
- **Answer relevance** — does the answer address the question asked,
  independent of whether it's *correct*. New judgment, no existing component
  to reuse.
- **Context relevance** — what fraction of retrieved chunks were actually
  pertinent. Close to what `DocumentGrader` (Milestone 10) already judges
  per-passage; running it as an eval metric over the retrieval eval set gives
  a number without needing CRAG enabled at runtime.
- MAP (mean average precision) is a smaller, cheaper addition to
  `rag/eval/metrics.py` alongside the existing NDCG@k — worth adding in the
  same pass since both are pure functions over gains.

### Milestone 23 — Eval regression gate in CI

CI runs `ruff`, `mypy` and `pytest`, all hermetic, so nothing stops a change
that makes answers or retrieval worse from merging. The eval harness exists
and the noise floor is measured; nothing runs it automatically. Pull this
ahead once the [eval harness plan](eval-harness-plan.md) provides verified
index provenance and immutable score revisions. Nightly answer tracking also
needs its Phase 4 judge calibration and repeat analysis.

- **Retrieval, gated on every PR.** `retrieval_eval` on the EDGAR eval set,
  failing the build when the **paired** 95% CI on Δ hit rate or Δ NDCG against
  a committed baseline run and pinned score revision (with per-sample scores)
  lies entirely below zero — `rag/eval/paired.py`, as `run_matrix.py` reports it. Not a fixed
  threshold of "SE ≈ 2.5pp": that is the unpaired error of one rate, not of a
  difference between two runs on the same questions. A threshold tighter than
  the real noise fails on chance; a looser one lets real regressions through.
- **The index is the hard part.** A GitHub runner has no Ollama, no EDGAR
  documents (gitignored and deliberately not redistributed), and no GPU. Build
  it once from `manifest.json` in a CI job and store it with `actions/cache`,
  selected by the harness's index recipe and corpus content hashes, with the
  restored build identity and inventory validated before reuse. Recipe
  equality alone does not prove the indexed data is unchanged. Then, so
  the per-PR run needs no Ollama, add a precomputed-vector path to
  `retrieval_eval` and the retriever: load the
  committed 174 query embeddings and pass each vector directly to retrieval,
  bypassing the configured embedding provider. Validate the vector metadata
  (model, dimension and query ordering) before running the gate; the vectors
  are derived from our own questions, not the corpus, so the per-PR run needs
  no embedding server.
- **Check the reranker fits the time budget.** `bge-reranker-v2-m3` took
  ~1.1s/query on Apple Silicon; a CPU runner will be slower. If the full set is
  too slow per PR, gate on a fixed stratified subset and run all 174 nightly.
  Report the subset's own noise floor; it is wider than the full set's.
- **Answer quality, nightly or on demand, not per PR.** `answer_eval` needs a
  generator and a judge (both local LLMs), and run-to-run variance is already
  ±2 samples at n=40. A nightly job on a self-hosted runner with Ollama can
  track it and flag a regression; making it block merges would mostly block
  on noise.
- **The quick alternative, and why it isn't enough:** gating on the committed
  `baseline` corpus with a small embedder is cheap and fully hermetic, but it
  gates a 31-chunk toy corpus that none of the measured results come from.
  Worth having as a smoke test alongside, not instead.
- The baseline result file gets updated deliberately, in the same PR as a
  change that intends to move the numbers, never automatically, or the gate
  ratchets toward whatever the last merge scored.

### Milestone 24 — Latency percentiles and cost per query

The measured results report total wall-clock seconds per eval run. That hides
the tail: a mean can look fine while one query in twenty takes 10x longer.
Every turn already records per-stage timings (`TurnRecord.stage_ms`), and
`answer_eval` times each sample, but nothing summarizes either.

- p50/p95 (and max) per stage and end to end, reported by `retrieval_eval`,
  `answer_eval` and `run_matrix.py` next to the quality metrics, and by
  `python -m rag.cli turns` over logged traffic. `retrieval_eval` needs
  per-query timing added first; it doesn't time samples today.
- Add latency columns to the tables in `docs/measured-results.md`, starting
  with the reranker comparison, where the tradeoff (v2-m3: +10.4pp hit for
  ~4x latency) is currently stated as a mean.
- Cost per 1,000 queries: tokens are already metered (`UsageMeter`); price
  them per model from config. With local Ollama the dollar figure is $0 and
  the meaningful number is tokens and seconds per query. It becomes a real
  cost once Milestone 18's hosted adapters land, so price the hosted models
  in config rather than hardcoding a $0.
- Use the same percentile code for evals and logged turns, so an eval latency
  and a production latency are measured the same way.

### Milestone 25 — Chunk size and overlap sweep

**Planned:** see [Chunking and indexing plan](chunking-indexing-plan.md), Phase 7: run the sweep
on whichever chunker Phase 5 leaves as the default.

`chunking.chunk_size` and `chunk_overlap` are in `run_matrix.py`'s fingerprint
but have never been varied. The matrix excludes them because each value needs
its own index build, not a config flip. The shipped values were never tuned
on EDGAR.

- A small grid (e.g. size 500 / 1000 / 2000 × overlap 0 / 150 / 300), each
  built into its own `index_dir` the way the contextual comparison did it
  (`data/eval/config_contextual.yaml`), since the collection name is derived
  from the corpus selection alone and would otherwise be overwritten.
- **Chunk size changes what a hit means.** The eval set matches answer spans,
  and a larger chunk contains more spans by construction, so hit rate goes up
  with size even if the reranker gets no better. Report NDCG, and the
  prompt-token cost of `rerank_top_k` chunks of that size, alongside hit rate.
- Record chunk-boundary failures: samples whose answer span is split across
  two chunks at one size and whole at another. This is the concrete
  before/after for the failure-analysis section, and it tells Milestone 15
  (semantic chunking) whether boundaries are actually the problem.
- Run after Milestone 23 if possible, so the chosen defaults become the CI
  baseline. Check at least one size against a second reranker rather than
  assuming the best size is independent of it: the MiniLM-vs-BGE pool-size
  reversal in `docs/measured-results.md` is the precedent.

### Milestone 26 — OpenTelemetry trace export

The turn log is local JSONL, read by `python -m rag.cli turns`. That is
deliberate (Milestone 12), and the sink sits behind an interface so an
exporter can be another adapter (`rag/observability/sink.py`). What the JSONL
can't give: a trace view of the stages as spans, search over many turns, and
comparing latency across deployments.

- An OpenTelemetry exporter behind the existing sink interface, one span per
  pipeline stage (the `stage_ms` keys), LLM calls as child spans carrying
  token counts, and the turn's retrieved/cited chunk ids as attributes.
- Point it at a self-hosted Arize Phoenix or Langfuse via config. Both accept
  OTLP, so the adapter is written against OTel and not either vendor's SDK.
  That keeps it one dependency (the OTel SDK) and swappable between them.
- JSONL stays the default. The OTel sink is off unless configured, and the
  eval runners still log nothing.
- Justify the OTel SDK against the minimal-dependencies rule. It is the
  standard, and hand-writing OTLP would be worse, but it is a new dependency.
- Most useful after Milestone 18, when there is a deployment to trace, and
  after Milestone 19, when a turn becomes 1–N tool calls a flat record reads
  poorly.

### Milestone 27 — Eval coverage and judge reliability

Implementation order and acceptance criteria now live in the
[evaluation rigor plan](evaluation-rigor-plan.md). Prioritize a fresh grouped
holdout, judge calibration, and evidence-aware scoring before further broad
quality claims or extensive tuning. The existing sets remain development and
regression benchmarks; the plan below is not evidence that these additions
have shipped.

PR #15 fixed how differences are *tested* (paired CIs) and made answer failures
attributable to retrieval or generation. What remains is what the eval sets
*measure* and how far the judge can be trusted. Several "measured-off" verdicts
hold only for the kind of question the eval set contains, so this matters as
much as any new feature. Judge calibration and repeat analysis are now planned
as acceptance requirements of [eval harness Phase 4](eval-harness-plan.md#phase-4--answer-side-generate-and-judge-as-separate-steps-15-days),
ahead of Milestone 23's nightly answer tracking. Coverage additions and an
untouched confirmation set remain dataset work here, and must precede the
broader measurement claims they support. None of this is marked shipped by
the harness plan's publication.

- **A question tier that doesn't name its own entities.** In
  `edgar_eval_set.json`, all 174 questions name company and period (by
  construction of the generator). 128 start with "What was/were", 124 spans
  contain a figure, and a median 58% of a question's content words appear in
  its answer span. That favors lexical matching and may limit the measured
  benefit of contextual chunking and query expansion; the results do not
  establish the cause. Broader questions include "which airline's fuel costs
  rose most" and "how exposed is Apple to China". Add a separate,
  hand-reviewed tier of underspecified and paraphrased questions. Report it
  apart from the generated set, and re-measure contextual chunking and
  expansion on it before treating either verdict as general. *Built and
  baselined in [Chunking and indexing plan](chunking-indexing-plan.md)
  Phase 0 (2026-09-27): `edgar_underspecified_set.json`, 118 questions,
  retrieval hit 0.833 `implicit` / 0.500 `paraphrase` against 0.908 on the
  generated set. The `period` tier shipped with it; `table` waits for Phase 4.
  Every chunking-plan phase since has been judged on all three sets. Contextual
  chunking is not currently prioritized for re-measurement: Phase 2's
  deterministic header replaced it (+14.4pp answer pass on this tier).
  That implementation choice does not establish that generated context
  cannot help another workload. Query expansion and CRAG
  haven't been re-measured on it.*
- **Turn log → eval candidates.** Milestone 12 called logged queries with
  feedback "the cheapest source of new eval samples", but nothing converts
  them. Add a `rag.cli turns --export-candidates` path that writes thumbs-down
  and uncited turns as draft samples (query, retrieved chunk ids, answer) for
  a human to add spans and a reference answer to. Never auto-label: a silently
  wrong label is worse than a smaller set, the rule `generate_eval_set.py`
  already follows. This is also the natural source for the tier above.
- **Question review and annotation procedure.** Follow the
  [harness measurement requirements](eval-harness-plan.md#measurement-quality-requirements):
  record answerability, target-user usefulness and clarity with decisions and
  reasons before accepting new samples. Apply tier-specific criteria so valid
  refusals and history-dependent questions are retained. LLM critique can flag
  drafts; human review decides acceptance. Write worked labeling examples,
  independently review a subset across tiers/outcomes, and preserve reviewer
  decisions, agreement counts and adjudications. Use the same procedure for
  benchmark labels and judge-calibration labels; document a single-reviewer
  limitation when independent review is unavailable.
- **Required answer points.** Harness Phase 4a extends compound questions with
  reviewed facts and evidence, including company, period, quantity and units
  where applicable. Reuse the multi-hop parts/conclusion pattern to report
  coverage and omissions beside pass/fail, leaving one-fact questions simple.
  Store per-point judgments and version label changes; correctness and
  completeness do not substitute for checking grounding.
- **Conversational coverage.** Owned by
  [harness Phase 4c](eval-harness-plan.md#phase-4c--conversational-dataset-and-runner-estimate-pending).
  Add a separate reviewed conversation dataset covering elliptical follow-ups,
  company/topic switches, corrections and unanswerable turns. Evaluate fixed
  reference history separately from histories containing each variant's own
  answers. Record ordered turns, actual history, exact generator messages and
  rewritten retrieval queries; report per-turn and conversation outcomes.
  Keep conversations together in splits and uncertainty calculations. Multi-hop
  questions alone do not test these behaviors.
- **Judge calibration set.** The judge's accuracy has been spot-checked once
  (15 complete multi-hop answers read by hand, one false pass), never
  measured. Hand-label ~60 answers once: single-hop pass and fail, refusals,
  and multi-hop parts, including the known false-pass shape (a figure
  attached to the wrong period). Commit them as a fixture, and add a runner
  that reports the judge's agreement with them and its false-pass and
  false-fail rates. Re-run it whenever `eval.judge` or a judge prompt changes.
  Include category counts, uncertainty and human-label rationale; set acceptable
  error bounds before testing a new judge. Also recheck parser changes and
  retain repeated judgments of fixed answers to measure consistency separately
  from accuracy. Phase 4 stores each attempt with rubric, reference and model
  provenance. Without calibration, a pass-rate change can't be separated from
  judge drift; consistency alone does not establish correctness.
- **Reference answers that are answers.** 61 of 174 `expected_answer`s are the
  expected span copied verbatim, a sentence fragment rather than an answer.
  Regenerate them as one-sentence answers (a human reviews the diff) so the
  judge compares like with like.
- **Near-miss negatives in the refusal set.** 15 samples, and every CRAG
  variant scored 14–15/15: a set everything aces can't catch a regression.
  The realistic hard case here is the right company in a period the corpus
  doesn't hold (the 27b's Apple FY2015 parametric leak). Generate
  period-shifted negatives from answerable samples, and verify programmatically
  that the answer is absent from the corpus before keeping one. That check is
  what the two removed Costco negatives lacked. The cross-company superlative
  is already covered by the Milestone 19 plan's open question. Human-review
  and freeze this tier before comparing variants, and require its results
  before claiming reliable refusal behavior.
- **Generation settings and repeat analysis.** The generator runs at
  `llm.temperature: 0.2` during evals, which is where the measured ±2-sample
  run-to-run noise on answer eval comes from. Either add an `eval.generator`
  override (temperature 0) and check its observed variability, or keep the
  deployment settings and repeat them. Harness Phase 4 adds
  `run_answer_matrix.py --repeats k` and reports per-run rates, mean, standard
  deviation, range and paired repeat deltas. Start with at least three repeats
  per compared variant, predeclaring count and pairing/order; this does not
  guarantee enough power for small effects. Retain repeat IDs and available
  seeds. The paired CI covers which questions were sampled, not a single
  run's sampling luck. Do not count repeats of a question as independent new
  questions. Repeat judging of fixed outputs separately from generation;
  any combined interval must account for repeated observations per question.
  With paired CIs the full 174-sample set (~45 min per variant at the phase-0
  latency) is also worth making the default over `--limit 40`.
- **Coverage before generalization.** Report generated, period, underspecified
  (and each kind), refusal and multi-hop tiers separately, with counts and
  remaining coverage gaps. The table tier is owned by chunking-plan Phase 4
  and is required before evaluating its new chunker; harder refusal cases
  are required as above. Audit label alternatives and reference-answer quality
  alongside adding questions. Shipping the harness does not complete coverage.
- **Untouched confirmation set for default selection.** Reserve reviewed,
  frozen questions before further tuning, covering the intended tiers and
  grouping related source facts/document families so paraphrases do not cross
  the tuning/confirmation boundary. Use existing matrices for exploration;
  keep confirmation outcomes out of candidate selection. Predeclare primary
  metrics, slices and practical improvement/regression criteria, then compare
  the selected candidate with the baseline and publish that result separately.
  Freeze rules protect labels but do not prevent tuning to a repeatedly
  inspected benchmark. Record confirmation-set use and replenish after its
  results have informed later tuning. Harness completion alone cannot satisfy
  this requirement for selecting new defaults.
- **Missing labels.** Spans are checked to be unique across filings, but not
  checked for *other* chunks stating the same fact in different words (a
  table next to prose). Judge the top-5 chunks of every retrieval miss with
  the LLM once, to estimate how often a "miss" retrieved a correct answer. If
  it is more than a few percent, add the alternates as extra spans.
- **Metrics that duplicate each other.** Every generated sample has one
  grade-1 span, so hit rate equals recall and NDCG reduces to a rank discount.
  Graded relevance is supported but unused. Either have the generator also
  label a grade-2 span (the sentence stating the figure versus one discussing
  the topic), or stop reporting the duplicates for single-span sets.
- **Per-citation support.** Milestone 21 marks which passages an answer cited;
  Milestone 22 scores faithfulness for the whole answer. Neither checks that
  a cited passage `[n]` supports the claim attached to it. An eval-only metric
  (judge each claim–citation pair) is now part of harness Phase 4a alongside
  Milestone 22's faithfulness. Preserve markers, mappings and passage text;
  distinguish incorrect citations from missing citations under a declared
  citation requirement. Calibrate on human-reviewed examples and keep this
  score separate from correctness and whole-answer grounding.
- **Smaller harness fixes.** `recall_by_k` averages only over samples that
  returned at least k results, so the denominators differ when `min_score`
  prunes lists. Pad to k instead. The `baseline` corpus's `eval_set.json` is
  document-matched only and hasn't been measured since the reranker change:
  add spans or retire it as a control.

### Milestone 28 — Production hardening

Milestone 18 makes the components swappable for hosted ones. It doesn't make
the service safe to expose. Today nothing is authenticated: `/chat`,
`/feedback` and `/mcp` are open to anyone who can reach the port, and
`/mcp`'s `allowed_hosts` only blocks DNS rebinding. `ChatRequest.query` and
`history` have no length limit. `/chat` logs every raw query at INFO. `/health`
returns `ok` while Ollama or Chroma is down. `docker-compose.yml` runs
`uvicorn --reload` over a bind-mounted checkout as root. This milestone
closes those gaps.

**Target: one internal team, stated up front.** That means a known set of
callers, one shared corpus, one deployment, and no per-user access control.
Write the target and a short threat model in `docs/operations.md` before any
code. Every item below is measured against that target, and anything outside
it is listed as deliberately out of scope rather than silently missing. A
public demo URL changes the threat model (anyone can call it), and auth plus
rate limiting are what cover that case.

- **Auth.** An API-key check as a FastAPI dependency (`APIKeyHeader`,
  constant-time compare, keys from the environment, never `config.yaml`) on
  `/chat` and `/feedback`. `/health` stays open. `/mcp` is a mounted
  sub-app, which router dependencies don't reach, so it needs the same check
  as ASGI middleware. Test it: a request to `/mcp` without a key must fail.
  *The right mechanism for more than one team* is an auth proxy or gateway
  (OIDC) in front, with the app trusting it. Key auth is the step before that
  and won't have to be removed when the proxy arrives.
- **Input bounds and rate limiting.** `max_length` on `query`, and a cap on
  `history` turns and total characters, so a single request can't buy an
  unbounded prompt. Per-key rate limiting: an in-process token bucket needs
  no new dependency but is per replica. Say so, and name the gateway as the
  place it belongs once there is more than one replica.
- **Prompt-injection delimiters.** Wrap each passage and the query in
  explicit tags in `build_rag_prompt`, and tell the system prompt that tagged
  content is data, not instructions. This changes the grounded prompt, so
  re-measure answer quality with the `measure-change` skill before and after.
  Also add a few adversarial documents (planted "ignore previous
  instructions" text) to a separate eval tier, and report how often they
  work. Delimiters reduce injection; they don't prevent it. Record the rate
  rather than claiming a fix.
- **Failure behaviour.** A `/ready` check, separate from `/health`
  (liveness), that asks the configured components rather than a fixed list,
  so each deployment checks only what it runs. Locally that is Ollama and
  Chroma. On the GCP plan (Milestone 18) it is the loaded index and the
  in-process embedder and reranker. Do not call a hosted API from `/ready`:
  probes run every few seconds, and a `generateContent` probe would spend the
  Gemini free quota. A hosted-API outage would also mark every replica
  unready at once, when it should show up as per-request 503s and an alert.
  An upstream failure maps to a fast 503 with a clear message instead of a
  500. A hung Ollama currently holds a
  thread-pool worker for up to `llm.timeout_s` (120s), and Starlette's
  default pool is 40 threads. Add a test per dependency-down case.
- **Production container.** A non-editable install, a non-root user, no
  `--reload`, no bind mount, and a pinned Ollama image instead of `:latest`.
  Keep the dev compose file as it is and add a production one alongside it.
  Use persistent in-process Chroma for this deployment; it pins the API to one
  replica.
- **Log hygiene and retention.** Query text reaches the application log from
  15 call sites, not just `/chat`: the condenser (`query_rewriter.py`), CRAG's
  retry and groundedness paths (`crag.py`), HyDE and multi-query expansion
  (`expansion.py`), `chat_service.py`, and web search (`websearch.py`).
  Several of those are at WARNING, which production keeps. Move raw query,
  rewrite and passage text to DEBUG at every site. At INFO and WARNING, log
  the `turn_id`, lengths and counts instead. The turn log is then the only
  place query text is stored, and the retention setting below governs it.
  Add a test that runs a turn at INFO with every query path enabled and
  asserts the query string never appears in captured logs, so a new call
  site can't reintroduce it. Add a retention/rotation setting to
  `observability.turn_log`. `JsonlTurnSink`'s lock covers threads, not
  processes, so either run one worker per log file or verify multi-worker
  appends of multi-KB records don't interleave.
- **Load test with published numbers.** A small script over `httpx` (already
  a dependency, so no locust) that ramps concurrent `/chat` calls against the
  EDGAR index and reports p50/p95 and error rate per concurrency level, plus
  the saturation point. Use Milestone 24's percentile code if it has landed.
  Record the results in `docs/measured-results.md` with the hardware and
  config attached, the same as any other measured result. The expected
  finding is that one local Ollama is the bottleneck. The number is what
  justifies Milestone 18's hosted adapters.
- **Runbook.** In `docs/operations.md`: deploy, rotate a key, rebuild the
  index, what `/ready` failing means, and restoring the index from its
  corpus manifest.

Out of scope, stated so it doesn't read as forgotten: multi-tenancy,
per-document access control, SSO in the app itself, autoscaling, and
zero-downtime reindexing (building a new collection while the old one serves,
then switching over). The last is the natural next milestone once the index
is rebuilt on a schedule rather than by hand.

Ordering: do this ahead of Milestone 13. Auth, input bounds and the
production container come first, because they're what make any deployment
safe to expose. Milestone 23's retrieval gate pairs with it: "production
ready" includes "a change can't quietly make answers worse", and that gate
has no dependencies either.
