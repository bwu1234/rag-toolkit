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
- [x] Milestone 9 — Contextual chunking (index-time enrichment)
- [x] Milestone 10 — Corrective RAG (grade / retry / groundedness)
- [x] Milestone 11 — Measure Milestones 9 & 10 (before/after eval numbers)
      — see [Measured results](docs/measured-results.md). Headline: the reranker
      was the bottleneck (swapped, +10.4pp hit rate); contextual chunking, CRAG,
      query expansion and `min_score` all measured as no better than noise on
      this corpus and stay off.
- [x] MCP server — retrieval exposed to external agents as read-only tools
      (`rag_search`, `rag_list_corpora`) over stdio and streamable HTTP.
      Not Milestone 19: that is *this* system calling search as a tool;
      this is an outside agent calling ours. See [MCP server](docs/mcp-server.md).
- [ ] Milestone 12 — Observability (query logs, latency/cost, feedback)
- [ ] Milestone 13 — Query result caching
- [ ] Milestone 14 — Richer document parsing (tables, layout, OCR fallback)
- [ ] Milestone 15 — Semantic chunking
- [ ] Milestone 16 — Async ingestion (queue + worker)
- [ ] Milestone 17 — PII detection & redaction
- [ ] Milestone 18 — Deployment (container, hosted providers)
- [ ] Milestone 19 — Agentic retrieval (search as a tool the model calls)
- [ ] Milestone 20 — Search-quality layer (query filters, ranking signals, dedup/MMR, BM25F)
- [ ] Milestone 21 — Streaming & citation fidelity
- [ ] Milestone 22 — Reference-free eval metrics (RAG triad, MAP)

See [Backlog](docs/backlog.md) for what each of these means and why it's
ordered where it is.

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

### Corpora

`corpora.registry` names bodies of documents; `corpora.active` picks which ones
a command operates on, overridden per-invocation by `--corpus NAME` (repeatable,
supported by the CLI and both eval runners).

Listing **several** names pools them into one index, and that is the point of
the registry rather than an afterthought:

- **isolated** (one name) — retrieval quality *within* a corpus;
- **pooled** (several) — robustness to plausible-but-wrong neighbours.

The gap between the two is the **cross-corpus interference cost**, and it can't
be observed with a single corpus. It's also what finally makes
`retrieval.min_score` and CRAG's document grader measurable: on a small,
topically-distinct corpus nothing irrelevant is ever nearby, so the grader has
no job to do.

`config.corpus_selection(names)` resolves this into a `CorpusSelection`, which
**derives its own storage names** — `rag_corpus__edgar` vs.
`rag_corpus__baseline+edgar`, and `bm25_index__<slug>.json` alongside. Isolated
and pooled indexes therefore coexist instead of silently overwriting each other,
which matters because comparing them is the exercise. Selections are sorted and
deduped, so `--corpus a --corpus b` and `--corpus b --corpus a` are one index.

- An unknown corpus name **raises** rather than indexing nothing — a typo would
  otherwise produce an empty index and a plausible-looking all-zero eval run.
- Pooling refuses **duplicate `Document.id`s**. Ids are corpus-relative paths,
  so two corpora each containing `faq.txt` would produce colliding chunk ids and
  the store would silently upsert one over the other. Namespacing ids by corpus
  was rejected as the fix: it would change every `document_id`, invalidating the
  `expected_doc_ids` already recorded in the eval sets, to solve a problem the
  current corpora don't have.
- An empty registry falls back to a single implicit corpus at
  `paths.corpus_dir`, so a config predating the registry keeps working.
- **Existing indexes need one rebuild.** Collection and BM25 names now carry the
  selection slug, so a pre-registry `data/index` won't be found. Re-run
  `python -m rag.cli index --corpus <name>`; the contextual cache is keyed by
  prompt content, not by index name, so contexts are not re-paid for.
- On-disk convention: each registry entry's `documents_dir` lives at
  `data/corpora/<name>/documents/`. `baseline` (the small hand-written sample
  corpus) is committed to the repo; other corpora (e.g. `edgar`) are fetched
  and gitignored, reproduced from a `manifest.json` alongside their
  `documents/` — see `.gitignore` for the exemption that keeps `baseline`
  committed despite the general `data/corpora/*/documents/` ignore rule.

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
  mcp/          MCP server (tools.py defines the surface; server.py = SDK
                transports, fallback.py = zero-dep stdio JSON-RPC)
  eval/         retrieval & answer evaluation scripts + eval set
  ui/           Streamlit app
tests/          pytest suite, mirrors rag/ layout
data/corpora/   named corpora (see Corpora below); each is <name>/documents/
                plus an optional manifest.json. `baseline` is committed sample
                content; other corpora (e.g. `edgar`) are gitignored and
                reproduced by a fetch script.
data/index/     persisted Chroma index (gitignored — rebuildable)
docs/           design rationale, measured results, backlog, known limitations
                (see Documentation below) — not always-loaded, read on demand
```

## Running things

(Filled in as each milestone lands.)

- Run tests: `pytest`
- See the whole pipeline run: `scripts/demo.sh` (ingest → chunk → index →
  retrieve → chat, on the active corpus). Takes `--corpus NAME` (repeatable) and
  `--question "..."`; resolves document directories from `corpora` in config, so
  it checks the corpus is actually present before starting.
- Ingest & inspect the corpus: `python -m rag.cli ingest --show 3`
  (add `--corpus edgar`, or repeat it to pool: `--corpus baseline --corpus edgar`;
  works on every command below and on both eval runners)
- Chunk & inspect chunk sizes: `python -m rag.cli chunk --show 3`
- Build the index: `python -m rag.cli index` (add `--reset` to rebuild from scratch;
  required after changing anything under `chunking.contextual`). Generated chunk
  contexts are checkpointed and survive `--reset` on purpose, so a rebuild does
  not re-pay for them; add `--clear-context-cache` to force regeneration.
- Retrieve & rerank for a query: `python -m rag.cli retrieve "your question"`
- Ask a question end to end (retrieve → rerank → generate, with citations): `python -m rag.cli chat "your question"`
- Start the API: `uvicorn rag.api.main:app --reload` (then `POST /chat` with `{"query": "..."}`, or check `/health`)
- Serve retrieval to an external agent over MCP: `python -m rag.mcp` (stdio), or
  `POST /mcp` on the running API (streamable HTTP). Exposes two read-only tools,
  `rag_search` and `rag_list_corpora` — see [MCP server](docs/mcp-server.md).
- Start the UI: `streamlit run rag/ui/app.py`
- Run retrieval eval: `python -m rag.eval.retrieval_eval` (add `-v` for per-sample detail;
  `--eval-set data/eval/edgar_eval_set.json --corpus edgar` for the EDGAR set)
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

## Documentation

This file is the operational core, kept lean on purpose since it loads into
every conversation. Design rationale, history, and forward-looking planning
live in `docs/`, read on demand rather than always-loaded:

- **`docs/milestone-notes.md`** — why each shipped milestone (2–10: ingestion
  through Corrective RAG) is built the way it is. Read before touching a
  component to see what tradeoff its current shape already encodes.
- **`docs/measured-results.md`** — Milestone 11: retrieval, reranker-model,
  contextual-chunking and CRAG numbers on the EDGAR corpus, with the caveats
  needed to read them safely. Read before turning on anything that is off by
  default — several features measured as no better than noise.
- **`docs/backlog.md`** — Milestones 11–22, planned work not yet started.
- **`docs/known-limitations.md`** — known gaps and failure modes in what's
  shipped, worth checking before recommending a feature that's off by default.
- **`docs/mcp-server.md`** — the MCP tool contract, both transports, how to
  point an external agent at it, and why indexing is not exposed as a tool.
