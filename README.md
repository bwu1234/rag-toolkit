# RAG Toolkit

A local-first Retrieval-Augmented Generation (RAG) system in Python. Ollama
serves both embeddings and chat, and Chroma is the vector store.

The pipeline runs ingest → clean → chunk → embed → index at index time, and
hybrid retrieval (dense vector search + BM25, fused with reciprocal rank
fusion) → cross-encoder rerank → cited generation at query time. Each chunk is
indexed behind a deterministic header naming its document (e.g. "Apple Inc.
(AAPL) 10-K, period ended 2024-09-28"), and retrieval can be restricted by
document metadata (company, period, form, …). An opt-in agentic mode lets the
model call search as a tool instead of retrieving once. Each stage
sits behind an interface (`EmbeddingModel`, `VectorStore`, `Reranker`,
`LLMClient`, `QueryExpander`, `Chunker`) and is selected in
[rag/config/config.yaml](rag/config/config.yaml), so swapping an implementation
is a config change rather than a code change.

The same pipeline is reachable from a CLI, a FastAPI service, a Streamlit UI,
and an MCP server. By default, every chat turn is logged with its retrieved and cited
passages, per-stage latency, LLM token counts, and any thumbs up/down feedback.

## Prerequisites

- Python 3.10+ (CI runs 3.14)
- [Ollama](https://ollama.com) running at `http://localhost:11434`, with the two
  default models pulled:

  ```bash
  ollama pull qwen3-embedding:0.6b
  ollama pull qwen3.5:9b-mlx
  ```

- Network access on first query: the reranker (`BAAI/bge-reranker-v2-m3`) is
  downloaded from Hugging Face by `sentence-transformers` the first time it
  loads, and cached after that. It is the only default component that doesn't
  run through Ollama.
- Optional: a `GEMINI_API_KEY` to run the generator, agent or eval judge on
  the Gemini API (`llm.provider: gemini`; overlays in
  `rag/config/gemini-*.yaml`). Nothing uses it by default.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'      # quotes stop zsh from globbing the brackets
pytest -q                    # no Ollama needed; HTTP calls are mocked
```

Then build an index over the bundled `baseline` corpus and ask it something:

```bash
python -m rag.cli index
python -m rag.cli retrieve "What does this project do?"   # ranked passages only
python -m rag.cli chat "What does this project do?"       # generated answer with citations
```

`ingest --show N` and `chunk --show N` print what the loader and chunker
produce without touching the index — useful when a document doesn't seem to be
contributing anything. `index-report` is the read-only health check: chunk
sizes, chunks starting mid-table, duplicates, how many documents got a chunk
header, and whether the built index is in sync with the corpus and config.

`index` is incremental: unchanged chunks are skipped and chunks of deleted or
shortened documents are removed. Changing the embedder or anything under
`chunking.contextual`, `chunking.carry_metadata` or `chunking.header` needs
`index --reset`, and the index manifest refuses the run until you do.

`scripts/demo.sh` runs the whole sequence (ingest → chunk → index → retrieve →
chat) and accepts `--corpus NAME` and `--question "..."`.

## Corpora

A corpus is a named directory of documents registered under
`corpora.registry` in the config. `corpora.active` picks which ones a command
uses; `--corpus NAME` overrides it and is accepted by the CLI, `demo.sh`, and
the eval runners.

| Corpus | Contents | Where it comes from |
|---|---|---|
| `baseline` | 8 hand-written sample docs | Committed at `data/corpora/baseline/documents/` |
| `edgar` | 61 SEC 10-K/10-Q MD&A sections (~4,200 chunks) | Fetched from its manifest (below); gitignored |

```bash
# The SEC requires a User-Agent identifying the requester
SEC_USER_AGENT="Your Name you@example.com" \
  python scripts/fetch_edgar.py --manifest data/corpora/edgar/manifest.json
python -m rag.cli index --corpus edgar
```

The fetcher writes YAML front matter (`company`, `ticker`, `form`,
`period_end`, `filed`, `accession`) that the chunk header and metadata filters
read. Filings fetched before it did need `scripts/fetch_edgar.py
--backfill-front-matter`; `index-report` shows how many documents have a header.

Passing `--corpus` more than once **pools** the corpora into a single index.
Each selection gets its own collection (`rag_corpus__edgar` vs.
`rag_corpus__baseline+edgar`), so isolated and pooled indexes coexist and can
be compared.

To add your own documents, either drop them into
`data/corpora/baseline/documents/` or register a new corpus in the config, then
re-run `python -m rag.cli index`. Add `--reset` to rebuild from scratch.

## Supported document formats

Ingestion dispatches on file extension. Files with any other extension are skipped (at `DEBUG` log level — check the `N file(s) skipped` count in the ingest summary if a document seems missing). That count also includes files whose loader raised (e.g. a corrupt PDF), which are logged at `ERROR` with a traceback rather than failing the run:

| Extension | Loader | Granularity |
|---|---|---|
| `.pdf` | `PdfLoader` (pypdf) | one document **per page**, so citations can reference a page number |
| `.md`, `.markdown` | `MarkdownLoader` | one document per file |
| `.txt` | `TextLoader` | one document per file |

Formats **not** currently supported include `.docx`, `.pptx`, `.xlsx`, `.csv`, `.html`, `.json`, and `.epub`.

Some limits worth knowing before pointing the pipeline at a corpus:

- **No OCR.** PDF text comes from `pypdf`'s text layer. A scanned or image-only PDF extracts as empty text and contributes zero chunks — it loads without error but adds nothing to the index.
- **Text only.** Images and layout structure are dropped, and PDF tables are linearized into running text rather than preserved as tables.
- **Chunking is format-blind** — fixed-size character windows over the extracted text (see `chunking` in [rag/config/config.yaml](rag/config/config.yaml)), which suits prose better than highly structured content. On EDGAR about 13% of chunks start partway through a table; a structure-aware chunker is planned ([chunking plan](docs/chunking-indexing-plan.md), Phases 4–5).
- **Markdown front matter becomes metadata.** `MarkdownLoader` parses a YAML front-matter block into `Document.metadata` and strips it from the text; `chunking.carry_metadata` picks which keys ride on each chunk (and so which are filterable).
- **Ingestion is an offline CLI step.** There is no upload endpoint or widget; the API and UI only query an index that `python -m rag.cli index` already built.

### Adding a format

The loader interface is the extension point — everything downstream operates on `Document.text` uniformly, so no other pipeline code changes:

1. Subclass `Loader` ([rag/ingestion/models.py](rag/ingestion/models.py)), set `extensions`, and implement `load()`, building ids with `make_document_id`.
2. Add an instance to the loader tuple that builds the `_LOADERS` extension table in [rag/ingestion/loaders.py](rag/ingestion/loaders.py).
3. Add the parsing dependency to [pyproject.toml](pyproject.toml).

## Interfaces

### API

```bash
uvicorn rag.api.main:app --reload
```

| Endpoint | Purpose |
|---|---|
| `POST /chat` | `{"query": "...", "history": [...], "filters": {...}}` → answer with citations and a `turn_id`. The API is stateless; clients replay their own history. `filters` (optional) restricts retrieval by document metadata, e.g. `{"equals": {"ticker": "AAPL"}, "range": {"period_end": {"gte": "2025-01-01"}}}`. |
| `POST /feedback` | `{"turn_id": "...", "rating": "up" \| "down", "comment": "..."}` |
| `GET /health` | Liveness check |
| `/docs` | Interactive OpenAPI docs |
| `/mcp` | MCP over streamable HTTP (requires the `mcp` extra) |

### UI

```bash
streamlit run rag/ui/app.py
```

A chat interface over the same pipeline, with citations and thumbs up/down.

### MCP server

Exposes retrieval to an external agent as two read-only tools: `rag_search`
(ranked passages, no generated answer; takes the same optional `filters`) and
`rag_list_corpora`.

```bash
python -m rag.mcp              # stdio; works on the core install
pip install -e '.[mcp]'        # adds streamable HTTP at /mcp on the API
```

It serves MCP revision 2026-07-28 only. Client configuration, the tool
contract, and why indexing isn't exposed are in
[docs/mcp-server.md](docs/mcp-server.md).

### Logged turns

With the default `jsonl` provider, the API, UI, and `cli chat` append every
turn to `observability.turn_log.path` — `data/logs/turns.jsonl` unless you
change it (gitignored there); the eval runners never do. `cli turns` reads back
from the same configured path. Set `observability.turn_log.provider: none` to
turn logging off — `POST /feedback` then returns 503, since there is no turn
record to attach the rating to.

```bash
python -m rag.cli turns --show 20 --feedback down
```

## Configuration

Everything — component selection, model names, chunk size, retrieval depth —
lives in [rag/config/config.yaml](rag/config/config.yaml) and is validated by
pydantic models in [rag/config/settings.py](rag/config/settings.py). The YAML
is commented with why each default is what it is.

Any single key can be overridden with an environment variable named
`RAG__SECTION__KEY`, for example `RAG__LLM__BASE_URL=http://ollama:11434`.
This is meant for per-deployment values such as service URLs; lists and
mappings still have to be set in the YAML.

A config file can start with `base: <path>` to inherit another and list only
what it changes (mappings merge, lists replace), and any command takes
`--config`. [rag/config/vanilla.yaml](rag/config/vanilla.yaml) uses this for
a plain dense-RAG baseline (no BM25, no reranker, no chunk header, plain
prompt) on its own index; `rag/config/gemini-*.yaml` swap only the generator.

`chat.mode` selects how an answer is produced: `pipeline` (the default:
retrieve once, then generate) or `agentic` (the model calls `rag_search` as a
tool under the guards in `agent:`). Agentic mode stays off until
[Milestone 19](docs/milestone-19-plan.md)'s phase 4 measures it.

Several features ship **disabled** because they measured as no better than
noise on the EDGAR corpus: contextual chunking, corrective RAG (CRAG), query
expansion (HyDE / multi-query), the `retrieval.min_score` relevance floor, and
the embedder's query instruction (`embedding.query_instruction`). Document
routing (`retrieval.document_routing`) gained on the period-specific question
tiers but didn't clear McNemar's test, so it is off too. Live web search via
SearxNG is also off by default, since enabled queries leave the machine.
Re-measure before turning any of these on — see
[docs/measured-results.md](docs/measured-results.md).

## Evaluation

```bash
python -m rag.eval.retrieval_eval -v                  # hit rate, MRR, NDCG on the baseline set
python -m rag.eval.retrieval_eval --eval-set data/eval/edgar_eval_set.json --corpus edgar
python -m rag.eval.answer_eval                        # LLM-as-judge answer quality
python -m rag.eval.multihop_eval --corpus edgar       # questions that need several documents
```

EDGAR's question sets in `data/eval/` are reported separately rather than
averaged: `edgar_eval_set.json` (174 generated questions that name their
company and period), `edgar_period_set.json` (the same text appears in another
period's filing), `edgar_underspecified_set.json` (paraphrased questions, or
questions that name the company only indirectly), `edgar_multihop_set.json`
and `edgar_refusal_set.json` (questions the corpus can't answer). To
compare config variants question by question, with paired confidence
intervals and McNemar's test, use `scripts/run_matrix.py` (retrieval) and
`scripts/run_answer_matrix.py` (answers). The `measure-change` skill in
`.claude/skills/` walks through it.

The answer judge is `eval.judge` in the config. When that is unset, the
generator grades itself and the runner warns, because swapping the generator
then swaps the judge too. `--judge-model` overrides it per run, and
`--judge-provider` sets the judge's provider when it differs from the one it
inherits.

## Project layout

```text
rag/
  api/            FastAPI app and routes
  chunking/       chunkers and the contextual-chunking cache
  config/         config.yaml and its pydantic settings
  embedding/      embedding adapters
  eval/           retrieval, answer, and multi-hop evaluation
  generation/     prompts, LLM clients (Ollama, Gemini), chat service, CRAG, agent
  ingestion/      document loaders and cleaners
  mcp/            MCP server and transports
  observability/  turn records, LLM usage metering, feedback sink
  retrieval/      dense/sparse retrieval, RRF, rerankers, query expansion, document routing
  ui/             Streamlit UI
  vectorstore/    vector store adapters
  tools.py        the rag_search / rag_list_corpora tools, shared by MCP and the agent
  query_filter.py typed metadata filter (QueryFilter)
  index_manifest.py, index_report.py   index settings guard and health report
  cli.py          ingest / chunk / index / index-report / retrieve / chat / turns
scripts/          demo, EDGAR fetcher, eval-set builders and tier drafter, eval matrices
tests/            one test module per component
data/
  corpora/        corpus documents (and manifests for fetched corpora)
  eval/           eval sets and recorded results
```

## Development

CI runs these three; run them before pushing:

```bash
ruff check .
mypy --ignore-missing-imports rag   # scoped to rag/: tests use structural fakes
pytest -q
```

## Further reading

- [docs/architecture.md](docs/architecture.md) — how the index-time and query-time paths fit together, and which entrypoint uses which layer
- [docs/milestone-notes.md](docs/milestone-notes.md) — why each component is built the way it is
- [docs/measured-results.md](docs/measured-results.md) — eval numbers and how to read them
- [docs/known-limitations.md](docs/known-limitations.md) — known gaps and failure modes
- [docs/backlog.md](docs/backlog.md) — what's planned next and in what order
- [docs/mcp-server.md](docs/mcp-server.md) — MCP tool contract and client setup
- [docs/chunking-indexing-plan.md](docs/chunking-indexing-plan.md) — the phased chunking/indexing plan and where each phase stands
- [docs/milestone-19-plan.md](docs/milestone-19-plan.md) — the phased plan for agentic retrieval
