# RAG Toolkit

A local-first Retrieval-Augmented Generation (RAG) system in Python. Ollama
serves both embeddings and chat, and Chroma is the vector store.

The pipeline runs ingest → clean → chunk → embed → index at index time, and
hybrid retrieval (dense vector search + BM25, fused with reciprocal rank
fusion) → cross-encoder rerank → cited generation at query time. Each stage
sits behind an interface (`EmbeddingModel`, `VectorStore`, `Reranker`,
`LLMClient`, `QueryExpander`, `Chunker`) and is selected in
[rag/config/config.yaml](rag/config/config.yaml), so swapping an implementation
is a config change rather than a code change.

The same pipeline is reachable from a CLI, a FastAPI service, a Streamlit UI,
and an MCP server. Every chat turn is logged with its retrieved and cited
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
contributing anything.

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

Passing `--corpus` more than once **pools** the corpora into a single index.
Each selection gets its own collection (`rag_corpus__edgar` vs.
`rag_corpus__baseline+edgar`), so isolated and pooled indexes coexist and can
be compared.

To add your own documents, either drop them into
`data/corpora/baseline/documents/` or register a new corpus in the config, then
re-run `python -m rag.cli index`. Add `--reset` to rebuild from scratch.

## Supported document formats

Ingestion dispatches on file extension. Files with any other extension are skipped (at `DEBUG` log level — check the `N file(s) skipped` count in the ingest summary if a document seems missing):

| Extension | Loader | Granularity |
|---|---|---|
| `.pdf` | `PdfLoader` (pypdf) | one document **per page**, so citations can reference a page number |
| `.md`, `.markdown` | `MarkdownLoader` | one document per file |
| `.txt` | `TextLoader` | one document per file |

Formats **not** currently supported include `.docx`, `.pptx`, `.xlsx`, `.csv`, `.html`, `.json`, and `.epub`.

Some limits worth knowing before pointing the pipeline at a corpus:

- **No OCR.** PDF text comes from `pypdf`'s text layer. A scanned or image-only PDF extracts as empty text and contributes zero chunks — it loads without error but adds nothing to the index.
- **Text only.** Images and layout structure are dropped, and PDF tables are linearized into running text rather than preserved as tables.
- **Chunking is format-blind** — fixed-size character windows over the extracted text (see `chunking` in [rag/config/config.yaml](rag/config/config.yaml)), which suits prose better than highly structured content.
- **Ingestion is an offline CLI step.** There is no upload endpoint or widget; the API and UI only query an index that `python -m rag.cli index` already built.

### Adding a format

The loader interface is the extension point — everything downstream operates on `Document.text` uniformly, so no other pipeline code changes:

1. Subclass `Loader` ([rag/ingestion/models.py](rag/ingestion/models.py)), set `extensions`, and implement `load()`, building ids with `make_document_id`.
2. Register an instance in the `_LOADERS` table in [rag/ingestion/loaders.py](rag/ingestion/loaders.py).
3. Add the parsing dependency to [pyproject.toml](pyproject.toml).

## Interfaces

### API

```bash
uvicorn rag.api.main:app --reload
```

| Endpoint | Purpose |
|---|---|
| `POST /chat` | `{"query": "...", "history": [...]}` → answer with citations and a `turn_id`. The API is stateless; clients replay their own history. |
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
(ranked passages, no generated answer) and `rag_list_corpora`.

```bash
python -m rag.mcp              # stdio; works on the core install
pip install -e '.[mcp]'        # adds streamable HTTP at /mcp on the API
```

It serves MCP revision 2026-07-28 only. Client configuration, the tool
contract, and why indexing isn't exposed are in
[docs/mcp-server.md](docs/mcp-server.md).

### Logged turns

The API, UI, and `cli chat` append every turn to `data/logs/turns.jsonl`
(gitignored); the eval runners never do.

```bash
python -m rag.cli turns --show 20 --feedback down
```

## Configuration

Everything — component selection, model names, chunk size, retrieval depth —
lives in [rag/config/config.yaml](rag/config/config.yaml) and is validated by
pydantic models in [rag/config/settings.py](rag/config/settings.py). The YAML
is commented with why each default is what it is.

Several features ship **disabled** because they measured as no better than
noise on the EDGAR corpus: contextual chunking, corrective RAG (CRAG), query
expansion (HyDE / multi-query), and the `retrieval.min_score` relevance floor.
Live web search via SearxNG is also off by default, since enabled queries
leave the machine. Re-measure before turning any of these on — see
[docs/measured-results.md](docs/measured-results.md).

## Evaluation

```bash
python -m rag.eval.retrieval_eval -v                  # hit rate, MRR, NDCG on the baseline set
python -m rag.eval.retrieval_eval --eval-set data/eval/edgar_eval_set.json --corpus edgar
python -m rag.eval.answer_eval                        # LLM-as-judge answer quality
python -m rag.eval.multihop_eval --corpus edgar       # questions that need several documents
```

The answer judge is `eval.judge` in the config. When that is unset, the
generator grades itself and the runner warns, because swapping the generator
then swaps the judge too. `--judge-model` overrides it per run.

## Project layout

```text
rag/
  api/            FastAPI app and routes
  chunking/       chunkers and the contextual-chunking cache
  config/         config.yaml and its pydantic settings
  embedding/      embedding adapters
  eval/           retrieval, answer, and multi-hop evaluation
  generation/     prompts, LLM clients, chat service, CRAG
  ingestion/      document loaders and cleaners
  mcp/            MCP server and tools
  observability/  turn records, LLM usage metering, feedback sink
  retrieval/      dense/sparse retrieval, RRF, rerankers, query expansion
  ui/             Streamlit UI
  vectorstore/    vector store adapters
  cli.py          ingest / chunk / index / retrieve / chat / turns
scripts/          demo, EDGAR fetcher, eval-set builders, eval matrices
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
