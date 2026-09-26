# RAG Toolkit

A local-first Retrieval-Augmented Generation (RAG) system in Python.

This project implements the full RAG loop from document ingestion to answer generation: ingest → clean → chunk → embed → index → retrieve → rerank → generate. It is designed to be modular, testable, and easy to swap out piece by piece without rewriting the whole pipeline.

## Design goals

- End-to-end: the repo covers the complete RAG stack, from ingestion through evaluation, not one isolated component.
- Swappable components: pluggable interfaces, typed config, and a clear separation of concerns.
- Local-first defaults: Ollama handles embeddings and generation; Chroma provides local vector storage.

## What it does

The system takes a corpus of documents, turns them into searchable chunks, stores them in a vector index, and answers questions grounded in those passages. The stack includes:

- document ingestion for PDFs and text/Markdown files
- text cleaning and chunking
- embedding and vector indexing
- retrieval with optional reranking
- a chat pipeline with citations
- a FastAPI web API
- a Streamlit UI
- evaluation scripts for retrieval and answer quality
- per-turn observability: every chat turn logged with its retrieved and cited passages, CRAG verdicts, per-stage latency and LLM token counts, plus thumbs up/down feedback (`python -m rag.cli turns`)

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
- **Ingestion is an offline CLI step.** There is no upload endpoint or widget; the API and UI only query an index that `python -m rag.cli index` already built. Add documents by placing files under `data/corpus` and re-running the indexer.

### Adding a format

The loader interface is the extension point — everything downstream operates on `Document.text` uniformly, so no other pipeline code changes:

1. Subclass `Loader` ([rag/ingestion/models.py](rag/ingestion/models.py)), set `extensions`, and implement `load()`, building ids with `make_document_id`.
2. Register an instance in the `_LOADERS` table in [rag/ingestion/loaders.py](rag/ingestion/loaders.py).
3. Add the parsing dependency to [pyproject.toml](pyproject.toml).

## Architecture at a glance

The code is organized around small interfaces so components can be swapped without rewriting the pipeline:

- [rag/config](rag/config) holds configuration and validated settings
- [rag/ingestion](rag/ingestion) handles document loading and cleaning
- [rag/chunking](rag/chunking) creates searchable chunks
- [rag/embedding](rag/embedding) and [rag/vectorstore](rag/vectorstore) manage embeddings and indexing
- [rag/retrieval](rag/retrieval) covers retrieval and reranking
- [rag/generation](rag/generation) handles prompt construction and LLM calls
- [rag/api](rag/api) exposes the chat service over FastAPI
- [rag/ui](rag/ui) provides a simple interactive interface
- [tests](tests) mirrors the package layout and exercises the core behavior

For how these connect at runtime — the index-time and query-time paths, and which entrypoint uses which layer — see [docs/architecture.md](docs/architecture.md).

## Tech stack

- Python 3.10+
- Pydantic + Pydantic Settings for config validation
- FastAPI + Uvicorn for the API layer
- Streamlit for the demo UI
- Chroma for local vector storage
- Ollama for local embeddings and LLM generation
- Pytest + Ruff + Mypy for quality checks

## Prerequisites

To run the full pipeline locally, you will need:

- Python 3.10+
- A virtual environment (recommended)
- Ollama running locally at http://localhost:11434
- A corpus placed under [data/corpora/baseline/documents](data/corpora/baseline/documents) (or any corpus registered in `rag/config/config.yaml`)

The project is designed to work locally by default, and the tests are structured to avoid depending on a running Ollama or Chroma server whenever possible.

## Quick start

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install the package and development extras:

```bash
pip install -e .[dev]
```

3. Run the test suite:

```bash
pytest -q
```

4. Add your own documents under [data/corpora/baseline/documents](data/corpora/baseline/documents) and run the pipeline:

```bash
python -m rag.cli ingest --show 3
python -m rag.cli chunk --show 3
python -m rag.cli index
python -m rag.cli retrieve "What does this project do?"
python -m rag.cli chat "What does this project do?"
```

## Demo script

A ready-to-run demo flow is available in [scripts/demo.sh](scripts/demo.sh). It walks through the core experience from ingestion to chat.

```bash
bash scripts/demo.sh
```

If you prefer to run the steps manually, use the following flow:

```bash
python -m rag.cli ingest --show 3
python -m rag.cli chunk --show 3
python -m rag.cli index
python -m rag.cli retrieve "Summarize the key ideas in this repository"
python -m rag.cli chat "Summarize the key ideas in this repository"
```

## Run the API

```bash
uvicorn rag.api.main:app --reload
```

Then open:

- http://localhost:8000/health for a health check
- http://localhost:8000/docs for the interactive API docs

## Run the UI

```bash
streamlit run rag/ui/app.py
```

## Configuration

Most runtime behavior is controlled by [rag/config/config.yaml](rag/config/config.yaml) and validated by [rag/config/settings.py](rag/config/settings.py). Key settings include:

- embedding model selection
- LLM model selection
- chunk size and overlap
- vector store settings
- retrieval depth and reranking behavior

## Project structure

```text
rag/
  api/          FastAPI app and routes
  chunking/     chunking strategies
  config/       configuration and settings
  embedding/    embedding adapters
  eval/         retrieval and answer evaluation
  generation/   generation and prompt logic
  observability/ turn records, LLM usage metering, feedback sink
  ingestion/    document loaders and cleaners
  retrieval/    retrievers and rerankers
  ui/           Streamlit UI
  vectorstore/  vector store adapters
tests/          unit/integration tests
scripts/        demo helpers
```

## Validation and quality

This repository includes:

- a CI workflow in [.github/workflows/ci.yml](.github/workflows/ci.yml)
- linting with Ruff
- type checking with Mypy
- unit and integration-style tests under [tests](tests)

## Roadmap ideas

Possible next steps:

- add Docker support for one-command startup
- add a richer sample corpus and demo data
- export turn records to OpenTelemetry (the `TurnSink` interface is the seam)
- add deployment notes for cloud hosting or container deployment
