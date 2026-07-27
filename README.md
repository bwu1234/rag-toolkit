# RAG_Project

Retrieval-Augmented Generation (RAG) system built from scratch.

This repository implements a local-first RAG pipeline (ingest → chunk → embed → index → retrieve → generate) with pluggable components and a small FastAPI chat API and Streamlit UI.

Status
- Milestones 1–8 implemented (scaffolding, ingestion, chunking, embedding, indexing, retrieval, chat API, evaluation, Streamlit UI).

Why this repo
- Clean interfaces and strong typing (pydantic v2) make it easy to swap components.
- Local-first defaults: Ollama for embeddings/LLM and Chroma for vector storage.
- Extensive unit tests that mirror the package layout for confident refactors.

Quick links
- Config: [rag/config/config.yaml](/Users/bwu19/RAG_Project/rag/config/config.yaml)
- Typed settings: [rag/config/settings.py](/Users/bwu19/RAG_Project/rag/config/settings.py)
- CLI entrypoint: [rag/cli.py](/Users/bwu19/RAG_Project/rag/cli.py)
- FastAPI app: [rag/api/main.py](/Users/bwu19/RAG_Project/rag/api/main.py)
- Packaging: [pyproject.toml](/Users/bwu19/RAG_Project/pyproject.toml)
- Tests: [tests/](/Users/bwu19/RAG_Project/tests)
- Contribution/license: (none yet) — see "Before publishing" checklist below.

Prerequisites
- Python 3.10+
- Recommended: create a virtual environment (venv/conda)
- Local services for full end-to-end: Ollama daemon (default: http://localhost:11434) and Chroma (local persistent mode). The system is designed to run in fully-local mode, but tests mock external services where possible.

Install

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install the package (editable) and dev extras (recommended):

```bash
pip install -e .[dev]
```

Run tests

```bash
pytest -q
```

Note: the test suite is designed to be hermetic. Integration tests that require a running Ollama or Chroma are either mocked or explicitly documented.

Common commands (CLI)

- Ingest a small preview of the corpus:

```bash
python -m rag.cli ingest --show 3
```

- Chunk only (preview):

```bash
python -m rag.cli chunk --show 3
```

- Build the index (embeds all chunks and upserts into Chroma):

```bash
python -m rag.cli index
# use --reset to rebuild from scratch
```

- Retrieve + rerank (quick query):

```bash
python -m rag.cli retrieve "your question"
```

- End-to-end chat (retrieve → generate):

```bash
python -m rag.cli chat "your question"
```

Run the API

```bash
uvicorn rag.api.main:app --reload
# health: GET /health
# POST /chat with JSON {"query": "..."}
```

Run the UI (Streamlit)

```bash
streamlit run rag/ui/app.py
```

Configuration

All runtime configuration lives in [rag/config/config.yaml](/Users/bwu19/RAG_Project/rag/config/config.yaml) and is validated by [rag/config/settings.py](/Users/bwu19/RAG_Project/rag/config/settings.py).
Important knobs:
- embedding.model and llm.model — default to Ollama-served models
- chunking.chunk_size / chunk_overlap
- vector_store.collection_name
- retrieval.top_k / rerank_top_k

Where to put your corpus

Place documents under `data/corpus/` (gitignored by default). PDF loader emits one Document per page; markdown/text loaders emit one Document per file.