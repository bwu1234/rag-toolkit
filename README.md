# RAG Toolkit

A local-first Retrieval-Augmented Generation (RAG) system built from scratch in Python.

This project implements the full RAG loop from document ingestion to answer generation: ingest → clean → chunk → embed → index → retrieve → rerank → generate. It is designed to be modular, testable, and easy to swap out piece by piece without rewriting the whole pipeline.

The project is intentionally built as both a practical system and a portfolio piece: it shows end-to-end engineering across data loading, vector search, prompt construction, API design, UI wiring, and evaluation.

## Why this project is worth showing

- End-to-end architecture: the repo covers the complete RAG stack, not just one isolated component.
- Production-minded structure: pluggable interfaces, typed config, and a clear separation of concerns.
- Local-first defaults: Ollama handles embeddings and generation; Chroma provides local vector storage.
- Strong engineering hygiene: a real test suite, CI workflow, and typed Python code.
- Portfolio-friendly story: it is easy to explain as “I built a working retrieval system from first principles and containerized the experience around it.”

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
- A corpus placed under [data/corpus](data/corpus)

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

4. Add your own documents under [data/corpus](data/corpus) and run the pipeline:

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

If you want to push this project further as a portfolio piece, good next steps are:

- add Docker support for one-command startup
- add a richer sample corpus and demo data
- improve observability and tracing around retrieval quality
- add deployment notes for cloud hosting or container deployment

This repository is a strong example of building a real AI system end to end: the core ideas are grounded, the structure is deliberate, and the implementation is more than a single notebook or toy script.