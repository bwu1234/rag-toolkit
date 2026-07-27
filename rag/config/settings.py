"""Typed, config-driven settings for the RAG system.

All component choices (embedding model, vector store, reranker, LLM, chunking
parameters, etc.) live here as plain pydantic models loaded from a YAML file.
Swapping an implementation should mean: change a `provider` string in
config.yaml (and possibly add a small adapter class) — not editing pipeline
code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

# Repo root = two levels up from this file (rag/config/settings.py -> repo/)
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


class PathsConfig(BaseModel):
    """Filesystem locations used by the pipeline. Relative paths resolve against REPO_ROOT."""

    corpus_dir: Path = Path("data/corpus")
    index_dir: Path = Path("data/index")

    def resolved(self) -> "PathsConfig":
        return PathsConfig(
            corpus_dir=(REPO_ROOT / self.corpus_dir).resolve(),
            index_dir=(REPO_ROOT / self.index_dir).resolve(),
        )


class EmbeddingConfig(BaseModel):
    """Which embedding model to use and how to reach it.

    Default: Ollama-served `qwen3-embedding:0.6b`. Because both embeddings and
    chat go through Ollama, the only network dependency for a fully local setup
    is the Ollama daemon — no extra ML libraries required for the default path.
    """

    provider: Literal["ollama", "sentence_transformers"] = "ollama"
    model: str = "qwen3-embedding:0.6b"
    base_url: str = "http://localhost:11434"
    # Embedding dimensionality is provider/model-specific; recorded here so the
    # vector store can validate it rather than discovering mismatches at query time.
    dimensions: int | None = None


class LLMConfig(BaseModel):
    """Which chat/generation model to use and how to reach it.

    Default: Ollama-served `qwen3.5:4b`.
    """

    provider: Literal["ollama", "anthropic", "openai"] = "ollama"
    model: str = "qwen3.5:4b"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.2
    max_tokens: int = 1024


class ChunkingConfig(BaseModel):
    """Parameters for splitting documents into retrievable chunks.

    `fixed` = simple character-based windows with overlap. Deliberately the
    simplest strategy that could work; structure-aware/semantic chunking is a
    later milestone once the end-to-end pipeline is proven out.
    """

    strategy: Literal["fixed"] = "fixed"
    chunk_size: int = Field(default=1000, gt=0, description="Target characters per chunk")
    chunk_overlap: int = Field(default=150, ge=0, description="Characters of overlap between consecutive chunks")


class VectorStoreConfig(BaseModel):
    """Vector store selection and connection details.

    Default: Chroma in persistent local mode (single directory on disk, no
    server process). Bundles vector search with metadata storage, which keeps
    the dependency count low for a local-first setup.
    """

    provider: Literal["chroma"] = "chroma"
    collection_name: str = "rag_corpus"


class RerankerConfig(BaseModel):
    """Reranker selection. `none` disables reranking (pure vector retrieval)."""

    provider: Literal["none", "cross_encoder"] = "none"
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class RetrievalConfig(BaseModel):
    """Controls how many candidates are pulled from the vector store and kept after reranking."""

    top_k: int = Field(default=20, gt=0, description="Candidates retrieved from the vector store")
    rerank_top_k: int = Field(default=5, gt=0, description="Final number of chunks passed to the LLM after reranking")


class RagConfig(BaseModel):
    """Top-level config object — the single source of truth for component selection."""

    paths: PathsConfig = PathsConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    llm: LLMConfig = LLMConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    vector_store: VectorStoreConfig = VectorStoreConfig()
    reranker: RerankerConfig = RerankerConfig()
    retrieval: RetrievalConfig = RetrievalConfig()


def load_config(path: str | Path | None = None) -> RagConfig:
    """Load and validate config from a YAML file, falling back to defaults for missing keys.

    Passing no path loads `rag/config/config.yaml`. A missing file simply
    yields the default `RagConfig()` so the system runs out of the box.
    """

    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        return RagConfig()

    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    return RagConfig.model_validate(raw)
