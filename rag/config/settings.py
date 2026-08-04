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

    Default: Ollama-served `qwen3.5:9b-mlx`.
    """

    provider: Literal["ollama", "anthropic", "openai"] = "ollama"
    model: str = "qwen3.5:9b-mlx"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.2
    max_tokens: int = 1024
    # qwen3.5 reasoning models spend `max_tokens` on a hidden "thinking" trace
    # before the real answer; on a long RAG prompt that can exhaust the
    # budget and leave `content` empty. Off by default for reliable answers.
    think: bool = False


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
    # How to fold a candidate's per-query scores into one, when
    # `retrieval.expansion` produced several question phrasings. `max` keeps a
    # chunk that any phrasing found relevant -- the setting that lets multi-query
    # expansion actually change the final ranking rather than only stage 1.
    # `mean` requires broader agreement and suppresses chunks only one rewrite
    # liked. No effect with `retrieval.expansion.provider: none`.
    aggregate: Literal["max", "mean"] = "max"


class QueryExpansionConfig(BaseModel):
    """Pre-retrieval query transformation. `none` keeps retrieval LLM-free.

    - ``hyde`` — generate a hypothetical answer passage and embed *that*,
      closing the shape mismatch between a short question and the long
      declarative passage that answers it.
    - ``multi_query`` — generate several rephrasings, retrieve for each, and
      fuse the ranked lists, so the corpus's choice of vocabulary matters less.

    Both add an LLM call to the retrieval path (and `hyde` with
    ``num_documents > 1`` adds one per document), plus one embedding round trip
    per generated query. That's a real latency cost on every search, which is
    why the default is `none`: turn one on and measure with
    `python -m rag.eval.retrieval_eval` rather than assuming it helps.
    """

    provider: Literal["none", "hyde", "multi_query"] = "none"
    num_queries: int = Field(
        default=3, ge=1, le=10, description="multi_query: rephrasings to generate (excluding the original)"
    )
    num_documents: int = Field(
        default=1, ge=1, le=5, description="hyde: hypothetical passages to generate and embed"
    )
    include_original: bool = Field(
        default=True,
        description="hyde: also embed the user's real question, so a bad generation can't sink the search",
    )


class WebSearchConfig(BaseModel):
    """Optional live web search source, fused into retrieval alongside dense/BM25.

    Disabled by default -- enabling it means every query leaves the local
    machine (a SearxNG instance, self-hosted or not) and costs one extra
    embedding round trip per result. `min_similarity`/`dedup_threshold`
    mirror the fixed thresholds Perplexica/Vane use for the same job (scoring
    raw search snippets by embedding cosine similarity against the query, then
    dropping near-duplicate snippets) -- see `rag.retrieval.websearch`.
    """

    enabled: bool = False
    searxng_url: str = "http://localhost:8080"
    top_k: int = Field(default=10, gt=0, description="Search results fetched from SearxNG per query")
    min_similarity: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Drop results below this query-similarity score"
    )
    dedup_threshold: float = Field(
        default=0.75, ge=0.0, le=1.0, description="Drop a result if another kept result exceeds this similarity to it"
    )
    timeout_s: float = Field(default=10.0, gt=0, description="HTTP timeout for the SearxNG request")


class RetrievalConfig(BaseModel):
    """Controls candidate retrieval, fusion, and post-rerank width.

    `mode`:
      - ``dense``  — vector similarity only (original behavior)
      - ``hybrid`` — dense + BM25 keyword search fused with Reciprocal Rank
        Fusion (RRF), then optionally reranked. Recovers exact keyword hits
        (acronyms, IDs, names) that pure embedding search often misses.
    """

    top_k: int = Field(default=20, gt=0, description="Candidates retrieved per retriever (dense and/or BM25)")
    rerank_top_k: int = Field(default=5, gt=0, description="Final number of chunks passed to the LLM after reranking")
    mode: Literal["dense", "hybrid"] = "dense"
    # RRF constant from Cormack et al.; 60 is the widely used default.
    rrf_k: int = Field(default=60, gt=0, description="RRF rank constant: score += 1 / (rrf_k + rank)")
    # Relevance floor applied to the *final* results, after reranking. Without
    # it a vector store always returns its nearest `top_k` neighbours no matter
    # how distant they are, so an off-corpus question still arrives at the LLM
    # with a full set of irrelevant passages to "ground" itself in. Dropping
    # everything below the floor is what lets `ChatService` distinguish "no
    # relevant context" from "empty index".
    #
    # The threshold is read against whatever the last stage's score means (see
    # the `Reranker` score convention): a cross-encoder's sigmoid-squashed
    # logit with `reranker.provider: cross_encoder`, raw cosine similarity with
    # `none`. Those scales differ, so retune this when switching providers.
    # 0.0 (the default) disables filtering.
    min_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Drop final results scoring below this")
    expansion: QueryExpansionConfig = QueryExpansionConfig()
    web_search: WebSearchConfig = WebSearchConfig()


class ChatConfig(BaseModel):
    """Turn-level behaviour of the chat pipeline (as opposed to retrieval tuning).

    `condense_history` controls conversational query rewriting: with it on, a
    follow-up like "what about part-time staff?" is rewritten into a
    standalone question using the preceding turns before it's embedded.
    Retrieval has no memory of its own -- embedding the raw follow-up searches
    the corpus for the literal words "what about part-time staff", which is
    almost never what the user meant.

    Costs one extra LLM round trip per turn that has history; set to `false`
    for a purely one-shot assistant (the CLI's `chat` command, the eval
    pipeline) where there's never any history to condense anyway.
    """

    condense_history: bool = True
    max_history_turns: int = Field(
        default=6,
        ge=0,
        description="Most recent turns fed to the condenser (keeps the rewrite prompt bounded)",
    )


class RagConfig(BaseModel):
    """Top-level config object — the single source of truth for component selection."""

    paths: PathsConfig = PathsConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    llm: LLMConfig = LLMConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    vector_store: VectorStoreConfig = VectorStoreConfig()
    reranker: RerankerConfig = RerankerConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    chat: ChatConfig = ChatConfig()


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
