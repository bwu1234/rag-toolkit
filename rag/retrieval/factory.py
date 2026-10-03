"""Config-driven factories for `Reranker`, `QueryExpander` and `SparseIndex` implementations.

Mirrors `rag.embedding.factory.get_embedder` / `rag.vectorstore.factory.get_vector_store`:
one branch per provider string, so pipeline code never constructs a concrete
reranker directly.
"""

from __future__ import annotations

from pathlib import Path

from rag.config.settings import QueryExpansionConfig, RerankerConfig, SparseIndexConfig
from rag.llm.base import LLMClient
from rag.retrieval.cross_encoder_reranker import CrossEncoderReranker
from rag.retrieval.expansion import (
    HyDEQueryExpander,
    MultiQueryExpander,
    NoOpQueryExpander,
    QueryExpander,
)
from rag.retrieval.reranker import NoOpReranker, Reranker
from rag.retrieval.sparse import BM25Index, SparseIndex, bm25_index_path
from rag.retrieval.sqlite_fts5 import SqliteFts5Index


def get_reranker(config: RerankerConfig) -> Reranker:
    """Instantiate the `Reranker` selected by `config.provider`."""

    if config.provider == "none":
        return NoOpReranker()

    if config.provider == "cross_encoder":
        return CrossEncoderReranker(
            model=config.model,
            aggregate=config.aggregate,
            query_prefix=config.query_prefix,
            document_prefix=config.document_prefix,
            include_header=config.include_header,
            max_length=config.max_length,
        )

    raise ValueError(
        f"Unknown reranker provider: {config.provider!r}. "
        "Add an adapter and register it here to support a new provider."
    )


def get_query_expander(config: QueryExpansionConfig, llm_client: LLMClient | None) -> QueryExpander:
    """Instantiate the `QueryExpander` selected by `config.provider`.

    `llm_client` may be `None` only when the provider is `none` -- every other
    strategy generates text. Passing `None` with a generating provider is a
    wiring bug, so it raises here rather than failing later mid-query.
    """

    if config.provider == "none":
        return NoOpQueryExpander()

    if llm_client is None:
        raise ValueError(
            f"Query expansion provider {config.provider!r} needs an LLMClient, but none was supplied. "
            "Build the retriever with `build_retriever(config)`, which wires one in."
        )

    if config.provider == "hyde":
        return HyDEQueryExpander(
            llm_client,
            num_documents=config.num_documents,
            include_original=config.include_original,
        )

    if config.provider == "multi_query":
        return MultiQueryExpander(llm_client, num_queries=config.num_queries)

    raise ValueError(
        f"Unknown query expansion provider: {config.provider!r}. "
        "Add an expander and register it here to support a new provider."
    )


def sparse_index_path(config: SparseIndexConfig, index_dir: Path, slug: str | None = None) -> Path:
    """Where the `config.provider` sparse index for corpus selection `slug` lives.

    Each provider has its own file, so switching providers never reads an
    index written in another's format.
    """

    if config.provider == "bm25":
        return bm25_index_path(index_dir, slug)
    if config.provider == "sqlite_fts5":
        return index_dir / ("fts5_index.sqlite3" if slug is None else f"fts5_index__{slug}.sqlite3")
    raise ValueError(f"Unknown sparse index provider: {config.provider!r}")


def get_sparse_index(config: SparseIndexConfig, index_dir: Path, slug: str | None = None) -> SparseIndex:
    """Open (creating if needed) the `SparseIndex` selected by `config.provider`."""

    path = sparse_index_path(config, index_dir, slug)
    if config.provider == "bm25":
        return BM25Index(path)
    if config.provider == "sqlite_fts5":
        return SqliteFts5Index(path)
    raise ValueError(
        f"Unknown sparse index provider: {config.provider!r}. "
        "Add an adapter and register it here to support a new provider."
    )
