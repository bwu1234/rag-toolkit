"""Config-driven factory for `Reranker` implementations.

Mirrors `rag.embedding.factory.get_embedder` / `rag.vectorstore.factory.get_vector_store`:
one branch per provider string, so pipeline code never constructs a concrete
reranker directly.
"""

from __future__ import annotations

from rag.config.settings import QueryExpansionConfig, RerankerConfig
from rag.generation.llm import LLMClient
from rag.retrieval.cross_encoder_reranker import CrossEncoderReranker
from rag.retrieval.expansion import (
    HyDEQueryExpander,
    MultiQueryExpander,
    NoOpQueryExpander,
    QueryExpander,
)
from rag.retrieval.reranker import NoOpReranker, Reranker


def get_reranker(config: RerankerConfig) -> Reranker:
    """Instantiate the `Reranker` selected by `config.provider`."""

    if config.provider == "none":
        return NoOpReranker()

    if config.provider == "cross_encoder":
        return CrossEncoderReranker(model=config.model, aggregate=config.aggregate)

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
