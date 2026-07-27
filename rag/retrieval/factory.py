"""Config-driven factory for `Reranker` implementations.

Mirrors `rag.embedding.factory.get_embedder` / `rag.vectorstore.factory.get_vector_store`:
one branch per provider string, so pipeline code never constructs a concrete
reranker directly.
"""

from __future__ import annotations

from rag.config.settings import RerankerConfig
from rag.retrieval.cross_encoder_reranker import CrossEncoderReranker
from rag.retrieval.reranker import NoOpReranker, Reranker


def get_reranker(config: RerankerConfig) -> Reranker:
    """Instantiate the `Reranker` selected by `config.provider`."""

    if config.provider == "none":
        return NoOpReranker()

    if config.provider == "cross_encoder":
        return CrossEncoderReranker(model=config.model)

    raise ValueError(
        f"Unknown reranker provider: {config.provider!r}. "
        "Add an adapter and register it here to support a new provider."
    )
