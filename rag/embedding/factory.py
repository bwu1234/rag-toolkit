"""Config-driven factory for `EmbeddingModel` implementations.

Single point of truth for "provider string in config" -> "concrete adapter",
mirroring `rag.chunking.chunkers.get_chunker`.
"""

from __future__ import annotations

from rag.config.settings import EmbeddingConfig
from rag.embedding.base import EmbeddingModel
from rag.embedding.ollama_embedder import OllamaEmbedder
from rag.embedding.sentence_transformers_embedder import SentenceTransformersEmbedder


def get_embedder(config: EmbeddingConfig) -> EmbeddingModel:
    """Instantiate the `EmbeddingModel` selected by `config.provider`."""

    if config.provider == "ollama":
        return OllamaEmbedder(
            model=config.model,
            base_url=config.base_url,
            dimensions=config.dimensions,
            query_instruction=config.query_instruction,
        )

    if config.provider == "sentence_transformers":
        return SentenceTransformersEmbedder(
            model=config.model,
            dimensions=config.dimensions,
            revision=config.revision,
            query_instruction=config.query_instruction,
        )

    raise ValueError(
        f"Unknown embedding provider: {config.provider!r}. "
        "Add an adapter and register it here to support a new provider."
    )
