"""Config-driven factory for `VectorStore` implementations.

Mirrors `rag.embedding.factory.get_embedder`: one branch per provider string,
so pipeline code never constructs a concrete store directly.
"""

from __future__ import annotations

from pathlib import Path

from rag.config.settings import VectorStoreConfig
from rag.vectorstore.base import VectorStore
from rag.vectorstore.chroma_store import ChromaVectorStore


def get_vector_store(config: VectorStoreConfig, index_dir: Path) -> VectorStore:
    """Instantiate the `VectorStore` selected by `config.provider`.

    `index_dir` is passed separately (rather than living on `VectorStoreConfig`)
    because it comes from `PathsConfig.resolved()` -- keeping path resolution
    in one place avoids every config section needing its own `REPO_ROOT` logic.
    """

    if config.provider == "chroma":
        return ChromaVectorStore(persist_dir=index_dir, collection_name=config.collection_name)

    raise ValueError(
        f"Unknown vector store provider: {config.provider!r}. "
        "Add an adapter and register it here to support a new provider."
    )
