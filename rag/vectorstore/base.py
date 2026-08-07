"""Interface for persisting and searching chunk embeddings.

A `VectorStore` is responsible for two things: storing `(chunk, embedding)`
pairs durably, and returning the most similar chunks to a query embedding.
Pipeline code (indexing, retrieval) depends only on this interface and the
`ScoredChunk` result type -- never on a concrete provider's client or query
syntax -- so swapping Chroma for another store is a config change plus one
adapter.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rag.chunking.models import Chunk


@dataclass(frozen=True)
class ScoredChunk:
    """A chunk returned from similarity search, together with its score.

    Carries the same provenance fields as `Chunk` (flattened, rather than
    nesting a `Chunk`) so retrieval and the API layer can build citations
    directly from a search result without a second lookup.

    Attributes:
        score: Similarity score in `[0, 1]`, higher is more similar.
            Providers report distance in different units (cosine distance,
            L2, inner product); adapters are responsible for converting to
            this common similarity convention so downstream code never has to
            know which metric the underlying store used.
        context: The chunk's generated document context, when contextual
            chunking produced one (see `rag.chunking.contextualizer`). Carried
            through retrieval so the answering model can be shown where a
            passage sits in its document -- `text` remains the verbatim span.
    """

    chunk_id: str
    text: str
    document_id: str
    source: Path
    doc_type: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    context: str | None = None

    @property
    def contextual_text(self) -> str:
        """Context and chunk text as one string -- mirrors `Chunk.contextual_text`."""

        return f"{self.context}\n\n{self.text}" if self.context else self.text


class VectorStore(ABC):
    """Interface for storing chunk embeddings and searching them by similarity.

    To add a new backend: subclass `VectorStore`, implement these methods, and
    register it in `rag.vectorstore.factory.get_vector_store` based on
    `VectorStoreConfig.provider`.
    """

    @abstractmethod
    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Insert or overwrite `(chunk, embedding)` pairs, keyed by `chunk.id`.

        Upsert (rather than insert-only) makes re-running the indexing
        pipeline after a corpus or chunking-config change idempotent: chunks
        with ids that already exist are replaced in place rather than
        duplicated.

        Raises:
            ValueError: if `len(chunks) != len(embeddings)`.
        """
        raise NotImplementedError

    def get_metadatas(self, ids: list[str]) -> dict[str, dict[str, Any]]:
        """Return a mapping of chunk_id -> stored metadata for `ids`.

        Missing ids should be absent from the returned mapping. This helper
        allows the indexing pipeline to detect unchanged chunks (e.g. via a
        stored content hash in metadata) and skip re-embedding them.

        Default implementation returns an empty mapping for backends that do
        not support metadata lookups; adapters that can return stored metadata
        should override this method for incremental indexing support.
        """
        # By default, assume no metadata exists for any id.
        return {}

    @abstractmethod
    def query(self, embedding: list[float], top_k: int) -> list[ScoredChunk]:
        """Return the `top_k` chunks most similar to `embedding`, best first."""
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        """Number of chunks currently stored -- used for index sanity checks and CLI summaries."""
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Delete all stored chunks, leaving an empty collection.

        Used by the indexing CLI to support a clean rebuild (e.g. after a
        chunking-strategy change that invalidates old chunk ids) without
        leaving orphaned vectors behind.
        """
        raise NotImplementedError
