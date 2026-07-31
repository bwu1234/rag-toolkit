"""Interface for turning text into vectors.

Embeddings are the bridge between text (chunks, queries) and the vector
store's similarity search. Keeping this behind an ABC means the rest of the
pipeline (indexing, retrieval) calls `embed_documents` / `embed_query` and
never touches a concrete provider's SDK or HTTP API directly -- swapping
`qwen3-embedding:0.6b` via Ollama for, say, a `sentence-transformers` model is
a config change plus one adapter, not a pipeline rewrite.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingModel(ABC):
    """Interface for embedding text into fixed-size vectors.

    Two methods rather than one because some providers (and some models)
    distinguish between how a *document* should be embedded for indexing and
    how a *query* should be embedded for search -- e.g. asymmetric models that
    prepend different instruction prefixes to each. Implementations that don't
    need the distinction can simply have `embed_query` delegate to
    `embed_documents`.
    """

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of chunk texts for indexing.

        Returns one vector per input text, in the same order. Implementations
        should batch internally where the provider supports it -- callers pass
        whatever list size makes sense for their pipeline (e.g. one chunk
        batch at a time) without worrying about provider-side limits.
        """
        raise NotImplementedError

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string for similarity search."""
        raise NotImplementedError

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """The dimensionality of vectors this model produces.

        Exposed so the vector store can validate that an existing collection's
        dimensionality matches the configured embedder before silently
        producing nonsense similarity scores.
        """
        raise NotImplementedError
