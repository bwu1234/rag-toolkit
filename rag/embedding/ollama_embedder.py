"""Ollama-backed `EmbeddingModel` adapter.

Talks to a local Ollama daemon's `/api/embed` endpoint over HTTP. This is the
only network dependency for the default fully-local setup -- no embedding
library or model weights live in this process.
"""

from __future__ import annotations

import logging

import httpx

from rag.embedding.base import EmbeddingModel

logger = logging.getLogger(__name__)

# Ollama batches embedding requests server-side, but very large `input` lists
# can still produce slow, memory-heavy single requests. Chunking client-side
# keeps each HTTP call's size predictable regardless of corpus size.
_DEFAULT_BATCH_SIZE = 32


class OllamaEmbedder(EmbeddingModel):
    """Embeds text via a local Ollama daemon's `/api/embed` endpoint.

    Dimensionality is either taken from config (if the user knows it) or
    discovered lazily from the first response and cached -- this avoids
    hardcoding a model-specific constant that would silently go stale if the
    configured model changes.
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        dimensions: int | None = None,
        *,
        query_instruction: str | None = None,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        timeout: float = 60.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.query_instruction = query_instruction
        self.batch_size = batch_size
        # `trust_env=False`: a local Ollama daemon is reached over loopback,
        # so picking up the environment's HTTP(S)_PROXY/ALL_PROXY settings
        # would be both pointless and (in proxied environments) a source of
        # spurious connection failures or missing-dependency errors.
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout, trust_env=False)
        self._dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            vectors.extend(self._embed_batch(batch))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        # No space after "Query:" -- that is the model card's exact format, and
        # the model was trained on it.
        if self.query_instruction is not None:
            text = f"Instruct: {self.query_instruction}\nQuery:{text}"
        return self._embed_batch([text])[0]

    @property
    def dimensions(self) -> int:
        if self._dimensions is None:
            # Discover lazily rather than at construction time -- building an
            # `OllamaEmbedder` shouldn't require a running daemon (e.g. when
            # just loading config or wiring up the factory in tests).
            self._dimensions = len(self.embed_query("dimension probe"))
            logger.info("Discovered embedding dimensionality: %d", self._dimensions)
        return self._dimensions

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        try:
            response = self._client.post("/api/embed", json={"model": self.model, "input": texts})
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"Failed to get embeddings from Ollama at {self.base_url} "
                f"(model={self.model!r}): {exc}"
            ) from exc

        payload = response.json()
        embeddings = payload.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise RuntimeError(
                f"Unexpected response shape from Ollama /api/embed: "
                f"expected {len(texts)} embedding(s), got {payload!r}"
            )
        return embeddings

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "OllamaEmbedder":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
