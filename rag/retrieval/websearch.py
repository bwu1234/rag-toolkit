"""SearxNG-backed web search, fused into `Retriever` as a third ranked-list
source alongside dense and BM25 search.

Follows the reranking algorithm Perplexica/Vane publish for their "speed"/
"balanced" modes (see `docs/architecture` in that project): fetch raw results
from a metasearch backend, embed the query and each result snippet, score by
cosine similarity, drop anything below a floor, then drop near-duplicate
snippets by comparing result embeddings to each other. That's the entire
algorithm -- no learned reranker, no LLM call, just cosine similarity applied
twice.
"""

from __future__ import annotations

import logging
from pathlib import Path

import httpx

from rag.config.settings import WebSearchConfig
from rag.embedding.base import EmbeddingModel
from rag.retrieval.similarity import cosine_similarity
from rag.vectorstore.base import ScoredChunk

logger = logging.getLogger(__name__)


class SearxNGWebSearch:
    """Fetches results from a SearxNG instance and reranks them by embedding
    cosine similarity against the query.

    Unlike `VectorStore`, there is nothing pre-indexed here -- every call
    embeds the query plus every fetched result snippet fresh, so this is
    strictly more expensive per search than dense retrieval and should stay
    opt-in (`retrieval.web_search.enabled`).
    """

    def __init__(self, embedder: EmbeddingModel, config: WebSearchConfig) -> None:
        self._embedder = embedder
        self._config = config
        # SearxNG is expected to be a local/self-hosted instance (default
        # `http://localhost:8080`), reached directly -- same loopback
        # rationale as `OllamaEmbedder`'s `trust_env=False`.
        self._client = httpx.Client(timeout=config.timeout_s, trust_env=False)

    def search(self, query: str) -> list[ScoredChunk]:
        """Return up to `top_k` web results scored by similarity to `query`, best first.

        Returns an empty list on a request failure or zero results -- a dead
        or misconfigured SearxNG instance should degrade the pipeline to
        "no web results" (dense/BM25 sources still run), not fail the whole
        search.
        """

        try:
            response = self._client.get(
                f"{self._config.searxng_url.rstrip('/')}/search",
                params={"q": query, "format": "json"},
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("SearxNG request failed for query %r: %s", query, exc)
            return []

        raw_results = response.json().get("results", [])[: self._config.top_k]
        if not raw_results:
            return []

        contents = [r.get("content") or r.get("title", "") for r in raw_results]
        query_vector = self._embedder.embed_query(query)
        result_vectors = self._embedder.embed_documents(contents)

        scored = [
            ScoredChunk(
                chunk_id=r["url"],
                text=content,
                document_id=r["url"],
                source=Path(r["url"]),
                doc_type="web",
                score=cosine_similarity(query_vector, vector),
                metadata={"title": r.get("title", ""), "url": r["url"]},
            )
            for r, content, vector in zip(raw_results, contents, result_vectors)
        ]
        scored = [c for c in scored if c.score >= self._config.min_similarity]
        scored.sort(key=lambda c: c.score, reverse=True)

        return _dedup_by_similarity(scored, result_vectors, raw_results, self._config.dedup_threshold)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "SearxNGWebSearch":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def _dedup_by_similarity(
    scored: list[ScoredChunk],
    result_vectors: list[list[float]],
    raw_results: list[dict],
    threshold: float,
) -> list[ScoredChunk]:
    """Drop a chunk if its content embedding is near-identical to one already kept.

    Metasearch aggregation (SearxNG queries several engines at once) often
    surfaces the same underlying snippet from multiple sources; URL-based
    dedup alone misses that, so this compares content embeddings instead --
    the same second pass Vane/Perplexica run in `baseSearch.ts`.
    """

    url_to_vector = {r["url"]: v for r, v in zip(raw_results, result_vectors)}

    kept: list[ScoredChunk] = []
    kept_vectors: list[list[float]] = []
    for chunk in scored:
        vector = url_to_vector[chunk.metadata["url"]]
        if any(cosine_similarity(vector, kv) > threshold for kv in kept_vectors):
            continue
        kept.append(chunk)
        kept_vectors.append(vector)
    return kept
