"""End-to-end retrieval pipeline: query -> vector search -> rerank.

`Retriever` is the seam between "raw similarity search" and "what the chat API
/ eval pipeline actually consumes" -- it owns the two-stage retrieve-then-
rerank shape (`retrieval.top_k` candidates pulled from the vector store,
narrowed to `retrieval.rerank_top_k` after rescoring) so that shape lives in
exactly one place rather than being re-implemented by every caller.
"""

from __future__ import annotations

import logging

from rag.embedding.base import EmbeddingModel
from rag.retrieval.reranker import Reranker
from rag.vectorstore.base import ScoredChunk, VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    """Embeds a query, searches the vector store, and reranks the results.

    Each stage is an injected interface (`EmbeddingModel`, `VectorStore`,
    `Reranker`) -- `Retriever` contains no provider-specific logic of its own,
    only the orchestration and the two `top_k` values that shape it.
    """

    def __init__(
        self,
        embedder: EmbeddingModel,
        vector_store: VectorStore,
        reranker: Reranker,
        *,
        top_k: int,
        rerank_top_k: int,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._reranker = reranker
        self.top_k = top_k
        self.rerank_top_k = rerank_top_k

    def retrieve(self, query: str) -> list[ScoredChunk]:
        """Return the `rerank_top_k` chunks most relevant to `query`, best first.

        Stage 1 (`top_k` candidates from the vector store) is intentionally
        wider than stage 2's output -- it gives the reranker enough material
        to recover relevant chunks that pure vector similarity ranked lower,
        which is the entire reason to rerank at all.
        """

        if not query.strip():
            return []

        query_vector = self._embedder.embed_query(query)
        candidates = self._vector_store.query(query_vector, top_k=self.top_k)
        logger.info("Retrieved %d candidate(s) for query %r", len(candidates), query)

        results = self._reranker.rerank(query, candidates, top_k=self.rerank_top_k)
        logger.info("Reranked down to %d result(s)", len(results))
        return results
