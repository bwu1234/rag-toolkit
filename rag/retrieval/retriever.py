"""End-to-end retrieval pipeline: query -> (hybrid) search -> rerank.

`Retriever` is the seam between "raw similarity search" and "what the chat API
/ eval pipeline actually consumes" -- it owns the retrieve-then-rerank shape
so that shape lives in exactly one place rather than being re-implemented by
every caller.

Two retrieval modes (selected via ``retrieval.mode`` in config):

- **dense** — embed the query, pull ``top_k`` candidates from the vector store
  by cosine similarity (original behavior).
- **hybrid** — run dense *and* BM25 keyword search in parallel (each ``top_k``),
  fuse the ranked lists with Reciprocal Rank Fusion, then pass the fused
  candidates to the reranker. Hybrid recovers exact-term hits that pure
  vector search often misses.
"""

from __future__ import annotations

import logging
import time
from typing import Literal

from rag.embedding.base import EmbeddingModel
from rag.events import EventSink, emit
from rag.retrieval.reranker import Reranker
from rag.retrieval.rrf import DEFAULT_RRF_K, reciprocal_rank_fusion
from rag.retrieval.sparse import SparseIndex
from rag.vectorstore.base import ScoredChunk, VectorStore

logger = logging.getLogger(__name__)

RetrievalMode = Literal["dense", "hybrid"]


class Retriever:
    """Embeds a query, searches (dense or hybrid), and reranks the results.

    Each stage is an injected interface (`EmbeddingModel`, `VectorStore`,
    optional `SparseIndex`, `Reranker`) -- `Retriever` contains no
    provider-specific logic of its own, only the orchestration and the knobs
    that shape it (`top_k`, `rerank_top_k`, `mode`, `rrf_k`).
    """

    def __init__(
        self,
        embedder: EmbeddingModel,
        vector_store: VectorStore,
        reranker: Reranker,
        *,
        top_k: int,
        rerank_top_k: int,
        sparse_index: SparseIndex | None = None,
        mode: RetrievalMode = "dense",
        rrf_k: int = DEFAULT_RRF_K,
    ) -> None:
        if mode == "hybrid" and sparse_index is None:
            raise ValueError(
                "Retriever mode='hybrid' requires a sparse_index "
                "(e.g. BM25Index). Pass sparse_index=... or use mode='dense'."
            )
        self._embedder = embedder
        self._vector_store = vector_store
        self._reranker = reranker
        self._sparse_index = sparse_index
        self.top_k = top_k
        self.rerank_top_k = rerank_top_k
        self.mode: RetrievalMode = mode
        self.rrf_k = rrf_k

    def retrieve(self, query: str, *, on_event: EventSink | None = None) -> list[ScoredChunk]:
        """Return the `rerank_top_k` chunks most relevant to `query`, best first.

        Stage 1 (``top_k`` candidates, denser still for hybrid before fusion)
        is intentionally wider than stage 2's output -- it gives the reranker
        enough material to recover relevant chunks that pure first-stage
        ranking placed lower, which is the entire reason to rerank at all.

        `on_event`, if given, is called once per completed stage (embedding,
        search, fusion, rerank) with a `PipelineEvent` -- e.g. so the UI can
        show a live trace. It's purely an observation hook: omitting it
        changes nothing about retrieval behavior or its return value.
        """

        if not query.strip():
            return []

        if self.mode == "hybrid":
            candidates = self._hybrid_candidates(query, on_event)
        else:
            candidates = self._dense_candidates(query, on_event)

        logger.info(
            "Retrieved %d candidate(s) via mode=%s for query %r",
            len(candidates),
            self.mode,
            query,
        )

        start = time.monotonic()
        results = self._reranker.rerank(query, candidates, top_k=self.rerank_top_k)
        emit(
            on_event,
            start,
            "rerank",
            f"Reranked {len(candidates)} candidate(s) down to {len(results)} result(s)",
        )
        logger.info("Reranked down to %d result(s)", len(results))
        return results

    def _dense_candidates(self, query: str, on_event: EventSink | None) -> list[ScoredChunk]:
        start = time.monotonic()
        query_vector = self._embedder.embed_query(query)
        emit(on_event, start, "embed", f"Embedded query into a {len(query_vector)}-dim vector")

        start = time.monotonic()
        results = self._vector_store.query(query_vector, top_k=self.top_k)
        emit(on_event, start, "vector_search", f"Vector search returned {len(results)} candidate(s)")
        return results

    def _hybrid_candidates(self, query: str, on_event: EventSink | None) -> list[ScoredChunk]:
        """Dense + BM25, fused with RRF into a single ``top_k`` candidate list."""

        assert self._sparse_index is not None  # enforced in __init__ for hybrid

        dense = self._dense_candidates(query, on_event)

        start = time.monotonic()
        sparse = self._sparse_index.query(query, top_k=self.top_k)
        emit(on_event, start, "sparse_search", f"BM25 search returned {len(sparse)} candidate(s)")

        logger.info(
            "Hybrid stage-1: dense=%d sparse=%d (rrf_k=%d)",
            len(dense),
            len(sparse),
            self.rrf_k,
        )

        if not dense and not sparse:
            return []
        if not sparse:
            # Sparse index empty (e.g. never indexed after enabling hybrid) —
            # degrade gracefully to dense-only rather than returning nothing.
            logger.warning("BM25 returned no results; falling back to dense candidates only")
            return dense
        if not dense:
            logger.warning("Dense search returned no results; falling back to BM25 candidates only")
            return sparse

        start = time.monotonic()
        fused = reciprocal_rank_fusion([dense, sparse], top_k=self.top_k, k=self.rrf_k)
        emit(on_event, start, "fusion", f"Fused dense + BM25 into {len(fused)} candidate(s) via RRF")
        return fused
