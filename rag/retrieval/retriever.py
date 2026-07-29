"""End-to-end retrieval pipeline: query -> expand -> (hybrid) search -> fuse -> rerank.

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

Orthogonally, a `QueryExpander` (``retrieval.expansion``) can turn the one
query into several before any of that runs -- HyDE's hypothetical passages,
multi-query's rephrasings. Every (query, retriever) pair yields one ranked
list and RRF fuses them all, so expansion needs no new code path here: hybrid
mode was already fusing two lists, and expansion just makes it more.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Literal

from rag.embedding.base import EmbeddingModel
from rag.events import EventSink, emit
from rag.retrieval.expansion import ExpandedQuery, NoOpQueryExpander, QueryExpander
from rag.retrieval.reranker import Reranker
from rag.retrieval.rrf import DEFAULT_RRF_K, reciprocal_rank_fusion
from rag.retrieval.sparse import SparseIndex
from rag.vectorstore.base import ScoredChunk, VectorStore

logger = logging.getLogger(__name__)

RetrievalMode = Literal["dense", "hybrid"]


@dataclass(frozen=True)
class RetrievalResult:
    """The chunks retrieval settled on, plus how it got there.

    `retrieve` returns this rather than a bare list because "no results" has
    two very different causes that callers need to tell apart: nothing came
    back from the index at all (empty or unbuilt), versus candidates came back
    and every one of them scored below `min_score`. Only the retriever knows
    which happened, and the distinction is the difference between telling a
    user "run the indexer" and "the corpus doesn't cover this" -- so it's
    reported explicitly rather than reconstructed by guesswork downstream.
    """

    chunks: list[ScoredChunk] = field(default_factory=list)
    candidate_count: int = 0
    """Stage-1 candidates (post-fusion, pre-rerank). 0 means the index gave us nothing."""
    dropped_below_min_score: int = 0
    """Reranked results discarded by the `min_score` floor."""
    search_queries: list[str] = field(default_factory=list)
    """Every query actually searched, when expansion produced more than the one asked for."""


class Retriever:
    """Embeds a query, searches (dense or hybrid), and reranks the results.

    Each stage is an injected interface (`EmbeddingModel`, `VectorStore`,
    optional `SparseIndex`, `Reranker`) -- `Retriever` contains no
    provider-specific logic of its own, only the orchestration and the knobs
    that shape it (`top_k`, `rerank_top_k`, `mode`, `rrf_k`, `min_score`).

    Note that `top_k` is per *ranked list*, not per query: with expansion on,
    stage 1 pulls `top_k` for each generated query from each enabled retriever
    before fusion narrows the union back to `top_k`. More queries therefore
    means more recall to fuse over, not a bigger final candidate set.
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
        min_score: float = 0.0,
        query_expander: QueryExpander | None = None,
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
        self.min_score = min_score
        self._query_expander = query_expander or NoOpQueryExpander()

    def retrieve(self, query: str, *, on_event: EventSink | None = None) -> RetrievalResult:
        """Return the `rerank_top_k` chunks most relevant to `query`, best first.

        Stage 1 (``top_k`` candidates, denser still for hybrid before fusion)
        is intentionally wider than stage 2's output -- it gives the reranker
        enough material to recover relevant chunks that pure first-stage
        ranking placed lower, which is the entire reason to rerank at all.

        Results scoring below `min_score` are dropped, so this can legitimately
        return fewer than `rerank_top_k` chunks -- or none at all for a query
        the corpus has nothing to say about. Empty `chunks` is the signal
        `ChatService` uses to skip generation instead of grounding an answer in
        whatever happened to be nearest; the counts alongside it are what let
        `ChatService` explain *which* kind of nothing it got (see
        `RetrievalResult`).

        `on_event`, if given, is called once per completed stage (embedding,
        search, fusion, rerank) with a `PipelineEvent` -- e.g. so the UI can
        show a live trace. It's purely an observation hook: omitting it
        changes nothing about retrieval behavior or its return value.
        """

        if not query.strip():
            return RetrievalResult()

        start = time.monotonic()
        expanded = self._query_expander.expand(query)
        if expanded.is_expanded:
            emit(
                on_event,
                start,
                "expand",
                f"Expanded into {len(expanded.dense)} dense / {len(expanded.sparse)} sparse query/queries",
            )

        candidates = self._candidates(expanded, on_event)

        logger.info(
            "Retrieved %d candidate(s) via mode=%s for query %r",
            len(candidates),
            self.mode,
            query,
        )

        start = time.monotonic()
        # The reranker gets the expanded *question-shaped* queries, not just the
        # original: expansion that only widens stage 1 gets undone here, since a
        # vocabulary gap the rewrites closed is reintroduced the moment scoring
        # falls back to the user's original wording. See `ExpandedQuery.rerank`
        # for why HyDE's generated passage is deliberately excluded.
        reranked = self._reranker.rerank(expanded.rerank, candidates, top_k=self.rerank_top_k)
        results = [chunk for chunk in reranked if chunk.score >= self.min_score]
        dropped = len(reranked) - len(results)
        emit(
            on_event,
            start,
            "rerank",
            f"Reranked {len(candidates)} candidate(s) down to {len(results)} result(s)"
            + (f" ({dropped} dropped below min_score={self.min_score})" if dropped else ""),
        )
        logger.info("Reranked down to %d result(s) (%d below min_score)", len(results), dropped)
        return RetrievalResult(
            chunks=results,
            candidate_count=len(candidates),
            dropped_below_min_score=dropped,
            search_queries=expanded.all_queries() if expanded.is_expanded else [],
        )

    def _candidates(self, expanded: ExpandedQuery, on_event: EventSink | None) -> list[ScoredChunk]:
        """Run every expanded query through every enabled retriever and fuse the lot.

        Each (query, retriever) pair produces one ranked list, and RRF folds
        them all into one. That's the same operation hybrid mode always did
        across two lists -- expansion just means there are now more of them, so
        one dense + one BM25 list and four dense + four BM25 lists take the
        identical path. Rank-based fusion is what makes this safe: the lists
        come from different queries and different scoring functions, and RRF
        never compares their scores, only their positions.
        """

        ranked_lists = [self._dense_search(query, on_event) for query in expanded.dense]

        if self.mode == "hybrid":
            assert self._sparse_index is not None  # enforced in __init__ for hybrid
            ranked_lists.extend(self._sparse_search(query, on_event) for query in expanded.sparse)

        populated = [ranked for ranked in ranked_lists if ranked]
        logger.info(
            "Stage-1: %d/%d ranked list(s) non-empty via mode=%s (rrf_k=%d)",
            len(populated),
            len(ranked_lists),
            self.mode,
            self.rrf_k,
        )

        if not populated:
            return []
        if len(populated) == 1:
            # One list needs no fusion -- and skipping it keeps single-query
            # dense mode byte-for-byte what it was before expansion existed,
            # including its untouched similarity scores.
            if len(ranked_lists) > 1:
                logger.warning(
                    "Only 1 of %d ranked list(s) returned results; using it directly without fusion",
                    len(ranked_lists),
                )
            return populated[0]

        start = time.monotonic()
        fused = reciprocal_rank_fusion(populated, top_k=self.top_k, k=self.rrf_k)
        emit(
            on_event,
            start,
            "fusion",
            f"Fused {len(populated)} ranked list(s) into {len(fused)} candidate(s) via RRF",
        )
        return fused

    def _dense_search(self, query: str, on_event: EventSink | None) -> list[ScoredChunk]:
        start = time.monotonic()
        query_vector = self._embedder.embed_query(query)
        emit(on_event, start, "embed", f"Embedded {query[:60]!r} into a {len(query_vector)}-dim vector")

        start = time.monotonic()
        results = self._vector_store.query(query_vector, top_k=self.top_k)
        emit(on_event, start, "vector_search", f"Vector search returned {len(results)} candidate(s)")
        return results

    def _sparse_search(self, query: str, on_event: EventSink | None) -> list[ScoredChunk]:
        assert self._sparse_index is not None
        start = time.monotonic()
        results = self._sparse_index.query(query, top_k=self.top_k)
        emit(on_event, start, "sparse_search", f"BM25 search returned {len(results)} candidate(s)")
        return results
