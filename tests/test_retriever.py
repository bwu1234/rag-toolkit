"""Tests for `Retriever` orchestration and `build_retriever` wiring.

`Retriever` is tested against lightweight fakes for `EmbeddingModel`,
`VectorStore`, `SparseIndex`, and `Reranker` -- it's a thin orchestration
layer, and the fakes let us assert exactly what gets passed between stages
without spinning up Ollama, Chroma, or a cross-encoder model.
"""

from __future__ import annotations

from pathlib import Path

import dataclasses

import pytest

from rag.query_filter import QueryFilter
from rag.retrieval.document_router import RoutingDecision
from rag.retrieval.expansion import ExpandedQuery
from rag.retrieval.reranker import NoOpReranker, Reranker
from rag.retrieval.retriever import Retriever
from rag.retrieval.sparse import IndexedDocument, SparseIndex
from rag.vectorstore.base import ScoredChunk, VectorStore


def _scored(chunk_id: str, score: float = 0.5, text: str | None = None) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk_id,
        text=text if text is not None else f"text for {chunk_id}",
        document_id="doc.md",
        source=Path("doc.md"),
        doc_type="markdown",
        score=score,
        metadata={},
    )


class _FakeEmbedder:
    """Returns a fixed query vector and records what it was asked to embed."""

    def __init__(self, vector: list[float] | None = None) -> None:
        self.vector = vector if vector is not None else [1.0, 0.0, 0.0]
        self.queries: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return self.vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise AssertionError("Retriever should never call embed_documents")

    @property
    def dimensions(self) -> int:
        return len(self.vector)


def _filtered(candidates: list[ScoredChunk], query_filter: QueryFilter | None) -> list[ScoredChunk]:
    """What a real store does with a filter: drop non-matching chunks before the top-k cut."""
    if query_filter is None:
        return candidates
    return [c for c in candidates if query_filter.matches({**c.metadata, "document_id": c.document_id})]


class _FakeVectorStore(VectorStore):
    """Returns a canned candidate list and records the query it received."""

    def __init__(self, candidates: list[ScoredChunk]) -> None:
        self.candidates = candidates
        self.queries: list[tuple[list[float], int]] = []
        self.filters: list[QueryFilter | None] = []

    def upsert(self, chunks, vectors) -> None:
        raise AssertionError("Retriever should never call upsert")

    def query(self, vector: list[float], top_k: int, query_filter: QueryFilter | None = None) -> list[ScoredChunk]:
        self.queries.append((vector, top_k))
        self.filters.append(query_filter)
        return _filtered(self.candidates, query_filter)[:top_k]

    def count(self) -> int:
        return len(self.candidates)

    def ids(self) -> set[str]:
        return {c.chunk_id for c in self.candidates}

    def delete(self, ids: list[str]) -> None:
        raise AssertionError("Retriever should never call delete")

    def reset(self) -> None:
        raise AssertionError("Retriever should never call reset")


class _FakeSparseIndex(SparseIndex):
    """Returns a canned BM25 candidate list and records the query it received."""

    def __init__(self, candidates: list[ScoredChunk]) -> None:
        self.candidates = candidates
        self.queries: list[tuple[str, int]] = []
        self.filters: list[QueryFilter | None] = []

    def upsert(self, chunks) -> None:
        raise AssertionError("Retriever should never call sparse upsert")

    def query(self, query: str, top_k: int, query_filter: QueryFilter | None = None) -> list[ScoredChunk]:
        self.queries.append((query, top_k))
        self.filters.append(query_filter)
        return _filtered(self.candidates, query_filter)[:top_k]

    def count(self) -> int:
        return len(self.candidates)

    def has_chunk(self, chunk_id: str) -> bool:
        return any(c.chunk_id == chunk_id for c in self.candidates)

    def ids(self) -> set[str]:
        return {c.chunk_id for c in self.candidates}

    def documents(self) -> list[IndexedDocument]:
        return []

    def delete(self, ids: list[str]) -> None:
        raise AssertionError("Retriever should never call sparse delete")

    def reset(self) -> None:
        raise AssertionError("Retriever should never call sparse reset")


class _FakeWebSearch:
    """Returns a canned web-search candidate list and records the query it received."""

    def __init__(self, candidates: list[ScoredChunk]) -> None:
        self.candidates = candidates
        self.queries: list[str] = []

    def search(self, query: str) -> list[ScoredChunk]:
        self.queries.append(query)
        return self.candidates


class _FakeReranker(Reranker):
    """Reverses candidate order and records what it was asked to rerank."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], list[ScoredChunk], int]] = []

    def rerank(self, queries: list[str], candidates: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        self.calls.append((queries, candidates, top_k))
        return list(reversed(candidates))[:top_k]


def _retriever(
    *,
    candidates: list[ScoredChunk] | None = None,
    sparse_candidates: list[ScoredChunk] | None = None,
    web_search_candidates: list[ScoredChunk] | None = None,
    reranker: Reranker | None = None,
    top_k: int = 10,
    rerank_top_k: int = 3,
    mode: str = "dense",
    rrf_k: int = 60,
    min_score: float = 0.0,
    query_expander=None,
    document_router=None,
) -> tuple[Retriever, _FakeEmbedder, _FakeVectorStore, Reranker, _FakeSparseIndex | None, _FakeWebSearch | None]:
    embedder = _FakeEmbedder()
    candidates = candidates if candidates is not None else [_scored("a"), _scored("b"), _scored("c")]
    vector_store = _FakeVectorStore(candidates)
    reranker = reranker if reranker is not None else _FakeReranker()
    sparse: _FakeSparseIndex | None = None
    if mode == "hybrid" or sparse_candidates is not None:
        sparse = _FakeSparseIndex(
            sparse_candidates if sparse_candidates is not None else list(candidates)
        )
    web_search = _FakeWebSearch(web_search_candidates) if web_search_candidates is not None else None
    retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
        reranker=reranker,
        top_k=top_k,
        rerank_top_k=rerank_top_k,
        sparse_index=sparse,
        mode=mode,  # type: ignore[arg-type]
        rrf_k=rrf_k,
        min_score=min_score,
        query_expander=query_expander,
        web_search=web_search,
        document_router=document_router,
    )
    return retriever, embedder, vector_store, reranker, sparse, web_search


def test_retrieve_embeds_query_and_passes_vector_to_store() -> None:
    retriever, embedder, vector_store, _reranker, _sparse, _web = _retriever(top_k=7)

    retriever.retrieve("what is the refund policy")

    assert embedder.queries == ["what is the refund policy"]
    [(vector, top_k)] = vector_store.queries
    assert vector == embedder.vector
    assert top_k == 7


def test_retrieve_passes_vector_store_results_to_reranker_with_rerank_top_k() -> None:
    candidates = [_scored("a"), _scored("b"), _scored("c")]
    retriever, _embedder, _store, reranker, _sparse, _web = _retriever(
        candidates=candidates, rerank_top_k=2
    )

    retriever.retrieve("query")

    [(queries, passed_candidates, top_k)] = reranker.calls
    assert queries == ["query"]
    assert passed_candidates == candidates
    assert top_k == 2


def test_retrieve_returns_rerankers_output() -> None:
    candidates = [_scored("a"), _scored("b"), _scored("c")]
    retriever, *_ = _retriever(candidates=candidates, rerank_top_k=2)

    results = retriever.retrieve("query").chunks

    # _FakeReranker reverses and truncates to top_k=2.
    assert [r.chunk_id for r in results] == ["c", "b"]


def test_retrieve_reports_embed_vector_search_and_rerank_events_in_dense_mode() -> None:
    retriever, *_ = _retriever(mode="dense")

    events = []
    retriever.retrieve("query", on_event=events.append)

    assert [event.stage for event in events] == ["embed", "vector_search", "rerank"]
    assert all(event.elapsed_ms is not None for event in events)


def test_retrieve_reports_fusion_event_in_hybrid_mode() -> None:
    retriever, *_ = _retriever(mode="hybrid")

    events = []
    retriever.retrieve("query", on_event=events.append)

    assert [event.stage for event in events] == [
        "embed",
        "vector_search",
        "sparse_search",
        "fusion",
        "rerank",
    ]


def test_retrieve_on_blank_query_returns_empty_without_calling_anything() -> None:
    retriever, embedder, vector_store, reranker, sparse, _web = _retriever(mode="hybrid")

    assert retriever.retrieve("   ").chunks == []
    assert embedder.queries == []
    assert vector_store.queries == []
    assert reranker.calls == []
    assert sparse is not None and sparse.queries == []


def test_retrieve_with_noop_reranker_is_pure_vector_search_truncated_to_rerank_top_k() -> None:
    candidates = [_scored("a", 0.9), _scored("b", 0.5), _scored("c", 0.1)]
    retriever, *_ = _retriever(candidates=candidates, reranker=NoOpReranker(), rerank_top_k=2)

    results = retriever.retrieve("query").chunks

    assert [r.chunk_id for r in results] == ["a", "b"]


def test_retrieve_on_empty_index_returns_empty_list() -> None:
    retriever, *_ = _retriever(candidates=[])

    assert retriever.retrieve("query").chunks == []


def test_hybrid_mode_requires_sparse_index() -> None:
    with pytest.raises(ValueError, match="sparse_index"):
        Retriever(
            embedder=_FakeEmbedder(),
            vector_store=_FakeVectorStore([]),
            reranker=NoOpReranker(),
            top_k=5,
            rerank_top_k=3,
            mode="hybrid",
            sparse_index=None,
        )


def test_hybrid_queries_both_dense_and_sparse_then_fuses() -> None:
    # Dense ranks a > b > c; sparse ranks c > a. RRF should surface a and c
    # ahead of b (present in only one list / lower combined rank).
    dense = [_scored("a", 0.9), _scored("b", 0.5), _scored("c", 0.1)]
    sparse = [_scored("c", 0.95), _scored("a", 0.4)]
    retriever, embedder, vector_store, reranker, sparse_index, _web = _retriever(
        candidates=dense,
        sparse_candidates=sparse,
        mode="hybrid",
        top_k=10,
        rerank_top_k=10,
        reranker=NoOpReranker(),
        rrf_k=60,
    )

    results = retriever.retrieve("SKU-42 refund").chunks

    assert embedder.queries == ["SKU-42 refund"]
    assert len(vector_store.queries) == 1
    assert sparse_index is not None
    assert sparse_index.queries == [("SKU-42 refund", 10)]
    ids = [r.chunk_id for r in results]
    assert set(ids) == {"a", "b", "c"}
    # a appears in both lists near the top → highest RRF; c is #1 sparse + #3 dense.
    assert ids[0] == "a"
    assert "c" in ids[:2]


def test_hybrid_falls_back_to_dense_when_sparse_empty() -> None:
    dense = [_scored("a", 0.9), _scored("b", 0.5)]
    retriever, *_rest = _retriever(
        candidates=dense,
        sparse_candidates=[],
        mode="hybrid",
        reranker=NoOpReranker(),
        rerank_top_k=2,
    )

    results = retriever.retrieve("query").chunks

    assert [r.chunk_id for r in results] == ["a", "b"]


def test_dense_mode_does_not_query_sparse_index() -> None:
    sparse_candidates = [_scored("only-sparse")]
    retriever, _e, _v, _r, sparse, _web = _retriever(
        candidates=[_scored("dense-only")],
        sparse_candidates=sparse_candidates,
        mode="dense",
        reranker=NoOpReranker(),
        rerank_top_k=5,
    )

    results = retriever.retrieve("query").chunks

    assert [r.chunk_id for r in results] == ["dense-only"]
    assert sparse is not None and sparse.queries == []


# ---------------------------------------------------------------------------
# web_search — a third ranked-list source, fused in like BM25
# ---------------------------------------------------------------------------


def test_web_search_disabled_by_default_and_never_called() -> None:
    retriever, *_rest = _retriever(mode="dense")

    retriever.retrieve("query")

    # No web_search fake was configured (web_search_candidates=None), so the
    # only way this would fail is if Retriever tried to call it anyway.


def test_dense_mode_still_fuses_in_web_search_when_configured() -> None:
    # Web search runs independently of retrieval.mode -- it's gated on
    # whether a SearxNGWebSearch was wired in, not on "dense" vs "hybrid".
    dense = [_scored("a", 0.9)]
    web = [_scored("w", 0.8)]
    retriever, embedder, _v, _r, _s, web_search = _retriever(
        candidates=dense,
        web_search_candidates=web,
        mode="dense",
        reranker=NoOpReranker(),
        rerank_top_k=5,
    )

    results = retriever.retrieve("query").chunks

    assert web_search is not None
    assert web_search.queries == ["query"]
    assert set(r.chunk_id for r in results) == {"a", "w"}
    assert embedder.queries == ["query"]


def test_hybrid_mode_fuses_dense_sparse_and_web_search() -> None:
    dense = [_scored("a", 0.9)]
    sparse = [_scored("b", 0.9)]
    web = [_scored("c", 0.9)]
    retriever, *_rest = _retriever(
        candidates=dense,
        sparse_candidates=sparse,
        web_search_candidates=web,
        mode="hybrid",
        reranker=NoOpReranker(),
        rerank_top_k=5,
    )

    results = retriever.retrieve("query").chunks

    assert set(r.chunk_id for r in results) == {"a", "b", "c"}


def test_web_search_event_reported_when_configured() -> None:
    retriever, *_rest = _retriever(
        candidates=[_scored("a")],
        web_search_candidates=[_scored("w")],
        mode="dense",
        reranker=NoOpReranker(),
    )

    events = []
    retriever.retrieve("query", on_event=events.append)

    assert "web_search" in [event.stage for event in events]


# ---------------------------------------------------------------------------
# min_score — the relevance floor on final results
# ---------------------------------------------------------------------------


def test_retrieve_drops_results_below_min_score() -> None:
    candidates = [_scored("a", 0.9), _scored("b", 0.4), _scored("c", 0.05)]
    retriever, *_rest = _retriever(
        candidates=candidates, reranker=NoOpReranker(), rerank_top_k=5, min_score=0.3
    )

    results = retriever.retrieve("query").chunks

    assert [r.chunk_id for r in results] == ["a", "b"]


def test_retrieve_returns_nothing_when_every_result_is_below_min_score() -> None:
    # The case the floor exists for: a vector store always hands back its
    # nearest neighbours, however distant. Without a floor an off-corpus
    # question still arrives at the LLM with a full set of passages.
    candidates = [_scored("a", 0.1), _scored("b", 0.05)]
    retriever, *_rest = _retriever(
        candidates=candidates, reranker=NoOpReranker(), rerank_top_k=5, min_score=0.5
    )

    assert retriever.retrieve("an off-corpus question").chunks == []


def test_min_score_of_zero_keeps_every_reranked_result() -> None:
    candidates = [_scored("a", 0.9), _scored("b", 0.0)]
    retriever, *_rest = _retriever(
        candidates=candidates, reranker=NoOpReranker(), rerank_top_k=5, min_score=0.0
    )

    assert [r.chunk_id for r in retriever.retrieve("query").chunks] == ["a", "b"]


def test_min_score_applies_to_reranker_scores_not_vector_scores() -> None:
    """The floor runs *after* reranking, so it reads the reranker's judgment."""

    class _ConstantReranker(Reranker):
        def __init__(self, score: float) -> None:
            self.score = score

        def rerank(self, queries, candidates, top_k):  # type: ignore[no-untyped-def]
            from dataclasses import replace

            return [replace(c, score=self.score) for c in candidates][:top_k]

    # High vector scores, but the reranker judges everything irrelevant.
    retriever, *_rest = _retriever(
        candidates=[_scored("a", 0.99), _scored("b", 0.98)],
        reranker=_ConstantReranker(0.02),
        rerank_top_k=5,
        min_score=0.3,
    )

    assert retriever.retrieve("query").chunks == []


def test_rerank_event_reports_how_many_were_dropped() -> None:
    retriever, *_rest = _retriever(
        candidates=[_scored("a", 0.9), _scored("b", 0.1)],
        reranker=NoOpReranker(),
        rerank_top_k=5,
        min_score=0.5,
    )

    events = []
    retriever.retrieve("query", on_event=events.append)

    [rerank_event] = [e for e in events if e.stage == "rerank"]
    assert "1 dropped below min_score" in rerank_event.message


def test_retrieve_reports_candidate_count_and_drops() -> None:
    retriever, *_rest = _retriever(
        candidates=[_scored("a", 0.9), _scored("b", 0.1), _scored("c", 0.05)],
        reranker=NoOpReranker(),
        rerank_top_k=5,
        min_score=0.5,
    )

    outcome = retriever.retrieve("query")

    assert [c.chunk_id for c in outcome.chunks] == ["a"]
    assert outcome.candidate_count == 3
    assert outcome.dropped_below_min_score == 2


def test_retrieve_distinguishes_an_empty_index_from_an_all_filtered_result() -> None:
    """`candidate_count` is what lets ChatService tell the two apart."""
    empty_index, *_rest = _retriever(candidates=[], reranker=NoOpReranker())
    all_filtered, *_rest2 = _retriever(
        candidates=[_scored("a", 0.1)], reranker=NoOpReranker(), min_score=0.9
    )

    empty = empty_index.retrieve("query")
    filtered = all_filtered.retrieve("query")

    assert empty.chunks == [] and empty.candidate_count == 0 and empty.dropped_below_min_score == 0
    assert filtered.chunks == [] and filtered.candidate_count == 1
    assert filtered.dropped_below_min_score == 1


def test_retrieve_on_blank_query_reports_an_empty_result() -> None:
    retriever, *_rest = _retriever()

    outcome = retriever.retrieve("   ")

    assert outcome.chunks == []
    assert outcome.candidate_count == 0
    assert outcome.dropped_below_min_score == 0


# ---------------------------------------------------------------------------
# Query expansion (HyDE / multi-query) — N ranked lists through one RRF
# ---------------------------------------------------------------------------


class _FakeExpander:
    """Returns a canned expansion and records what it was asked to expand."""

    def __init__(
        self, dense: list[str], sparse: list[str], rerank: list[str] | None = None
    ) -> None:
        # Default rerank to the sparse (question-shaped) list, which is what
        # both real expanders do -- HyDE keeps its passage out of reranking.
        self.expansion = ExpandedQuery(
            dense=dense, sparse=sparse, rerank=rerank if rerank is not None else list(sparse)
        )
        self.calls: list[str] = []

    def expand(self, query: str) -> ExpandedQuery:
        self.calls.append(query)
        return self.expansion


def test_expansion_embeds_and_searches_every_dense_query() -> None:
    expander = _FakeExpander(dense=["hypothetical passage", "original"], sparse=["original"])
    retriever, embedder, vector_store, _r, _s, _web = _retriever(
        reranker=NoOpReranker(), query_expander=expander
    )

    retriever.retrieve("original")

    assert expander.calls == ["original"]
    assert embedder.queries == ["hypothetical passage", "original"]
    assert len(vector_store.queries) == 2


def test_expansion_sends_only_the_sparse_queries_to_bm25() -> None:
    # HyDE's whole asymmetry: BM25 must not see the invented passage.
    expander = _FakeExpander(dense=["hypothetical passage", "original"], sparse=["original"])
    retriever, _e, _v, _r, sparse, _web = _retriever(
        mode="hybrid", reranker=NoOpReranker(), query_expander=expander
    )

    retriever.retrieve("original")

    assert sparse is not None
    assert [q for q, _k in sparse.queries] == ["original"]


def test_expansion_fuses_every_ranked_list_through_rrf() -> None:
    expander = _FakeExpander(dense=["q1", "q2"], sparse=["q1", "q2"])
    retriever, *_rest = _retriever(
        mode="hybrid", reranker=NoOpReranker(), rerank_top_k=5, query_expander=expander
    )

    events = []
    results = retriever.retrieve("original", on_event=events.append)

    # 2 dense + 2 sparse lists collapse into one fused candidate list.
    fusion_events = [e for e in events if e.stage == "fusion"]
    assert len(fusion_events) == 1
    assert "4 ranked list(s)" in fusion_events[0].message
    assert results.chunks


def test_expansion_reports_the_queries_it_searched() -> None:
    expander = _FakeExpander(dense=["hypothetical", "original"], sparse=["original"])
    retriever, *_rest = _retriever(reranker=NoOpReranker(), query_expander=expander)

    outcome = retriever.retrieve("original")

    assert outcome.search_queries == ["hypothetical", "original"]


def test_no_expansion_leaves_search_queries_empty() -> None:
    # Nothing to report when the query was searched as typed.
    retriever, *_rest = _retriever(reranker=NoOpReranker())

    assert retriever.retrieve("query").search_queries == []


def test_single_dense_query_still_skips_fusion_entirely() -> None:
    # The pre-expansion path must stay byte-for-byte identical, scores included.
    candidates = [_scored("a", 0.9), _scored("b", 0.5)]
    retriever, *_rest = _retriever(
        candidates=candidates, reranker=NoOpReranker(), rerank_top_k=5, mode="dense"
    )

    events = []
    results = retriever.retrieve("query", on_event=events.append)

    assert [e.stage for e in events] == ["embed", "vector_search", "rerank"]
    assert [(c.chunk_id, c.score) for c in results.chunks] == [("a", 0.9), ("b", 0.5)]


def test_expansion_emits_an_expand_event_only_when_it_changed_something() -> None:
    expanded, *_rest = _retriever(
        reranker=NoOpReranker(), query_expander=_FakeExpander(dense=["a", "b"], sparse=["a"])
    )
    unchanged, *_rest2 = _retriever(reranker=NoOpReranker())

    expanded_events = []
    expanded.retrieve("q", on_event=expanded_events.append)
    unchanged_events = []
    unchanged.retrieve("q", on_event=unchanged_events.append)

    assert "expand" in [e.stage for e in expanded_events]
    assert "expand" not in [e.stage for e in unchanged_events]


def test_expansion_degrades_to_the_one_list_that_returned_results() -> None:
    # One bad generated query shouldn't sink the search.
    expander = _FakeExpander(dense=["q1", "q2"], sparse=[])
    retriever, *_rest = _retriever(
        candidates=[_scored("a", 0.9)], reranker=NoOpReranker(), rerank_top_k=5,
        mode="dense", query_expander=expander,
    )

    results = retriever.retrieve("original")

    assert [c.chunk_id for c in results.chunks] == ["a"]


def test_blank_query_skips_expansion_entirely() -> None:
    expander = _FakeExpander(dense=["should not run"], sparse=[])
    retriever, *_rest = _retriever(reranker=NoOpReranker(), query_expander=expander)

    assert retriever.retrieve("   ").chunks == []
    assert expander.calls == []


def test_reranker_receives_the_expanded_question_shaped_queries() -> None:
    # Expansion that only widens stage 1 gets undone at rerank time, so the
    # rephrasings have to reach the reranker too.
    expander = _FakeExpander(
        dense=["q1", "q2"], sparse=["q1", "q2"], rerank=["q1", "q2"]
    )
    retriever, _e, _v, reranker, _s, _web = _retriever(query_expander=expander)

    retriever.retrieve("q1")

    [(queries, _candidates, _top_k)] = reranker.calls  # type: ignore[attr-defined]
    assert queries == ["q1", "q2"]


def test_hyde_shaped_expansion_keeps_the_generated_passage_out_of_reranking() -> None:
    # A cross-encoder is trained on (question, passage) pairs -- a passage in
    # the question slot is off-distribution, so HyDE excludes it here.
    expander = _FakeExpander(
        dense=["a long invented passage", "original"], sparse=["original"], rerank=["original"]
    )
    retriever, _e, _v, reranker, _s, _web = _retriever(query_expander=expander)

    retriever.retrieve("original")

    [(queries, _candidates, _top_k)] = reranker.calls  # type: ignore[attr-defined]
    assert queries == ["original"]


# ---------------------------------------------------------------------------
# Metadata filters (chunking plan, Phase 3)
# ---------------------------------------------------------------------------


def _filed(chunk_id: str, ticker: str) -> ScoredChunk:
    return dataclasses.replace(_scored(chunk_id), metadata={"ticker": ticker})


def test_filter_reaches_both_stores_and_only_matching_chunks_come_back() -> None:
    candidates = [_filed("a", "AAPL"), _filed("m", "MSFT"), _filed("b", "AAPL")]
    retriever, _e, vector_store, _r, sparse, _w = _retriever(
        candidates=candidates, mode="hybrid", reranker=NoOpReranker(), rerank_top_k=5
    )
    retriever.filterable_fields = ("ticker",)
    aapl = QueryFilter(equals={"ticker": "AAPL"})

    results = retriever.retrieve("q", query_filter=aapl).chunks

    assert {c.chunk_id for c in results} == {"a", "b"}
    assert vector_store.filters == [aapl]
    assert sparse is not None and sparse.filters == [aapl]


def test_filter_on_an_uncarried_field_is_rejected_before_searching() -> None:
    retriever, embedder, *_ = _retriever()

    with pytest.raises(ValueError, match="ticker"):
        retriever.retrieve("q", query_filter=QueryFilter(equals={"ticker": "AAPL"}))
    assert embedder.queries == [], "nothing is searched with a filter that can match nothing"


def test_an_empty_filter_is_no_filter() -> None:
    retriever, _e, vector_store, *_ = _retriever()

    retriever.retrieve("q", query_filter=QueryFilter())

    assert vector_store.filters == [None]


def test_web_search_is_skipped_under_a_filter() -> None:
    retriever, _e, _v, _r, _s, web_search = _retriever(
        candidates=[_filed("a", "AAPL")], web_search_candidates=[_scored("w")], reranker=NoOpReranker()
    )
    retriever.filterable_fields = ("ticker",)

    results = retriever.retrieve("q", query_filter=QueryFilter(equals={"ticker": "AAPL"})).chunks

    assert web_search is not None and web_search.queries == []
    assert [c.chunk_id for c in results] == ["a"]


# ---------------------------------------------------------------------------
# Document routing (chunking plan, Phase 3b)
# ---------------------------------------------------------------------------


class _FakeRouter:
    """Returns a canned decision and records the queries it was asked to route."""

    def __init__(self, document_ids: list[str]) -> None:
        self.document_ids = document_ids
        self.queries: list[str] = []

    def route(self, query: str) -> RoutingDecision:
        self.queries.append(query)
        return RoutingDecision(self.document_ids, "fake")


def _in_doc(chunk_id: str, document_id: str) -> ScoredChunk:
    return dataclasses.replace(_scored(chunk_id), document_id=document_id)


def test_routing_filters_both_stores_to_the_routed_documents() -> None:
    candidates = [_in_doc("a", "A.md"), _in_doc("b", "B.md"), _in_doc("a2", "A.md")]
    router = _FakeRouter(["A.md"])
    retriever, _e, vector_store, _r, sparse, _w = _retriever(
        candidates=candidates, mode="hybrid", reranker=NoOpReranker(), rerank_top_k=5, document_router=router
    )

    result = retriever.retrieve("q")

    assert {c.chunk_id for c in result.chunks} == {"a", "a2"}
    assert result.routed_to == ["A.md"]
    routed = QueryFilter(any_of={"document_id": ["A.md"]})
    assert vector_store.filters == [routed]
    assert sparse is not None and sparse.filters == [routed]


def test_a_routing_fallback_searches_unfiltered() -> None:
    retriever, _e, vector_store, *_ = _retriever(document_router=_FakeRouter([]))

    result = retriever.retrieve("q")

    assert vector_store.filters == [None]
    assert result.routed_to == []


def test_a_caller_filter_wins_over_routing() -> None:
    router = _FakeRouter(["B.md"])
    retriever, _e, vector_store, *_ = _retriever(candidates=[_in_doc("a", "A.md")], document_router=router)
    caller = QueryFilter(any_of={"document_id": ["A.md"]})

    result = retriever.retrieve("q", query_filter=caller)

    assert router.queries == [], "routing never runs under a caller's filter"
    assert vector_store.filters == [caller]
    assert result.routed_to == []
