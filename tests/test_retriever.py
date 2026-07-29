"""Tests for `Retriever` orchestration and `build_retriever` wiring.

`Retriever` is tested against lightweight fakes for `EmbeddingModel`,
`VectorStore`, `SparseIndex`, and `Reranker` -- it's a thin orchestration
layer, and the fakes let us assert exactly what gets passed between stages
without spinning up Ollama, Chroma, or a cross-encoder model.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.retrieval.reranker import NoOpReranker, Reranker
from rag.retrieval.retriever import Retriever
from rag.retrieval.sparse import SparseIndex
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


class _FakeVectorStore(VectorStore):
    """Returns a canned candidate list and records the query it received."""

    def __init__(self, candidates: list[ScoredChunk]) -> None:
        self.candidates = candidates
        self.queries: list[tuple[list[float], int]] = []

    def upsert(self, chunks, vectors) -> None:
        raise AssertionError("Retriever should never call upsert")

    def query(self, vector: list[float], top_k: int) -> list[ScoredChunk]:
        self.queries.append((vector, top_k))
        return self.candidates[:top_k]

    def count(self) -> int:
        return len(self.candidates)

    def reset(self) -> None:
        raise AssertionError("Retriever should never call reset")


class _FakeSparseIndex(SparseIndex):
    """Returns a canned BM25 candidate list and records the query it received."""

    def __init__(self, candidates: list[ScoredChunk]) -> None:
        self.candidates = candidates
        self.queries: list[tuple[str, int]] = []

    def upsert(self, chunks) -> None:
        raise AssertionError("Retriever should never call sparse upsert")

    def query(self, query: str, top_k: int) -> list[ScoredChunk]:
        self.queries.append((query, top_k))
        return self.candidates[:top_k]

    def count(self) -> int:
        return len(self.candidates)

    def reset(self) -> None:
        raise AssertionError("Retriever should never call sparse reset")


class _FakeReranker(Reranker):
    """Reverses candidate order and records what it was asked to rerank."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[ScoredChunk], int]] = []

    def rerank(self, query: str, candidates: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        self.calls.append((query, candidates, top_k))
        return list(reversed(candidates))[:top_k]


def _retriever(
    *,
    candidates: list[ScoredChunk] | None = None,
    sparse_candidates: list[ScoredChunk] | None = None,
    reranker: Reranker | None = None,
    top_k: int = 10,
    rerank_top_k: int = 3,
    mode: str = "dense",
    rrf_k: int = 60,
    min_score: float = 0.0,
) -> tuple[Retriever, _FakeEmbedder, _FakeVectorStore, Reranker, _FakeSparseIndex | None]:
    embedder = _FakeEmbedder()
    candidates = candidates if candidates is not None else [_scored("a"), _scored("b"), _scored("c")]
    vector_store = _FakeVectorStore(candidates)
    reranker = reranker if reranker is not None else _FakeReranker()
    sparse: _FakeSparseIndex | None = None
    if mode == "hybrid" or sparse_candidates is not None:
        sparse = _FakeSparseIndex(
            sparse_candidates if sparse_candidates is not None else list(candidates)
        )
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
    )
    return retriever, embedder, vector_store, reranker, sparse


def test_retrieve_embeds_query_and_passes_vector_to_store() -> None:
    retriever, embedder, vector_store, _reranker, _sparse = _retriever(top_k=7)

    retriever.retrieve("what is the refund policy")

    assert embedder.queries == ["what is the refund policy"]
    [(vector, top_k)] = vector_store.queries
    assert vector == embedder.vector
    assert top_k == 7


def test_retrieve_passes_vector_store_results_to_reranker_with_rerank_top_k() -> None:
    candidates = [_scored("a"), _scored("b"), _scored("c")]
    retriever, _embedder, _store, reranker, _sparse = _retriever(
        candidates=candidates, rerank_top_k=2
    )

    retriever.retrieve("query")

    [(query, passed_candidates, top_k)] = reranker.calls
    assert query == "query"
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
    retriever, embedder, vector_store, reranker, sparse = _retriever(mode="hybrid")

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
    retriever, embedder, vector_store, reranker, sparse_index = _retriever(
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
    retriever, _e, _v, _r, sparse = _retriever(
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

        def rerank(self, query, candidates, top_k):  # type: ignore[no-untyped-def]
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
