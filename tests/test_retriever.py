"""Tests for `Retriever` orchestration and `build_retriever` wiring.

`Retriever` is tested against lightweight fakes for `EmbeddingModel`,
`VectorStore`, and `Reranker` -- it's a thin orchestration layer, and the
fakes let us assert exactly what gets passed between stages without spinning
up Ollama, Chroma, or a cross-encoder model.
"""

from __future__ import annotations

from pathlib import Path

from rag.retrieval.reranker import NoOpReranker, Reranker
from rag.retrieval.retriever import Retriever
from rag.vectorstore.base import ScoredChunk, VectorStore


def _scored(chunk_id: str, score: float = 0.5) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk_id,
        text=f"text for {chunk_id}",
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
    reranker: Reranker | None = None,
    top_k: int = 10,
    rerank_top_k: int = 3,
) -> tuple[Retriever, _FakeEmbedder, _FakeVectorStore, Reranker]:
    embedder = _FakeEmbedder()
    candidates = candidates if candidates is not None else [_scored("a"), _scored("b"), _scored("c")]
    vector_store = _FakeVectorStore(candidates)
    reranker = reranker if reranker is not None else _FakeReranker()
    retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
        reranker=reranker,
        top_k=top_k,
        rerank_top_k=rerank_top_k,
    )
    return retriever, embedder, vector_store, reranker


def test_retrieve_embeds_query_and_passes_vector_to_store() -> None:
    retriever, embedder, vector_store, _reranker = _retriever(top_k=7)

    retriever.retrieve("what is the refund policy")

    assert embedder.queries == ["what is the refund policy"]
    [(vector, top_k)] = vector_store.queries
    assert vector == embedder.vector
    assert top_k == 7


def test_retrieve_passes_vector_store_results_to_reranker_with_rerank_top_k() -> None:
    candidates = [_scored("a"), _scored("b"), _scored("c")]
    retriever, _embedder, _store, reranker = _retriever(candidates=candidates, rerank_top_k=2)

    retriever.retrieve("query")

    [(query, passed_candidates, top_k)] = reranker.calls
    assert query == "query"
    assert passed_candidates == candidates
    assert top_k == 2


def test_retrieve_returns_rerankers_output() -> None:
    candidates = [_scored("a"), _scored("b"), _scored("c")]
    retriever, *_ = _retriever(candidates=candidates, rerank_top_k=2)

    results = retriever.retrieve("query")

    # _FakeReranker reverses and truncates to top_k=2.
    assert [r.chunk_id for r in results] == ["c", "b"]


def test_retrieve_on_blank_query_returns_empty_without_calling_anything() -> None:
    retriever, embedder, vector_store, reranker = _retriever()

    assert retriever.retrieve("   ") == []
    assert embedder.queries == []
    assert vector_store.queries == []
    assert reranker.calls == []


def test_retrieve_with_noop_reranker_is_pure_vector_search_truncated_to_rerank_top_k() -> None:
    candidates = [_scored("a", 0.9), _scored("b", 0.5), _scored("c", 0.1)]
    retriever, *_ = _retriever(candidates=candidates, reranker=NoOpReranker(), rerank_top_k=2)

    results = retriever.retrieve("query")

    assert [r.chunk_id for r in results] == ["a", "b"]


def test_retrieve_on_empty_index_returns_empty_list() -> None:
    retriever, *_ = _retriever(candidates=[])

    assert retriever.retrieve("query") == []
