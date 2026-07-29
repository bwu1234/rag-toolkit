"""Unit tests for Reciprocal Rank Fusion (pure, no I/O)."""

from __future__ import annotations

from pathlib import Path

from rag.retrieval.rrf import reciprocal_rank_fusion
from rag.vectorstore.base import ScoredChunk


def _scored(chunk_id: str, score: float = 0.5) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk_id,
        text=f"text-{chunk_id}",
        document_id="doc.md",
        source=Path("doc.md"),
        doc_type="markdown",
        score=score,
        metadata={"origin": chunk_id},
    )


def test_rrf_empty_lists_return_empty() -> None:
    assert reciprocal_rank_fusion([], top_k=5) == []
    assert reciprocal_rank_fusion([[], []], top_k=5) == []


def test_rrf_single_list_preserves_order_and_normalizes_top_to_one() -> None:
    ranked = [_scored("a", 0.9), _scored("b", 0.5), _scored("c", 0.1)]
    results = reciprocal_rank_fusion([ranked], top_k=10, k=60)

    assert [r.chunk_id for r in results] == ["a", "b", "c"]
    # Rank-1 in the only list → score == 1.0 after normalization.
    assert results[0].score == pytest_approx_one()
    assert results[0].score > results[1].score > results[2].score


def pytest_approx_one() -> float:
    return 1.0


def test_rrf_prefers_items_ranked_high_in_multiple_lists() -> None:
    list_a = [_scored("shared"), _scored("only-a")]
    list_b = [_scored("shared"), _scored("only-b")]
    results = reciprocal_rank_fusion([list_a, list_b], top_k=10, k=60)

    assert results[0].chunk_id == "shared"
    # shared gets 1/(k+1) + 1/(k+1); only-a/only-b get a single 1/(k+2)
    assert results[0].score > results[1].score
    assert {r.chunk_id for r in results} == {"shared", "only-a", "only-b"}


def test_rrf_respects_top_k() -> None:
    ranked = [_scored("a"), _scored("b"), _scored("c"), _scored("d")]
    results = reciprocal_rank_fusion([ranked], top_k=2, k=60)
    assert [r.chunk_id for r in results] == ["a", "b"]


def test_rrf_keeps_first_seen_payload() -> None:
    # Same id, different text — first list wins for the representative payload.
    list_a = [_scored("x", 0.9)]
    list_a[0] = ScoredChunk(
        chunk_id="x",
        text="from-dense",
        document_id="doc.md",
        source=Path("doc.md"),
        doc_type="markdown",
        score=0.9,
        metadata={"src": "dense"},
    )
    list_b = [
        ScoredChunk(
            chunk_id="x",
            text="from-bm25",
            document_id="doc.md",
            source=Path("doc.md"),
            doc_type="markdown",
            score=0.8,
            metadata={"src": "bm25"},
        )
    ]
    [result] = reciprocal_rank_fusion([list_a, list_b], top_k=1, k=60)
    assert result.text == "from-dense"
    assert result.metadata == {"src": "dense"}


def test_rrf_score_capped_at_one_when_first_in_all_lists() -> None:
    lists = [[_scored("top")], [_scored("top")], [_scored("top")]]
    [result] = reciprocal_rank_fusion(lists, top_k=1, k=60)
    assert result.score == 1.0
