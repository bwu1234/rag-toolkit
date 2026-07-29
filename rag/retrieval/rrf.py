"""Reciprocal Rank Fusion (RRF) for combining ranked retrieval lists.

RRF is rank-based (not score-based), so it fuses heterogeneous retrievers —
e.g. cosine similarity and BM25 — without needing to calibrate their raw
score scales against each other.

Reference: Cormack, Clarke, Buettcher (2009), "Reciprocal Rank Fusion
outperforms Condorcet and individual Rank Learning Methods."
"""

from __future__ import annotations

from rag.vectorstore.base import ScoredChunk

# Canonical default from the RRF paper / common IR practice.
DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    ranked_lists: list[list[ScoredChunk]],
    *,
    top_k: int,
    k: int = DEFAULT_RRF_K,
) -> list[ScoredChunk]:
    """Fuse one or more ranked result lists via Reciprocal Rank Fusion.

    For each unique ``chunk_id`` appearing in any list:

        RRF(d) = Σ_m  1 / (k + rank_m(d))

    where ``rank_m`` is the 1-based position of ``d`` in list ``m`` (chunks
    absent from a list contribute nothing from that list).

    The fused ``score`` is normalized into ``[0, 1]`` by dividing by the
    theoretical maximum (every list ranking the chunk first):

        max_rrf = n_lists / (k + 1)

    so a chunk ranked #1 by every retriever scores ``1.0``. Chunk payload
    (text, provenance, metadata) is taken from the first occurrence across
    the input lists.

    Args:
        ranked_lists: Each inner list is already sorted best-first.
        top_k: Maximum number of fused results to return.
        k: RRF rank constant; larger ``k`` dampens the influence of top ranks.

    Returns:
        Up to ``top_k`` ``ScoredChunk``s sorted by fused score, best first.
        Empty input lists yield an empty result.
    """

    if top_k <= 0:
        return []

    non_empty = [lst for lst in ranked_lists if lst]
    if not non_empty:
        return []

    # chunk_id -> (rrf_raw_score, representative ScoredChunk)
    fused: dict[str, tuple[float, ScoredChunk]] = {}

    for ranked in non_empty:
        for rank, chunk in enumerate(ranked, start=1):
            contribution = 1.0 / (k + rank)
            existing = fused.get(chunk.chunk_id)
            if existing is None:
                fused[chunk.chunk_id] = (contribution, chunk)
            else:
                raw_score, representative = existing
                fused[chunk.chunk_id] = (raw_score + contribution, representative)

    # Theoretical max: rank 1 in every contributing list.
    max_rrf = len(non_empty) / (k + 1)
    scored = [
        ScoredChunk(
            chunk_id=rep.chunk_id,
            text=rep.text,
            document_id=rep.document_id,
            source=rep.source,
            doc_type=rep.doc_type,
            score=(raw / max_rrf) if max_rrf > 0 else 0.0,
            metadata=dict(rep.metadata),
        )
        for raw, rep in fused.values()
    ]
    scored.sort(key=lambda c: c.score, reverse=True)
    return scored[:top_k]
