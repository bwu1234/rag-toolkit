"""Interface for rescoring retrieved candidates against a query, plus a no-op default.

Vector similarity is a fast first pass but a weak relevance signal on its own
-- it compares a query embedding to a chunk embedding independently, never
looking at the two together. A reranker scores `(query, chunk)` *pairs*
directly, trading throughput for precision on a much smaller candidate set.
Kept behind an ABC so the pipeline can run with no reranker at all
(`provider: none`, pure vector retrieval) until a heavier model is justified.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import replace

from rag.vectorstore.base import ScoredChunk


class Reranker(ABC):
    """Interface for rescoring and re-ordering a candidate list against a query.

    To add a new strategy: subclass `Reranker`, implement `rerank`, and
    register it in `rag.retrieval.factory.get_reranker` based on
    `RerankerConfig.provider`. Pipeline code only depends on this interface.
    """

    @abstractmethod
    def rerank(self, query: str, candidates: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        """Return up to `top_k` of `candidates`, re-ordered by relevance to `query`.

        Implementations should return `ScoredChunk`s whose `.score` reflects
        *this* reranker's judgment (in `[0, 1]`, higher is more relevant) --
        not the vector store's similarity score, which is a different,
        incomparable signal. `NoOpReranker` is the one exception: it has no
        opinion of its own, so it preserves the incoming vector-similarity
        scores and ordering.
        """
        raise NotImplementedError


class NoOpReranker(Reranker):
    """Pass-through reranker used when `reranker.provider = "none"`.

    Candidates arrive from the vector store already ranked by similarity;
    this simply truncates to `top_k` without rescoring. Selecting it is what
    makes "pure vector retrieval" actually pure -- no extra model, no extra
    latency, nothing to download.
    """

    def rerank(self, query: str, candidates: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        return candidates[:top_k]


def _sigmoid(x: float) -> float:
    """Map an unbounded real-valued logit to `(0, 1)`.

    Cross-encoder models commonly output raw relevance logits (roughly
    -10..10 for `ms-marco-MiniLM`-style models), not calibrated probabilities.
    Squashing through a sigmoid keeps `ScoredChunk.score` in the same `[0, 1]`
    "higher is more relevant" convention the vector store uses, so callers
    (CLI output, the future API, eval scripts) never need to know which
    component produced a given score.
    """

    import math

    return 1.0 / (1.0 + math.exp(-x))


def normalize_rerank_score(raw_score: float) -> float:
    """Shared helper for reranker implementations to map a raw model score into `[0, 1]`.

    Exposed at module level (rather than buried in one adapter) so any future
    reranker backend that also produces unbounded logits can reuse it and stay
    consistent with `CrossEncoderReranker`.
    """

    return _sigmoid(raw_score)


def rescored(chunk: ScoredChunk, score: float) -> ScoredChunk:
    """Return a copy of `chunk` with `.score` replaced -- `ScoredChunk` is frozen."""

    return replace(chunk, score=score)
