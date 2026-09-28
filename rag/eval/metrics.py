"""Retrieval quality metrics for the RAG evaluation pipeline.

All functions are pure — they operate on relevance grades and return scalars.
No I/O, no dependencies beyond the stdlib.

Metrics take **gains**: the relevance grade of each retrieved result, in rank
order, as produced by :mod:`rag.eval.relevance`.  A grade of ``0`` means "not
relevant"; any positive value means relevant, and larger means more relevant.

Taking grades rather than id lists is what lets one set of metrics serve both
span-level and document-level ground truth (and any future scheme) without
knowing which is in play — the judging step decides what "relevant" means, and
these functions only rank-weight the answer.

Metrics
-------
``hit_rate``
    1.0 if any retrieved result was relevant, else 0.0.  Averaged across
    samples this is the fraction of queries that found *something* useful.

``recall_at_k``
    Fraction of the sample's expected items that the ranking covered.  Takes
    counts rather than gains, because "how many of the things I was supposed to
    find did I find" cannot be read off a rank-ordered gain list — a single
    expected item can be matched by several retrieved chunks.

``precision_at_k``
    Fraction of retrieved results that were relevant.

``reciprocal_rank``
    1 / rank of the first relevant result (0.0 if none).  Average across
    samples = MRR (Mean Reciprocal Rank).

``ndcg_at_k``
    Discounted cumulative gain against the best achievable ranking.  Unlike the
    others this rewards *ordering* and respects grades, so a run that surfaces
    the passage stating the exact figure above one merely discussing the topic
    scores higher — a distinction binary metrics discard entirely.
"""

from __future__ import annotations

from math import log2, sqrt

#: Two-sided 95% normal quantile.
_Z_95 = 1.959964


def hit_rate(gains: list[int]) -> float:
    """1.0 if any retrieved result was relevant, else 0.0."""
    return 1.0 if any(gain > 0 for gain in gains) else 0.0


def recall_at_k(covered: int, total_expected: int) -> float:
    """Fraction of expected items covered; 0.0 when nothing was expected."""
    if total_expected <= 0:
        return 0.0
    return covered / total_expected


def precision_at_k(gains: list[int]) -> float:
    """Fraction of retrieved results that were relevant."""
    if not gains:
        return 0.0
    return sum(1 for gain in gains if gain > 0) / len(gains)


def reciprocal_rank(gains: list[int]) -> float:
    """1 / rank of the first relevant result; 0.0 if none."""
    for rank, gain in enumerate(gains, start=1):
        if gain > 0:
            return 1.0 / rank
    return 0.0


def dcg(gains: list[int]) -> float:
    """Discounted cumulative gain with the standard ``2**g - 1`` numerator.

    The exponential numerator is what makes grades matter super-linearly: a
    grade-3 result is worth substantially more than three grade-1 results, which
    is the intended reading of "this passage actually states the answer".
    """
    return sum((2**gain - 1) / log2(rank + 1) for rank, gain in enumerate(gains, start=1))


def ndcg_at_k(gains: list[int], ideal_gains: list[int]) -> float:
    """DCG normalized by the DCG of the best achievable ranking.

    Returns 0.0 when no relevant result was achievable, so an unanswerable
    sample cannot inflate the mean.
    """
    ideal = dcg(ideal_gains)
    if ideal <= 0.0:
        return 0.0
    return dcg(gains) / ideal


def mean(values: list[float]) -> float:
    """Arithmetic mean of a non-empty list; returns 0.0 for an empty list."""
    return sum(values) / len(values) if values else 0.0


def wilson_interval(successes: int, n: int, *, z: float = _Z_95) -> tuple[float, float]:
    """95% Wilson score interval for a rate of ``successes`` out of ``n``.

    The noise floor of one rate on its own: how far a tier's hit or pass rate
    could move on a different draw of the same number of questions. Wilson
    rather than the textbook ``p ± z·sqrt(p(1-p)/n)``, which collapses to zero
    width at 0% or 100% and runs past [0, 1] near them -- both common on a
    50-question tier. Comparing two configs is :mod:`rag.eval.paired`'s job;
    this is the interval to read a single tier's baseline with.
    """
    if n <= 0:
        return 0.0, 1.0
    p = successes / n
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    half = z * sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)
