"""Paired comparison of two eval runs over the same samples.

Why paired
----------
Every comparison in this repo runs two configurations on the *same* questions.
The noise that matters is then "would the difference hold on other questions",
and it depends only on the questions where the two runs **disagree**. A
question both runs get right, or both get wrong, says nothing about which is
better.

The unpaired standard error of one hit rate -- ``sqrt(p(1-p)/n)``, about 2.5pp
at n=174 -- ignores that. It measures how precisely *one* run's rate is known,
not how precisely a *difference* is. When two configs agree on most questions,
the paired interval is much tighter than twice that; when they reshuffle which
questions they get right, it can be wider. Neither direction is safe to assume,
so the difference is computed rather than inferred.

What is reported
----------------
* **mean difference** (candidate minus baseline) with a 95% interval from the
  normal approximation to the per-sample differences. For a binary metric this
  is the Wald interval for the difference of paired proportions.
* **wins / losses** -- samples where the candidate scored higher / lower. For a
  binary metric these are the discordant pairs, and are the most useful single
  thing to look at: "+6pp" from 12 wins and 2 losses is a different claim from
  "+6pp" from 40 wins and 30 losses.
* **sign-test p-value**, exact and two-sided, on wins vs losses. On a binary
  metric this *is* McNemar's exact test; on a graded one it is the sign test.
  It is exact where the normal interval is not: with a handful of discordant
  pairs, trust the p-value over the interval.

The interval assumes the samples are independent. When they come in related
groups (several questions written from one source), :func:`grouped_difference`
widens it with a cluster-robust variance, and ``confidence`` sets a wider level
when several comparisons are read together (Bonferroni: ``1 - 0.05 / m``).

Pure functions, stdlib only, like :mod:`rag.eval.metrics`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import comb, sqrt
from statistics import NormalDist

#: Two-sided 95% normal quantile.
Z_95 = 1.959964


def z_for(confidence: float) -> float:
    """Two-sided normal quantile for ``confidence`` (0.95 -> 1.96)."""
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    return Z_95 if confidence == 0.95 else NormalDist().inv_cdf(0.5 + confidence / 2)


@dataclass(frozen=True)
class PairedDifference:
    """Candidate minus baseline, over samples both runs scored."""

    n: int
    mean_diff: float
    ci_low: float
    ci_high: float
    #: Samples where the candidate scored higher than the baseline.
    wins: int
    #: Samples where the candidate scored lower than the baseline.
    losses: int
    #: Exact two-sided sign test on wins vs losses (McNemar's, for a binary metric).
    p_value: float

    @property
    def excludes_zero(self) -> bool:
        """True when the 95% interval lies entirely on one side of zero."""
        return self.ci_low > 0.0 or self.ci_high < 0.0


def sign_test_p(wins: int, losses: int) -> float:
    """Exact two-sided sign-test p-value; 1.0 when no sample differs."""
    trials = wins + losses
    if trials == 0:
        return 1.0
    tail = sum(comb(trials, i) for i in range(min(wins, losses) + 1)) / 2**trials
    return min(1.0, 2.0 * tail)


def paired_difference(
    baseline: list[float], candidate: list[float], *, confidence: float = 0.95
) -> PairedDifference:
    """Compare two equal-length score lists, position ``i`` being the same sample in both."""
    if len(baseline) != len(candidate):
        raise ValueError(
            f"Paired scores must have equal length, got {len(baseline)} and {len(candidate)}"
        )
    n = len(baseline)
    if n == 0:
        raise ValueError("Cannot compare two empty runs")

    diffs = [c - b for b, c in zip(baseline, candidate)]
    mean_diff = sum(diffs) / n
    # Sample variance (n-1). With n=1 there is no spread to estimate, and an
    # interval of zero width would read as certainty, so it is left unbounded.
    if n > 1:
        variance = sum((d - mean_diff) ** 2 for d in diffs) / (n - 1)
        half_width = z_for(confidence) * sqrt(variance / n)
    else:
        half_width = float("inf")

    wins = sum(1 for d in diffs if d > 0)
    losses = sum(1 for d in diffs if d < 0)
    return PairedDifference(
        n=n,
        mean_diff=mean_diff,
        ci_low=mean_diff - half_width,
        ci_high=mean_diff + half_width,
        wins=wins,
        losses=losses,
        p_value=sign_test_p(wins, losses),
    )


def grouped_difference(
    baseline: list[float], candidate: list[float], groups: list[str], *, confidence: float = 0.95
) -> PairedDifference:
    """Like :func:`paired_difference`, with the interval widened for grouped samples.

    ``groups[i]`` names the group sample ``i`` belongs to. The variance of the
    mean difference is the cluster-robust (sandwich) one: residuals are summed
    within each group before squaring, with the ``G / (G - 1)`` small-sample
    correction. All-singleton groups give the plain interval, up to that
    correction. Wins, losses and the sign test stay per sample.
    """
    plain = paired_difference(baseline, candidate, confidence=confidence)
    if len(groups) != plain.n:
        raise ValueError(f"Need one group per sample, got {len(groups)} for {plain.n}")
    sums: dict[str, float] = {}
    for b, c, g in zip(baseline, candidate, groups):
        sums[g] = sums.get(g, 0.0) + (c - b - plain.mean_diff)
    g_count = len(sums)
    if g_count < 2:
        half_width = float("inf")
    else:
        variance = g_count / (g_count - 1) * sum(v * v for v in sums.values()) / plain.n**2
        half_width = z_for(confidence) * sqrt(variance)
    return PairedDifference(
        n=plain.n,
        mean_diff=plain.mean_diff,
        ci_low=plain.mean_diff - half_width,
        ci_high=plain.mean_diff + half_width,
        wins=plain.wins,
        losses=plain.losses,
        p_value=plain.p_value,
    )


def compare_by_id(
    baseline: dict[str, float],
    candidate: dict[str, float],
    *,
    confidence: float = 0.95,
    groups: Mapping[str, str] | None = None,
) -> PairedDifference:
    """Pair two runs' per-sample scores by sample id, then compare them.

    Refuses rather than silently comparing on the intersection when the sample
    sets differ: that happens when two runs used different eval sets or
    ``--limit`` values, and a difference over a quietly shrunk set is not the
    comparison anyone asked for. ``groups`` (sample id -> group) switches to
    :func:`grouped_difference`.
    """
    if baseline.keys() != candidate.keys():
        only_base = sorted(baseline.keys() - candidate.keys())
        only_cand = sorted(candidate.keys() - baseline.keys())
        raise ValueError(
            "Runs cover different samples, so they cannot be paired "
            f"({len(only_base)} only in baseline, e.g. {only_base[:3]}; "
            f"{len(only_cand)} only in candidate, e.g. {only_cand[:3]})"
        )
    ids = sorted(baseline)
    base, cand = [baseline[i] for i in ids], [candidate[i] for i in ids]
    if groups is None:
        return paired_difference(base, cand, confidence=confidence)
    return grouped_difference(base, cand, [groups[i] for i in ids], confidence=confidence)


def format_difference(diff: PairedDifference, *, binary: bool = False) -> str:
    """Compact table cell: ``+0.104 [+0.061, +0.147]*``, plus ``21W/3L p=0.0002`` when binary.

    The ``*`` marks an interval that excludes zero. The win/loss count and the
    p-value are shown only for binary metrics, where the counts are the
    discordant pairs and the p-value is McNemar's exact test -- the number to
    trust over the interval when few samples differ. On NDCG almost every sample
    differs slightly, and a direction-only test can disagree with the interval
    for reasons that are not a finding, so neither is shown.
    """
    marker = "*" if diff.excludes_zero else ""
    cell = f"{diff.mean_diff:+.3f} [{diff.ci_low:+.3f}, {diff.ci_high:+.3f}]{marker}"
    if binary:
        cell += f" {diff.wins}W/{diff.losses}L p={diff.p_value:.2g}"
    return cell
