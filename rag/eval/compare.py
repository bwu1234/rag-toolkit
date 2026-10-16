"""Comparing two pinned score revisions: per-sample scores, flips and paired differences.

Pure functions over `Pinned` revisions -- no tracker, no model, no index --
shared by the MLflow export (`rag.eval.tracking`) and usable by any report.
Eligibility is `report.check_pair`'s: a pair it refuses is refused here too,
whatever a tracker's UI would let you line up. See docs/eval-harness-plan.md,
"A viewer for comparing runs".

Each stage pairs the metrics its matrix table pairs: retrieval hit rate
(binary, with McNemar's test) and NDCG; answer pass (unparseable counts as a
fail, as `run_answer_matrix.paired_pass_delta` does); multi-hop completeness
(every part passed) and evidence recall.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rag.eval.paired import PairedDifference, compare_by_id, format_difference
from rag.eval.report import Pinned, check_pair, config_differences

#: A sample's outcome under a binary metric, and under a continuous one.
WIN, LOSS, BOTH_PASS, BOTH_FAIL, TIE = "win", "loss", "both pass", "both fail", "tie"
#: Losses first: they are what a reviewer reads before accepting a change.
OUTCOME_ORDER = (LOSS, WIN, BOTH_FAIL, BOTH_PASS, TIE)


@dataclass(frozen=True)
class Metric:
    """A per-sample score field, and whether it is pass/fail."""

    name: str
    binary: bool


PAIRED_METRICS: dict[str, tuple[Metric, ...]] = {
    "retrieval": (Metric("hit_rate", True), Metric("ndcg", False)),
    "answer": (Metric("passed", True),),
    "multihop": (Metric("complete", True), Metric("evidence_recall", False)),
}


def samples_by_id(scores: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """A revision's per-sample scores by id (retrieval keys them; answer stages list them)."""
    samples = scores["samples"]
    if isinstance(samples, dict):
        return {str(sid): s for sid, s in samples.items()}
    return {str(s["id"]): s for s in samples}


def sample_scores(scores: dict[str, Any], metric: Metric) -> dict[str, float]:
    """``metric`` per sample as a float; a binary metric's missing or unparseable value is a fail."""
    out = {}
    for sid, sample in samples_by_id(scores).items():
        value = sample.get(metric.name)
        if metric.binary:
            out[sid] = 1.0 if value is True or (not isinstance(value, bool) and value == 1) else 0.0
        elif value is None:
            raise ValueError(f"sample {sid} has no {metric.name}; it can't be paired")
        else:
            out[sid] = float(value)
    return out


def outcome(baseline: float, candidate: float, *, binary: bool) -> str:
    if candidate > baseline:
        return WIN
    if candidate < baseline:
        return LOSS
    if binary:
        return BOTH_PASS if candidate else BOTH_FAIL
    return TIE


@dataclass(frozen=True)
class MetricComparison:
    metric: Metric
    baseline_mean: float
    candidate_mean: float
    difference: PairedDifference
    #: Sample id -> `outcome`, for every paired sample.
    outcomes: dict[str, str]

    def cell(self) -> str:
        return format_difference(self.difference, binary=self.metric.binary)


@dataclass(frozen=True)
class Comparison:
    baseline: Pinned
    candidate: Pinned
    declared: tuple[str, ...]
    #: Undeclared differences `check_pair` flagged; empty when the pair is clean.
    confounders: list[str]
    config_diff: dict[str, tuple[Any, Any]]
    metrics: list[MetricComparison]

    @property
    def stage(self) -> str:
        return str(self.baseline.run.manifest["stage"])


def compare(baseline: Pinned, candidate: Pinned, declared: tuple[str, ...] = ()) -> Comparison:
    """Pair two revisions on their stage's metrics.

    Raises:
        Ineligible: `check_pair` refuses the pair.
        ValueError: the revisions cover different samples.
    """
    confounders = check_pair(baseline, candidate, list(declared))
    metrics = []
    for metric in PAIRED_METRICS[str(baseline.run.manifest["stage"])]:
        base, cand = sample_scores(baseline.scores, metric), sample_scores(candidate.scores, metric)
        diff = compare_by_id(base, cand)
        metrics.append(MetricComparison(
            metric=metric,
            baseline_mean=sum(base.values()) / len(base),
            candidate_mean=sum(cand.values()) / len(cand),
            difference=diff,
            outcomes={sid: outcome(base[sid], cand[sid], binary=metric.binary) for sid in sorted(base)},
        ))
    return Comparison(
        baseline=baseline,
        candidate=candidate,
        declared=declared,
        confounders=confounders,
        config_diff=config_differences(baseline.run, candidate.run),
        metrics=metrics,
    )


def flip_rows(comparison: Comparison) -> list[dict[str, Any]]:
    """One row per sample with its outcome under each metric, losses on the first metric first."""
    first = comparison.metrics[0]
    base, cand = samples_by_id(comparison.baseline.scores), samples_by_id(comparison.candidate.scores)
    rows = []
    for sid in sorted(first.outcomes, key=lambda s: (OUTCOME_ORDER.index(first.outcomes[s]), s)):
        row: dict[str, Any] = {"sample_id": sid, "kind": base[sid].get("kind") or cand[sid].get("kind")}
        for m in comparison.metrics:
            row[m.metric.name] = m.outcomes[sid]
        rows.append(row)
    return rows


def render_comparison(comparison: Comparison) -> str:
    """The comparison as markdown: paired metrics, pins, confounders, config diff, flips."""
    b, c = comparison.baseline, comparison.candidate
    lines = [
        f"# {c.run.manifest['variant']} vs {b.run.manifest['variant']}",
        "",
        f"- Stage: `{comparison.stage}`, eval set `{b.run.manifest['dataset']['path']}`",
        f"- Baseline: `{b.ref}`",
        f"- Candidate: `{c.ref}`",
        f"- Declared differences: {', '.join(f'`{d}`' for d in comparison.declared) or 'none'}",
        "",
        "| metric | baseline | candidate | Δ [95% CI] (W/L p for binary) | n |",
        "|---|---|---|---|---|",
        *(f"| {m.metric.name} | {m.baseline_mean:.3f} | {m.candidate_mean:.3f} | {m.cell()} | {m.difference.n} |"
          for m in comparison.metrics),
        "",
        "Δ is candidate minus baseline, paired by sample (`rag.eval.paired`). `*` marks a 95% "
        "interval that excludes zero; for a binary metric W/L counts the discordant samples and "
        "`p` is McNemar's exact test.",
    ]
    if comparison.confounders:
        lines += ["", "## Undeclared differences (possible confounders)", "",
                  *(f"- {item}" for item in comparison.confounders)]
    lines += ["", "## Config differences", ""]
    lines += ([f"- `{path}`: {old!r} -> {new!r}" for path, (old, new) in comparison.config_diff.items()]
              or ["- none"])
    first = comparison.metrics[0]
    flipped = [r for r in flip_rows(comparison) if r[first.metric.name] in (WIN, LOSS)]
    lines += ["", f"## Samples that flipped on {first.metric.name}", ""]
    lines += ([f"- {r[first.metric.name]}: `{r['sample_id']}`" for r in flipped] or ["- none"])
    return "\n".join(lines) + "\n"
