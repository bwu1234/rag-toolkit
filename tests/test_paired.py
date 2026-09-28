"""Tests for paired comparison of eval runs, and the matrix tables that use it."""

from __future__ import annotations

import sys
from math import sqrt
from pathlib import Path

import pytest

from rag.config.settings import RagConfig
from rag.eval.paired import (
    Z_95,
    compare_by_id,
    format_difference,
    paired_difference,
    sign_test_p,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_answer_matrix  # noqa: E402
import run_matrix  # noqa: E402

# ---------------------------------------------------------------------------
# Sign test / McNemar
# ---------------------------------------------------------------------------


def test_sign_test_is_1_when_nothing_differs() -> None:
    assert sign_test_p(0, 0) == 1.0


def test_sign_test_matches_the_exact_binomial() -> None:
    # 10 discordant pairs split 9/1: P(X<=1 | n=10, p=.5) = 11/1024, doubled.
    assert sign_test_p(9, 1) == pytest.approx(22 / 1024)
    assert sign_test_p(1, 9) == pytest.approx(22 / 1024)


def test_sign_test_even_split_is_capped_at_1() -> None:
    assert sign_test_p(5, 5) == 1.0


# ---------------------------------------------------------------------------
# paired_difference
# ---------------------------------------------------------------------------


def test_identical_runs_have_a_zero_width_interval() -> None:
    """The min_score case: byte-identical rows are an exact tie, not noise."""
    diff = paired_difference([1.0, 0.0, 1.0], [1.0, 0.0, 1.0])

    assert diff.mean_diff == 0.0
    assert (diff.ci_low, diff.ci_high) == (0.0, 0.0)
    assert (diff.wins, diff.losses) == (0, 0)
    assert diff.p_value == 1.0
    assert not diff.excludes_zero


def test_interval_is_the_normal_approximation_to_per_sample_differences() -> None:
    baseline = [0.0, 0.0, 1.0, 1.0]
    candidate = [1.0, 0.0, 1.0, 0.0]  # diffs: +1, 0, 0, -1
    diff = paired_difference(baseline, candidate)

    sd = sqrt((1 + 0 + 0 + 1) / 3)
    assert diff.mean_diff == 0.0
    assert diff.ci_high == pytest.approx(Z_95 * sd / 2)
    assert diff.ci_low == pytest.approx(-Z_95 * sd / 2)
    assert (diff.wins, diff.losses) == (1, 1)


def test_pairing_is_tighter_than_unpaired_when_runs_mostly_agree() -> None:
    """The reason this module exists: 174 samples, the candidate gains 10 and
    loses none. Unpaired, a 5.7pp gain on p≈0.8 is barely two standard errors
    of one rate; paired, it is unambiguous."""
    baseline = [1.0] * 140 + [0.0] * 34
    candidate = [1.0] * 150 + [0.0] * 24
    diff = paired_difference(baseline, candidate)

    unpaired_se = sqrt(0.8 * 0.2 / 174)
    assert diff.mean_diff == pytest.approx(10 / 174)
    assert (diff.ci_high - diff.ci_low) / 2 < Z_95 * unpaired_se
    assert diff.excludes_zero
    assert diff.p_value < 0.01


def test_a_reshuffle_is_not_a_gain() -> None:
    """Same net +2 as a clean 2-win run, but from 12 wins and 10 losses."""
    baseline = [1.0] * 10 + [0.0] * 12 + [1.0] * 100
    candidate = [0.0] * 10 + [1.0] * 12 + [1.0] * 100
    diff = paired_difference(baseline, candidate)

    assert (diff.wins, diff.losses) == (12, 10)
    assert not diff.excludes_zero
    assert diff.p_value > 0.5


def test_single_sample_has_an_unbounded_interval() -> None:
    diff = paired_difference([0.0], [1.0])
    assert diff.ci_low == float("-inf") and diff.ci_high == float("inf")


def test_unequal_lengths_are_rejected() -> None:
    with pytest.raises(ValueError, match="equal length"):
        paired_difference([1.0], [1.0, 0.0])


def test_empty_runs_are_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        paired_difference([], [])


# ---------------------------------------------------------------------------
# compare_by_id
# ---------------------------------------------------------------------------


def test_compare_by_id_pairs_by_id_not_by_order() -> None:
    diff = compare_by_id({"a": 0.0, "b": 1.0}, {"b": 1.0, "a": 1.0})
    assert (diff.wins, diff.losses) == (1, 0)


def test_compare_by_id_refuses_different_sample_sets() -> None:
    """A different --limit or eval set must not be compared on a silent intersection."""
    with pytest.raises(ValueError, match="different samples"):
        compare_by_id({"a": 1.0, "b": 1.0}, {"a": 1.0, "c": 0.0})


def test_format_difference_marks_significance_and_counts_only_for_binary() -> None:
    diff = paired_difference([0.0] * 10, [1.0] * 10)
    # 10 wins, 0 losses: p = 2 * (1/2)**10.
    assert format_difference(diff, binary=True) == "+1.000 [+1.000, +1.000]* 10W/0L p=0.002"
    assert "W/" not in format_difference(diff)
    assert "p=" not in format_difference(diff), "no direction-only p-value on graded metrics"


# ---------------------------------------------------------------------------
# Retrieval matrix table
# ---------------------------------------------------------------------------


def _row(variant: str, hits: dict[str, float] | None, axis: str = "x") -> dict:
    hit_rate = sum(hits.values()) / len(hits) if hits else 0.5
    row = {
        "variant": variant,
        "axis": axis,
        "elapsed_s": 1.0,
        "metrics": {"hit_rate": hit_rate, "recall": hit_rate, "precision": 0.1,
                    "mrr": 0.5, "ndcg": hit_rate},
    }
    if hits is not None:
        row["samples"] = {sid: {"hit_rate": h, "ndcg": h} for sid, h in hits.items()}
    return row


def test_retrieval_table_shows_a_paired_interval_against_baseline() -> None:
    base = _row("baseline", {"a": 0.0, "b": 1.0, "c": 1.0}, axis="baseline")
    variant = _row("v", {"a": 1.0, "b": 1.0, "c": 1.0})

    table = run_matrix.render_table([base, variant])

    assert "1W/0L p=1" in table, "the McNemar p-value is rendered, not just computed"
    assert "Δ hit [95% CI]" in table


def test_retrieval_table_marks_rows_without_per_sample_data() -> None:
    """Results recorded before per-sample storage must not be read as tested."""
    base = _row("baseline", None, axis="baseline")
    variant = _row("v", None)
    assert "(no CI)" in run_matrix.paired_delta(variant, base, "hit_rate")


def test_retrieval_table_refuses_to_pair_different_sample_sets() -> None:
    base = _row("baseline", {"a": 1.0}, axis="baseline")
    variant = _row("v", {"b": 1.0})
    assert "(unpaired)" in run_matrix.paired_delta(variant, base, "hit_rate")


# ---------------------------------------------------------------------------
# Answer matrix table
# ---------------------------------------------------------------------------


def _answer_run(passed: dict[str, bool | None], retrieval: int, generation: int) -> dict:
    return {
        "pass_rate": sum(1 for p in passed.values() if p) / len(passed),
        "num_evaluated": len(passed),
        "elapsed_s": 1.0,
        "failed_retrieval": retrieval,
        "failed_generation": generation,
        "failed_unattributed": 0,
        "samples": [{"id": sid, "passed": p, "evidence_retrieved": True}
                    for sid, p in passed.items()],
    }


def test_answer_table_splits_failures_and_pairs_against_the_first_variant() -> None:
    first = {"variant": "crag=off",
             "answerable": _answer_run({"a": True, "b": False, "c": None}, 1, 1)}
    second = {"variant": "crag=on",
              "answerable": _answer_run({"a": True, "b": True, "c": True}, 0, 0)}

    table = run_answer_matrix.render_table([first, second])

    assert "Δ vs `crag=off`" in table
    assert "| 1 / 1 |" in table
    assert "2W/0L" in table, "an unparseable verdict counts as a fail when pairing"


def test_answer_table_reports_failures_it_cannot_attribute() -> None:
    run = _answer_run({"a": False, "b": False}, 0, 0)
    run["failed_unattributed"] = 2
    table = run_answer_matrix.render_table([{"variant": "crag=off", "answerable": run}])
    assert "0 / 0 (+2 n/a)" in table


def test_multihop_table_reports_cost_and_marks_rows_recorded_before_it() -> None:
    new = {"num_evaluated": 34, "complete_rate": 0.5, "mean_completeness": 0.7,
           "evidence_recall": 0.6, "mean_latency_s": 12.0, "mean_llm_calls": 1.0,
           "mean_llm_s": 10.5, "mean_prompt_tokens": 4321.0, "mean_completion_tokens": None,
           "num_with_tokens": 30}
    old = {k: new[k] for k in ("num_evaluated", "complete_rate", "mean_completeness",
                               "evidence_recall", "mean_latency_s")}

    table = run_answer_matrix.render_table(
        [{"variant": "new", "multihop": new}, {"variant": "old", "multihop": old}]
    )

    assert "| 1.0 | 10.5 | 4,321 / ? (30 of 34) |" in table
    assert "| 12.0 | — | — | — |" in table


def test_matrix_fingerprint_keys_on_query_instruction_only_once_it_is_set() -> None:
    """Rows recorded before the field existed keep their fingerprint at its default."""
    unset = RagConfig()
    instructed = run_matrix.apply_overrides(
        unset, {"embedding.query_instruction": run_matrix.QWEN3_RETRIEVAL_INSTRUCTION}
    )

    digest, settings = run_matrix.fingerprint(unset)
    assert "embedding.query_instruction" not in settings
    assert run_matrix.fingerprint(instructed)[0] != digest
