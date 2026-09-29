"""Tests for the phase-4 BEIR runner's own logic: variants, grouping, readings."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from rag.eval.dataset import MODE_QRELS, EvalSample
from rag.eval.paired import PairedDifference

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_beir_stack  # noqa: E402


def _sample(qid: str, judged: dict[str, int]) -> EvalSample:
    return EvalSample.from_dict(
        {
            "id": qid,
            "query": f"query {qid}",
            "expected_doc_ids": [d if g == 1 else {"id": d, "grade": g} for d, g in judged.items()],
            "matching_mode": MODE_QRELS,
        }
    )


def _diff(mean: float, low: float, high: float) -> PairedDifference:
    return PairedDifference(n=10, mean_diff=mean, ci_low=low, ci_high=high, wins=0, losses=0, p_value=1.0)


def test_every_family_comparison_names_known_variants_differing_from_their_baseline() -> None:
    for c in run_beir_stack.FAMILY:
        candidate, baseline = run_beir_stack.VARIANTS[c.candidate], run_beir_stack.VARIANTS[c.baseline]
        assert (candidate.config, candidate.overrides) != (baseline.config, baseline.overrides)


def test_every_variant_runs_at_the_benchmark_depths() -> None:
    for variant in run_beir_stack.VARIANTS.values():
        config = run_beir_stack.variant_config(variant)
        assert (config.retrieval.top_k, config.retrieval.rerank_top_k) == (100, 10)
        # Near-exact dense search, so rows measure the retriever and not HNSW.
        assert config.vector_store.hnsw_ef_search == 1600


def test_reranked_variants_use_the_shipped_cross_encoder() -> None:
    shipped = run_beir_stack.load_config().reranker.model
    for name in ("bge-hybrid-rerank", "qwen-hybrid-rerank"):
        config = run_beir_stack.variant_config(run_beir_stack.VARIANTS[name])
        assert config.reranker.provider == "cross_encoder" and config.reranker.model == shipped


def test_queries_sharing_a_relevant_document_share_a_group() -> None:
    samples = [
        _sample("1", {"a": 1}),
        _sample("2", {"a": 1, "b": 1}),
        _sample("3", {"b": 2}),
        _sample("4", {"c": 1}),
        # A shared grade-0 judgment does not relate queries.
        _sample("5", {"c": 0, "d": 1}),
    ]
    groups = run_beir_stack.query_groups(samples)

    assert groups["1"] == groups["2"] == groups["3"]
    assert len({groups["1"], groups["4"], groups["5"]}) == 3


@pytest.mark.parametrize(
    ("diff", "reading"),
    [
        (_diff(0.03, 0.01, 0.05), "improved"),
        (_diff(0.005, 0.001, 0.009), "improved (below 0.01)"),
        (_diff(-0.03, -0.05, -0.01), "worse"),
        (_diff(0.0, -0.008, 0.008), "no worthwhile effect"),
        (_diff(0.01, -0.01, 0.03), "not shown"),
    ],
)
def test_readings_follow_the_frozen_rules(diff: PairedDifference, reading: str) -> None:
    assert run_beir_stack._reading(diff) == reading
