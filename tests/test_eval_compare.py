"""Pairing two pinned revisions: per-sample scores, flips and the paired difference.

What these protect: a comparison over run folders gives the same paired
cell (n, W/L, McNemar's p) as the CLI report's table; a pair the report
refuses is refused here; flips list losses first; and an answer revision's
unparseable verdict counts as a fail, as the answer matrix counts it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from rag.config.settings import RagConfig
from rag.eval.compare import (
    BOTH_FAIL,
    BOTH_PASS,
    LOSS,
    WIN,
    Metric,
    compare,
    flip_rows,
    render_comparison,
    sample_scores,
)
from rag.eval.dataset import EvalDataset
from rag.eval.report import Ineligible, Pinned, paired_delta, resolve
from rag.eval.retrieval_eval import evaluate_into_run
from rag.eval.runs import Run, open_run
from rag.eval.score import rescore
from rag.retrieval.retriever import RetrievalResult
from rag.vectorstore.base import ScoredChunk
from tests.test_eval_runs import VERIFIED, _dataset, _Retriever
from tests.test_eval_score_report import _edited, run_matrix


class _Shifted:
    """Finds the right document for every third question instead of every second."""

    def retrieve(self, query: str, *, query_filter: object = None, on_event: object = None) -> RetrievalResult:
        i = int(query.split()[-1])
        doc = f"doc{i}" if i % 3 == 0 else "other"
        chunk = ScoredChunk(
            chunk_id=f"{doc}::chunk0", text=f"It says fact {i if i % 3 == 0 else 'none'}.", document_id=doc,
            source=Path("/abs") / doc, doc_type="markdown", score=0.5,
        )
        return RetrievalResult(chunks=[chunk], candidate_count=1)


def retrieval_run(tmp_path: Path, dataset: EvalDataset, *, variant: str = "baseline",
                  overrides: dict[str, Any] | None = None, retriever: Any = None) -> Run:
    overrides = overrides or {}
    config = run_matrix.apply_overrides(RagConfig(), overrides)
    run = open_run(
        tmp_path, stage="retrieval", experiment="test", variant=variant, config=config,
        selection=config.corpus_selection(None), dataset_path=Path("data/eval/set.json"),
        dataset=dataset, selected=dataset, selection_method="all", index=VERIFIED, code="abc123",
        extra={"axis": variant, "overrides": overrides},
    )
    evaluate_into_run(run, dataset, retriever or _Retriever())  # type: ignore[arg-type]
    return run


def pair(tmp_path: Path) -> tuple[Pinned, Pinned]:
    """baseline hits q0, q2, q4; the candidate hits q0 and q3."""
    dataset = _dataset()
    base = retrieval_run(tmp_path, dataset)
    cand = retrieval_run(tmp_path, dataset, variant="top_k=3", overrides={"retrieval.top_k": 3}, retriever=_Shifted())
    return resolve(str(base.path)), resolve(str(cand.path))


def test_the_paired_cell_is_the_cli_reports(tmp_path: Path) -> None:
    base, cand = pair(tmp_path)
    comparison = compare(base, cand)
    hit = comparison.metrics[0]
    assert (hit.metric.name, hit.difference.n, hit.difference.wins, hit.difference.losses) == ("hit_rate", 6, 1, 2)
    assert hit.cell() == paired_delta(cand.row(), base.row(), "hit_rate", binary=True)
    assert comparison.config_diff == {"retrieval.top_k": (20, 3)} and comparison.confounders == []


def test_flips_list_losses_first(tmp_path: Path) -> None:
    rows = flip_rows(compare(*pair(tmp_path)))
    assert [(r["sample_id"], r["hit_rate"]) for r in rows] == [
        ("q2", LOSS), ("q4", LOSS), ("q3", WIN), ("q1", BOTH_FAIL), ("q5", BOTH_FAIL), ("q0", BOTH_PASS),
    ]
    assert "- loss: `q2`" in render_comparison(compare(*pair(tmp_path / "again")))


def test_a_pair_the_report_refuses_is_refused(tmp_path: Path) -> None:
    dataset = _dataset()
    base = retrieval_run(tmp_path, dataset)
    other = retrieval_run(tmp_path, dataset, variant="other")
    relabeled, _ = rescore(other, _edited(dataset, "q2", expected_spans=["a fact nobody wrote"]))
    with pytest.raises(Ineligible, match="different labels"):
        compare(resolve(str(base.path)), resolve(f"{other.path}:{relabeled}"))


def test_an_undeclared_difference_is_carried_as_a_confounder(tmp_path: Path) -> None:
    dataset = _dataset()
    base = resolve(str(retrieval_run(tmp_path, dataset).path))
    run = retrieval_run(tmp_path, dataset, variant="sneaky")
    run.manifest["config"]["retrieval"]["top_k"] = 3  # changed without declaring it
    sneaky = Pinned(run, base.revision, run.read_scores(run.score_revisions()[0]))
    assert compare(base, sneaky).confounders == ["`retrieval.top_k`: 20 -> 3"]


def test_an_unparseable_verdict_counts_as_a_fail() -> None:
    scores = {"samples": [{"id": "a", "passed": True}, {"id": "b", "passed": None}, {"id": "c", "passed": False}]}
    assert sample_scores(scores, Metric("passed", True)) == {"a": 1.0, "b": 0.0, "c": 0.0}
    with pytest.raises(ValueError, match="no evidence_recall"):
        sample_scores({"samples": [{"id": "a", "evidence_recall": None}]}, Metric("evidence_recall", False))
