"""Offline rescoring and pinned reports over run folders.

What these protect: a label fix rescores exactly the samples it touched and
leaves every earlier revision as it was; a changed input is refused rather
than rescored; runs that measured or scored different things can't be
paired, while variants differing in declared config can; a report over a
matrix's runs is the matrix's own table; and a report pinned to revisions
doesn't move when the run is rescored later.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from rag.config.settings import RagConfig
from rag.eval.dataset import EvalDataset, subsample
from rag.eval.report import Ineligible, build_report, check_pair, resolve
from rag.eval.retrieval_eval import evaluate_into_run
from rag.eval.runs import Run, open_run
from rag.eval.score import InputChanged, rescore
from tests.test_eval_runs import VERIFIED, _dataset, _Retriever

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import run_matrix  # noqa: E402


def _run(tmp_path: Path, dataset: EvalDataset, *, variant: str = "baseline",
         overrides: dict[str, Any] | None = None, selected: EvalDataset | None = None,
         code: str = "abc123", filters: str = "") -> Run:
    overrides = overrides or {}
    config = run_matrix.apply_overrides(RagConfig(), overrides)
    run = open_run(
        tmp_path, stage="retrieval", experiment="test", variant=variant, config=config,
        selection=config.corpus_selection(None), dataset_path=Path("data/eval/set.json"),
        dataset=dataset, selected=selected or dataset, selection_method="all", index=VERIFIED,
        code=code, extra={"axis": variant, "overrides": overrides, **({"filters": filters} if filters else {})},
    )
    evaluate_into_run(run, selected or dataset, _Retriever())  # type: ignore[arg-type]
    return run


def _edited(dataset: EvalDataset, sample_id: str, **fields: Any) -> EvalDataset:
    items = [s.to_dict() for s in dataset]
    for item in items:
        if item["id"] == sample_id:
            item.update(fields)
    return EvalDataset.from_dicts(items)


# ---------------------------------------------------------------------------
# Rescoring
# ---------------------------------------------------------------------------


def test_rescoring_unchanged_labels_names_the_existing_revision(tmp_path: Path) -> None:
    run = _run(tmp_path, _dataset())
    [initial] = run.score_revisions()
    revision, changes = rescore(run)
    assert revision == initial and changes == {} and run.score_revisions() == [initial]


def test_an_edited_span_label_rescored_changes_only_that_sample(tmp_path: Path) -> None:
    dataset = _dataset()
    run = _run(tmp_path, dataset)
    [initial] = run.score_revisions()
    initial_file = (run.path / "scores" / f"{initial}.json").read_bytes()

    # q2 was a hit on "fact 2"; a span no chunk contains makes it a miss.
    fixed = _edited(dataset, "q2", expected_spans=["a fact nobody wrote"])
    revision, changes = rescore(run, fixed)

    assert revision != initial and changes == {"q2": ["expected_spans"]}
    assert (run.path / "scores" / f"{initial}.json").read_bytes() == initial_file
    before, after = run.read_scores(initial), run.read_scores(revision)
    assert [sid for sid in before["samples"] if before["samples"][sid] != after["samples"][sid]] == ["q2"]
    assert after["samples"]["q2"]["hit_rate"] == 0.0
    assert after["unmatchable_spans"] is None  # not the old count carried onto new labels
    assert run.labels(after["dataset_hash"]).samples[2].expected_spans[0].text == "a fact nobody wrote"
    assert rescore(run, fixed)[0] == revision  # same labels again: same revision


@pytest.mark.parametrize("edit, match", [
    (lambda d: _edited(d, "q1", query="a different question"), "different inputs for q1"),
    (lambda d: EvalDataset(samples=d.samples[:-1]), "no longer has samples this run evaluated: q5"),
])
def test_a_changed_input_needs_a_fresh_run(tmp_path: Path, edit: Any, match: str) -> None:
    dataset = _dataset()
    run = _run(tmp_path, dataset)
    with pytest.raises(InputChanged, match=match):
        rescore(run, edit(dataset))
    assert len(run.score_revisions()) == 1


def test_document_labels_are_inputs_to_a_run_filtered_by_them(tmp_path: Path) -> None:
    dataset = _dataset()
    moved = _edited(dataset, "q0", expected_doc_ids=["doc9"])
    rescore(_run(tmp_path / "plain", dataset), moved)
    with pytest.raises(InputChanged, match="q0"):
        rescore(_run(tmp_path / "filtered", dataset, filters="company"), moved)


# ---------------------------------------------------------------------------
# Eligibility
# ---------------------------------------------------------------------------


def test_declared_config_variants_pair_and_undeclared_differences_are_flagged(tmp_path: Path) -> None:
    dataset = _dataset()
    base = resolve(str(_run(tmp_path, dataset).path))
    declared = resolve(str(_run(tmp_path, dataset, variant="rerank_top_k=10",
                                overrides={"retrieval.rerank_top_k": 10}).path))
    assert check_pair(base, declared, []) == []

    # Same change, made in a config file rather than as a declared override.
    config = run_matrix.apply_overrides(RagConfig(), {"retrieval.rerank_top_k": 10})
    run = open_run(tmp_path, stage="retrieval", experiment="t", variant="from-file", config=config,
                   selection=config.corpus_selection(None), dataset_path=Path("data/eval/set.json"),
                   dataset=dataset, selected=dataset, selection_method="all", index=VERIFIED, code="other")
    evaluate_into_run(run, dataset, _Retriever())  # type: ignore[arg-type]
    undeclared = resolve(str(run.path))
    assert check_pair(base, undeclared, []) == [
        "`retrieval.rerank_top_k`: 5 -> 10", "code: abc123 -> other"
    ]
    assert check_pair(base, undeclared, ["retrieval.*", "code"]) == []


def test_runs_on_different_samples_or_labels_cannot_be_paired(tmp_path: Path) -> None:
    dataset = _dataset()
    base_run = _run(tmp_path, dataset)
    base = resolve(str(base_run.path))
    limited = resolve(str(_run(tmp_path, dataset, variant="limited", selected=subsample(dataset, 3)).path))
    with pytest.raises(Ineligible, match="different samples evaluated"):
        check_pair(base, limited, [])

    other_run = _run(tmp_path, dataset, variant="other")
    fixed = _edited(dataset, "q2", expected_spans=["a fact nobody wrote"])
    relabeled, _ = rescore(other_run, fixed)
    with pytest.raises(Ineligible, match="different labels"):
        check_pair(base, resolve(f"{other_run.path}:{relabeled}"), [])
    # Bringing the baseline onto the same labels makes them comparable.
    base_fixed, _ = rescore(base_run, fixed)
    assert check_pair(resolve(f"{base_run.path}:{base_fixed}"), resolve(f"{other_run.path}:{relabeled}"), []) == []


def test_a_pinned_report_is_unchanged_by_later_rescoring(tmp_path: Path) -> None:
    dataset = _dataset()
    base_run, cand_run = _run(tmp_path, dataset), _run(tmp_path, dataset, variant="cand")
    refs = [f"{r.run_id}:{r.score_revisions()[0]}" for r in (base_run, cand_run)]
    before = build_report(resolve(refs[0], tmp_path), [resolve(refs[1], tmp_path)])

    rescore(cand_run, _edited(dataset, "q2", expected_spans=["a fact nobody wrote"]))
    assert build_report(resolve(refs[0], tmp_path), [resolve(refs[1], tmp_path)]) == before
    with pytest.raises(ValueError, match="has 2 score revisions"):
        resolve(cand_run.run_id, tmp_path)


# ---------------------------------------------------------------------------
# The report reproduces the matrix's table
# ---------------------------------------------------------------------------


def test_a_report_over_matrix_runs_reproduces_its_markdown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = _dataset()
    eval_set = tmp_path / "set.json"
    dataset.save(eval_set)

    def chunk(config: Any, corpora: Any) -> tuple[Any, list[Any], list[Any]]:
        return config.corpus_selection(corpora), [], []

    monkeypatch.setattr(run_matrix, "chunk_selected_corpora", chunk)
    monkeypatch.setattr(run_matrix, "checked_index", lambda *a, **k: VERIFIED)
    monkeypatch.setattr(run_matrix, "build_retriever", lambda *a, **k: _Retriever())
    monkeypatch.setattr(sys, "argv", [
        "run_matrix.py", "--eval-set", str(eval_set), "--results-dir", str(tmp_path / "results"),
        "--runs-dir", str(tmp_path / "runs"), "--variant", "baseline", "--variant", "rerank_top_k=10",
        "--variant", "mode=dense",
    ])
    assert run_matrix.main() == 0

    [legacy_json] = (tmp_path / "results").glob("retrieval_*.json")
    rows = json.loads(legacy_json.read_text())["results"]
    refs = [f"{r['run']}:{r['score_revision']}" for r in rows]
    report = build_report(resolve(refs[0], tmp_path / "runs"), [resolve(r, tmp_path / "runs") for r in refs[1:]])
    legacy_md = legacy_json.with_suffix(".md").read_text()
    assert report.startswith(legacy_md)
    assert "## Undeclared differences" not in report
