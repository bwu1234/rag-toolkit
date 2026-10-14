"""Run folders for retrieval: what is recorded, when a run resumes, when it may complete.

What these protect: a resumed run scores exactly as an uninterrupted one and
never re-retrieves a finished sample; it resumes only into a run with the
same identity on the same verified index build; a limited run can't stand in
for a full one; and a run whose outputs don't cover its samples once each is
never marked complete or scored.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from rag.config.settings import RagConfig
from rag.eval.dataset import EvalDataset, subsample
from rag.eval.provenance import IndexProvenance
from rag.eval.retrieval_eval import (
    evaluate_into_run,
    ranking_from_output,
    report_scores,
    run_retrieval_eval,
)
from rag.eval.runs import COMPLETE, RUNNING, Run, RunIncomplete, open_run
from rag.retrieval.retriever import RetrievalResult
from rag.vectorstore.base import ScoredChunk

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import run_matrix  # noqa: E402

VERIFIED = IndexProvenance("recipe", "corpus", "build-1")


def _dataset(n: int = 6) -> EvalDataset:
    return EvalDataset.from_dicts([
        {
            "id": f"q{i}",
            "query": f"question {i}",
            "expected_doc_ids": [f"doc{i}"],
            "expected_spans": [f"fact {i}"],
            **({"kind": "implicit"} if i % 2 else {}),
        }
        for i in range(n)
    ])


class _Retriever:
    """Finds the right document for even-numbered questions; raises on ``fail_on``."""

    def __init__(self, fail_on: str | None = None) -> None:
        self.asked: list[str] = []
        self._fail_on = fail_on

    def retrieve(self, query: str, *, query_filter: object = None, on_event: object = None) -> RetrievalResult:
        if query == self._fail_on:
            raise RuntimeError("interrupted")
        self.asked.append(query)
        i = int(query.split()[-1])
        doc = f"doc{i}" if i % 2 == 0 else "other"
        chunk = ScoredChunk(
            chunk_id=f"{doc}::chunk0", text=f"It says fact {i if i % 2 == 0 else 'none'}.", document_id=doc,
            source=Path("/abs") / doc, doc_type="markdown", score=0.5 + i / 100,
            metadata={"ticker": "X"}, header="X 10-K",
        )
        return RetrievalResult(chunks=[chunk], candidate_count=1)


def _open(tmp_path: Path, dataset: EvalDataset, *, selected: EvalDataset | None = None,
          index: IndexProvenance = VERIFIED, **kwargs: Any) -> Run:
    config = RagConfig()
    return open_run(
        tmp_path / "runs", stage="retrieval", experiment="test", variant="baseline", config=config,
        selection=config.corpus_selection(None), dataset_path=Path("data/eval/set.json"),
        dataset=dataset, selected=selected or dataset, selection_method="all", index=index,
        code="abc123", **kwargs,
    )


def test_a_run_records_its_outputs_and_an_initial_score_revision(tmp_path: Path) -> None:
    dataset = _dataset()
    run = _open(tmp_path, dataset)
    report, revision = evaluate_into_run(run, dataset, _Retriever())

    manifest = json.loads((run.path / "manifest.json").read_text())
    assert manifest["status"] == COMPLETE and "pid" not in manifest
    assert manifest["dataset"]["selected_ids"] == [s.id for s in dataset]
    assert manifest["index"]["index_build_id"] == "build-1"
    assert manifest["run_id"] == run.path.name
    assert run.path.name.endswith(f"-retrieval-set-{manifest['config_hash'][:8]}")
    assert len(json.loads((run.path / "dataset.json").read_text())) == len(dataset)

    lines = run.outputs_path.read_text().splitlines()
    assert json.loads(lines[0]) == {"fingerprint": manifest["identity"]}
    first = json.loads(lines[1])
    assert first["results"][0]["text"] == "It says fact 0." and first["results"][0]["rank"] == 1
    assert "total" in first["latency_ms"] and "source" not in first["results"][0]

    scores = json.loads((run.path / "scores" / f"{revision}.json").read_text())
    assert scores["metrics"] == report_scores(report)["metrics"]
    assert scores["samples"] == report_scores(report)["samples"]
    assert scores["dataset_hash"] == manifest["dataset"]["dataset_hash"]
    assert scores["metrics"]["hit_rate"] == 0.5


def test_scoring_from_stored_outputs_matches_scoring_live_chunks() -> None:
    dataset = _dataset()
    live = run_retrieval_eval(dataset, _Retriever())  # type: ignore[arg-type]
    stored: dict[str, dict[str, Any]] = {}
    run_retrieval_eval(dataset, _Retriever(), on_output=lambda o: stored.__setitem__(o["sample_id"], o))  # type: ignore[arg-type]
    replayed = json.loads(json.dumps(stored))  # what a JSON Lines round trip hands back
    offline = run_retrieval_eval(dataset, None, outputs=replayed)
    assert report_scores(offline) == report_scores(live)
    assert ranking_from_output(replayed["q0"])[0].header == "X 10-K"


def test_an_interrupted_run_resumes_and_scores_as_an_uninterrupted_one(tmp_path: Path) -> None:
    dataset = _dataset()
    run = _open(tmp_path, dataset)
    with pytest.raises(RuntimeError, match="interrupted"):
        evaluate_into_run(run, dataset, _Retriever(fail_on="question 3"))
    assert json.loads((run.path / "manifest.json").read_text())["status"] == RUNNING

    resumed = _open(tmp_path, dataset)
    assert resumed.path == run.path
    retriever = _Retriever()
    report, _ = evaluate_into_run(resumed, dataset, retriever)
    assert retriever.asked == ["question 3", "question 4", "question 5"]

    clean, _ = evaluate_into_run(_open(tmp_path / "clean", dataset), dataset, _Retriever())
    assert report_scores(report) == report_scores(clean)


@pytest.mark.parametrize("index", [
    IndexProvenance("recipe", "corpus", None, unverified="no build record"),
    IndexProvenance("recipe", "corpus", "build-2"),
])
def test_a_run_resumes_only_on_the_same_verified_index_build(tmp_path: Path, index: IndexProvenance) -> None:
    dataset = _dataset()
    first = _open(tmp_path, dataset, index=index if index.build_id is None else VERIFIED)
    with pytest.raises(RuntimeError):
        evaluate_into_run(first, dataset, _Retriever(fail_on="question 3"))
    assert _open(tmp_path, dataset, index=index).path != first.path


def test_a_limited_run_is_not_resumed_as_a_full_one(tmp_path: Path) -> None:
    dataset = _dataset()
    limited = subsample(dataset, 3)
    run = _open(tmp_path, dataset, selected=limited)
    evaluate_into_run(run, limited, _Retriever())
    # Interrupt the full run at the first sample the limited one didn't have.
    full = _open(tmp_path, dataset)
    with pytest.raises(RuntimeError):
        evaluate_into_run(full, dataset, _Retriever(fail_on="question 1"))
    assert full.path != run.path
    assert full.manifest["identity"]["selection_hash"] != run.manifest["identity"]["selection_hash"]


def test_fresh_starts_a_new_run_beside_the_interrupted_one(tmp_path: Path) -> None:
    dataset = _dataset()
    run = _open(tmp_path, dataset)
    with pytest.raises(RuntimeError):
        evaluate_into_run(run, dataset, _Retriever(fail_on="question 3"))
    assert _open(tmp_path, dataset, resume=False).path != run.path


def test_a_run_another_live_process_holds_is_not_resumed(tmp_path: Path) -> None:
    dataset = _dataset()
    run = _open(tmp_path, dataset)
    run.manifest["pid"] = 1  # launchd/init: alive, and not this process
    (run.path / "manifest.json").write_text(json.dumps(run.manifest))
    assert _open(tmp_path, dataset).path != run.path


@pytest.mark.parametrize("edit, problem", [
    (lambda lines: lines + [lines[-1]], "duplicate outputs for q5"),
    (lambda lines: lines[:-1], "no output for q5"),
])
def test_completion_needs_each_selected_sample_exactly_once(tmp_path: Path, edit: Any, problem: str) -> None:
    dataset = _dataset()
    run = _open(tmp_path, dataset)
    evaluate_into_run(run, dataset, _Retriever())
    run.manifest["status"] = RUNNING
    run.outputs_path.write_text("\n".join(edit(run.outputs_path.read_text().splitlines())) + "\n")
    with pytest.raises(RunIncomplete, match=problem):
        run.complete()
    with pytest.raises(RunIncomplete, match="not complete"):
        run.write_scores(scorer="retrieval/1", settings={}, scores={})


def test_the_matrix_row_equals_its_runs_score_revision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = _dataset()

    def chunk(config: Any, corpora: Any) -> tuple[Any, list[Any], list[Any]]:
        return config.corpus_selection(corpora), [], []

    monkeypatch.setattr(run_matrix, "chunk_selected_corpora", chunk)
    monkeypatch.setattr(run_matrix, "checked_index", lambda *a, **k: VERIFIED)
    monkeypatch.setattr(run_matrix, "build_retriever", lambda *a, **k: _Retriever())
    row = run_matrix.run_variant(
        run_matrix.Variant("baseline", "baseline"), RagConfig(), dataset, None,
        dataset_path=Path("set.json"), code="abc", runs_dir=tmp_path,
    )
    scores = json.loads((tmp_path / row["run"] / "scores" / f"{row['score_revision']}.json").read_text())
    for key in ("num_samples", "metrics", "by_kind", "recall_by_k", "unmatchable_spans", "samples"):
        assert row[key] == scores[key], key
    manifest = json.loads((tmp_path / row["run"] / "manifest.json").read_text())
    assert manifest["config_hash"] == row["provenance"]["config_hash"]
    assert manifest["extra"] == {"overrides": {}}
