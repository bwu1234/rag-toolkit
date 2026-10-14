"""Answer runs: generation stored once, judged as separate attempts, scored offline.

What these protect: an answer run keeps the passages each answer was shown
and every model call verbatim, so evidence is rescored without an index and a
trace needs no current template to read; judging again -- the same judge or
another -- adds an attempt and never overwrites one; a changed reference
answer refuses the old judgments while a changed span label doesn't need
them redone; a judging stopped halfway resumes; and the answer matrix's row
is exactly the score revision it names. No model runs: chat and judge are fakes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from rag.config.settings import LLMConfig, RagConfig
from rag.eval.answer_eval import generate_into_run
from rag.eval.dataset import EvalDataset
from rag.eval.judge import ReferencesChanged, attempts, judge_run
from rag.eval.report import Ineligible, check_pair, resolve
from rag.eval.runs import REVISION_FIELDS, Run, RunIncomplete, open_run
from rag.eval.score import InputChanged, rescore
from rag.generation.chat_service import ChatService
from rag.observability.usage import MeteredLLMClient
from tests.fakes import skip_index_checks
from tests.test_eval_runs import VERIFIED
from tests.test_observability import _Retriever, _result, _UsageLLM

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import run_answer_matrix as matrix  # noqa: E402

JUDGE = LLMConfig(model="judge-a")
OTHER_JUDGE = LLMConfig(model="judge-b")


class _Judge:
    """Replies from a script, then PASS; counts its calls."""

    def __init__(self, *replies: str, fail_after: int | None = None) -> None:
        self.replies = list(replies)
        self.calls = 0
        self.fail_after = fail_after

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        if self.fail_after is not None and self.calls >= self.fail_after:
            raise RuntimeError("judge went away")
        self.calls += 1
        return self.replies.pop(0) if self.replies else "PASS ok"


def _answers(n: int = 3) -> EvalDataset:
    return EvalDataset.from_dicts([
        {"id": f"s{i}", "query": f"q{i}", "expected_answer": f"answer {i}", "expected_spans": ["text of a"]}
        for i in range(n)
    ])


def _multihop() -> EvalDataset:
    return EvalDataset.from_dicts([{
        "id": "mh", "query": "compare", "expected_answer": "A 1; B 2", "kind": "cross_company",
        "parts": [{"label": "A", "answer": "1", "spans": ["text of a"]},
                  {"label": "B", "answer": "2", "spans": ["text of b"]}],
        "conclusion": "B is larger",
    }])


def _chat() -> ChatService:
    """The real pipeline over a canned retriever and model, so the turn's transcript is real."""
    return ChatService(retriever=_Retriever(_result("a")), llm_client=MeteredLLMClient(_UsageLLM("It is a [1].")))


def _run(tmp_path: Path, dataset: EvalDataset, stage: str = "answer") -> Run:
    config = RagConfig()
    run = open_run(
        tmp_path, stage=stage, experiment="test", variant="v", config=config,
        selection=config.corpus_selection(None), dataset_path=Path("data/eval/set.json"), dataset=dataset,
        selected=dataset, selection_method="all", index=VERIFIED, code="abc",
    )
    generate_into_run(run, dataset, _chat())
    return run


def _edited(dataset: EvalDataset, sample_id: str, **fields: Any) -> EvalDataset:
    items = [s.to_dict() for s in dataset]
    for item in items:
        if item["id"] == sample_id:
            item.update(fields)
    return EvalDataset.from_dicts(items)


# ---------------------------------------------------------------------------
# What a run stores
# ---------------------------------------------------------------------------


def test_outputs_keep_the_shown_passages_and_every_model_call_verbatim(tmp_path: Path) -> None:
    run = _run(tmp_path, _answers(1))
    [output] = run.read_outputs()

    assert output["answer"] == "It is a [1]." and "passed" not in output  # generated, not judged
    assert [(p["chunk_id"], p["document_id"], p["text"]) for p in output["passages"]] == [("a", "a-doc", "text of a")]
    [call] = output["turn"]["llm_exchanges"]
    # The prompt as the model received it: the passage text is in it, rendered.
    assert call["system"] and "text of a" in call["prompt"] and call["response"] == "It is a [1]."


def test_judging_needs_a_complete_run(tmp_path: Path) -> None:
    config = RagConfig()
    run = open_run(
        tmp_path, stage="answer", experiment="t", variant="v", config=config,
        selection=config.corpus_selection(None), dataset_path=Path("s.json"), dataset=_answers(),
        selected=_answers(), selection_method="all", index=VERIFIED, code="abc",
    )
    with pytest.raises(RunIncomplete, match="only a complete run can be judged"):
        judge_run(run, _Judge(), JUDGE)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Attempts
# ---------------------------------------------------------------------------


def test_repeated_attempts_and_a_second_judge_coexist(tmp_path: Path) -> None:
    run = _run(tmp_path, _answers())
    first = judge_run(run, _Judge("FAIL wrong"), JUDGE)  # type: ignore[arg-type]
    first_lines = first.records_path.read_text()
    again = judge_run(run, _Judge(), JUDGE)  # type: ignore[arg-type]
    other = judge_run(run, _Judge(), OTHER_JUDGE)  # type: ignore[arg-type]

    assert first.judge_hash == again.judge_hash != other.judge_hash
    assert len({first.attempt_id, again.attempt_id, other.attempt_id}) == 3
    assert first.records_path.read_text() == first_lines  # never rewritten
    assert first.records()["s0"]["verdicts"][0]["passed"] is False
    assert again.records()["s0"]["verdicts"][0]["passed"] is True
    # Every judge call keeps its exact inputs and full reply.
    verdict = first.records()["s0"]["verdicts"][0]
    assert "answer 0" in verdict["prompt"] and verdict["reply"] == "FAIL wrong" and verdict["system"]

    # Two attempts against these references: scoring needs to be told which.
    with pytest.raises(ValueError, match="pick one with --judgment"):
        rescore(run)
    revisions = {rescore(run, judgment=a.ref)[0] for a in (first, again, other)}
    assert len(revisions) == 3
    assert run.read_scores(rescore(run, judgment=first.ref)[0])["num_passed"] == 2


def test_an_interrupted_judging_resumes_without_rejudging(tmp_path: Path) -> None:
    run = _run(tmp_path, _answers())
    with pytest.raises(RuntimeError, match="judge went away"):
        judge_run(run, _Judge(fail_after=2), JUDGE)  # type: ignore[arg-type]
    [stopped] = attempts(run)
    assert stopped.manifest["status"] == "running" and len(stopped.records()) == 2

    judge = _Judge()
    resumed = judge_run(run, judge, JUDGE)  # type: ignore[arg-type]
    assert resumed.ref == stopped.ref and judge.calls == 1
    assert resumed.manifest["status"] == "complete"


def test_a_generation_failure_is_recorded_as_a_failure_without_a_judge_call(tmp_path: Path) -> None:
    run = _run(tmp_path, _answers(1))
    lines = run.outputs_path.read_text().splitlines()
    record = json.loads(lines[1])
    record["generation_failure"] = "deadline"
    run.outputs_path.write_text("\n".join([lines[0], json.dumps(record)]) + "\n")

    judge = _Judge()
    attempt = judge_run(run, judge, JUDGE)  # type: ignore[arg-type]
    [verdict] = attempt.records()["s0"]["verdicts"]
    assert judge.calls == 0 and verdict["passed"] is False and "generation failed" in verdict["judge_output"]


# ---------------------------------------------------------------------------
# Label changes
# ---------------------------------------------------------------------------


def test_a_span_fix_rescores_evidence_offline_with_the_same_judgment(tmp_path: Path) -> None:
    dataset = _answers()
    run = _run(tmp_path, dataset)
    attempt = judge_run(run, _Judge(), JUDGE)  # type: ignore[arg-type]
    before = run.read_scores(rescore(run)[0])

    # No retriever, index or judge is involved from here on.
    fixed = _edited(dataset, "s1", expected_spans=["a span no passage holds"])
    revision, changes = rescore(run, fixed)
    after = run.read_scores(revision)

    assert changes == {"s1": ["expected_spans"]}
    assert after["settings"]["judgment"]["attempt_id"] == attempt.attempt_id
    assert [s["evidence_retrieved"] for s in before["samples"]] == [True, True, True]
    assert [s["evidence_retrieved"] for s in after["samples"]] == [True, False, True]
    assert after["num_passed"] == before["num_passed"]


def test_a_changed_reference_refuses_the_old_judgment_until_judged_again(tmp_path: Path) -> None:
    dataset = _answers()
    run = _run(tmp_path, dataset)
    old = judge_run(run, _Judge(), JUDGE)  # type: ignore[arg-type]
    fixed = _edited(dataset, "s2", expected_answer="a corrected reference")

    with pytest.raises(ReferencesChanged):
        rescore(run, fixed)
    with pytest.raises(ReferencesChanged):
        rescore(run, fixed, judgment=old.ref)

    new = judge_run(run, _Judge(), JUDGE, fixed)  # type: ignore[arg-type]
    assert new.judge_hash != old.judge_hash  # other references, other protocol
    assert "a corrected reference" in new.records()["s2"]["verdicts"][0]["prompt"]
    rescore(run, fixed)


def test_a_changed_query_needs_a_fresh_run_not_a_judgment(tmp_path: Path) -> None:
    dataset = _answers()
    run = _run(tmp_path, dataset)
    with pytest.raises(InputChanged, match="s0"):
        judge_run(run, _Judge(), JUDGE, _edited(dataset, "s0", query="another question"))  # type: ignore[arg-type]


def test_multihop_parts_are_judged_and_scored_from_stored_passages(tmp_path: Path) -> None:
    run = _run(tmp_path, _multihop(), stage="multihop")
    judge = _Judge("PASS", "FAIL not found", "PASS")
    attempt = judge_run(run, judge, JUDGE)  # type: ignore[arg-type]

    assert [v["label"] for v in attempt.records()["mh"]["verdicts"]] == [
        "A", "B", "The overall comparison or conclusion the question asks for"]
    scores = run.read_scores(rescore(run)[0])
    assert scores["settings"]["evidence"] == "retrieved union"
    assert scores["evidence_recall"] == 0.5  # "text of a" was shown, "text of b" never
    assert (scores["complete_rate"], round(scores["mean_completeness"], 3)) == (0.0, 0.667)


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------


def test_runs_judged_under_different_protocols_cannot_be_paired(tmp_path: Path) -> None:
    dataset = _answers()
    base, cand = _run(tmp_path, dataset), _run(tmp_path, dataset)
    base_rev = rescore(base, judgment=judge_run(base, _Judge(), JUDGE).ref)[0]  # type: ignore[arg-type]
    same = rescore(cand, judgment=judge_run(cand, _Judge(), JUDGE).ref)[0]  # type: ignore[arg-type]
    other = rescore(cand, judgment=judge_run(cand, _Judge(), OTHER_JUDGE).ref)[0]  # type: ignore[arg-type]

    baseline = resolve(f"{base.path}:{base_rev}")
    assert check_pair(baseline, resolve(f"{cand.path}:{same}"), []) == []  # different attempts, same protocol
    with pytest.raises(Ineligible, match="judged under different protocols"):
        check_pair(baseline, resolve(f"{cand.path}:{other}"), [])


# ---------------------------------------------------------------------------
# The answer matrix
# ---------------------------------------------------------------------------


class _CountingChat(ChatService):
    def __init__(self) -> None:
        self.inner = _chat()
        self.asked: list[str] = []

    def ask(self, query: str, **kwargs: Any) -> Any:  # type: ignore[override]
        self.asked.append(query)
        return self.inner.ask(query, **kwargs)


def _matrix(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, chat: ChatService, judge: _Judge, *extra: str) -> int:
    eval_set = tmp_path / "refusals.json"
    if not eval_set.exists():
        eval_set.write_text(json.dumps([s.to_dict() for s in _answers()]))
    monkeypatch.setattr(matrix, "build_chat_service", lambda config, corpora=None, retriever=None: chat)
    monkeypatch.setattr(matrix, "get_llm_client", lambda config: judge)
    skip_index_checks(monkeypatch, matrix, verified=True)
    monkeypatch.setattr(sys, "argv", [
        "run_answer_matrix.py", "--family", "m19", "--variant", "pipeline / 9b", "--sets", "refusals",
        "--refusals", str(eval_set), "--results-dir", str(tmp_path / "results"),
        "--runs-dir", str(tmp_path / "runs"), *extra,
    ])
    return matrix.main()


def _rows(tmp_path: Path) -> list[dict[str, Any]]:
    [results] = (tmp_path / "results").glob("*.json")
    rows: list[dict[str, Any]] = json.loads(results.read_text())["results"]
    return rows


def test_a_matrix_row_is_the_score_revision_it_names(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    assert _matrix(monkeypatch, tmp_path, _CountingChat(), _Judge("FAIL no")) == 0

    [row] = _rows(tmp_path)
    saved = row["refusals"]
    run = Run(tmp_path / "runs" / saved["run"], json.loads((tmp_path / "runs" / saved["run"] / "manifest.json").read_text()))
    revision = run.read_scores(saved["score_revision"])
    assert {k: v for k, v in revision.items() if k not in REVISION_FIELDS} == {
        k: v for k, v in saved.items()
        if k not in {"elapsed_s", "resumed_samples", "run", "judgment", "score_revision", "provenance"}
    }
    assert saved["judgment"] == f"{revision['settings']['judge_hash']}/{revision['settings']['judgment']['attempt_id']}"
    assert (saved["num_passed"], saved["num_evaluated"]) == (2, 3)


def test_a_rerun_after_judging_stopped_judges_instead_of_regenerating(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with pytest.raises(RuntimeError, match="judge went away"):
        _matrix(monkeypatch, tmp_path, _CountingChat(), _Judge(fail_after=1))

    chat, judge = _CountingChat(), _Judge()
    assert _matrix(monkeypatch, tmp_path, chat, judge) == 0
    assert chat.asked == [] and judge.calls == 2  # answers kept; the two unjudged samples judged
    assert _rows(tmp_path)[0]["refusals"]["num_evaluated"] == 3


def test_repeats_record_their_group_and_number(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    assert _matrix(monkeypatch, tmp_path, _CountingChat(), _Judge(), "--repeat", "2") == 0

    manifests = [json.loads(p.read_text()) for p in sorted((tmp_path / "runs").glob("*/manifest.json"))]
    assert sorted((m["variant"], m["extra"]["repeat"]["number"]) for m in manifests) == [
        ("pipeline / 9b #1", 1), ("pipeline / 9b #2", 2)]
    assert {m["extra"]["repeat"]["group"] for m in manifests} == {"pipeline / 9b"}


def test_the_answer_eval_cli_writes_a_judged_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from rag.eval import answer_eval, judge

    eval_set = tmp_path / "set.json"
    eval_set.write_text(json.dumps([s.to_dict() for s in _answers()] + [{"id": "unjudgeable", "query": "q"}]))
    monkeypatch.setattr(answer_eval, "build_chat_service", lambda config, corpora=None, retriever=None: _chat())
    monkeypatch.setattr(answer_eval, "get_llm_client", lambda config: _Judge("FAIL no"))
    skip_index_checks(monkeypatch, judge, verified=True)

    assert answer_eval.main(["--eval-set", str(eval_set), "--runs-dir", str(tmp_path / "runs"),
                             "--judge-model", "judge-a"]) == 0

    [run_dir] = (tmp_path / "runs").iterdir()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["stage"] == "answer" and manifest["dataset"]["selected_ids"] == ["s0", "s1", "s2"]
    assert "Pass rate   0.667  (2/3)" in capsys.readouterr().out
