"""Tests for per-sample eval checkpoints and resume.

What these protect: a resumed run must produce the same report a clean run
would, must never re-ask a finished sample, and must refuse to mix samples
from two different setups into one number.
"""

from __future__ import annotations

from typing import Any

import json
import sys
from pathlib import Path

import pytest

from rag.eval.answer_eval import run_answer_eval, trajectory
from rag.eval.checkpoint import CheckpointMismatch, SampleCheckpoint
from rag.eval.dataset import EvalDataset
from rag.eval.multihop_eval import MultihopSampleResult, run_multihop_eval
from rag.generation.chat_service import ChatAnswer, ChatService
from rag.observability.records import AgentToolCall
from tests.fakes import skip_index_checks

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import run_answer_matrix  # noqa: E402

FINGERPRINT = {"config": "a", "judge": "b", "dataset": "c", "corpus": "edgar", "code": "d"}


class _CountingChat(ChatService):
    """Answers every query, recording each; raises on the query named in `fail_on`."""

    def __init__(self, fail_on: str | None = None) -> None:
        self.asked: list[str] = []
        self._fail_on = fail_on

    def ask(self, query: str, **_: Any) -> ChatAnswer:  # type: ignore[override]
        if query == self._fail_on:
            raise RuntimeError("Ollama went away")
        self.asked.append(query)
        return ChatAnswer(answer=f"answer to {query}", citations=[], llm_calls=1, prompt_tokens=10)


class _PassJudge:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        self.calls += 1
        return "PASS fine"


def _multihop(n: int) -> EvalDataset:
    return EvalDataset.from_dicts([
        {
            "id": f"mh-{i}", "query": f"q{i}", "expected_answer": "A: 1; B: 2", "kind": "cross_company",
            "parts": [{"label": "A", "answer": "1", "spans": ["a1"]},
                      {"label": "B", "answer": "2", "spans": ["b2"]}],
        }
        for i in range(n)
    ])


# ---------------------------------------------------------------------------
# The checkpoint file
# ---------------------------------------------------------------------------


def test_appended_samples_load_back_by_id(tmp_path: Path) -> None:
    checkpoint = SampleCheckpoint(tmp_path / "p" / "c.jsonl", FINGERPRINT)

    checkpoint.append({"sample_id": "x", "passed": True})
    checkpoint.append({"sample_id": "y", "passed": False})

    assert checkpoint.load() == {"x": {"sample_id": "x", "passed": True},
                                 "y": {"sample_id": "y", "passed": False}}
    header = json.loads(checkpoint.path.read_text().splitlines()[0])
    assert header == {"fingerprint": FINGERPRINT}


def test_no_checkpoint_loads_as_nothing_done(tmp_path: Path) -> None:
    assert SampleCheckpoint(tmp_path / "none.jsonl", FINGERPRINT).load() == {}


def test_a_checkpoint_from_a_different_setup_is_refused_naming_what_changed(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    SampleCheckpoint(path, FINGERPRINT).append({"sample_id": "x"})

    with pytest.raises(CheckpointMismatch, match=r"changed: code, judge"):
        SampleCheckpoint(path, {**FINGERPRINT, "judge": "other", "code": "new"}).load()


def test_a_truncated_last_line_is_dropped_and_the_next_append_starts_clean(tmp_path: Path) -> None:
    checkpoint = SampleCheckpoint(tmp_path / "c.jsonl", FINGERPRINT)
    checkpoint.append({"sample_id": "x"})
    with checkpoint.path.open("a") as f:
        f.write('{"sample_id": "y", "ans')  # killed mid-write

    assert list(checkpoint.load()) == ["x"]
    checkpoint.append({"sample_id": "z"})
    assert list(checkpoint.load()) == ["x", "z"]


def test_a_corrupt_line_before_the_end_is_an_error_not_a_skip(tmp_path: Path) -> None:
    checkpoint = SampleCheckpoint(tmp_path / "c.jsonl", FINGERPRINT)
    checkpoint.append({"sample_id": "x"})
    with checkpoint.path.open("a") as f:
        f.write("garbage\n")
    checkpoint.append({"sample_id": "z"})

    with pytest.raises(CheckpointMismatch, match="corrupt"):
        checkpoint.load()


# ---------------------------------------------------------------------------
# Runner hooks
# ---------------------------------------------------------------------------


def test_multihop_result_survives_a_json_round_trip() -> None:
    report = run_multihop_eval(_multihop(1), _CountingChat(), _PassJudge())
    original = report.sample_results[0]

    restored = MultihopSampleResult.from_dict(json.loads(json.dumps(original.to_dict())))

    assert restored == original


def test_an_agent_trace_survives_the_checkpoint_and_lands_in_the_matrix_record() -> None:
    searches = [
        AgentToolCall(step=1, query="fuel", filters={"equals": {"ticker": "DAL"}},
                    passages=[1, 2], new_passages=[1, 2], chunk_ids=["a", "b"]),
        AgentToolCall(step=2, query="fuel", status="refused", note="Already searched"),
    ]

    class _Agent(_CountingChat):
        def ask(self, query: str, **_: Any) -> ChatAnswer:  # type: ignore[override]
            return ChatAnswer(answer="a", agent_calls=searches)

    multihop = run_multihop_eval(_multihop(1), _Agent(), _PassJudge()).sample_results[0]
    answer = run_answer_eval(
        EvalDataset.from_dicts([{"id": "s", "query": "q", "expected_answer": "a"}]), _Agent(), _PassJudge()
    ).sample_results[0]

    for result in (multihop, answer):
        restored = type(result).from_dict(json.loads(json.dumps(result.to_dict())))
        assert restored == result and restored.agent_calls == searches
    record = trajectory(multihop.agent_calls)
    assert record["calls"][0]["query"] == "fuel" and record["calls"][1]["status"] == "refused"
    assert trajectory([]) == {}


def test_multihop_reuses_completed_samples_without_asking_or_judging_them() -> None:
    dataset = _multihop(3)
    first = run_multihop_eval(dataset, _CountingChat(), _PassJudge())
    completed = {r.sample_id: r for r in first.sample_results[:2]}
    chat, judge, fresh = _CountingChat(), _PassJudge(), []

    report = run_multihop_eval(dataset, chat, judge, completed=completed, on_result=fresh.append)

    assert chat.asked == ["q2"]
    assert judge.calls == 2  # the one new sample's two parts
    assert [r.sample_id for r in fresh] == ["mh-2"]
    assert [r.sample_id for r in report.sample_results] == ["mh-0", "mh-1", "mh-2"]
    assert report.sample_results[:2] == first.sample_results[:2]  # reused as saved
    # Scores match a clean run's; latency is re-measured, so it can't.
    assert (report.complete_rate, report.evidence_recall, report.mean_llm_calls) == (
        first.complete_rate, first.evidence_recall, first.mean_llm_calls)


def test_answer_eval_reuses_completed_samples_and_reports_each_new_one() -> None:
    dataset = EvalDataset.from_dicts([
        {"id": f"s{i}", "query": f"q{i}", "expected_answer": "A."} for i in range(3)
    ])
    first = run_answer_eval(dataset, _CountingChat(), _PassJudge())
    chat, fresh = _CountingChat(), []

    report = run_answer_eval(
        dataset, chat, _PassJudge(),
        completed={"s0": first.sample_results[0]}, on_result=fresh.append,
    )

    assert chat.asked == ["q1", "q2"]
    assert [r.sample_id for r in fresh] == ["s1", "s2"]
    assert report.sample_results[0] == first.sample_results[0]
    assert (report.num_evaluated, report.num_passed) == (first.num_evaluated, first.num_passed)


# ---------------------------------------------------------------------------
# The answer matrix, end to end
# ---------------------------------------------------------------------------


def _run_matrix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, chat: ChatService, *extra: str
) -> int:
    dataset_path = tmp_path / "multihop.json"
    if not dataset_path.exists():
        dataset_path.write_text(json.dumps([s.to_dict() for s in _multihop(3)]))
    monkeypatch.setattr(run_answer_matrix, "build_chat_service", lambda config, corpora=None: chat)
    monkeypatch.setattr(run_answer_matrix, "get_llm_client", lambda config: _PassJudge())
    skip_index_checks(monkeypatch, run_answer_matrix, verified=True)
    monkeypatch.setattr(sys, "argv", [
        "run_answer_matrix.py", "--sets", "multihop", "--variant", "crag=off",
        "--multihop", str(dataset_path), "--results-dir", str(tmp_path / "results"),
        "--runs-dir", str(tmp_path / "runs"), *extra,
    ])
    return run_answer_matrix.main()


def _manifests(tmp_path: Path) -> list[dict[str, Any]]:
    return [json.loads(p.read_text()) for p in sorted((tmp_path / "runs").glob("*/manifest.json"))]


def test_matrix_resumes_a_crashed_set_from_its_run_folder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with pytest.raises(RuntimeError, match="went away"):
        _run_matrix(monkeypatch, tmp_path, _CountingChat(fail_on="q2"))

    [crashed] = _manifests(tmp_path)
    assert crashed["status"] == "running"
    assert not list((tmp_path / "results").glob("*.json"))  # no half set in the results file

    chat = _CountingChat()
    assert _run_matrix(monkeypatch, tmp_path, chat) == 0

    assert chat.asked == ["q2"]
    [finished] = _manifests(tmp_path)
    assert finished["run_id"] == crashed["run_id"] and finished["status"] == "complete"
    [results_file] = (tmp_path / "results").glob("*.json")
    saved = json.loads(results_file.read_text())["results"][0]["multihop"]
    assert (saved["num_evaluated"], saved["resumed_samples"], saved["complete_rate"]) == (3, 2, 1.0)
    assert saved["run"] == crashed["run_id"]


def test_matrix_never_resumes_a_run_from_another_setup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with pytest.raises(RuntimeError):
        _run_matrix(monkeypatch, tmp_path, _CountingChat(fail_on="q1"))
    monkeypatch.setattr(run_answer_matrix, "code_version", lambda: "older")

    chat = _CountingChat()
    assert _run_matrix(monkeypatch, tmp_path, chat) == 0
    assert chat.asked == ["q0", "q1", "q2"]  # a new run; the other setup's is left as it was
    assert sorted(m["status"] for m in _manifests(tmp_path)) == ["complete", "running"]


def test_fresh_starts_a_new_run_instead_of_resuming(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    with pytest.raises(RuntimeError):
        _run_matrix(monkeypatch, tmp_path, _CountingChat(fail_on="q1"))

    chat = _CountingChat()
    assert _run_matrix(monkeypatch, tmp_path, chat, "--fresh") == 0
    assert chat.asked == ["q0", "q1", "q2"]


def test_matrix_runs_a_tier_in_full_and_counts_each_kind(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tier = tmp_path / "underspecified.json"
    tier.write_text(json.dumps([
        {"id": f"u{i}", "query": f"q{i}", "expected_answer": "x", "tier": "underspecified",
         "kind": "implicit" if i % 2 else "paraphrase"}
        for i in range(4)
    ]))
    monkeypatch.setattr(run_answer_matrix, "build_chat_service", lambda config, corpora=None: _CountingChat())
    monkeypatch.setattr(run_answer_matrix, "get_llm_client", lambda config: _PassJudge())
    skip_index_checks(monkeypatch, run_answer_matrix)
    monkeypatch.setattr(sys, "argv", [
        "run_answer_matrix.py", "--sets", "underspecified", "--variant", "crag=off",
        "--underspecified", str(tier), "--limit", "1", "--results-dir", str(tmp_path / "results"),
    ])

    assert run_answer_matrix.main() == 0

    [results_file] = (tmp_path / "results").glob("*.json")
    saved = json.loads(results_file.read_text())["results"][0]
    assert "answerable" not in saved
    run = saved["underspecified"]
    assert run["num_evaluated"] == 4, "--limit subsamples the answerable set only"
    assert {k: v["num_evaluated"] for k, v in run["by_kind"].items()} == {"implicit": 2, "paraphrase": 2}
    assert run["pass_ci"][1] == pytest.approx(1.0)
