"""The optional MLflow export (`rag.eval.tracking`), against a throwaway local store.

What these protect: an exported revision carries our identities and one
trace per sample with its expectations and scores; re-exporting is a no-op
and a new score revision is a new MLflow run beside the old one; a partial
export is cleaned up rather than duplicated; and an exported comparison
holds `rag.eval.compare`'s numbers, with a pair the harness refuses
exporting nothing; and an agent turn exports each LLM call verbatim at its
recorded time, with its reconstructed prompt and a Markdown transcript. Skipped when MLflow (the ``tracking`` extra) is absent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("mlflow")

from rag.agent.service import AgentService  # noqa: E402
from rag.config.settings import LLMConfig, RagConfig  # noqa: E402
from rag.eval import tracking  # noqa: E402
from rag.eval.answer_eval import generate_into_run  # noqa: E402
from rag.eval.compare import compare  # noqa: E402
from rag.eval.dataset import EvalDataset  # noqa: E402
from rag.eval.judge import judge_run  # noqa: E402
from rag.eval.report import resolve  # noqa: E402
from rag.eval.runs import Run, open_run  # noqa: E402
from rag.eval.score import rescore  # noqa: E402
from rag.llm.base import AssistantTurn, ToolCall  # noqa: E402
from rag.observability.usage import metered_client  # noqa: E402
from tests.fakes import ScriptedToolLLM  # noqa: E402
from tests.test_agent import _chunk, _FakeTools  # noqa: E402
from tests.test_eval_compare import pair, retrieval_run  # noqa: E402
from tests.test_eval_judge import JUDGE, _Judge  # noqa: E402
from tests.test_eval_runs import VERIFIED, _dataset  # noqa: E402
from tests.test_eval_score_report import _edited  # noqa: E402


@pytest.fixture
def store(tmp_path: Path) -> tracking.Store:
    return tracking.Store(tmp_path / "mlflow")


def _traces(store: tracking.Store, exported: tracking.Exported) -> list[Any]:
    run = store.client.get_run(exported.run_id)
    return list(store.mlflow.search_traces(locations=[run.info.experiment_id], run_id=exported.run_id,
                                           return_type="list"))


def test_a_revision_exports_with_our_identities_and_a_trace_per_sample(tmp_path: Path, store: tracking.Store) -> None:
    run = retrieval_run(tmp_path / "runs", _dataset())
    pinned = resolve(str(run.path))
    exported = tracking.export_run(store, pinned)

    assert exported.created and exported.experiment == "retrieval/set"
    mlflow_run = store.client.get_run(exported.run_id)
    tags = mlflow_run.data.tags
    assert tags["rag.key"] == pinned.ref and tags["rag.export_status"] == "complete"
    assert tags["rag.dataset_hash"] == pinned.scores["dataset_hash"] and tags["rag.index_build_id"] == "build-1"
    assert mlflow_run.data.metrics["metrics.hit_rate"] == pinned.scores["metrics"]["hit_rate"]
    assert mlflow_run.data.params["retrieval.top_k"] == "20"
    assert "Traces" in store.client.get_experiment(mlflow_run.info.experiment_id).tags[tracking.NOTE]

    traces = _traces(store, exported)
    assert sorted(t.info.tags["rag.sample_id"] for t in traces) == [f"q{i}" for i in range(6)]
    q2 = next(t for t in traces if t.info.tags["rag.sample_id"] == "q2")
    assessments = {a.name: a for a in q2.info.assessments}
    assert assessments["hit_rate"].feedback.value == 1.0
    assert assessments["expected_spans"].expectation.value == ["fact 2"]
    assert q2.data.spans[0].outputs[0]["metadata"]["doc_uri"] == "doc2"


def test_reexport_is_a_noop_and_a_rescore_is_a_new_run(tmp_path: Path, store: tracking.Store) -> None:
    dataset = _dataset()
    run = retrieval_run(tmp_path / "runs", dataset)
    first = tracking.export_run(store, resolve(str(run.path)))
    again = tracking.export_run(store, resolve(str(run.path)))
    assert (again.run_id, again.created) == (first.run_id, False)
    assert len(_traces(store, first)) == 6

    revision, _ = rescore(run, _edited(dataset, "q2", expected_spans=["a fact nobody wrote"]))
    rescored = tracking.export_run(store, resolve(f"{run.path}:{revision}"))
    assert rescored.created and rescored.run_id != first.run_id
    assert store.client.get_run(first.run_id).data.tags["rag.export_status"] == "complete"


def test_a_partial_export_is_replaced_not_duplicated(tmp_path: Path, store: tracking.Store) -> None:
    pinned = resolve(str(retrieval_run(tmp_path / "runs", _dataset()).path))
    first = tracking.export_run(store, pinned)
    store.client.set_tag(first.run_id, tracking.STATUS, "exporting")  # as if interrupted

    redone = tracking.export_run(store, pinned)
    experiment_id = store.experiment(redone.experiment)
    assert redone.created and redone.run_id != first.run_id
    assert [r.info.run_id for r in store.client.search_runs([experiment_id])] == [redone.run_id]
    assert len(_traces(store, redone)) == 6


def test_a_comparison_exports_the_harness_numbers(tmp_path: Path, store: tracking.Store) -> None:
    base, cand = pair(tmp_path / "runs")
    exported = tracking.export_comparison(store, compare(base, cand))
    assert exported.experiment == "comparisons/retrieval/set"
    assert "no traces" in store.client.get_experiment_by_name(exported.experiment).tags[tracking.NOTE]
    run = store.client.get_run(exported.run_id)
    m = run.data.metrics
    assert (m["hit_rate.n"], m["hit_rate.wins"], m["hit_rate.losses"]) == (6, 1, 2)
    assert m["hit_rate.mcnemar_p"] == compare(base, cand).metrics[0].difference.p_value
    assert run.data.tags["rag.baseline"] == base.ref and "compareToRunUuid=" in run.data.tags["mlflow.note.content"]
    assert tracking.export_comparison(store, compare(base, cand)).created is False


def test_a_refused_pair_exports_nothing(tmp_path: Path) -> None:
    dataset = _dataset()
    base = retrieval_run(tmp_path / "runs", dataset)
    other = retrieval_run(tmp_path / "runs", dataset, variant="other")
    relabeled, _ = rescore(other, _edited(dataset, "q2", expected_spans=["a fact nobody wrote"]))
    code = tracking.main(["--runs-dir", str(tmp_path / "runs"), "--tracking-dir", str(tmp_path / "mlflow"),
                          "compare", "--baseline", str(base.path), "--candidate", f"{other.path}:{relabeled}"])
    assert code == 1 and not (tmp_path / "mlflow").exists()


def _agent_run(tmp_path: Path) -> Run:
    """A judged answer run of one agent turn: a search, then the answer; the agent on a qwen3.8 model."""
    config = RagConfig()
    agent_llm = LLMConfig(model="qwen3.8:27b-mlx", think="low")
    config = config.model_copy(update={"agent": config.agent.model_copy(update={"llm": agent_llm})})
    dataset = EvalDataset.from_dicts([{"id": "s0", "query": "Which airline?", "expected_answer": "Delta"}])
    run = open_run(
        tmp_path, stage="answer", experiment="test", variant="agent", config=config,
        selection=config.corpus_selection(None), dataset_path=Path("data/eval/set.json"), dataset=dataset,
        selected=dataset, selection_method="all", index=VERIFIED, code="abc",
    )
    llm = ScriptedToolLLM([
        AssistantTurn(content="", tool_calls=(ToolCall(name="rag_search", arguments={"query": "fuel"}),)),
        AssistantTurn(content="Delta, up 67% [1]."),
    ])
    tools = _FakeTools({"fuel": [_chunk("c1", text="Aircraft fuel | 4,109 | 2,458", doc="DAL.md")]})
    agent = AgentService(metered_client(llm), tools, system_prompt="AGENT SYSTEM PROMPT", corpora=["baseline"])
    generate_into_run(run, dataset, agent)  # type: ignore[arg-type]
    judge_run(run, _Judge(), JUDGE)  # type: ignore[arg-type]
    rescore(run)
    return run


def test_an_agent_turn_exports_each_llm_call_and_its_transcript(tmp_path: Path, store: tracking.Store) -> None:
    exported = tracking.export_run(store, resolve(str(_agent_run(tmp_path / "runs").path)))
    [trace] = _traces(store, exported)
    spans = {s.name: s for s in trace.data.spans}
    first, second, search = spans["chat 1"], spans["chat 2"], spans["rag_search"]

    # OpenAI chat format, which MLflow's trace view renders as a conversation.
    assert first.inputs["messages"][0]["role"] == "system"
    assert "AGENT SYSTEM PROMPT" in first.inputs["messages"][0]["content"]
    [call] = first.outputs["choices"][0]["message"]["tool_calls"]
    [result] = [m for m in second.inputs["messages"] if m["role"] == "tool"]
    assert call["function"]["name"] == "rag_search" and result["tool_call_id"] == call["id"]
    assert "Aircraft fuel | 4,109" in result["content"]
    assert first.attributes["rendered_prompt (reconstructed)"].startswith("```text\n<|im_start|>system\n")
    # At their recorded times: the search after the call that asked for it, before the next.
    assert first.end_time_ns <= search.start_time_ns <= second.start_time_ns

    path = store.client.download_artifacts(exported.run_id, trace.info.tags["rag.transcript"], str(tmp_path))
    transcript = Path(path).read_text(encoding="utf-8")
    assert transcript.startswith("# Trace: s0") and "## Expected answer" in transcript
    assert "#### Rendered prompt, reconstructed from the messages" in transcript
