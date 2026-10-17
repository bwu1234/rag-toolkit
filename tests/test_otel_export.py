"""The eval-run OTLP export (`rag.eval.otel_export`), against the SDK's in-memory exporter.

Run folders are built by the harness itself (`open_run`, the runners, the
judge, `rescore`), so the export reads exactly what a real run stores.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path

import pytest
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from rag.agent.service import AgentService
from rag.config.settings import RagConfig
from rag.eval import otel_export
from rag.eval.answer_eval import generate_into_run
from rag.eval.dataset import EvalDataset
from rag.eval.judge import judge_run
from rag.eval.report import resolve
from rag.eval.runs import Run, open_run
from rag.eval.score import rescore
from rag.events import emit
from rag.llm.base import AssistantTurn, ToolCall
from rag.observability.otel import TraceWriter
from rag.observability.usage import metered_client
from rag.retrieval.retriever import RetrievalResult
from tests.fakes import ScriptedToolLLM
from tests.test_eval_compare import retrieval_run
from tests.test_eval_judge import JUDGE, _Judge
from tests.test_eval_runs import VERIFIED, _dataset, _Retriever
from tests.test_otel import _StagedTools


class _StagedRetriever(_Retriever):
    """The harness's fake retriever, emitting a timed embed first as `Retriever` does."""

    def retrieve(self, query: str, *, query_filter: object = None, on_event: object = None) -> RetrievalResult:
        emit(on_event, time.monotonic(), "embed", "embedded")  # type: ignore[arg-type]
        return super().retrieve(query, query_filter=query_filter, on_event=on_event)


def _writer(*, capture_content: bool = False) -> tuple[TraceWriter, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    return TraceWriter(SimpleSpanProcessor(exporter), service_name="eval", capture_content=capture_content), exporter


def _agent_run(tmp_path: Path) -> Run:
    """A judged answer run of one agent turn: a search, then the answer."""
    config = RagConfig()
    dataset = EvalDataset.from_dicts([{"id": "s0", "query": "Which airline?", "expected_answer": "Delta"}])
    run = open_run(
        tmp_path, stage="answer", experiment="test", variant="agent", config=config,
        selection=config.corpus_selection(None), dataset_path=Path("data/eval/set.json"), dataset=dataset,
        selected=dataset, selection_method="all", index=VERIFIED, code="abc",
    )
    llm = ScriptedToolLLM([
        AssistantTurn(content="", tool_calls=(ToolCall(name="rag_search", arguments={"query": "fuel"}),)),
        AssistantTurn(content="Delta [1]."),
    ])
    agent = AgentService(metered_client(llm), _StagedTools(), system_prompt="SYSTEM", corpora=["baseline"],  # type: ignore[arg-type]
                         turn_metadata={"mode": "agentic:react", "llm": "ollama:agent-27b"})
    generate_into_run(run, dataset, agent)  # type: ignore[arg-type]
    judge_run(run, _Judge("PASS names Delta"), JUDGE)  # type: ignore[arg-type]
    rescore(run)
    return run


def _roots(spans: Sequence[ReadableSpan]) -> dict[str, ReadableSpan]:
    return {str((s.attributes or {})["rag.sample_id"]): s for s in spans if s.parent is None}


def test_a_retrieval_revision_is_a_trace_per_sample_with_its_scores_and_stages(tmp_path: Path) -> None:
    pinned = resolve(str(retrieval_run(tmp_path / "runs", _dataset(), retriever=_StagedRetriever()).path))
    writer, exporter = _writer()

    assert otel_export.export_revision(writer, pinned) == 6

    spans = exporter.get_finished_spans()
    roots = _roots(spans)
    assert sorted(roots) == [f"q{i}" for i in range(6)]
    q2 = roots["q2"].attributes or {}
    assert q2["rag.eval.hit_rate"] == 1.0 and q2["rag.run_id"] == pinned.run.run_id
    assert q2["rag.score_revision"] == pinned.revision and q2["rag.index_build_id"] == "build-1"
    assert list(q2["rag.eval.expected_doc_ids"]) == ["doc2"]  # type: ignore[arg-type]
    assert "rag.query" not in q2 and "rag.eval.expected_spans" not in q2
    assert "rag.key" not in q2  # MLflow's idempotency tag means nothing here
    # Stages say what they are, as they do inside an agent's search.
    embeds = [s for s in spans if s.name == "embed"]
    assert embeds and all((s.attributes or {})["gen_ai.operation.name"] == "embeddings" for s in embeds)
    # Each root's stages inside it, and the samples one after another.
    for root in roots.values():
        children = [s for s in spans if s.parent is not None and s.parent.span_id == root.context.span_id]
        assert all(root.start_time <= c.start_time <= c.end_time <= root.end_time for c in children)  # type: ignore[operator]


def test_exporting_a_revision_again_sends_the_same_ids(tmp_path: Path) -> None:
    pinned = resolve(str(retrieval_run(tmp_path / "runs", _dataset()).path))
    first, second = _writer(), _writer()
    otel_export.export_revision(first[0], pinned)
    otel_export.export_revision(second[0], pinned)

    def ids(exporter: InMemorySpanExporter) -> list[tuple[int, int, int | None]]:
        return sorted((s.context.trace_id, s.context.span_id, s.parent.span_id if s.parent else None)
                      for s in exporter.get_finished_spans() if s.context)

    assert ids(first[1]) == ids(second[1])
    assert len({trace for trace, _, _ in ids(first[1])}) == 6
    # Other settings draw other spans, so they must not reuse the ids.
    content = _writer(capture_content=True)
    otel_export.export_revision(content[0], pinned)
    assert not {t for t, _, _ in ids(first[1])} & {t for t, _, _ in ids(content[1])}


def test_an_answer_revision_is_the_stored_turn_with_its_verdict(tmp_path: Path) -> None:
    pinned = resolve(str(_agent_run(tmp_path / "runs").path))
    writer, exporter = _writer()
    otel_export.export_revision(writer, pinned)

    spans = exporter.get_finished_spans()
    root = _roots(spans)["s0"]
    assert root.name == "invoke_agent rag"
    assert (root.attributes or {})["rag.eval.passed"] is True
    assert {"execute_tool rag_search", "embed", "rerank", "chat agent-27b"} <= {s.name for s in spans}
    assert not any(key in (s.attributes or {}) for s in spans
                   for key in ("rag.eval.judge_reply", "rag.eval.expected_answer", "rag.answer"))


def test_capture_content_adds_the_judge_reply_and_the_expected_answer(tmp_path: Path) -> None:
    pinned = resolve(str(_agent_run(tmp_path / "runs").path))
    writer, exporter = _writer(capture_content=True)
    otel_export.export_revision(writer, pinned)

    root = _roots(exporter.get_finished_spans())["s0"].attributes or {}
    assert root["rag.eval.expected_answer"] == "Delta" and root["rag.answer"] == "Delta [1]."
    assert "names Delta" in str(root["rag.eval.judge_reply"])


class _Refusing(SpanExporter):
    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        return SpanExportResult.FAILURE

    def shutdown(self) -> None:
        pass


def test_the_cli_reports_failed_exports_and_needs_an_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run = retrieval_run(tmp_path / "runs", _dataset())
    monkeypatch.setattr(otel_export, "otlp_exporter", lambda endpoint, protocol: _Refusing())

    code = otel_export.main([str(run.path), "--endpoint", "http://127.0.0.1:9/v1/traces"])

    assert code == 1
    assert "failed to export" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        otel_export.main([str(run.path), "--config", str(tmp_path / "no-otel.yaml")])


def test_the_key_holds_across_processes() -> None:
    # The SDK gives each process a random service.instance.id; a key holding it
    # would make every export new traces, duplicating them in the backend.
    import subprocess
    import sys

    code = ("from opentelemetry.sdk.trace.export import SimpleSpanProcessor as P;"
            "from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter as E;"
            "from rag.observability.otel import TraceWriter as W;"
            "print(W(P(E()), service_name='eval').identity)")
    runs = [subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
            for _ in range(2)]
    assert runs[0] == runs[1] and "service.instance.id" not in runs[0]
