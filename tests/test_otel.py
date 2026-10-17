"""The OpenTelemetry turn sink (`rag.observability.otel`), against the SDK's in-memory exporter.

Turns run through the real `AgentService`/`ChatService` with scripted models,
so the spans are drawn from the timings a live turn records. The fakes sleep a
few milliseconds per stage so that which span contains which is decided by
the clock, as it is in production, not by ties.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path

import pytest
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from rag.agent.service import AgentService
from rag.config.settings import CorpusSelection, OtelExportConfig, RagConfig, TurnLogConfig
from rag.events import EventSink, emit
from rag.generation.chat_service import ChatService
from rag.llm.base import AssistantTurn, LLMClient, LLMUsage, Message, ToolCall, ToolDefinition
from rag.observability.factory import get_turn_sink
from rag.observability.otel import OtelTurnSink
from rag.observability.records import AgentToolCall, FeedbackRecord, StageEvent, TurnRecord
from rag.observability.sink import FanoutTurnSink, JsonlTurnSink, TurnSink, read_turn_log
from rag.observability.usage import MeteredLLMClient, metered_client
from rag.retrieval.retriever import RetrievalResult
from rag.tools import RagTools
from rag.vectorstore.base import ScoredChunk
from tests.fakes import ScriptedToolLLM

_PAUSE_S = 0.003


def _chunk(chunk_id: str) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk_id, text=f"text of {chunk_id}", document_id=f"{chunk_id}-doc",
        source=Path(f"{chunk_id}.md"), doc_type="markdown", score=0.9,
    )


class _SlowScriptedLLM(ScriptedToolLLM):
    def chat(self, messages: Sequence[Message], tools: Sequence[ToolDefinition] = ()) -> AssistantTurn:
        time.sleep(_PAUSE_S)
        return super().chat(messages, tools)


class _StagedTools(RagTools):
    """`RagTools` whose search emits a timed embed and rerank, like `Retriever` does."""

    def __init__(self) -> None:
        super().__init__(config=RagConfig())

    def retrieve(  # type: ignore[override]
        self, query: str, corpus: str | Sequence[str] | None = None, top_k: int | None = None, *,
        query_filter: object = None, on_event: EventSink | None = None,
    ) -> tuple[CorpusSelection, RetrievalResult]:
        chunks = [_chunk("a"), _chunk("b")]
        for stage in ("embed", "rerank"):
            start = time.monotonic()
            time.sleep(_PAUSE_S)
            emit(on_event, start, stage, f"{stage} {query!r}", hits=chunks if stage == "rerank" else None)
        return self.config.corpus_selection(None), RetrievalResult(chunks=chunks, candidate_count=2)


class _SlowLLM(LLMClient):
    def __init__(self, reply: str) -> None:
        self.reply = reply

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        return self.generate_with_usage(prompt, system=system)[0]

    def generate_with_usage(self, prompt: str, *, system: str | None = None) -> tuple[str, LLMUsage | None]:
        time.sleep(_PAUSE_S)
        return self.reply, LLMUsage(100, 20)


class _Retriever:
    def retrieve(
        self, query: str, *, query_filter: object = None, on_event: EventSink | None = None
    ) -> RetrievalResult:
        start = time.monotonic()
        time.sleep(_PAUSE_S)
        chunks = [_chunk("a"), _chunk("b")]
        emit(on_event, start, "rerank", "reranked", hits=chunks)
        return RetrievalResult(chunks=chunks, candidate_count=2)


def _sink(*, capture_content: bool = False) -> tuple[OtelTurnSink, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    sink = OtelTurnSink(SimpleSpanProcessor(exporter), service_name="test", capture_content=capture_content)
    return sink, exporter


def _agentic_turn(sink: TurnSink) -> str:
    llm = _SlowScriptedLLM([
        AssistantTurn(content="", tool_calls=(ToolCall(name="rag_search", arguments={"query": "q1"}),),
                      usage=LLMUsage(100, 10)),
        AssistantTurn(content="It is so [1].", usage=LLMUsage(200, 20)),
    ])
    agent = AgentService(
        metered_client(llm), _StagedTools(), system_prompt="SYSTEM", corpora=["baseline"],  # type: ignore[arg-type]
        turn_sink=sink, turn_metadata={"mode": "agentic:react", "llm": "ollama:agent-27b"},
    )
    answer = agent.ask("what is it?")
    assert answer.turn_id is not None
    return answer.turn_id


def _by_name(spans: Sequence[ReadableSpan]) -> dict[str, list[ReadableSpan]]:
    named: dict[str, list[ReadableSpan]] = {}
    for span in spans:
        named.setdefault(span.name, []).append(span)
    return named


def _parent(span: ReadableSpan, spans: Sequence[ReadableSpan]) -> str | None:
    assert span.parent is not None
    return next(s.name for s in spans if s.context is not None and s.context.span_id == span.parent.span_id)


def test_an_agentic_turn_is_one_trace_keyed_by_the_turn_id_with_calls_and_stages_nested() -> None:
    sink, exporter = _sink()
    turn_id = _agentic_turn(sink)

    spans = exporter.get_finished_spans()
    named = _by_name(spans)
    [root] = named["invoke_agent rag"]
    assert {s.context.trace_id for s in spans if s.context} == {int(turn_id, 16)}
    assert root.parent is None
    # Each model call inside the agent step that made it (react answers on a step call too).
    chats = named["chat agent-27b"]
    assert [_parent(c, spans) for c in chats] == ["agent_step", "agent_step"]
    assert len({c.parent.span_id for c in chats if c.parent}) == 2
    assert [c.attributes["gen_ai.usage.input_tokens"] for c in chats if c.attributes] == [100, 200]
    # The search's stages inside the tool span, and the tool's call details on it.
    [tool] = named["execute_tool rag_search"]
    assert _parent(named["embed"][0], spans) == _parent(named["rerank"][0], spans) == "execute_tool rag_search"
    assert tool.attributes is not None
    assert (tool.attributes["rag.tool.status"], tool.attributes["rag.tool.step"]) == ("ran", 1)
    assert list(named["rerank"][0].attributes["rag.hits.chunk_ids"]) == ["a", "b"]  # type: ignore[arg-type, index]
    assert named["embed"][0].attributes["gen_ai.operation.name"] == "embeddings"  # type: ignore[index]
    # OpenInference kinds name what GenAI can't: Phoenix shows a reranker, not UNKNOWN.
    kinds = {s.name: (s.attributes or {}).get("openinference.span.kind") for s in spans}
    assert kinds == {"invoke_agent rag": "AGENT", "agent_step": "CHAIN", "chat agent-27b": "LLM",
                     "execute_tool rag_search": "TOOL", "embed": "EMBEDDING", "rerank": "RERANKER"}
    # Every span within the turn.
    assert root.start_time is not None and root.end_time is not None
    assert all(root.start_time <= s.start_time <= s.end_time <= root.end_time  # type: ignore[operator]
               for s in spans)


def test_no_text_leaves_without_capture_content() -> None:
    sink, exporter = _sink()
    _agentic_turn(sink)

    attributes = [dict(s.attributes or {}) for s in exporter.get_finished_spans()]
    assert not any(key in a for a in attributes
                   for key in ("rag.query", "rag.answer", "gen_ai.input.messages", "rag.message", "rag.tool.query"))
    assert not any("what is it" in str(v) or "q1" in str(v) for a in attributes for v in a.values())


def test_capture_content_sends_the_query_prompts_and_tool_arguments() -> None:
    sink, exporter = _sink(capture_content=True)
    _agentic_turn(sink)

    named = _by_name(exporter.get_finished_spans())
    root = named["invoke_agent rag"][0].attributes or {}
    assert (root["rag.query"], root["rag.answer"]) == ("what is it?", "It is so [1].")
    assert named["execute_tool rag_search"][0].attributes["rag.tool.query"] == "q1"  # type: ignore[index]
    first, second = (json.loads(str(c.attributes["gen_ai.input.messages"])) for c in named["chat agent-27b"])  # type: ignore[index]
    assert first[0] == {"role": "system", "parts": [{"type": "text", "content": "SYSTEM"}]}
    # The tool result answers the call the first step made, by the id it was numbered with.
    [call] = [p for m in second for p in m["parts"] if p["type"] == "tool_call"]
    [result] = [p for m in second for p in m["parts"] if p["type"] == "tool_call_response"]
    assert (call["name"], call["arguments"], result["id"]) == ("rag_search", {"query": "q1"}, call["id"])
    reply = json.loads(str(named["chat agent-27b"][0].attributes["gen_ai.output.messages"]))  # type: ignore[index]
    assert reply[0]["parts"][0]["id"] == call["id"]  # the step's reply names the same call
    assert json.loads(str(root["gen_ai.output.messages"])) == [
        {"role": "assistant", "parts": [{"type": "text", "content": "It is so [1]."}], "finish_reason": "stop"}
    ]


def test_a_pipeline_turn_and_its_feedback_share_a_trace() -> None:
    sink, exporter = _sink()
    service = ChatService(
        retriever=_Retriever(), llm_client=MeteredLLMClient(_SlowLLM("See [1].")), turn_sink=sink,
        turn_metadata={"llm": "ollama:gen-9b"},
    )
    answer = service.ask("q")
    assert answer.turn_id is not None
    sink.record_feedback(FeedbackRecord(turn_id=answer.turn_id, rating="up"))

    spans = exporter.get_finished_spans()
    named = _by_name(spans)
    assert {s.context.trace_id for s in spans if s.context} == {int(answer.turn_id, 16)}
    [completion] = named["text_completion gen-9b"]
    assert _parent(completion, spans) == "generate"
    assert named["invoke_workflow rag"][0].attributes["rag.cited_chunk_ids"] == ("a",)  # type: ignore[index]
    assert named["feedback"][0].attributes["rag.feedback.rating"] == "up"  # type: ignore[index]


def _record(**overrides: object) -> TurnRecord:
    fields: dict[str, object] = {
        "turn_id": "0" * 31 + "1", "timestamp": "2026-10-17T12:00:00.000+00:00", "query": "q",
        "outcome": "answered", "total_ms": 100.0,
    }
    fields.update(overrides)
    return TurnRecord(**fields)  # type: ignore[arg-type]


def test_a_step_whose_calls_and_events_differ_is_drawn_without_call_details() -> None:
    # A call rejected for its filters emits no event, so step 1 has two calls and one tool event.
    sink, exporter = _sink()
    sink.record_turn(_record(
        events=[StageEvent("agent_step", "m", 10.0, ended_ms=10.0), StageEvent("search", "s", 5.0, ended_ms=20.0)],
        agent_calls=[AgentToolCall(step=1, status="error"), AgentToolCall(step=1, query="q")],
    ))

    [tool] = _by_name(exporter.get_finished_spans())["execute_tool rag_search"]
    assert tool.attributes is not None and "rag.tool.status" not in tool.attributes


def test_a_failed_turn_and_a_failed_model_call_are_errors() -> None:
    from rag.observability.records import LLMExchange

    sink, exporter = _sink()
    sink.record_turn(_record(
        outcome="error", error="RuntimeError: boom",
        llm_exchanges=[LLMExchange(kind="generate", started_ms=1.0, elapsed_ms=2.0, error="TimeoutError: slow")],
    ))

    named = _by_name(exporter.get_finished_spans())
    assert named["invoke_workflow rag"][0].status.status_code is StatusCode.ERROR
    [call] = named["text_completion"]
    assert call.status.status_code is StatusCode.ERROR and call.attributes["error.type"] == "TimeoutError"  # type: ignore[index]


def test_stages_without_end_times_are_laid_end_to_end() -> None:
    sink, exporter = _sink()
    sink.record_turn(_record(events=[StageEvent("embed", "e", 10.0), StageEvent("rerank", "r", 30.0)]))

    named = _by_name(exporter.get_finished_spans())
    embed, rerank = named["embed"][0], named["rerank"][0]
    assert rerank.start_time == embed.end_time
    assert _parent(rerank, exporter.get_finished_spans()) == "invoke_workflow rag"


# ---------------------------------------------------------------------------
# Fan-out, the JSONL sink, and the factory
# ---------------------------------------------------------------------------


class _BrokenSink(TurnSink):
    def record_turn(self, record: TurnRecord) -> None:
        raise OSError("down")

    def record_feedback(self, record: FeedbackRecord) -> None:
        raise OSError("down")


def test_fanout_reaches_every_sink_past_a_failure_and_the_jsonl_never_gets_the_transcript(tmp_path: Path) -> None:
    from rag.observability.records import LLMExchange

    otel, exporter = _sink()
    jsonl = JsonlTurnSink(tmp_path / "turns.jsonl")
    fanout = FanoutTurnSink([_BrokenSink(), jsonl, otel])
    assert fanout.wants_exchanges

    with pytest.raises(OSError, match="down"):
        fanout.record_turn(_record(llm_exchanges=[LLMExchange(kind="generate", started_ms=0.0, elapsed_ms=1.0)]))

    [line] = list(read_turn_log(jsonl.path))
    assert line["llm_exchanges"] is None
    assert "text_completion" in _by_name(exporter.get_finished_spans())


def test_factory_adds_the_exporter_beside_the_log_or_alone(tmp_path: Path) -> None:
    pytest.importorskip("opentelemetry.exporter.otlp.proto.http")
    otel = OtelExportConfig(endpoint="http://127.0.0.1:9/v1/traces")

    both = get_turn_sink(TurnLogConfig(path=tmp_path / "t.jsonl", otel=otel))
    alone = get_turn_sink(TurnLogConfig(provider="none", otel=otel))

    assert isinstance(both, FanoutTurnSink)
    assert [type(s) for s in both.sinks] == [JsonlTurnSink, OtelTurnSink]
    assert isinstance(alone, OtelTurnSink)
    assert get_turn_sink(TurnLogConfig(provider="none")) is None
    for sink in (both.sinks[1], alone):
        assert isinstance(sink, OtelTurnSink)
        sink.shutdown()


def test_a_sink_that_wants_no_exchanges_gets_none() -> None:
    seen: list[TurnRecord] = []

    class _Plain(TurnSink):
        def record_turn(self, record: TurnRecord) -> None:
            seen.append(record)

        def record_feedback(self, record: FeedbackRecord) -> None:
            pass

    ChatService(retriever=_Retriever(), llm_client=MeteredLLMClient(_SlowLLM("x")), turn_sink=_Plain()).ask("q")
    assert seen[0].llm_exchanges is None
    assert all(e.ended_ms is not None for e in seen[0].events)



def test_a_stored_agent_turn_without_end_times_is_laid_out_from_its_model_calls() -> None:
    # The shape of a turn an eval run stored before stages kept their end time:
    # a step, a search's stages and then its event, a second step, all at
    # recorded call times with an unmeasured gap before the second call.
    from rag.observability.records import LLMExchange

    sink, exporter = _sink()
    sink.record_turn(_record(
        metadata={"mode": "agentic:react", "llm": "ollama:agent-27b"}, total_ms=200.0,
        events=[StageEvent("agent_step", "m", 40.2), StageEvent("embed", "e", 3.0), StageEvent("rerank", "r", 10.0),
                StageEvent("search", "s", 14.0), StageEvent("agent_step", "m", 50.0)],
        llm_exchanges=[LLMExchange(kind="chat", started_ms=0.1, elapsed_ms=40.0),
                       LLMExchange(kind="chat", started_ms=100.0, elapsed_ms=49.9)],
        agent_calls=[AgentToolCall(step=1, query="q")],
    ))

    spans = exporter.get_finished_spans()
    named = _by_name(spans)
    first, second = named["agent_step"]
    assert [_parent(c, spans) for c in named["chat agent-27b"]] == ["agent_step", "agent_step"]
    assert _parent(named["embed"][0], spans) == _parent(named["rerank"][0], spans) == "execute_tool rag_search"
    # The step's own length (40.2) outlasts its call (0.1 + 40.0), so it starts at the turn's start.
    assert first.start_time == named["invoke_agent rag"][0].start_time
    # The search starts as the first step ends; the second step ends with its call.
    assert named["execute_tool rag_search"][0].start_time == first.end_time
    assert second.end_time is not None and first.start_time is not None
    assert second.end_time - first.start_time == int(149.9 * 1e6)
    assert named["embed"][0].attributes["rag.timing"] == "reconstructed"  # type: ignore[index]
