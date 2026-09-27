"""Tests for Milestone 12: turn records, LLM usage metering, citation parsing, feedback.

Everything runs against fakes and a `tmp_path` JSONL file -- no Ollama, no
Chroma -- in keeping with the rest of the suite.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from rag.api.main import app
from rag.api.routes.chat import get_chat_service, get_turn_sink
from rag.config.settings import AgentConfig, LLMConfig, RagConfig, TurnLogConfig
from rag.events import EventSink, PipelineEvent
from rag.generation.chat_service import ChatAnswer, ChatService
from rag.generation.crag import GradedChunks
from rag.generation.builder import build_agent_llm
from rag.generation.llm import AssistantTurn, ChatMessage, LLMClient, LLMUsage, ToolCallingLLM
from rag.generation.ollama_llm import OllamaLLMClient
from rag.generation.prompts import parse_cited_passages
from rag.observability.factory import config_fingerprint, get_turn_sink as build_turn_sink
from rag.observability.records import FeedbackRecord, TurnRecord
from rag.observability.sink import JsonlTurnSink, TurnSink, read_turn_log, turns_with_feedback
from rag.observability.usage import MeteredLLMClient, MeteredToolCallingLLM, metered, metered_client
from rag.retrieval.retriever import RetrievalResult
from rag.ui.helpers import format_turn_metrics, rating_from_feedback_widget
from rag.vectorstore.base import ScoredChunk
from tests.fakes import ScriptedToolLLM


def _scored(chunk_id: str, score: float = 0.5) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk_id,
        text=f"text of {chunk_id}",
        document_id=f"{chunk_id}-doc",
        source=Path(f"{chunk_id}.md"),
        doc_type="markdown",
        score=score,
        metadata={},
    )


class _Retriever:
    """Returns one canned result per call (the last one repeats), emitting a timed event."""

    def __init__(self, *results: RetrievalResult) -> None:
        self.results = list(results)
        self.calls = 0

    def retrieve(self, query: str, *, on_event: EventSink | None = None) -> RetrievalResult:
        result = self.results[min(self.calls, len(self.results) - 1)]
        self.calls += 1
        if on_event is not None:
            on_event(PipelineEvent(stage="rerank", message="reranked", elapsed_ms=10.0))
        return result


class _UsageLLM(LLMClient):
    """Replies in order and reports fixed token counts, like `OllamaLLMClient` does."""

    def __init__(self, *replies: str, usage: LLMUsage | None = LLMUsage(100, 20)) -> None:
        self.replies = list(replies) or ["answer [1]"]
        self.usage = usage
        self.calls = 0

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        return self.generate_with_usage(prompt, system=system)[0]

    def generate_with_usage(self, prompt: str, *, system: str | None = None) -> tuple[str, LLMUsage | None]:
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        return reply, self.usage


class _RecordingSink(TurnSink):
    def __init__(self) -> None:
        self.turns: list[TurnRecord] = []
        self.feedback: list[FeedbackRecord] = []

    def record_turn(self, record: TurnRecord) -> None:
        self.turns.append(record)

    def record_feedback(self, record: FeedbackRecord) -> None:
        self.feedback.append(record)


class _BrokenSink(TurnSink):
    def record_turn(self, record: TurnRecord) -> None:
        raise OSError("disk full")

    def record_feedback(self, record: FeedbackRecord) -> None:
        raise OSError("disk full")


def _result(*chunk_ids: str, candidate_count: int | None = None) -> RetrievalResult:
    chunks = [_scored(cid) for cid in chunk_ids]
    return RetrievalResult(
        chunks=chunks, candidate_count=len(chunks) if candidate_count is None else candidate_count
    )


# ---------------------------------------------------------------------------
# parse_cited_passages
# ---------------------------------------------------------------------------


def test_parse_cited_passages_reads_single_adjacent_and_comma_markers_in_first_cited_order() -> None:
    assert parse_cited_passages("B is true [3]. A too [1][2], and again [3, 1].", 3) == [3, 1, 2]


def test_parse_cited_passages_drops_out_of_range_numbers() -> None:
    assert parse_cited_passages("See [0], [4] and [2].", 3) == [2]


def test_parse_cited_passages_ignores_ranges_and_uncited_answers() -> None:
    assert parse_cited_passages("Passages [1-3] say so.", 3) == []
    assert parse_cited_passages("No markers here.", 3) == []


# ---------------------------------------------------------------------------
# Usage metering
# ---------------------------------------------------------------------------


def test_metered_client_counts_calls_and_tokens_only_inside_a_meter() -> None:
    client = MeteredLLMClient(_UsageLLM("x"))
    client.generate("outside")  # not recorded anywhere, and must not fail
    with metered() as meter:
        client.generate("one")
        client.generate("two")
    assert meter.calls == 2
    assert meter.prompt_tokens == 200
    assert meter.completion_tokens == 40


def test_tokens_stay_unknown_when_the_provider_reports_none() -> None:
    client = MeteredLLMClient(_UsageLLM("x", usage=None))
    with metered() as meter:
        client.generate("one")
    assert meter.calls == 1
    assert meter.prompt_tokens is None and meter.completion_tokens is None


def test_default_generate_with_usage_reports_no_usage() -> None:
    class _Plain(LLMClient):
        def generate(self, prompt: str, *, system: str | None = None) -> str:
            return "hi"

    assert _Plain().generate_with_usage("q") == ("hi", None)


def test_metered_tool_client_records_chat_usage_into_the_active_meter() -> None:
    inner = ScriptedToolLLM([AssistantTurn("a", usage=LLMUsage(1000, 50)), AssistantTurn("b", usage=None)])
    client = metered_client(inner)
    with metered() as meter:
        client.chat([ChatMessage("user", "q")])
        client.chat([ChatMessage("user", "q")])
    assert meter.calls == 2  # a call with unknown usage still counts as a call
    assert (meter.prompt_tokens, meter.completion_tokens) == (1000, 50)
    assert meter.llm_ms >= 0


def test_metered_tool_client_meters_generate_too_and_passes_through_outside_a_meter() -> None:
    client = metered_client(ScriptedToolLLM([AssistantTurn("a")], generate_reply="g"))
    assert client.chat([ChatMessage("user", "outside")]).content == "a"  # nothing active; must not fail
    with metered() as meter:
        assert client.generate("q") == "g"
    assert meter.calls == 1


def test_metered_client_keeps_the_tool_calling_capability() -> None:
    # Plain `MeteredLLMClient` would hide `chat()` and fail the agent's
    # build-time check; the helper picks the wrapper that matches the inner type.
    assert isinstance(metered_client(ScriptedToolLLM([])), MeteredToolCallingLLM)
    assert isinstance(metered_client(ScriptedToolLLM([])), ToolCallingLLM)
    plain = metered_client(_UsageLLM())
    assert type(plain) is MeteredLLMClient and not isinstance(plain, ToolCallingLLM)


def test_build_agent_llm_is_metered_tool_calling_with_num_ctx_and_falls_back_to_llm() -> None:
    config = RagConfig(llm=LLMConfig(model="pipeline-9b"))
    client = build_agent_llm(config)
    assert isinstance(client, MeteredToolCallingLLM)
    assert isinstance(client.inner, OllamaLLMClient)
    assert client.inner.model == "pipeline-9b"
    assert client.inner.num_ctx == config.agent.num_ctx


def test_build_agent_llm_uses_agent_llm_when_set() -> None:
    config = RagConfig(agent=AgentConfig(llm=LLMConfig(model="agent-27b"), num_ctx=16384))
    client = build_agent_llm(config)
    assert isinstance(client, MeteredToolCallingLLM) and isinstance(client.inner, OllamaLLMClient)
    assert (client.inner.model, client.inner.num_ctx) == ("agent-27b", 16384)


def test_build_agent_llm_fails_at_build_time_for_a_provider_without_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    config = RagConfig(agent=AgentConfig(llm=LLMConfig(provider="gemini", model="gemma-4-31b-it")))
    with pytest.raises(ValueError, match="agent.llm selects provider 'gemini'"):
        build_agent_llm(config)


def test_nested_meters_do_not_double_count() -> None:
    client = MeteredLLMClient(_UsageLLM("x"))
    with metered() as outer:
        client.generate("outer")
        with metered() as inner:
            client.generate("inner")
        client.generate("outer again")
    assert (outer.calls, inner.calls) == (2, 1)


def test_ollama_client_reports_prompt_and_eval_counts() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": "hi"}, "prompt_eval_count": 26, "eval_count": 282},
        )

    client = OllamaLLMClient(model="m", base_url="http://fake:11434")
    client._client = httpx.Client(base_url=client.base_url, transport=httpx.MockTransport(handler))
    assert client.generate_with_usage("q") == ("hi", LLMUsage(prompt_tokens=26, completion_tokens=282))


def test_ollama_client_treats_missing_counts_as_unknown() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "hi"}})

    client = OllamaLLMClient(model="m", base_url="http://fake:11434")
    client._client = httpx.Client(base_url=client.base_url, transport=httpx.MockTransport(handler))
    assert client.generate_with_usage("q")[1] == LLMUsage(None, None)


# ---------------------------------------------------------------------------
# ChatService: metrics on ChatAnswer, and the persisted TurnRecord
# ---------------------------------------------------------------------------


def test_answer_carries_turn_id_latency_usage_and_cited_ids() -> None:
    llm = MeteredLLMClient(_UsageLLM("Refunds take 30 days [2]."))
    sink = _RecordingSink()
    service = ChatService(retriever=_Retriever(_result("a", "b")), llm_client=llm, turn_sink=sink)

    answer = service.ask("refunds?")

    assert answer.turn_id is not None and answer.turn_id == sink.turns[0].turn_id
    assert answer.cited_chunk_ids == ["b"]
    assert [c.chunk_id for c in answer.citations] == ["a", "b"]  # every passage shown stays a citation
    assert answer.llm_calls == 1
    assert answer.llm_ms == sink.turns[0].llm_ms
    assert (answer.prompt_tokens, answer.completion_tokens) == (100, 20)
    assert answer.stage_ms["rerank"] == pytest.approx(10.0)
    assert {"prompt", "generate"} <= answer.stage_ms.keys()
    assert answer.total_ms is not None and answer.total_ms >= 0


def test_turn_record_captures_attempts_shown_cited_and_events() -> None:
    sink = _RecordingSink()
    service = ChatService(
        retriever=_Retriever(_result("a", "b")),
        llm_client=MeteredLLMClient(_UsageLLM("See [1].")),
        turn_sink=sink,
        turn_metadata={"corpus": "baseline"},
    )

    service.ask("q", history=[])

    record = sink.turns[0]
    assert record.outcome == "answered"
    assert record.answer == "See [1]."
    assert record.shown_chunk_ids == ["a", "b"]
    assert record.cited_chunk_ids == ["a"]
    assert len(record.attempts) == 1
    assert [p.chunk_id for p in record.attempts[0].retrieved] == ["a", "b"]
    assert record.attempts[0].kept is None  # no grader ran
    assert [e.stage for e in record.events] == ["rerank", "prompt", "generate"]
    assert record.llm_calls == 1 and record.prompt_tokens == 100
    assert record.metadata == {"corpus": "baseline"}
    json.dumps(record.to_dict())  # must be JSON-serializable as-is


def test_turn_record_captures_grader_verdicts_retries_and_groundedness() -> None:
    class _Grader:
        def grade(self, query: str, chunks: list[ScoredChunk]) -> GradedChunks:
            kept = [c for c in chunks if c.chunk_id != "bad"]
            return GradedChunks(kept=kept, graded_out=len(chunks) - len(kept))

    class _Rewriter:
        def rewrite(self, query: str, *, attempt: int) -> str:
            return f"reworded {attempt}"

    class _Checker:
        def __init__(self) -> None:
            self.verdicts = [False, True]

        def check(self, query: str, chunks: list[ScoredChunk], answer: str) -> bool | None:
            return self.verdicts.pop(0)

    sink = _RecordingSink()
    service = ChatService(
        retriever=_Retriever(_result("bad"), _result("good", "bad")),
        llm_client=MeteredLLMClient(_UsageLLM("first [1]", "second [1]")),
        grader=_Grader(),  # type: ignore[arg-type]
        retry_rewriter=_Rewriter(),  # type: ignore[arg-type]
        groundedness_checker=_Checker(),  # type: ignore[arg-type]
        max_retries=1,
        max_regenerations=1,
        turn_sink=sink,
    )

    answer = service.ask("q")

    record = sink.turns[0]
    assert [a.query for a in record.attempts] == ["q", "reworded 1"]
    assert record.attempts[0].kept == []
    assert record.attempts[1].kept == ["good"]
    assert record.retry_queries == ["reworded 1"]
    assert record.groundedness_checks == [False, True]
    assert record.grounded is True
    assert record.answer == "second [1]"
    assert record.cited_chunk_ids == ["good"]
    # Grader/checker fakes don't call the LLM here, so only the two generations count.
    assert answer.llm_calls == 2


@pytest.mark.parametrize(
    ("result", "outcome"),
    [
        (_result(candidate_count=0), "empty_index"),
        (RetrievalResult(chunks=[], candidate_count=4, dropped_below_min_score=4), "below_min_score"),
    ],
)
def test_no_context_turns_are_recorded_with_their_outcome(result: RetrievalResult, outcome: str) -> None:
    sink = _RecordingSink()
    service = ChatService(retriever=_Retriever(result), llm_client=MeteredLLMClient(_UsageLLM()), turn_sink=sink)

    answer = service.ask("q")

    assert sink.turns[0].outcome == outcome
    assert sink.turns[0].shown_chunk_ids == []
    assert answer.llm_calls == 0


def test_blank_query_is_recorded() -> None:
    sink = _RecordingSink()
    service = ChatService(retriever=_Retriever(_result("a")), llm_client=_UsageLLM(), turn_sink=sink)
    service.ask("   ")
    assert sink.turns[0].outcome == "blank_query"


def test_a_failing_turn_is_recorded_as_an_error_and_still_raises() -> None:
    class _Down(LLMClient):
        def generate(self, prompt: str, *, system: str | None = None) -> str:
            raise RuntimeError("ollama is down")

    sink = _RecordingSink()
    service = ChatService(retriever=_Retriever(_result("a")), llm_client=MeteredLLMClient(_Down()), turn_sink=sink)

    with pytest.raises(RuntimeError, match="ollama is down"):
        service.ask("q")

    record = sink.turns[0]
    assert record.outcome == "error"
    assert record.error == "RuntimeError: ollama is down"
    assert record.answer is None
    assert record.shown_chunk_ids == ["a"]  # how far it got is part of the record


def test_a_broken_sink_does_not_fail_the_turn(caplog: pytest.LogCaptureFixture) -> None:
    service = ChatService(retriever=_Retriever(_result("a")), llm_client=_UsageLLM("ok [1]"), turn_sink=_BrokenSink())
    answer = service.ask("q")
    assert answer.answer == "ok [1]"
    assert "Failed to record turn" in caplog.text


def test_caller_event_sink_still_receives_every_event() -> None:
    seen: list[str] = []
    service = ChatService(retriever=_Retriever(_result("a")), llm_client=_UsageLLM())
    service.ask("q", on_event=lambda e: seen.append(e.stage))
    assert seen == ["rerank", "prompt", "generate"]


# ---------------------------------------------------------------------------
# JSONL sink, reading back, factory
# ---------------------------------------------------------------------------


def test_jsonl_sink_round_trips_turns_and_joins_latest_feedback(tmp_path: Path) -> None:
    path = tmp_path / "logs" / "turns.jsonl"
    sink = JsonlTurnSink(path)
    sink.record_turn(TurnRecord(turn_id="t1", timestamp="now", query="q1", outcome="answered"))
    sink.record_turn(TurnRecord(turn_id="t2", timestamp="now", query="q2", outcome="answered"))
    sink.record_feedback(FeedbackRecord(turn_id="t1", rating="down"))
    sink.record_feedback(FeedbackRecord(turn_id="t1", rating="up", comment="changed my mind"))
    sink.record_feedback(FeedbackRecord(turn_id="unknown", rating="down"))

    pairs = turns_with_feedback(read_turn_log(path))

    assert [(turn["turn_id"], fb and fb["rating"]) for turn, fb in pairs] == [("t1", "up"), ("t2", None)]
    assert pairs[0][1] is not None and pairs[0][1]["comment"] == "changed my mind"


def test_read_turn_log_skips_a_torn_line_and_tolerates_a_missing_file(tmp_path: Path) -> None:
    path = tmp_path / "turns.jsonl"
    assert list(read_turn_log(path)) == []
    path.write_text('{"kind": "turn", "turn_id": "t1"}\n{"kind": "tu\n', encoding="utf-8")
    assert [r["turn_id"] for r in read_turn_log(path)] == ["t1"]


def test_factory_builds_jsonl_sink_or_none(tmp_path: Path) -> None:
    sink = build_turn_sink(TurnLogConfig(provider="jsonl", path=tmp_path / "t.jsonl"))
    assert isinstance(sink, JsonlTurnSink) and sink.path == tmp_path / "t.jsonl"
    assert build_turn_sink(TurnLogConfig(provider="none")) is None


def test_config_fingerprint_ignores_where_logs_go_but_not_behaviour() -> None:
    base = RagConfig()
    moved_log = base.model_copy(
        update={"observability": base.observability.model_copy(update={"turn_log": TurnLogConfig(provider="none")})}
    )
    other_top_k = base.model_copy(update={"retrieval": base.retrieval.model_copy(update={"top_k": 7})})
    assert config_fingerprint(base) == config_fingerprint(moved_log)
    assert config_fingerprint(base) != config_fingerprint(other_top_k)


def test_config_fingerprint_ignores_the_agent_section_while_nothing_reads_it() -> None:
    base = RagConfig()
    other_agent = base.model_copy(update={"agent": AgentConfig(max_tool_calls=3)})
    assert config_fingerprint(base) == config_fingerprint(other_agent)


def test_config_fingerprint_keys_on_thinking_level_only_once_it_is_set() -> None:
    gemini = LLMConfig(provider="gemini", model="gemini-3.5-flash-lite")
    unset = RagConfig(llm=gemini)
    # Neither `llm.thinking_level` nor `agent` existed when the turns already
    # logged were hashed.
    as_before = hashlib.sha256(
        unset.model_dump_json(
            exclude={"observability": True, "eval": True, "agent": True, "llm": {"thinking_level"}}
        ).encode()
    ).hexdigest()[:12]
    minimal = RagConfig(llm=gemini.model_copy(update={"thinking_level": "minimal"}))

    # Unset hashes as it did before the field existed, so logged turns keep their key.
    assert config_fingerprint(unset) == as_before
    assert config_fingerprint(minimal) != config_fingerprint(unset)


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


class _FakeChatService:
    def ask(self, query: str, *, history: object = None) -> ChatAnswer:
        return ChatAnswer(
            answer="a [1]",
            cited_chunk_ids=["c1"],
            turn_id="t-123",
            stage_ms={"generate": 12.5},
            total_ms=20.0,
            llm_calls=1,
            prompt_tokens=50,
            completion_tokens=5,
        )


@pytest.fixture()
def api_sink() -> _RecordingSink:
    sink = _RecordingSink()
    app.dependency_overrides[get_chat_service] = lambda: _FakeChatService()
    app.dependency_overrides[get_turn_sink] = lambda: sink
    try:
        yield sink
    finally:
        app.dependency_overrides.pop(get_chat_service, None)
        app.dependency_overrides.pop(get_turn_sink, None)


def test_chat_response_surfaces_turn_metrics(api_sink: _RecordingSink) -> None:
    body = TestClient(app).post("/chat", json={"query": "q"}).json()
    assert body["turn_id"] == "t-123"
    assert body["cited_chunk_ids"] == ["c1"]
    assert body["stage_ms"] == {"generate": 12.5}
    assert (body["total_ms"], body["llm_calls"], body["prompt_tokens"], body["completion_tokens"]) == (20.0, 1, 50, 5)


def test_feedback_is_written_to_the_sink(api_sink: _RecordingSink) -> None:
    response = TestClient(app).post("/feedback", json={"turn_id": "t-123", "rating": "down", "comment": "wrong year"})
    assert response.status_code == 201
    assert response.json()["feedback_id"] == api_sink.feedback[0].feedback_id
    assert (api_sink.feedback[0].turn_id, api_sink.feedback[0].rating) == ("t-123", "down")
    assert api_sink.feedback[0].comment == "wrong year"


def test_feedback_rejects_an_unknown_rating(api_sink: _RecordingSink) -> None:
    assert TestClient(app).post("/feedback", json={"turn_id": "t", "rating": "meh"}).status_code == 422


def test_feedback_is_refused_when_turn_logging_is_off() -> None:
    app.dependency_overrides[get_turn_sink] = lambda: None
    try:
        response = TestClient(app).post("/feedback", json={"turn_id": "t", "rating": "up"})
    finally:
        app.dependency_overrides.pop(get_turn_sink, None)
    assert response.status_code == 503


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------


def test_format_turn_metrics() -> None:
    from rag.generation.chat_service import Citation

    citations = [Citation(chunk_id=c, document_id="d", text="t", score=0.5) for c in ("a", "b", "c")]
    answer = ChatAnswer(
        answer="x", citations=citations, cited_chunk_ids=["b"], total_ms=3210.0,
        llm_calls=2, prompt_tokens=1234, completion_tokens=None,
    )
    assert format_turn_metrics(answer) == "⏱️ 3.2 s · 2 LLM call(s) · 1,234 → ? tokens · cited 1 of 3 passage(s)"
    assert format_turn_metrics(ChatAnswer(answer="x")) is None


def test_rating_from_feedback_widget() -> None:
    assert rating_from_feedback_widget(1) == "up"
    assert rating_from_feedback_widget(0) == "down"
    assert rating_from_feedback_widget(None) is None
