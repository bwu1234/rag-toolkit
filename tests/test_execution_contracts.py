"""Execution and output contracts (`docs/milestone-19-plan.md`): deadlines, budgets, failures, citations.

Hermetic. Time is a fake clock the agent and `rag.deadline` both read, and
`_TimedLLM` behaves like a deadline-bound adapter: a call that would outlast
the active scope advances the clock to the deadline and raises
`DeadlineExceeded`, as a request httpx cuts off does. The adapter tests check
the real adapters against `httpx.MockTransport`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import httpx
import pytest

from rag import deadline
from rag.agent.prompts import AGENT_SYNTHESIS_INSTRUCTION
from rag.agent.service import AgentService
from rag.config.settings import AgentConfig
from rag.deadline import DeadlineExceeded, deadline_scope
from rag.embedding.ollama_embedder import OllamaEmbedder
from rag.generation.chat_service import GENERATION_FAILURE_ANSWERS, ChatAnswer, ChatService
from rag.generation.prompts import invalid_citations, parse_cited_passages
from rag.generation.query_rewriter import ChatTurn
from rag.llm.base import (
    AssistantTurn,
    ChatMessage,
    ContextLimits,
    LLMClient,
    LLMUsage,
    Message,
    ToolDefinition,
    ToolResult,
)
from rag.llm.gemini_llm import GeminiLLMClient
from rag.llm.ollama_llm import OllamaLLMClient
from rag.observability.records import TurnRecord
from rag.observability.usage import metered_client
from rag.retrieval.retriever import RetrievalResult
from tests.fakes import ScriptedToolLLM
from tests.test_agent import _answer, _chunk, _FakeTools, _search, _step


class _Clock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class _TimedLLM(ScriptedToolLLM):
    """Replays turns, each taking `seconds[i]` of fake time, cut off by the active deadline.

    Records the time the deadline left each call, so a test can check which
    scope bound it. `limits` is what `context_limits()` reports.
    """

    def __init__(
        self,
        turns: Sequence[AssistantTurn],
        clock: _Clock,
        seconds: Sequence[float] = (),
        *,
        limits: ContextLimits | None = None,
    ) -> None:
        super().__init__(turns)
        self.clock = clock
        self.seconds = list(seconds)
        self.limits = limits
        self.remaining: list[float | None] = []

    def context_limits(self) -> ContextLimits | None:
        return self.limits

    def chat(self, messages: Sequence[Message], tools: Sequence[ToolDefinition] = ()) -> AssistantTurn:
        index = len(self.remaining)
        remaining = deadline.remaining_s()
        self.remaining.append(remaining)
        cost = self.seconds[index] if index < len(self.seconds) else 1.0
        if remaining is not None and cost >= remaining:
            self.calls.append((list(messages), list(tools)))
            self.clock.now += max(remaining, 0.0)
            raise DeadlineExceeded("cut off", label="test")
        self.clock.now += cost
        return super().chat(messages, tools)


class _TimedTools(_FakeTools):
    """`_FakeTools` whose every search takes `search_s` of fake time, cut off like the LLM's calls."""

    def __init__(self, results: dict[str, list[Any]], clock: _Clock, search_s: float = 1.0) -> None:
        super().__init__(results)
        self.clock = clock
        self.search_s = search_s
        self.remaining: list[float | None] = []

    def retrieve(self, query, corpus=None, top_k=None, *, query_filter=None, on_event=None):  # type: ignore[no-untyped-def]
        remaining = deadline.remaining_s()
        self.remaining.append(remaining)
        if remaining is not None and self.search_s >= remaining:
            self.clock.now += max(remaining, 0.0)
            raise DeadlineExceeded("search cut off", label="search")
        self.clock.now += self.search_s
        return super().retrieve(query, corpus, top_k, query_filter=query_filter, on_event=on_event)


def _agent(llm: ScriptedToolLLM, tools: _FakeTools, clock: _Clock, **overrides: object) -> AgentService:
    options: dict[str, object] = {
        "system_prompt": "SYSTEM", "corpora": ["baseline"], "max_tool_calls": 8, "clock": clock,
    }
    options.update(overrides)
    return AgentService(llm, tools, **options)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# rag.deadline
# --------------------------------------------------------------------------


def test_a_scope_bounds_request_timeouts_and_an_inner_scope_cannot_extend_it() -> None:
    clock = _Clock(100.0)
    assert deadline.request_timeout(120.0, "x") == 120.0  # no scope: the adapter's own
    with deadline_scope(130.0, label="turn", clock=clock):
        assert deadline.request_timeout(120.0, "x") == 30.0
        with deadline_scope(200.0, label="later", clock=clock):
            assert deadline.request_timeout(120.0, "x") == 30.0
        with deadline_scope(110.0, label="search", clock=clock):
            assert deadline.request_timeout(120.0, "x") == 10.0
        clock.now = 130.0
        with pytest.raises(DeadlineExceeded) as excinfo:
            deadline.request_timeout(120.0, "the answer")
        assert excinfo.value.label == "turn"
    assert deadline.remaining_s() is None


def test_a_timeout_is_blamed_on_the_deadline_only_when_the_deadline_set_it() -> None:
    clock = _Clock()
    cause = httpx.ReadTimeout("slow")
    with deadline_scope(10.0, label="turn", clock=clock):
        with pytest.raises(DeadlineExceeded):
            deadline.raise_if_cut_short("a call", 10.0, 120.0, cause)
        # The adapter's own timeout ran out first: an upstream failure, not the deadline.
        deadline.raise_if_cut_short("a call", 120.0, 120.0, cause)


def test_a_wait_longer_than_the_time_left_is_refused() -> None:
    clock = _Clock()
    with deadline_scope(5.0, label="turn", clock=clock):
        deadline.check_wait(4.0, "a retry")
        with pytest.raises(DeadlineExceeded, match="a retry needs 6.0s"):
            deadline.check_wait(6.0, "a retry")


# --------------------------------------------------------------------------
# Adapters: the deadline reaches the request, and a cut-off is reported as one
# --------------------------------------------------------------------------


def _ollama(handler: Any, **kwargs: Any) -> OllamaLLMClient:
    client = OllamaLLMClient(model="m", base_url="http://fake-ollama:11434", timeout=120.0, **kwargs)
    client._client = httpx.Client(base_url=client.base_url, transport=httpx.MockTransport(handler))
    return client


def test_ollama_sends_the_time_left_as_the_request_timeout() -> None:
    seen: list[dict[str, float]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.extensions["timeout"])
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "ok"}})

    client = _ollama(handler)
    clock = _Clock()
    with deadline_scope(25.0, label="turn", clock=clock):
        client.chat([ChatMessage("user", "q")])
    client.chat([ChatMessage("user", "q")])

    assert seen[0]["read"] == 25.0
    assert seen[1]["read"] == 120.0


def test_ollama_reports_a_request_cut_off_by_the_deadline_as_deadline_exceeded() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    client = _ollama(handler)
    with deadline_scope(5.0, label="turn", clock=_Clock()):
        with pytest.raises(DeadlineExceeded):
            client.chat([ChatMessage("user", "q")])
    # Outside a deadline the same timeout is the daemon failing to answer.
    with pytest.raises(RuntimeError, match="Failed to get a chat completion"):
        client.chat([ChatMessage("user", "q")])


def test_ollama_refuses_a_request_with_no_time_left_without_sending_it() -> None:
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "ok"}})

    clock = _Clock(10.0)
    with deadline_scope(10.0, label="turn", clock=clock), pytest.raises(DeadlineExceeded):
        _ollama(handler).chat([ChatMessage("user", "q")])
    assert sent == []


def test_ollama_reports_its_window_and_output_limit() -> None:
    assert _ollama(lambda r: None, num_ctx=8192, max_tokens=512).context_limits() == ContextLimits(8192, 512)
    assert _ollama(lambda r: None).context_limits() == ContextLimits(None, 1024)


def test_the_ollama_embedder_is_bounded_by_the_deadline_too() -> None:
    seen: list[dict[str, float]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.extensions["timeout"])
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2]]})

    embedder = OllamaEmbedder(model="e", base_url="http://fake-ollama:11434", timeout=60.0)
    embedder._client = httpx.Client(base_url=embedder.base_url, transport=httpx.MockTransport(handler))
    with deadline_scope(3.0, label="search", clock=_Clock()):
        embedder.embed_query("q")
    assert seen[0]["read"] == 3.0


class _Counter:
    def __init__(self) -> None:
        self.taken = 0

    def take(self) -> None:
        self.taken += 1


def _gemini(handler: Any, **kwargs: Any) -> GeminiLLMClient:
    client = GeminiLLMClient(model="g", base_url="https://fake-gemini", api_key="k", timeout=120.0, **kwargs)
    client._client = httpx.Client(base_url=client.base_url, transport=httpx.MockTransport(handler))
    client._sleep = lambda seconds: pytest.fail(f"slept {seconds}s past the deadline")
    return client


def test_gemini_bounds_the_request_and_spends_no_daily_request_once_time_is_up() -> None:
    seen: list[dict[str, float]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.extensions["timeout"])
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "ok"}]}}]})

    counter = _Counter()
    client = _gemini(handler, daily_counter=counter)  # type: ignore[arg-type]
    clock = _Clock()
    with deadline_scope(40.0, label="turn", clock=clock):
        client.chat([ChatMessage("user", "q")])
        clock.now = 40.0
        with pytest.raises(DeadlineExceeded):
            client.chat([ChatMessage("user", "q")])

    assert seen[0]["read"] == 40.0
    assert counter.taken == 1  # the refused request was never charged


def test_gemini_gives_up_on_a_retry_that_would_outlast_the_deadline() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": {"message": "overloaded"}}, headers={"Retry-After": "30"})

    with deadline_scope(10.0, label="turn", clock=_Clock()):
        with pytest.raises(DeadlineExceeded, match="retrying HTTP 503"):
            _gemini(handler).chat([ChatMessage("user", "q")])


def test_gemini_reports_its_output_limit_and_no_window() -> None:
    assert _gemini(lambda r: None, max_tokens=2048).context_limits() == ContextLimits(None, 2048)


def test_a_metered_client_passes_the_limits_through() -> None:
    llm = _TimedLLM([], _Clock(), limits=ContextLimits(4096, 256))
    assert metered_client(llm).context_limits() == ContextLimits(4096, 256)  # type: ignore[attr-defined]


# --------------------------------------------------------------------------
# The turn deadline
# --------------------------------------------------------------------------


def test_a_stalled_step_is_cut_off_at_the_reserve_and_the_answer_gets_the_reserve() -> None:
    clock = _Clock()
    tools = _FakeTools({"q1": [_chunk("c1")]})
    # Step one takes 10 s; step two stalls (500 s) and is cut off at 100 - 30 = 70 s.
    llm = _TimedLLM([_step(_search("q1")), _step(_search("q2")), _answer("from [1]")], clock, [10, 500, 5])

    answer = _agent(llm, tools, clock, turn_deadline_s=100.0, synthesis_reserve_s=30.0).ask("question")

    assert (answer.answer, answer.stopped_reason, answer.generation_failure) == ("from [1]", "deadline", None)
    assert llm.remaining == [70.0, 60.0, 30.0]  # each step bound by the reserve, the answer by the deadline
    assert llm.calls[-1][1] == [] and llm.calls[-1][0][-1] == ChatMessage("user", AGENT_SYNTHESIS_INSTRUCTION)
    assert [c.chunk_id for c in answer.citations] == ["c1"]


def test_a_search_cut_off_at_the_reserve_shows_nothing_and_the_turn_answers() -> None:
    clock = _Clock()
    tools = _TimedTools({"q1": [_chunk("c1")]}, clock, search_s=200.0)
    llm = _TimedLLM([_step(_search("q1")), _answer("nothing found")], clock, [10, 5])

    answer = _agent(llm, tools, clock, turn_deadline_s=100.0, synthesis_reserve_s=30.0).ask("question")

    assert (answer.stopped_reason, answer.generation_failure) == ("deadline", None)
    assert tools.remaining == [60.0]
    call = answer.agent_calls[0]
    assert (call.query, call.status, call.passages) == ("q1", "refused", [])
    assert call.note is not None and "cut off" in call.note
    assert answer.citations == [] and answer.tool_calls == 1


def test_no_search_starts_that_the_longest_so_far_would_carry_into_the_reserve() -> None:
    clock = _Clock()
    tools = _TimedTools({"q1": [_chunk("c1")], "q2": [_chunk("c2")]}, clock, search_s=25.0)
    # 1 s step, 25 s search (t=26), 1 s step (t=27): 27 + 25 >= 50 -- q2 never starts.
    llm = _TimedLLM([_step(_search("q1")), _step(_search("q2")), _answer("from [1]")], clock)

    answer = _agent(llm, tools, clock, turn_deadline_s=80.0, synthesis_reserve_s=30.0).ask("question")

    assert [q for q, _ in tools.searches] == ["q1"]
    assert answer.stopped_reason == "deadline"
    assert answer.agent_calls[-1].status == "refused"


def test_expiry_before_the_answer_is_a_deadline_failure_without_a_call() -> None:
    clock = _Clock()
    llm = _TimedLLM([_step(_search("q1"))], clock, [10])

    class _SlowTools(_FakeTools):
        def retrieve(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            clock.now += 200.0  # in-process work the deadline can't interrupt
            return super().retrieve(*args, **kwargs)

    answer = _agent(llm, _SlowTools({"q1": [_chunk("c1")]}), clock, turn_deadline_s=100.0,
                    synthesis_reserve_s=30.0).ask("question")

    assert answer.generation_failure == "deadline"
    assert answer.stopped_reason == "deadline"
    assert answer.answer == GENERATION_FAILURE_ANSWERS["deadline"]
    assert len(llm.calls) == 1  # no answering call was started past the deadline


def test_expiry_during_the_answer_is_a_deadline_failure_that_keeps_its_trigger() -> None:
    clock = _Clock()
    llm = _TimedLLM([_step(_search("q1")), _answer("too late")], clock, [10, 500])

    answer = _agent(llm, _FakeTools({"q1": [_chunk("c1")]}), clock, max_tool_calls=1,
                    turn_deadline_s=100.0, synthesis_reserve_s=30.0).ask("question")

    # The cap stopped the searching; the deadline is why there's no answer.
    assert (answer.stopped_reason, answer.generation_failure) == ("cap", "deadline")
    assert llm.remaining[-1] == 90.0
    assert answer.cited_chunk_ids == []


def test_the_search_budget_still_stops_searching_between_calls() -> None:
    clock = _Clock()
    llm = _TimedLLM([_step(_search("q1")), _step(_search("q2")), _answer("[1]")], clock, [40, 40, 1])

    answer = _agent(llm, _FakeTools({"q1": [_chunk("c1")]}), clock, timeout_s=60.0,
                    turn_deadline_s=900.0, synthesis_reserve_s=120.0).ask("question")

    assert answer.stopped_reason == "timeout"
    assert answer.generation_failure is None


def test_the_groundedness_check_runs_inside_the_deadline() -> None:
    clock = _Clock()

    class _Checker:
        def __init__(self) -> None:
            self.remaining: list[float | None] = []

        def check(self, query: str, chunks: list[Any], answer: str) -> bool | None:
            self.remaining.append(deadline.remaining_s())
            return True

    checker = _Checker()
    llm = _TimedLLM([_step(_search("q1")), _answer("[1]")], clock, [10, 20])
    answer = _agent(llm, _FakeTools({"q1": [_chunk("c1")]}), clock, turn_deadline_s=100.0,
                    synthesis_reserve_s=30.0, groundedness_checker=checker).ask("question")

    assert answer.grounded is True and checker.remaining == [70.0]


def test_a_groundedness_check_cut_off_by_the_deadline_is_inconclusive(caplog: pytest.LogCaptureFixture) -> None:
    from rag.generation.crag import GroundednessChecker

    class _CutOff(LLMClient):
        def generate(self, prompt: str, *, system: str | None = None) -> str:
            raise DeadlineExceeded("turn deadline reached", label="turn")

    verdict = GroundednessChecker(_CutOff()).check("q", [_chunk("c1")], "answer [1]")

    assert verdict is None
    assert "cut off" in caplog.text and "Traceback" not in caplog.text


def test_a_groundedness_check_the_token_budget_cannot_afford_is_skipped() -> None:
    class _Checker:
        def check(self, query: str, chunks: list[Any], answer: str) -> bool | None:
            raise AssertionError("ran over budget")

    # The answer fits the 1,500 tokens (~1,110 by estimate), and costs 1,020;
    # re-sending its 2,000-character passage to the checker would not fit.
    llm = ScriptedToolLLM([_step(_search("q1")), AssistantTurn("[1]", usage=LLMUsage(1000, 20))])
    events: list[str] = []

    answer = _agent(llm, _FakeTools({"q1": [_chunk("c1", text="x" * 2000)]}), _Clock(), max_turn_tokens=1500,
                    max_passage_chars=2000, groundedness_checker=_Checker()).ask("q", on_event=lambda e: events.append(e.message))

    assert answer.grounded is None and answer.generation_failure is None
    assert "Groundedness check skipped: over the token budget" in events


def test_the_reserve_must_be_shorter_than_the_deadline() -> None:
    with pytest.raises(ValueError, match="synthesis_reserve_s"):
        AgentConfig(turn_deadline_s=60.0, synthesis_reserve_s=60.0)
    with pytest.raises(ValueError, match="synthesis_reserve_s"):
        _agent(ScriptedToolLLM([]), _FakeTools(), _Clock(), turn_deadline_s=10.0, synthesis_reserve_s=20.0)
    assert AgentConfig(turn_deadline_s=None, synthesis_reserve_s=500.0).turn_deadline_s is None


# --------------------------------------------------------------------------
# Context preflight and token budgets
# --------------------------------------------------------------------------


def test_a_step_whose_prompt_would_not_fit_is_rolled_back_before_it_is_sent() -> None:
    clock = _Clock()
    big = _chunk("c2", text="9" * 4000)
    tools = _FakeTools({"q1": [_chunk("c1")], "q2": [big]})
    # The provider counted 300 prompt tokens for each call; q2's 4,000 digits
    # estimate at 1,600 more, over a 2,000-token window with 256 kept for output.
    turns = [
        AssistantTurn("", (_search("q1"),), usage=LLMUsage(300, 10)),
        AssistantTurn("", (_search("q2"),), usage=LLMUsage(300, 10)),
        _answer("from [1]"),
    ]
    llm = _TimedLLM(turns, clock, limits=ContextLimits(2000, 256))

    answer = _agent(llm, tools, clock, max_passage_chars=5000).ask("question")

    assert answer.stopped_reason == "context"
    assert len(llm.calls) == 3  # two steps and the answer: the oversized step was never sent
    final = llm.calls[-1][0]
    assert [r.content.startswith("Results not shown") for r in final if isinstance(r, ToolResult)] == [False, True]
    assert [c.chunk_id for c in answer.citations] == ["c1"]


def test_long_history_is_trimmed_oldest_first_to_fit_the_window() -> None:
    clock = _Clock()
    history = [ChatTurn("user", "old " * 400), ChatTurn("assistant", "older answer " * 100),
               ChatTurn("user", "recent question"), ChatTurn("assistant", "recent answer")]
    llm = _TimedLLM([_answer("hi")], clock, limits=ContextLimits(1000, 256))
    events: list[str] = []

    answer = _agent(llm, _FakeTools(), clock).ask("question", history=history, on_event=lambda e: events.append(e.stage))

    sent = llm.calls[0][0]
    assert [type(m).__name__ for m in sent] == ["ChatMessage", "ChatMessage", "AssistantTurn", "ChatMessage"]
    assert sent[1] == ChatMessage("user", "recent question")
    assert "history_trimmed" in events and answer.generation_failure is None


def test_a_question_that_cannot_fit_even_alone_fails_without_a_call() -> None:
    llm = _TimedLLM([], _Clock(), limits=ContextLimits(1000, 256))

    answer = _agent(llm, _FakeTools(), _Clock()).ask("x" * 5000)

    assert (answer.generation_failure, answer.stopped_reason, llm.calls) == ("context", "context", [])


def test_missing_usage_is_counted_by_estimate_and_reported() -> None:
    clock = _Clock()
    turns = [AssistantTurn("", (_search("q1"),)), AssistantTurn("from [1]")]
    llm = _TimedLLM(turns, clock)
    records: list[TurnRecord] = []

    answer = _agent(llm, _FakeTools({"q1": [_chunk("c1")]}), clock).ask("question", on_record=records.append)

    assert answer.generation_failure is None
    assert records[0].estimated_usage_calls == 2
    # Unknown stays unknown in the reported totals: estimates feed the budget only.
    assert answer.prompt_tokens is None


def test_the_token_budget_stops_searching_while_the_answer_still_fits() -> None:
    clock = _Clock()
    turns = [AssistantTurn("", (_search("q1"),), usage=LLMUsage(1000, 50)), _answer("from [1]")]
    llm = _TimedLLM(turns, clock, limits=ContextLimits(None, 200))
    tools = _FakeTools({"q1": [_chunk("c1", text="x" * 1000)], "q2": [_chunk("c2")]})

    # After step one (1,050 used), a second step at ~1,500 tokens plus its
    # answer would pass 4,000: the agent answers instead.
    answer = _agent(llm, tools, clock, max_turn_tokens=4000).ask("question")

    assert (answer.stopped_reason, answer.generation_failure) == ("tokens", None)
    assert [q for q, _ in tools.searches] == ["q1"]
    assert len(llm.calls) == 2


def test_an_answer_the_token_budget_cannot_afford_is_a_token_budget_failure() -> None:
    clock = _Clock()
    turns = [AssistantTurn("", (_search("q1"),), usage=LLMUsage(1000, 50))]
    llm = _TimedLLM(turns, clock, limits=ContextLimits(None, 200))

    # The first step was affordable; the passages it returned make the answer not.
    answer = _agent(llm, _FakeTools({"q1": [_chunk("c1", text="x" * 6000)]}), clock,
                    max_turn_tokens=3500, max_tool_calls=1, max_passage_chars=6000).ask("question")

    assert (answer.stopped_reason, answer.generation_failure) == ("cap", "token_budget")
    assert len(llm.calls) == 1


# --------------------------------------------------------------------------
# Empty output, after each guard
# --------------------------------------------------------------------------


def _blank(stop_reason: str = "length") -> AssistantTurn:
    return AssistantTurn("", usage=LLMUsage(100, 4096), stop_reason=stop_reason)


@pytest.mark.parametrize(
    ("trigger", "turns", "options"),
    [
        ("cap", [_step(_search("q1")), _blank()], {"max_tool_calls": 1}),
        ("timeout", [_step(_search("q1")), _blank()], {"timeout_s": 5.0}),
        ("empty", [_step(_search("q1")), _blank(), _blank()], {}),
    ],
)
def test_an_empty_answer_after_any_guard_is_a_generation_failure(
    trigger: str, turns: list[AssistantTurn], options: dict[str, object]
) -> None:
    clock = _Clock()
    llm = _TimedLLM(turns, clock, [10.0] * len(turns))
    records: list[TurnRecord] = []

    answer = _agent(llm, _FakeTools({"q1": [_chunk("c1")]}), clock, **options).ask("q", on_record=records.append)

    assert (answer.stopped_reason, answer.generation_failure) == (trigger, "empty_output")
    record = records[0]
    assert (record.outcome, record.generation_failure, record.stopped_reason) == (
        "generation_failed", "empty_output", trigger,
    )
    assert record.final_stop_reason == "length"
    assert record.answer == GENERATION_FAILURE_ANSWERS["empty_output"]


def test_an_empty_answer_after_a_context_rollback_is_a_generation_failure() -> None:
    from rag.llm.base import ContextOverflowError

    class _Overflow(ScriptedToolLLM):
        def chat(self, messages, tools=()):  # type: ignore[no-untyped-def]
            if len(self.calls) == 1 and not getattr(self, "overflowed", False):
                self.overflowed = True
                raise ContextOverflowError("too long")
            return super().chat(messages, tools)

    llm = _Overflow([_step(_search("q1")), _blank()])
    answer = _agent(llm, _FakeTools({"q1": [_chunk("c1")]}), _Clock()).ask("q")

    assert (answer.stopped_reason, answer.generation_failure) == ("context", "empty_output")


# --------------------------------------------------------------------------
# Citation validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "valid", "invalid"),
    [
        ("Revenue rose [1][2].", [1, 2], []),
        ("Revenue rose [4].", [], [4]),
        ("Revenue rose [1, 7] and fell [0].", [1], [7, 0]),
        ("Revenue rose.", [], []),
    ],
)
def test_citation_markers_split_into_valid_and_invalid(text: str, valid: list[int], invalid: list[int]) -> None:
    assert parse_cited_passages(text, 3) == valid
    assert invalid_citations(text, 3) == invalid


def test_the_agent_reports_invalid_markers_and_keeps_them_in_the_text() -> None:
    llm = ScriptedToolLLM([_step(_search("q1")), _answer("Up 5% [1], per [9].")])

    answer = _agent(llm, _FakeTools({"q1": [_chunk("c1")]}), _Clock()).ask("q")

    assert answer.answer == "Up 5% [1], per [9]."
    assert (answer.cited_chunk_ids, answer.invalid_citations) == (["c1"], [9])


def test_two_turns_citing_1_mean_each_turns_own_first_passage() -> None:
    tools = _FakeTools({"apple": [_chunk("a1", doc="AAPL.md")], "msft": [_chunk("m1", doc="MSFT.md")]})
    first = _agent(ScriptedToolLLM([_step(_search("apple")), _answer("Apple [1].")]), tools, _Clock()).ask("apple?")
    llm = ScriptedToolLLM([_step(_search("msft")), _answer("Microsoft [1].")])

    second = _agent(llm, tools, _Clock()).ask(
        "and msft?", history=[ChatTurn("user", "apple?"), ChatTurn("assistant", first.answer)]
    )

    assert first.cited_chunk_ids == ["a1"] and second.cited_chunk_ids == ["m1"]
    replayed = llm.calls[0][0][2]
    assert isinstance(replayed, AssistantTurn) and "[1]" not in replayed.content


# --------------------------------------------------------------------------
# The pipeline, and everything that reads the outcome
# --------------------------------------------------------------------------


class _BlankLLM(LLMClient):
    def generate(self, prompt: str, *, system: str | None = None) -> str:
        return "  "


class _OneChunkRetriever:
    def retrieve(self, query: str, top_k: int | None = None, *, query_filter: Any = None, on_event: Any = None):  # type: ignore[no-untyped-def]
        return RetrievalResult(chunks=[_chunk("c1")], candidate_count=1)


def test_a_blank_pipeline_generation_is_a_generation_failure() -> None:
    records: list[TurnRecord] = []

    answer = ChatService(_OneChunkRetriever(), _BlankLLM()).ask("q", on_record=records.append)  # type: ignore[arg-type]

    assert answer.generation_failure == "empty_output"
    assert answer.answer == GENERATION_FAILURE_ANSWERS["empty_output"]
    assert records[0].outcome == "generation_failed"


class _FixedResponder:
    def __init__(self, answer: ChatAnswer) -> None:
        self.answer = answer

    def ask(self, query: str, **_: Any) -> ChatAnswer:
        return self.answer


class _NoJudge(LLMClient):
    def generate(self, prompt: str, *, system: str | None = None) -> str:
        raise AssertionError("a generation failure must not be judged")


_FAILED = ChatAnswer(answer=GENERATION_FAILURE_ANSWERS["deadline"], generation_failure="deadline")


def test_answer_eval_fails_a_generation_failure_even_on_a_refusal_sample() -> None:
    from rag.eval.answer_eval import run_answer_eval
    from rag.eval.dataset import EvalDataset, EvalSample

    dataset = EvalDataset(samples=[
        EvalSample(id="r1", query="q", expected_doc_ids=[], expected_answer="Refuse.", extra={"tier": "refusal"}),
    ])

    report = run_answer_eval(dataset, _FixedResponder(_FAILED), _NoJudge())  # type: ignore[arg-type]

    result = report.sample_results[0]
    assert (result.passed, result.generation_failure, report.num_empty) == (False, "deadline", 1)


def test_multihop_and_musique_score_a_generation_failure_without_judging_it() -> None:
    from rag.eval.dataset import EvalSample
    from rag.eval.multihop_eval import run_multihop_eval
    from rag.eval.musique_eval import score_sample
    from rag.eval.dataset import EvalDataset

    sample = EvalSample(
        id="m1", query="q", expected_doc_ids=[], expected_answer="Paris",
        extra={"kind": "2hop", "parts": [{"label": "p", "answer": "x", "spans": []}]},
    )
    report = run_multihop_eval(EvalDataset(samples=[sample]), _FixedResponder(_FAILED), _NoJudge())  # type: ignore[arg-type]
    assert [p.passed for p in report.sample_results[0].part_results] == [False]
    assert report.num_empty == 1

    scored = score_sample(sample, _FAILED, _NoJudge(), 1.0)
    assert (scored.em, scored.contains, scored.generation_failure) == (0, False, "deadline")


def test_the_api_reports_the_failure_stop_reason_and_invalid_citations() -> None:
    from fastapi.testclient import TestClient

    from rag.api.main import app
    from rag.api.routes.chat import get_chat_service

    fake = _FixedResponder(ChatAnswer(
        answer=GENERATION_FAILURE_ANSWERS["deadline"], generation_failure="deadline",
        stopped_reason="cap", invalid_citations=[7],
    ))
    app.dependency_overrides[get_chat_service] = lambda: fake
    try:
        body = TestClient(app).post("/chat", json={"query": "q"}).json()
    finally:
        app.dependency_overrides.pop(get_chat_service, None)

    assert (body["generation_failure"], body["stopped_reason"], body["invalid_citations"]) == ("deadline", "cap", [7])
