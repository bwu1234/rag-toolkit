"""The agent loop (`rag.agent.service`), against a scripted model and a fake search.

Hermetic: `ScriptedToolLLM` replays fixed turns and records what each call was
sent, and `_FakeTools` answers `retrieve` from a table, so each guard can be
driven exactly to its edge.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import pytest

from rag.config.settings import AgentConfig, ChatConfig, CorpusSelection, CragConfig, LLMConfig, RagConfig
from rag.events import EventSink, PipelineEvent
from rag.agent.service import AgentService, PassageLedger
from rag.chat import build_chat_service
from rag.generation.chat_service import BLANK_QUERY_ANSWER, GENERATION_FAILURE_ANSWERS
from rag.llm.base import (
    AssistantTurn,
    ChatMessage,
    ContextOverflowError,
    LLMUsage,
    Message,
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from rag.agent.prompts import AGENT_SYNTHESIS_INSTRUCTION
from rag.observability.usage import metered_client
from rag.query_filter import QueryFilter
from rag.retrieval.retriever import RetrievalResult
from rag.tools import RagTools
from rag.vectorstore.base import ScoredChunk
from tests.fakes import ScriptedToolLLM


def _chunk(chunk_id: str, text: str = "passage text", doc: str = "doc.md") -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk_id,
        text=text,
        document_id=doc,
        source=Path(f"data/{doc}"),
        doc_type="markdown",
        score=0.9,
    )


class _FakeTools(RagTools):
    """`RagTools` whose `retrieve` reads a table instead of building a retriever."""

    def __init__(self, results: dict[str, list[ScoredChunk]] | None = None) -> None:
        super().__init__(config=RagConfig())
        self.results = results or {}
        self.searches: list[tuple[str, list[str] | None]] = []
        self.filters: list[object] = []

    def retrieve(  # type: ignore[override]
        self,
        query: str,
        corpus: str | Sequence[str] | None = None,
        top_k: int | None = None,
        *,
        query_filter: object = None,
        on_event: EventSink | None = None,
    ) -> tuple[CorpusSelection, RetrievalResult]:
        self.filters.append(query_filter)
        if not query.strip():
            raise ValueError("query must not be empty")
        self.searches.append((query, list(corpus) if corpus is not None else None))
        chunks = self.results.get(query, [])
        selection = self.config.corpus_selection(None)
        return selection, RetrievalResult(chunks=chunks, candidate_count=len(chunks) or 3)


def _search(query: str, call_id: str | None = None) -> ToolCall:
    return ToolCall(name="rag_search", arguments={"query": query}, id=call_id)


def _step(*calls: ToolCall) -> AssistantTurn:
    return AssistantTurn(content="", tool_calls=calls, usage=LLMUsage(100, 10))


def _answer(text: str) -> AssistantTurn:
    return AssistantTurn(content=text, usage=LLMUsage(200, 20))


def _agent(
    llm: ScriptedToolLLM,
    tools: _FakeTools,
    **overrides: object,
) -> AgentService:
    options: dict[str, object] = {"system_prompt": "SYSTEM", "corpora": ["baseline"], "max_tool_calls": 8}
    options.update(overrides)
    return AgentService(llm, tools, **options)  # type: ignore[arg-type]


def _tool_results(messages: list[Message]) -> list[ToolResult]:
    return [m for m in messages if isinstance(m, ToolResult)]


# --------------------------------------------------------------------------
# Passage ledger
# --------------------------------------------------------------------------


def test_ledger_numbers_in_first_seen_order_and_keeps_a_repeat_on_its_first_number() -> None:
    ledger = PassageLedger(max_chars=100)

    first = ledger.add([_chunk("a"), _chunk("b")])
    second = ledger.add([_chunk("c"), _chunk("a")])

    assert [(n, c.chunk_id, new) for n, c, new in first] == [(1, "a", True), (2, "b", True)]
    assert [(n, c.chunk_id, new) for n, c, new in second] == [(3, "c", True), (1, "a", False)]
    assert [c.chunk_id for c in ledger.chunks] == ["a", "b", "c"]


def test_ledger_keeps_the_text_the_model_was_shown() -> None:
    ledger = PassageLedger(max_chars=5)
    ledger.add([_chunk("a", text="0123456789")])
    assert ledger.chunks[0].text == "01234 [...]"


def test_ledger_truncate_forgets_numbers_so_they_are_reassigned() -> None:
    ledger = PassageLedger(max_chars=100)
    ledger.add([_chunk("a"), _chunk("b")])
    ledger.truncate(1)
    assert [n for n, _, _ in ledger.add([_chunk("b")])] == [2]


# --------------------------------------------------------------------------
# react
# --------------------------------------------------------------------------


def test_react_searches_then_answers_with_citations_from_the_ledger() -> None:
    tools = _FakeTools({
        "delta revenue 2025": [_chunk("d1", doc="DAL.md"), _chunk("d2", doc="DAL.md")],
        "united revenue 2025": [_chunk("u1", doc="UAL.md"), _chunk("d2", doc="DAL.md")],
    })
    llm = ScriptedToolLLM([
        _step(_search("delta revenue 2025"), _search("united revenue 2025")),
        _answer("Delta grew [1]; United grew [3], see also [9]."),
    ])

    answer = _agent(llm, tools).ask("Compare Delta and United revenue")

    assert answer.answer == "Delta grew [1]; United grew [3], see also [9]."
    assert [c.chunk_id for c in answer.citations] == ["d1", "d2", "u1"]
    # [9] points at nothing shown, so it maps to nothing.
    assert answer.cited_chunk_ids == ["d1", "u1"]
    assert answer.search_queries == ["delta revenue 2025", "united revenue 2025"]
    assert (answer.tool_calls, answer.retrieval_attempts, answer.stopped_reason) == (2, 1, "answered")
    # The turn's corpus selection is pinned on every search.
    assert all(corpus == ["baseline"] for _, corpus in tools.searches)
    # A chunk the second search found again is referenced, not repeated.
    united = _tool_results(llm.calls[1][0])[1].content
    assert "Passage [4]" not in united and "Passage [2] (source: DAL.md): already shown above." in united


def test_react_offers_the_search_tool_with_pinned_arguments_removed() -> None:
    llm = ScriptedToolLLM([_answer("Hello!")])
    agent = _agent(llm, _FakeTools())

    answer = agent.ask("hi")

    [tool] = llm.calls[0][1]
    assert tool.name == "rag_search"
    assert set(tool.parameters["properties"]) == {"query"}
    # Declining to search is the routing decision; it costs one call and no search.
    assert (answer.answer, answer.tool_calls, answer.stopped_reason) == ("Hello!", 0, "answered")


def test_cap_forces_a_synthesis_turn_with_tools_removed() -> None:
    tools = _FakeTools({f"q{i}": [_chunk(f"c{i}")] for i in range(5)})
    llm = ScriptedToolLLM([
        _step(_search("q0"), _search("q1")),
        _step(_search("q2"), _search("q3"), _search("q4")),
        _answer("final [1]"),
    ])

    answer = _agent(llm, tools, max_tool_calls=3).ask("question")

    assert answer.search_queries == ["q0", "q1", "q2"]
    assert (answer.tool_calls, answer.stopped_reason) == (3, "cap")
    messages, offered = llm.calls[-1]
    assert offered == []
    # Every call got a result -- the over-budget ones say why -- and the
    # instruction to answer now is its own message, after them.
    results = _tool_results(messages)
    assert len(results) == 5
    assert "was not run" in results[3].content
    assert messages[-1] == ChatMessage("user", AGENT_SYNTHESIS_INSTRUCTION)
    assert answer.answer == "final [1]"


def test_cap_is_reached_by_steps_when_every_call_is_refused() -> None:
    tools = _FakeTools({"q": [_chunk("c")]})
    llm = ScriptedToolLLM([_step(_search("q")), _step(_search("q")), _answer("done")])

    answer = _agent(llm, tools, max_tool_calls=2).ask("question")

    # One real search; the model then only repeated itself, and ran out of steps.
    assert (answer.tool_calls, answer.stopped_reason) == (1, "cap")
    assert llm.calls[-1][1] == []


def test_a_duplicate_query_spends_no_search_and_points_at_the_earlier_passages() -> None:
    tools = _FakeTools({"Tesla margin": [_chunk("t1"), _chunk("t2")]})
    llm = ScriptedToolLLM([
        _step(_search("Tesla margin")),
        _step(_search("  tesla   MARGIN ")),
        _answer("Not covered."),
    ])

    answer = _agent(llm, tools).ask("Tesla's margin?")

    assert [q for q, _ in tools.searches] == ["Tesla margin"]
    assert answer.tool_calls == 1
    refusal = _tool_results(llm.calls[2][0])[-1].content
    assert refusal.startswith("Already searched for") and "[1], [2]" in refusal


def test_an_unknown_tool_or_missing_query_is_answered_without_a_search() -> None:
    tools = _FakeTools()
    llm = ScriptedToolLLM([
        _step(ToolCall(name="rag_list_corpora", arguments={}), ToolCall(name="rag_search", arguments={})),
        _answer("ok"),
    ])

    answer = _agent(llm, tools).ask("question")

    assert tools.searches == []
    first, second = _tool_results(llm.calls[1][0])
    assert "Unknown tool" in first.content and "non-empty 'query'" in second.content
    assert (answer.tool_calls, answer.retrieval_attempts) == (0, 0)


def test_timeout_takes_the_synthesis_turn_with_what_was_found() -> None:
    now = [0.0]
    tools = _FakeTools({"q1": [_chunk("c1")]})

    class _SlowLLM(ScriptedToolLLM):
        def chat(self, messages, tools=()):  # type: ignore[no-untyped-def]
            now[0] += 40.0
            return super().chat(messages, tools)

    llm = _SlowLLM([_step(_search("q1")), _step(_search("q2")), _answer("best so far [1]")])

    answer = _agent(llm, tools, timeout_s=60.0, clock=lambda: now[0]).ask("question")

    # 40 s for step one, 80 s after step two: q2 is refused, then one tool-free call.
    assert [q for q, _ in tools.searches] == ["q1"]
    assert (answer.answer, answer.stopped_reason) == ("best so far [1]", "timeout")
    assert llm.calls[-1][1] == []
    assert [c.chunk_id for c in answer.citations] == ["c1"]


def test_context_overflow_rolls_back_the_last_step_and_answers_from_earlier_passages() -> None:
    tools = _FakeTools({"q1": [_chunk("c1")], "q2": [_chunk("c2")]})

    class _OverflowOnce(ScriptedToolLLM):
        def chat(self, messages, tools=()):  # type: ignore[no-untyped-def]
            if len(self.calls) == 2 and not getattr(self, "overflowed", False):
                self.overflowed = True
                raise ContextOverflowError("too long")
            return super().chat(messages, tools)

    llm = _OverflowOnce([_step(_search("q1")), _step(_search("q2")), _answer("from [1]")])

    answer = _agent(llm, tools).ask("question")

    assert answer.stopped_reason == "context"
    # q2 ran (and counts), but its passage was never shown, so it can't be cited.
    assert answer.tool_calls == 2
    assert [c.chunk_id for c in answer.citations] == ["c1"]
    final_messages = llm.calls[-1][0]
    assert _tool_results(final_messages)[-1].content.startswith("Results not shown")
    assert final_messages[-1] == ChatMessage("user", AGENT_SYNTHESIS_INSTRUCTION)
    # The trace says the same: q2 searched, but showed the model nothing.
    rolled_back = answer.agent_calls[-1]
    assert (rolled_back.query, rolled_back.status, rolled_back.passages) == ("q2", "ran", [])
    assert rolled_back.note is not None and rolled_back.note.startswith("Results not shown")
    assert answer.agent_calls[0].chunk_ids == ["c1"]


def test_overflow_with_nothing_to_roll_back_is_a_context_generation_failure() -> None:
    class _Overflow(ScriptedToolLLM):
        def chat(self, messages, tools=()):  # type: ignore[no-untyped-def]
            raise ContextOverflowError("too long")

    answer = _agent(_Overflow([]), _FakeTools()).ask("question")

    assert (answer.generation_failure, answer.stopped_reason) == ("context", "context")
    assert answer.answer == GENERATION_FAILURE_ANSWERS["context"]


# --------------------------------------------------------------------------
# An empty reply
# --------------------------------------------------------------------------


def _empty(stop_reason: str | None = "length", thinking: str | None = "x" * 900) -> AssistantTurn:
    return AssistantTurn(content="", thinking=thinking, usage=LLMUsage(300, 4096), stop_reason=stop_reason)


def test_an_empty_reply_after_searching_gets_the_forced_answer_turn() -> None:
    # The Milestone 19 re-run case: one search, then neither text nor a tool call.
    tools = _FakeTools({"eli lilly incretin": [_chunk("m1", doc="MRK.md")]})
    llm = ScriptedToolLLM([_step(_search("eli lilly incretin")), _empty(), _answer("Eli Lilly is not covered.")])

    answer = _agent(llm, tools).ask("What did Eli Lilly say about incretin capacity?")

    assert (answer.answer, answer.stopped_reason) == ("Eli Lilly is not covered.", "empty")
    messages, offered = llm.calls[-1]
    assert offered == []
    assert messages[-1] == ChatMessage("user", AGENT_SYNTHESIS_INSTRUCTION)
    # The empty turn isn't sent back: the forced turn follows the tool results.
    assert isinstance(messages[-2], ToolResult)
    assert not any(isinstance(m, AssistantTurn) and m.thinking == "x" * 900 for m in messages)


def test_an_empty_plan_gets_the_forced_answer_turn() -> None:
    llm = ScriptedToolLLM([_empty(), _answer("Hello!")])

    answer = _agent(llm, _FakeTools(), strategy="planned").ask("hi")

    assert (answer.answer, answer.stopped_reason, len(llm.calls)) == ("Hello!", "empty", 2)
    assert llm.calls[-1][1] == []


def test_an_empty_forced_turn_is_not_retried_again(caplog: pytest.LogCaptureFixture) -> None:
    llm = ScriptedToolLLM([_step(_search("q")), _empty(), _empty()])

    with caplog.at_level(logging.WARNING, logger="rag.agent.service"):
        answer = _agent(llm, _FakeTools({"q": [_chunk("c")]})).ask("question")

    # One recovery attempt, then a generation failure rather than a loop -- or
    # an empty answer passed off as one.
    assert (answer.generation_failure, answer.stopped_reason, len(llm.calls)) == ("empty_output", "empty", 3)
    assert answer.answer == GENERATION_FAILURE_ANSWERS["empty_output"]
    assert answer.cited_chunk_ids == [] and answer.invalid_citations == []
    # The log says why, as far as the provider reported it.
    assert "stop_reason=length, 900 reasoning chars" in caplog.text
    assert "raise the agent model's max_tokens or set think: low" in caplog.text


def test_an_empty_reply_without_a_length_stop_gets_no_max_tokens_hint(caplog: pytest.LogCaptureFixture) -> None:
    llm = ScriptedToolLLM([_empty(stop_reason="stop", thinking=None), _answer("ok")])

    with caplog.at_level(logging.WARNING, logger="rag.agent.service"):
        _agent(llm, _FakeTools()).ask("question")

    assert "stop_reason=stop, 0 reasoning chars" in caplog.text
    assert "max_tokens" not in caplog.text


# --------------------------------------------------------------------------
# planned
# --------------------------------------------------------------------------


@pytest.mark.parametrize("sub_queries", [1, 3, 6])
def test_planned_calls_the_model_exactly_twice(sub_queries: int) -> None:
    queries = [f"q{i}" for i in range(sub_queries)]
    tools = _FakeTools({q: [_chunk(q)] for q in queries})
    llm = ScriptedToolLLM([_step(*(_search(q) for q in queries)), _answer("answer")])

    answer = _agent(llm, tools, strategy="planned").ask("question")

    assert len(llm.calls) == 2
    assert llm.calls[1][1] == []
    assert (answer.tool_calls, answer.retrieval_attempts, answer.stopped_reason) == (sub_queries, 1, "answered")


def test_planned_truncates_a_plan_longer_than_the_cap_and_dedups_it() -> None:
    tools = _FakeTools()
    plan = [_search("a"), _search("A"), _search("b"), _search("c"), _search("d")]
    llm = ScriptedToolLLM([_step(*plan), _answer("answer")])

    answer = _agent(llm, tools, strategy="planned", max_tool_calls=2).ask("question")

    assert [q for q, _ in tools.searches] == ["a", "b"]
    assert (len(llm.calls), answer.stopped_reason) == (2, "cap")


def test_planned_with_exactly_cap_queries_is_not_reported_as_cut_off() -> None:
    tools = _FakeTools()
    llm = ScriptedToolLLM([_step(_search("a"), _search("b")), _answer("answer")])
    answer = _agent(llm, tools, strategy="planned", max_tool_calls=2).ask("question")
    assert answer.stopped_reason == "answered"


# --------------------------------------------------------------------------
# The turn around the loop
# --------------------------------------------------------------------------


def test_history_goes_to_the_model_without_old_citation_markers() -> None:
    from rag.generation.query_rewriter import ChatTurn

    llm = ScriptedToolLLM([_answer("sure")])
    history = [ChatTurn("user", "old q"), ChatTurn("assistant", "Revenue rose [1][2].")]

    _agent(llm, _FakeTools()).ask("and margins?", history=history)

    system, user, assistant, question = llm.calls[0][0]
    assert system == ChatMessage("system", "SYSTEM")
    assert user == ChatMessage("user", "old q")
    assert isinstance(assistant, AssistantTurn) and assistant.content == "Revenue rose."
    assert question == ChatMessage("user", "and margins?")


def test_blank_query_makes_no_model_call() -> None:
    llm = ScriptedToolLLM([])
    assert _agent(llm, _FakeTools()).ask("   ").answer == BLANK_QUERY_ANSWER


def test_ask_meters_every_agent_call_and_emits_one_event_per_step() -> None:
    tools = _FakeTools({"q": [_chunk("c")]})
    llm = metered_client(ScriptedToolLLM([_step(_search("q")), _answer("done [1]")]))
    events: list[PipelineEvent] = []

    answer = _agent(llm, tools).ask("question", on_event=events.append)  # type: ignore[arg-type]

    assert (answer.llm_calls, answer.prompt_tokens, answer.completion_tokens) == (2, 300, 30)
    assert [e.stage for e in events] == ["agent_step", "search", "agent_step"]
    assert set(answer.stage_ms) == {"agent_step", "search"}


def test_groundedness_is_checked_against_the_ledger_when_configured() -> None:
    class _Checker:
        def __init__(self) -> None:
            self.seen: list[list[str]] = []

        def check(self, query: str, chunks: list[ScoredChunk], answer: str) -> bool | None:
            self.seen.append([c.chunk_id for c in chunks])
            return False

    checker = _Checker()
    tools = _FakeTools({"q": [_chunk("c")]})
    llm = ScriptedToolLLM([_step(_search("q")), _answer("done [1]")])

    answer = _agent(llm, tools, groundedness_checker=checker).ask("question")

    assert (answer.grounded, checker.seen) == (False, [["c"]])


# --------------------------------------------------------------------------
# Builder
# --------------------------------------------------------------------------


def test_build_chat_service_builds_the_agent_in_agentic_mode() -> None:
    config = RagConfig(
        chat=ChatConfig(mode="agentic"),
        agent=AgentConfig(strategy="planned", llm=LLMConfig(provider="ollama", model="qwen3.8:27b-mlx")),
        crag=CragConfig(enabled=True, check_groundedness=True, grade_documents=False, max_retries=0),
    )

    # Nothing is loaded yet: RagTools builds its retriever on the first search.
    agent = build_chat_service(config)

    assert isinstance(agent, AgentService)
    assert agent.strategy == "planned"
    assert agent._groundedness_checker is not None
    assert agent._turn_metadata["llm"] == "ollama:qwen3.8:27b-mlx"
    assert agent._turn_metadata["mode"] == "agentic:planned"


def test_agentic_mode_fails_at_build_time_for_a_model_without_tools() -> None:
    config = RagConfig(chat=ChatConfig(mode="agentic"), agent=AgentConfig(llm=LLMConfig(provider="openai", model="x")))
    with pytest.raises(ValueError):
        build_chat_service(config)


def test_the_search_tool_the_agent_offers_is_the_mcp_schema_projected() -> None:
    from rag.tools import build_tool_specs

    tools = _FakeTools()
    mcp = next(s for s in build_tool_specs(tools) if s.name == "rag_search").definition
    offered: ToolDefinition = _agent(ScriptedToolLLM([]), tools).search_tool

    assert offered.name == mcp.name
    # The MCP text plus notes for tools the agent doesn't offer (rag_read_document).
    assert mcp.description.startswith(offered.description)
    assert offered.parameters["properties"]["query"] == mcp.parameters["properties"]["query"]
    assert offered.parameters["required"] == mcp.parameters["required"] == ["query"]


@pytest.mark.parametrize("offered_tools", [("rag_search",), ("rag_search", "rag_list_documents", "calculator")])
def test_offered_descriptions_never_name_a_tool_or_argument_the_agent_lacks(offered_tools: tuple[str, ...]) -> None:
    """The shared descriptions point MCP clients at reading and paging; the agent has neither."""

    agent = _agent(ScriptedToolLLM([]), _ListingTools(_AIRLINES), offered_tools=offered_tools)

    for tool in agent.tool_definitions:
        for absent in ("rag_read_document", "rag_find", "char_start", "offset"):
            assert absent not in tool.description, (tool.name, absent)


def test_a_turn_filter_applies_to_every_agent_search_and_the_model_cannot_set_one() -> None:
    tools = _FakeTools({"q0": [_chunk("c0")], "q1": [_chunk("c1")]})
    llm = ScriptedToolLLM([_step(_search("q0"), _search("q1")), _answer("Done [1][2].")])
    aapl = QueryFilter(equals={"ticker": "AAPL"})

    _agent(llm, tools).ask("question", query_filter=aapl)

    assert tools.filters == [aapl, aapl]
    [tool] = llm.calls[0][1]
    assert "filters" not in tool.parameters["properties"]


# --------------------------------------------------------------------------
# Model-chosen filters (`agent.model_filters`) and the search trace
# --------------------------------------------------------------------------


def _filtered(query: str, filters: object) -> ToolCall:
    return ToolCall(name="rag_search", arguments={"query": query, "filters": filters})


def test_with_model_filters_off_the_tool_carries_no_filter_types() -> None:
    tool_off = _agent(ScriptedToolLLM([]), _FakeTools()).search_tool
    tool_on = _agent(ScriptedToolLLM([]), _FakeTools(), model_filters=True).search_tool

    assert "$defs" not in tool_off.parameters
    assert set(tool_on.parameters["properties"]) == {"query", "filters"}
    assert set(tool_on.parameters["$defs"]) == {"QueryFilter", "IntRange"}


def test_a_model_filter_reaches_retrieval_and_narrows_the_turn_filter() -> None:
    tools = _FakeTools({"fuel": [_chunk("dal")]})
    llm = ScriptedToolLLM([
        _step(_filtered("fuel", {"range": {"period_end": {"gte": "2026-06-30", "lte": "2026-06-30"}}})),
        _answer("[1]"),
    ])
    turn_filter = QueryFilter(equals={"ticker": "DAL"})

    answer = _agent(llm, tools, model_filters=True).ask("question", query_filter=turn_filter)

    assert tools.filters == [
        QueryFilter(equals={"ticker": "DAL"}, range={"period_end": {"gte": 20260630, "lte": 20260630}})
    ]
    [search] = answer.agent_calls
    # The trace keeps the model's own filter, as applied and as sent; the
    # turn's is on the record separately.
    assert search.filters == {"range": {"period_end": {"gte": 20260630, "lte": 20260630}}}
    assert search.filters_raw == {"range": {"period_end": {"gte": "2026-06-30", "lte": "2026-06-30"}}}


def test_a_model_filter_that_contradicts_the_turn_filter_is_an_error_not_a_search() -> None:
    tools = _FakeTools({"fuel": [_chunk("dal")]})
    llm = ScriptedToolLLM([_step(_filtered("fuel", {"equals": {"ticker": "UAL"}})), _answer("Nothing.")])

    answer = _agent(llm, tools, model_filters=True).ask(
        "question", query_filter=QueryFilter(equals={"ticker": "DAL"})
    )

    assert tools.searches == [] and answer.tool_calls == 0
    [search] = answer.agent_calls
    assert search.status == "error" and "ticker" in (search.note or "")
    assert _tool_results(llm.calls[1][0])[0].content.startswith("Search error:")


def test_the_same_query_under_another_filter_is_a_new_search_and_the_same_one_is_refused() -> None:
    tools = _FakeTools({"fuel": [_chunk("c")]})
    dal, ual = {"equals": {"ticker": "DAL"}}, {"equals": {"ticker": "UAL"}}
    llm = ScriptedToolLLM([
        _step(_filtered("fuel", dal), _filtered("fuel", ual), _filtered("Fuel ", dal)),
        _answer("[1]"),
    ])

    answer = _agent(llm, tools, model_filters=True).ask("question")

    assert answer.tool_calls == 2
    assert [s.status for s in answer.agent_calls] == ["ran", "ran", "refused"]
    assert "[filter: ticker=DAL]" in (answer.agent_calls[2].note or "")


def test_the_same_filter_written_in_another_order_is_a_repeat() -> None:
    tools = _FakeTools({"fuel": [_chunk("c")]})
    first = {"equals": {"ticker": "DAL", "form": "10-Q"}, "any_of": {"company": ["A", "B"]}}
    reordered = {"any_of": {"company": ["B", "A"]}, "equals": {"form": "10-Q", "ticker": "DAL"}}
    llm = ScriptedToolLLM([_step(_filtered("fuel", first), _filtered("fuel", reordered)), _answer("[1]")])

    answer = _agent(llm, tools, model_filters=True).ask("question")

    assert answer.tool_calls == 1
    assert [s.status for s in answer.agent_calls] == ["ran", "refused"]


def test_an_invalid_filter_is_reported_to_the_model_without_spending_a_search() -> None:
    tools = _FakeTools({"fuel": [_chunk("c")]})
    llm = ScriptedToolLLM([_step(_filtered("fuel", {"equals": {"ticker": 7}, "bogus": {}})), _answer("?")])

    answer = _agent(llm, tools, model_filters=True).ask("question")

    assert tools.searches == []
    [search] = answer.agent_calls
    assert search.status == "error" and (search.note or "").startswith("Search error: invalid filters")


def test_a_filter_sent_while_model_filters_is_off_is_rejected_not_dropped() -> None:
    tools = _FakeTools({"fuel": [_chunk("c")]})
    llm = ScriptedToolLLM([_step(_filtered("fuel", {"equals": {"ticker": "DAL"}})), _answer("?")])

    answer = _agent(llm, tools).ask("question")

    assert tools.searches == []
    assert answer.agent_calls[0].status == "error"


def test_a_filtered_search_matching_nothing_blames_the_filter_not_the_index() -> None:
    class _NoCandidates(_FakeTools):
        def retrieve(self, query, corpus=None, top_k=None, *, query_filter=None, on_event=None):  # type: ignore[no-untyped-def]
            selection, _ = super().retrieve(query, corpus, top_k, query_filter=query_filter, on_event=on_event)
            return selection, RetrievalResult(chunks=[], candidate_count=0)

    llm = ScriptedToolLLM([_step(_filtered("fuel", {"equals": {"ticker": "DLA"}})), _answer("?")])

    _agent(llm, _NoCandidates(), model_filters=True).ask("question")

    [result] = _tool_results(llm.calls[1][0])
    assert result.content.startswith("No passages match the filter (ticker=DLA)")


def test_the_search_trace_records_every_call_with_its_step_passages_and_repeats() -> None:
    tools = _FakeTools({"q1": [_chunk("a"), _chunk("b")], "q2": [_chunk("b"), _chunk("c")]})
    llm = ScriptedToolLLM([
        _step(_search("q1")),
        _step(_search("q2"), _search("q1"), ToolCall(name="rag_search", arguments={})),
        _answer("[1][3]"),
    ])

    answer = _agent(llm, tools).ask("question")

    rows = [(s.step, s.query, s.status, s.passages, s.new_passages, s.chunk_ids) for s in answer.agent_calls]
    assert rows == [
        (1, "q1", "ran", [1, 2], [1, 2], ["a", "b"]),
        (2, "q2", "ran", [2, 3], [3], ["b", "c"]),
        (2, "q1", "refused", [], [], []),
        (2, "", "refused", [], [], []),
    ]
    assert answer.search_queries == ["q1", "q2"]


def test_a_filter_sent_as_a_json_string_is_decoded() -> None:
    tools = _FakeTools({"fuel": [_chunk("dal")]})
    llm = ScriptedToolLLM([_step(_filtered("fuel", '{"equals": {"ticker": "DAL"}}')), _answer("[1]")])

    answer = _agent(llm, tools, model_filters=True).ask("question")

    assert tools.filters == [QueryFilter(equals={"ticker": "DAL"})]
    assert answer.agent_calls[0].filters == {"equals": {"ticker": "DAL"}}
    assert answer.agent_calls[0].filters_raw == '{"equals": {"ticker": "DAL"}}'


def test_an_invalid_filter_error_names_the_field_without_pydantic_boilerplate() -> None:
    llm = ScriptedToolLLM([_step(_filtered("fuel", "not json")), _step(_filtered("fuel", {"equals": {"ticker": 7}})), _answer("?")])

    answer = _agent(llm, _FakeTools(), model_filters=True).ask("question")

    first, second = (s.note or "" for s in answer.agent_calls)
    # A rejected filter has no applied form; what the model sent is the only record.
    assert [(s.filters, s.filters_raw) for s in answer.agent_calls] == [
        (None, "not json"), (None, {"equals": {"ticker": 7}}),
    ]
    assert first == "Search error: invalid filters: not a JSON object."
    assert second == "Search error: invalid filters: equals.ticker: Input should be a valid string"


# --------------------------------------------------------------------------
# rag_list_documents (`agent.tools`)
# --------------------------------------------------------------------------


_AIRLINES = [
    {"document_id": "DAL_10-Q_2026-06-30.md", "company": "DELTA AIR LINES, INC.", "ticker": "DAL",
     "form": "10-Q", "period_end": "2026-06-30", "chars": 57000},
    {"document_id": "UAL_10-Q_2026-06-30.md", "company": "United Airlines Holdings, Inc.", "ticker": "UAL",
     "form": "10-Q", "period_end": "2026-06-30", "chars": 61000},
]


class _ListingTools(_FakeTools):
    """`_FakeTools` that also lists `documents`, recording the filter of each listing."""

    def __init__(self, documents: list[dict[str, object]], results: dict[str, list[ScoredChunk]] | None = None) -> None:
        super().__init__(results)
        self.documents = documents
        self.listings: list[object] = []

    def list_documents(self, corpus=None, filters=None, limit=None):  # type: ignore[no-untyped-def]
        self.listings.append(filters)
        matched = [d for d in self.documents if filters is None or filters.matches(d)]
        return {"corpora": ["edgar_md"], "total": len(matched), "returned": len(matched), "documents": matched}


def _list_call(filters: object = None) -> ToolCall:
    return ToolCall(name="rag_list_documents", arguments={} if filters is None else {"filters": filters})


def test_the_listing_tool_is_offered_only_when_configured_with_corpus_and_limit_pinned() -> None:
    default = _agent(ScriptedToolLLM([]), _ListingTools(_AIRLINES))
    with_list = _agent(ScriptedToolLLM([]), _ListingTools(_AIRLINES), offered_tools=("rag_search", "rag_list_documents"))

    assert [t.name for t in default.tool_definitions] == ["rag_search"]
    assert [t.name for t in with_list.tool_definitions] == ["rag_search", "rag_list_documents"]
    assert set(with_list.tool_definitions[1].parameters["properties"]) == {"filters"}
    with pytest.raises(ValueError):
        _agent(ScriptedToolLLM([]), _ListingTools(_AIRLINES), offered_tools=("rag_list_documents",))


def test_a_listing_call_when_not_offered_is_refused_naming_the_tools() -> None:
    tools = _ListingTools(_AIRLINES)
    llm = ScriptedToolLLM([_step(_list_call()), _answer("?")])

    answer = _agent(llm, tools).ask("question")

    assert tools.listings == []
    assert answer.agent_calls[0].status == "refused"
    assert "The tools are: rag_search." in (answer.agent_calls[0].note or "")


def test_a_listing_renders_a_line_per_document_spends_budget_but_no_search_round() -> None:
    tools = _ListingTools(_AIRLINES)
    llm = ScriptedToolLLM([_step(_list_call()), _answer("Delta and United.")])

    answer = _agent(llm, tools, offered_tools=("rag_search", "rag_list_documents")).ask("Which airlines?")

    [result] = _tool_results(llm.calls[1][0])
    assert result.content.splitlines()[0] == "2 document(s):"
    assert result.content.splitlines()[1] == (
        "- DAL_10-Q_2026-06-30.md: company DELTA AIR LINES, INC.; ticker DAL; form 10-Q; "
        "period_end 2026-06-30; 57,000 chars"
    )
    assert (answer.tool_calls, answer.retrieval_attempts, answer.citations) == (1, 0, [])
    [call] = answer.agent_calls
    assert (call.tool, call.status, call.documents) == (
        "rag_list_documents", "ran", ["DAL_10-Q_2026-06-30.md", "UAL_10-Q_2026-06-30.md"],
    )


def test_a_listing_filter_narrows_the_turn_filter_and_a_repeat_is_refused() -> None:
    tools = _ListingTools(_AIRLINES)
    llm = ScriptedToolLLM([
        _step(_list_call('{"equals": {"ticker": "UAL"}}'), _list_call({"equals": {"ticker": "UAL"}})),
        _answer("United."),
    ])

    answer = _agent(llm, tools, offered_tools=("rag_search", "rag_list_documents")).ask(
        "question", query_filter=QueryFilter(equals={"form": "10-Q"})
    )

    assert tools.listings == [QueryFilter(equals={"form": "10-Q", "ticker": "UAL"})]
    assert [c.status for c in answer.agent_calls] == ["ran", "refused"]
    assert answer.agent_calls[0].documents == ["UAL_10-Q_2026-06-30.md"]
    assert "Already listed documents [filter: ticker=UAL]" in (answer.agent_calls[1].note or "")


def test_a_filtered_listing_with_no_match_blames_the_filter() -> None:
    llm = ScriptedToolLLM([_step(_list_call({"equals": {"ticker": "AAL"}})), _answer("No American.")])

    _agent(llm, _ListingTools(_AIRLINES), offered_tools=("rag_search", "rag_list_documents")).ask("question")

    [result] = _tool_results(llm.calls[1][0])
    assert result.content.startswith("No documents match the filter (ticker=AAL)")


def test_listings_and_searches_share_one_budget() -> None:
    tools = _ListingTools(_AIRLINES, {"fuel": [_chunk("c")]})
    llm = ScriptedToolLLM([_step(_list_call(), _search("fuel")), _answer("From the list.")])

    answer = _agent(llm, tools, offered_tools=("rag_search", "rag_list_documents"), max_tool_calls=1).ask("q")

    assert tools.searches == []
    assert [c.status for c in answer.agent_calls] == ["ran", "refused"]
    assert "Tool budget for this question is used up; this search was not run." == answer.agent_calls[1].note
    assert answer.stopped_reason == "cap"


def test_build_chat_service_offers_the_configured_tools_and_says_so_in_the_prompt() -> None:
    config = RagConfig(
        chat=ChatConfig(mode="agentic"),
        agent=AgentConfig(llm=LLMConfig(provider="ollama", model="m"), tools=["rag_search", "rag_list_documents"]),
    )

    agent = build_chat_service(config)

    assert isinstance(agent, AgentService)
    assert [t.name for t in agent.tool_definitions] == ["rag_search", "rag_list_documents"]
    assert "call rag_list_documents" in agent._system_prompt


# --------------------------------------------------------------------------
# Calculator
# --------------------------------------------------------------------------


def _calc(expression: object) -> ToolCall:
    return ToolCall(name="calculator", arguments={"expression": expression})


_WITH_CALC = ("rag_search", "calculator")


def test_the_calculator_is_offered_only_when_configured() -> None:
    default = _agent(ScriptedToolLLM([]), _FakeTools())
    with_calc = _agent(ScriptedToolLLM([]), _FakeTools(), offered_tools=_WITH_CALC)

    assert [t.name for t in default.tool_definitions] == ["rag_search"]
    assert [t.name for t in with_calc.tool_definitions] == ["rag_search", "calculator"]
    assert with_calc.tool_definitions[1].parameters["required"] == ["expression"]


def test_a_calculation_returns_its_value_and_spends_no_search_budget() -> None:
    tools = _FakeTools({"fuel": [_chunk("c")]})
    llm = ScriptedToolLLM([
        _step(_calc("(5110 - 2775) / 2775 * 100"), _search("fuel")),
        _answer("Up 84.1% [1]."),
    ])

    answer = _agent(llm, tools, offered_tools=_WITH_CALC, max_tool_calls=1).ask("q")

    [calc_result, _] = _tool_results(llm.calls[1][0])
    assert calc_result.content == "(5110 - 2775) / 2775 * 100 = 84.1441441441"
    assert tools.searches == [("fuel", ["baseline"])]  # the one budgeted call went to search
    assert answer.tool_calls == 1
    calc = answer.agent_calls[0]
    assert (calc.tool, calc.status, calc.expression, calc.result) == (
        "calculator", "ran", "(5110 - 2775) / 2775 * 100", "84.1441441441",
    )


def test_a_bad_expression_tells_the_model_what_to_fix() -> None:
    llm = ScriptedToolLLM([_step(_calc("4,109 - 2,458"), _calc(42)), _answer("?")])

    answer = _agent(llm, _FakeTools(), offered_tools=_WITH_CALC).ask("q")

    first, second = _tool_results(llm.calls[1][0])
    assert first.content.startswith("Calculator error: remove thousands separators")
    assert second.content == "calculator needs an 'expression' string."
    assert [(c.status, c.result) for c in answer.agent_calls] == [("error", None), ("error", None)]


def test_a_calculator_call_when_not_offered_is_refused() -> None:
    llm = ScriptedToolLLM([_step(_calc("1 + 1")), _answer("2")])

    answer = _agent(llm, _FakeTools()).ask("q")

    assert answer.agent_calls[0].status == "refused"
    assert "Unknown tool 'calculator'" in (answer.agent_calls[0].note or "")


def test_the_calculator_hint_is_in_the_prompt_only_when_offered() -> None:
    def prompt(tools: list[str]) -> str:
        config = RagConfig(
            chat=ChatConfig(mode="agentic"),
            agent=AgentConfig(llm=LLMConfig(provider="ollama", model="m"), tools=tools),  # type: ignore[arg-type]
        )
        agent = build_chat_service(config)
        assert isinstance(agent, AgentService)
        return agent._system_prompt

    with_calc, without = prompt(["rag_search", "calculator"]), prompt(["rag_search"])
    assert "call calculator" in with_calc
    assert "calculator" not in without
    # Off, the prompt is the one the measured rows ran with.
    assert "citing them inline as [n], e.g. [2] or [3][5]. Do not add facts" in without
