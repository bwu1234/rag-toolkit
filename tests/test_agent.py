"""The agent loop (`rag.generation.agent`), against a scripted model and a fake search.

Hermetic: `ScriptedToolLLM` replays fixed turns and records what each call was
sent, and `_FakeTools` answers `retrieve` from a table, so each guard can be
driven exactly to its edge.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from rag.config.settings import AgentConfig, ChatConfig, CorpusSelection, CragConfig, LLMConfig, RagConfig
from rag.events import EventSink, PipelineEvent
from rag.generation.agent import AgentService, PassageLedger
from rag.generation.builder import build_chat_service
from rag.generation.chat_service import BLANK_QUERY_ANSWER
from rag.generation.llm import (
    AssistantTurn,
    ChatMessage,
    ContextOverflowError,
    LLMUsage,
    Message,
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from rag.generation.prompts import AGENT_SYNTHESIS_INSTRUCTION
from rag.observability.usage import metered_client
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

    def retrieve(  # type: ignore[override]
        self,
        query: str,
        corpus: str | Sequence[str] | None = None,
        top_k: int | None = None,
        *,
        on_event: EventSink | None = None,
    ) -> tuple[CorpusSelection, RetrievalResult]:
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


def test_overflow_with_nothing_to_roll_back_raises() -> None:
    class _Overflow(ScriptedToolLLM):
        def chat(self, messages, tools=()):  # type: ignore[no-untyped-def]
            raise ContextOverflowError("too long")

    with pytest.raises(ContextOverflowError):
        _agent(_Overflow([]), _FakeTools()).ask("question")


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

    assert (offered.name, offered.description) == (mcp.name, mcp.description)
    assert offered.parameters["properties"]["query"] == mcp.parameters["properties"]["query"]
    assert offered.parameters["required"] == mcp.parameters["required"] == ["query"]
