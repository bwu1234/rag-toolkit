"""Turn transcripts: verbatim LLM-call capture, `ask(on_record=...)`, and the Markdown render.

Hermetic: scripted models and fake retrieval. What these pin is that the
transcript is what the model was actually sent -- snapshotted before the agent
appends to its conversation, the system prompt included -- and that asking for
one changes nothing about the turn log.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from rag.agent.service import AgentService
from rag.generation.chat_service import ChatService
from rag.llm.base import AssistantTurn, ChatMessage, LLMUsage, Message, ToolCall, ToolDefinition
from rag.observability.records import LLMExchange, TurnRecord
from rag.observability.transcript import _fence, render_turn_markdown, write_trace
from rag.observability.usage import MeteredLLMClient, metered, metered_client
from tests.fakes import ScriptedToolLLM
from tests.test_agent import _FakeTools, _chunk
from tests.test_observability import _RecordingSink, _Retriever, _UsageLLM, _result

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import trace_question  # noqa: E402

_TOOL = ToolDefinition(name="rag_search", description="Search.", parameters={"type": "object"})


def test_capture_is_off_by_default_and_records_generate_calls_verbatim_when_on() -> None:
    client = MeteredLLMClient(_UsageLLM("reply"))

    with metered() as meter:
        client.generate("p0", system="s0")
    assert meter.exchanges is None

    with metered(capture=True) as meter:
        client.generate("p1", system="s1")
    assert meter.exchanges is not None
    [exchange] = meter.exchanges
    assert (exchange.kind, exchange.system, exchange.prompt, exchange.response) == ("generate", "s1", "p1", "reply")
    assert (exchange.prompt_tokens, exchange.completion_tokens) == (100, 20)
    assert meter.calls == 1


def test_chat_capture_snapshots_the_conversation_before_the_caller_appends_to_it() -> None:
    reply = AssistantTurn(
        content="", tool_calls=(ToolCall(name="rag_search", arguments={"query": "q"}),),
        thinking="hmm", usage=LLMUsage(5, 1), stop_reason="stop",
    )
    client = metered_client(ScriptedToolLLM([reply]))
    messages: list[Message] = [ChatMessage("system", "SYS"), ChatMessage("user", "Q")]

    with metered(capture=True) as meter:
        client.chat(messages, [_TOOL])
        messages.append(reply)

    assert meter.exchanges is not None
    [exchange] = meter.exchanges
    assert exchange.messages == [{"role": "system", "content": "SYS"}, {"role": "user", "content": "Q"}]
    assert exchange.tools == [{"name": "rag_search", "description": "Search.", "parameters": {"type": "object"}}]
    assert exchange.tool_calls == [{"name": "rag_search", "arguments": {"query": "q"}}]
    assert (exchange.thinking, exchange.stop_reason) == ("hmm", "stop")


def test_a_call_that_raises_is_captured_with_its_error() -> None:
    client = metered_client(ScriptedToolLLM([]))

    with metered(capture=True) as meter, pytest.raises(AssertionError):
        client.chat([ChatMessage("user", "Q")])

    assert meter.exchanges is not None
    assert meter.exchanges[0].error is not None and "ran out of turns" in meter.exchanges[0].error


def test_on_record_gets_the_transcript_and_the_turn_log_does_not() -> None:
    sink = _RecordingSink()
    service = ChatService(
        retriever=_Retriever(_result("a")), llm_client=MeteredLLMClient(_UsageLLM("ans [1]")), turn_sink=sink
    )
    records: list[TurnRecord] = []

    service.ask("question", on_record=records.append)

    [record] = records
    assert record.llm_exchanges is not None and len(record.llm_exchanges) == 1
    exchange = record.llm_exchanges[0]
    assert exchange.system and "text of a" in (exchange.prompt or "")
    assert sink.turns[0].llm_exchanges is None
    assert sink.turns[0].turn_id == record.turn_id


def test_on_record_works_without_a_turn_sink_and_on_a_failed_turn() -> None:
    class _Down(_UsageLLM):
        def generate_with_usage(self, prompt: str, *, system: str | None = None) -> tuple[str, LLMUsage | None]:
            raise TimeoutError("model down")

    service = ChatService(retriever=_Retriever(_result("a")), llm_client=MeteredLLMClient(_Down()))
    records: list[TurnRecord] = []

    with pytest.raises(TimeoutError):
        service.ask("question", on_record=records.append)

    [record] = records
    assert record.outcome == "error"
    assert record.llm_exchanges is not None and record.llm_exchanges[0].error == "TimeoutError: model down"
    assert "## Error" in render_turn_markdown(record)


def _agent_record() -> TurnRecord:
    llm = ScriptedToolLLM([
        AssistantTurn(content="", tool_calls=(ToolCall(name="rag_search", arguments={"query": "fuel"}),)),
        AssistantTurn(content="Delta, up 67% [1]."),
    ])
    tools = _FakeTools({"fuel": [_chunk("c1", text="Aircraft fuel | 4,109 | 2,458", doc="DAL.md")]})
    agent = AgentService(metered_client(llm), tools, system_prompt="AGENT SYSTEM PROMPT", corpora=["baseline"])
    records: list[TurnRecord] = []
    agent.ask("Which airline?", on_record=records.append)
    return records[0]


def test_an_agent_turn_renders_end_to_end() -> None:
    record = _agent_record()
    markdown = render_turn_markdown(record, title="Trace: ad-airline-fuel", expected_answer="Delta")

    assert record.llm_exchanges is not None and [e.kind for e in record.llm_exchanges] == ["chat", "chat"]
    for expected in (
        "# Trace: ad-airline-fuel",
        "AGENT SYSTEM PROMPT",  # the system prompt, in the first call's conversation
        "Aircraft fuel | 4,109 | 2,458",  # the tool result the model read
        "Added since the previous call** (1 message(s); the earlier 3 are unchanged)",
        "Delta, up 67% [1].",
        "## Expected answer",
        "## Tool calls",
        "| [1] | `c1` | yes |",
    ):
        assert expected in markdown
    # Sent once in the first call, not repeated in the second.
    assert markdown.count("AGENT SYSTEM PROMPT") == 1


def test_write_trace_writes_markdown_or_the_raw_record(tmp_path: Path) -> None:
    record = _agent_record()
    write_trace(record, tmp_path / "t.md")
    write_trace(record, tmp_path / "nested" / "t.json")
    assert (tmp_path / "t.md").read_text().startswith("# Turn trace")
    assert '"llm_exchanges"' in (tmp_path / "nested" / "t.json").read_text()


def test_fence_outgrows_any_backtick_run_in_the_text() -> None:
    fenced = _fence("before\n````\nafter")
    assert fenced.startswith("`````text\n") and fenced.endswith("\n`````")


def test_find_sample_by_id_across_sets(tmp_path: Path) -> None:
    one = tmp_path / "a.json"
    two = tmp_path / "b.json"
    one.write_text('[{"id": "x", "query": "q1"}]')
    two.write_text('[{"id": "y", "query": "q2"}, {"id": "x", "query": "q3"}]')

    assert trace_question.find_sample("y", [one, two]) == ({"id": "y", "query": "q2"}, two)
    with pytest.raises(SystemExit, match="several sets"):
        trace_question.find_sample("x", [one, two])
    with pytest.raises(SystemExit, match="no sample"):
        trace_question.find_sample("z", [one, two])


def test_variant_overrides_match_the_matrix_row_and_refuse_oracle_rows() -> None:
    import run_answer_matrix as matrix

    row = next(v for v in matrix.M19_VARIANTS if v.name == "agentic react / 27b")
    assert trace_question.variant_overrides("m19", "agentic react / 27b") == row.overrides
    with pytest.raises(SystemExit, match="oracle"):
        trace_question.variant_overrides("m19", "oracle / 9b")
    with pytest.raises(SystemExit, match="no variant"):
        trace_question.variant_overrides("m19", "nope")


def _raw_exchange(prompt: str, output: str, *, chat_tokens: int | None = 10) -> LLMExchange:
    return LLMExchange(
        kind="chat", started_ms=0.0, elapsed_ms=1.0, messages=[{"role": "user", "content": "Q"}],
        tools=[{"name": "rag_search", "description": "Search.", "parameters": {}}],
        prompt_tokens=10, completion_tokens=3, raw_prompt=prompt, raw_output=output, chat_prompt_tokens=chat_tokens,
    )


def _raw_record(*exchanges: LLMExchange) -> TurnRecord:
    return TurnRecord(turn_id="t", timestamp="now", query="Q", outcome="answered", answer="A", llm_exchanges=list(exchanges))


def test_raw_calls_show_the_whole_first_prompt_then_only_what_each_call_added() -> None:
    first = "<|im_start|>user\nQ<|im_end|>\n<|im_start|>assistant\n<think>\n"
    output = "Search.\n</think>\n\n<tool_call>\n<function=rag_search>\n</function>\n</tool_call>"
    added = "<|im_end|>\n<|im_start|>user\n<tool_response>\nPASSAGE\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n"

    text = render_turn_markdown(_raw_record(_raw_exchange(first, output), _raw_exchange(first + output + added, "A")))

    assert "| Raw mode | 2 call(s) rendered client-side; 2 checked against /api/chat, all token counts match |" in text
    assert "#### Rendered prompt (10 tokens, as /api/chat counts it)\n\n```text\n" + first + "\n```" in text
    assert "#### Raw output (3 tokens, unparsed)\n\n```text\n" + output + "\n```" in text
    assert f"then {len(added):,} new chars" in text
    assert "```text\n" + added + "\n```" in text
    assert text.count("<|im_start|>user\nQ") == 1  # the first prompt isn't repeated


def test_a_raw_prompt_that_doesnt_extend_the_last_one_says_where() -> None:
    first = "<|im_start|>user\nQ<|im_end|>\n<|im_start|>assistant\n<think>\n"
    # The model wrote text after its tool call, which the server drops when it re-renders the turn.
    output = "Go.\n</think>\n\n<tool_call>\nX\n</tool_call> trailing"
    second = first + "Go.\n</think>\n\n<tool_call>\nX\n</tool_call><|im_end|>\n"

    text = render_turn_markdown(_raw_record(_raw_exchange(first, output), _raw_exchange(second, "A", chat_tokens=12)))

    diverge = len(first + "Go.\n</think>\n\n<tool_call>\nX\n</tool_call>")
    assert f"differs from the previous prompt and output at char {diverge:,}, inside the previous output" in text
    assert "```text\n trailing\n```" in text
    assert "**/api/chat counts 12** for the same messages" in text
    assert "1 mismatched" in text


def test_with_raw_agent_llm_turns_on_raw_for_the_agents_model_only() -> None:
    from rag.config.settings import ChatConfig, RagConfig, load_config

    config = trace_question.with_raw_agent_llm(load_config())

    assert config.agent.llm is not None and config.agent.llm.raw
    assert not config.llm.raw
    with pytest.raises(SystemExit, match="pipeline"):
        trace_question.with_raw_agent_llm(RagConfig(chat=ChatConfig(mode="pipeline")))
