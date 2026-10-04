"""Render one turn's `TurnRecord`, transcript included, as a Markdown document.

The record is what `ChatResponder.ask(on_record=...)` hands back: the
structured trace every turn already logs (events, retrieval attempts, the
agent's tool calls) plus `llm_exchanges`, every LLM call verbatim. This module
lays them out for a person reading one turn end to end -- summary and answer
first, then how it got there, then each model call exactly as sent.

An agent resends its whole conversation on every step, so each `chat` call
shows only the messages added since the previous call; the first shows them
all, the system prompt included.

A turn run in raw mode (`llm.raw`) also has each call's rendered prompt -- the
chat template applied, special tokens as text -- and the model's unparsed
output. The first call shows the whole prompt; each later one shows only what
follows the previous prompt and output, which is what the model's prompt cache
can reuse. Where the next prompt doesn't start with them (the server
re-rendered the previous reply differently from how the model wrote it), the
trace says where and shows both sides.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rag.observability.records import LLMExchange, TurnRecord

_BACKTICK_RUN = re.compile(r"`+")


def _fence(text: str, lang: str = "text") -> str:
    """`text` in a code fence longer than any backtick run inside it, so nothing escapes it."""

    longest = max((len(run) for run in _BACKTICK_RUN.findall(text)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}{lang}\n{text}\n{fence}"


def _json(value: Any) -> str:
    return _fence(json.dumps(value, indent=2, ensure_ascii=False), "json")


def _cell(text: object) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _seconds(ms: float | None) -> str:
    return "-" if ms is None else f"{ms / 1000:.1f} s"


def _tokens(prompt: int | None, completion: int | None) -> str:
    if prompt is None and completion is None:
        return "tokens not reported"
    def count(value: int | None) -> str:
        return "?" if value is None else f"{value:,}"

    return f"{count(prompt)} prompt / {count(completion)} completion tokens"


def render_turn_markdown(
    record: TurnRecord,
    *,
    title: str | None = None,
    context: Sequence[tuple[str, str]] = (),
    expected_answer: str | None = None,
) -> str:
    """The whole turn as Markdown.

    `context` adds rows to the summary table (the eval sample id, the config
    file, a matrix variant); `expected_answer` adds the gold answer under the
    model's, for a turn replayed from an eval set.
    """

    parts = [f"# {title or 'Turn trace'}", _summary(record, context)]
    parts += ["## Question", _fence(record.query)]
    parts += ["## Answer", _fence(record.answer) if record.answer is not None else "_No answer: the turn raised._"]
    if record.error:
        parts += ["## Error", _fence(record.error)]
    if expected_answer is not None:
        parts += ["## Expected answer", _fence(expected_answer)]
    if record.events:
        parts += ["## Timeline", _events(record)]
    if record.agent_calls:
        parts += ["## Tool calls", _agent_calls(record)]
    if record.attempts:
        parts += ["## Retrieval", _attempts(record)]
    if record.shown_chunk_ids:
        parts += ["## Passages shown and cited", _shown(record)]
    parts += ["## LLM calls", _exchanges(record.llm_exchanges)]
    return "\n\n".join(parts) + "\n"


def write_trace(
    record: TurnRecord,
    path: Path,
    *,
    title: str | None = None,
    context: Sequence[tuple[str, str]] = (),
    expected_answer: str | None = None,
) -> None:
    """Write the turn to `path`: the raw record as JSON for a `.json` path, else rendered Markdown."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".json":
        path.write_text(json.dumps(record.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return
    path.write_text(
        render_turn_markdown(record, title=title, context=context, expected_answer=expected_answer),
        encoding="utf-8",
    )


def _summary(record: TurnRecord, context: Sequence[tuple[str, str]]) -> str:
    rows: list[tuple[str, str]] = [*context, ("Turn id", record.turn_id), ("Started", record.timestamp)]
    rows.append(("Outcome", record.outcome + (f" (agent stopped: {record.stopped_reason})" if record.stopped_reason else "")))
    rows += [(key, value) for key, value in record.metadata.items()]
    if record.query_filter:
        rows.append(("Filter", json.dumps(record.query_filter)))
    if record.rewritten_query:
        rows.append(("Rewritten query", record.rewritten_query))
    if record.grounded is not None or record.groundedness_checks:
        rows.append(("Grounded", f"{record.grounded} (checks: {record.groundedness_checks})"))
    rows.append(("Total", _seconds(record.total_ms)))
    rows.append(("LLM", f"{record.llm_calls} call(s), {_seconds(record.llm_ms)}, {_tokens(record.prompt_tokens, record.completion_tokens)}"))
    raw = [e for e in record.llm_exchanges or () if e.raw_prompt is not None]
    if raw:
        checked = [e for e in raw if e.chat_prompt_tokens is not None]
        drifted = sum(e.chat_prompt_tokens != e.prompt_tokens for e in checked)
        verdict = (
            "not checked" if not checked
            else f"{len(checked)} checked against /api/chat, "
            + ("all token counts match" if not drifted else f"**{drifted} mismatched**")
        )
        rows.append(("Raw mode", f"{len(raw)} call(s) rendered client-side; {verdict}"))
    if record.stopped_reason is not None:
        rows.append(("Tool calls run", str(record.tool_calls)))
    return "| | |\n|---|---|\n" + "\n".join(f"| {_cell(key)} | {_cell(value)} |" for key, value in rows)


def _events(record: TurnRecord) -> str:
    lines = ["| # | Stage | Time | What happened |", "|---|---|---|---|"]
    for number, event in enumerate(record.events, start=1):
        elapsed = "" if event.elapsed_ms is None else f"{event.elapsed_ms:,.0f} ms"
        lines.append(f"| {number} | {event.stage} | {elapsed} | {_cell(event.message)} |")
    return "\n".join(lines)


def _agent_calls(record: TurnRecord) -> str:
    lines = ["| Step | Tool | Query or expression | Filter | Status | Result |", "|---|---|---|---|---|---|"]
    for call in record.agent_calls:
        filters = json.dumps(call.filters) if call.filters else (
            f"as sent: {json.dumps(call.filters_raw)}" if call.filters_raw is not None else ""
        )
        if call.status != "ran":
            result = call.note or ""
        elif call.tool == "rag_list_documents":
            result = f"{len(call.documents)} document(s): {', '.join(call.documents)}"
        elif call.tool == "calculator":
            result = f"= {call.result}"
        else:
            new = set(call.new_passages)
            result = " ".join(f"[{n}]" if n in new else f"([{n}])" for n in call.passages) or "nothing"
        lines.append(
            f"| {call.step} | {call.tool} | {_cell(call.expression or call.query)} | {_cell(filters)} | {call.status} | {_cell(result)} |"
        )
    lines.append("\nPassage numbers in parentheses were already shown by an earlier search.")
    return "\n".join(lines)


def _attempts(record: TurnRecord) -> str:
    blocks = []
    for number, attempt in enumerate(record.attempts, start=1):
        head = f"**{number}. {_cell(attempt.query)}** -- {len(attempt.retrieved)} passage(s) of {attempt.candidate_count} candidate(s)"
        if attempt.dropped_below_min_score:
            head += f", {attempt.dropped_below_min_score} below min_score"
        if attempt.routed_to:
            head += f", routed to {', '.join(attempt.routed_to)}"
        kept = set(attempt.kept) if attempt.kept is not None else None
        rows = [
            f"| {rank} | {passage.score:.3f} | {passage.document_id} | `{passage.chunk_id}` |"
            + ("" if kept is None else f" {'kept' if passage.chunk_id in kept else 'graded out'} |")
            for rank, passage in enumerate(attempt.retrieved, start=1)
        ]
        table_head = "| Rank | Score | Document | Chunk |" + ("" if kept is None else " Grader |")
        rule = "|---|---|---|---|" + ("" if kept is None else "---|")
        blocks.append("\n".join([head, "", table_head, rule, *rows]) if rows else head)
    return "\n\n".join(blocks)


def _shown(record: TurnRecord) -> str:
    cited = set(record.cited_chunk_ids)
    lines = ["| [n] | Chunk | Cited |", "|---|---|---|"]
    lines += [
        f"| [{number}] | `{chunk_id}` | {'yes' if chunk_id in cited else ''} |"
        for number, chunk_id in enumerate(record.shown_chunk_ids, start=1)
    ]
    return "\n".join(lines)


def _exchanges(exchanges: Sequence[LLMExchange] | None) -> str:
    if exchanges is None:
        return "_No transcript: the turn was not run with one._"
    if not exchanges:
        return "_The turn made no LLM calls._"
    blocks = []
    previous_chat: LLMExchange | None = None
    previous_raw: LLMExchange | None = None
    previous_tools: list[dict[str, Any]] | None = None
    for number, exchange in enumerate(exchanges, start=1):
        head = (
            f"### Call {number}: {exchange.kind} -- at {_seconds(exchange.started_ms)}, "
            f"took {_seconds(exchange.elapsed_ms)}, {_tokens(exchange.prompt_tokens, exchange.completion_tokens)}"
            + (f", stop: {exchange.stop_reason}" if exchange.stop_reason else "")
        )
        body = [head]
        if exchange.kind == "generate":
            body += ["#### System prompt", _fence(exchange.system) if exchange.system else "_none_"]
            body += ["#### Prompt", _fence(exchange.prompt or "")]
        else:
            body.append(_tools_offered(exchange.tools, previous_tools))
            messages = _new_messages(exchange, previous_chat)
            if exchange.raw_prompt is None:
                body += messages
            else:
                body.append(_details("Messages sent (structured)", "\n\n".join(messages), fence=False))
                body += _raw_prompt(exchange, previous_raw)
                body += _raw_output(exchange)
                previous_raw = exchange
            previous_chat, previous_tools = exchange, exchange.tools
        body += _reply(exchange)
        blocks.append("\n\n".join(body))
    return "\n\n".join(blocks)


def _token_check(exchange: LLMExchange) -> str:
    if exchange.chat_prompt_tokens is None or exchange.prompt_tokens is None:
        return f"{exchange.prompt_tokens or '?'} tokens"
    if exchange.chat_prompt_tokens == exchange.prompt_tokens:
        return f"{exchange.prompt_tokens:,} tokens, as /api/chat counts it"
    return (
        f"{exchange.prompt_tokens:,} tokens, but **/api/chat counts {exchange.chat_prompt_tokens:,}** for the "
        "same messages: the client-side renderer no longer matches Ollama's"
    )


def _common_prefix(a: str, b: str) -> int:
    size = 0
    for x, y in zip(a, b):
        if x != y:
            break
        size += 1
    return size


def _raw_prompt(exchange: LLMExchange, previous: LLMExchange | None) -> list[str]:
    """The rendered prompt: whole the first time, then only what follows the previous prompt and output."""

    prompt = exchange.raw_prompt or ""
    if previous is None or previous.raw_prompt is None:
        return [f"#### Rendered prompt ({_token_check(exchange)})", _fence(prompt)]
    before = previous.raw_prompt + (previous.raw_output or "")
    same = _common_prefix(before, prompt)
    if same == len(before):
        return [
            f"#### Rendered prompt: the previous prompt and output, then {len(prompt) - same:,} new chars "
            f"({_token_check(exchange)})",
            _fence(prompt[same:]),
        ]
    where = "the previous prompt" if same < len(previous.raw_prompt) else "the previous output"
    return [
        f"#### Rendered prompt: differs from the previous prompt and output at char {same:,}, inside "
        f"{where} ({_token_check(exchange)})",
        "The server re-rendered the conversation, so this prompt doesn't extend the last one there "
        "(and the prompt cache can't reuse anything after that point). The previous prompt and output, "
        "from that char on:",
        _fence(before[same:]),
        "This prompt, from that char on:",
        _fence(prompt[same:]),
    ]


def _raw_output(exchange: LLMExchange) -> list[str]:
    if exchange.error:
        return []
    tokens = "?" if exchange.completion_tokens is None else f"{exchange.completion_tokens:,}"
    return [f"#### Raw output ({tokens} tokens, unparsed)", _fence(exchange.raw_output or "")]


def _tools_offered(tools: list[dict[str, Any]], previous: list[dict[str, Any]] | None) -> str:
    if not tools:
        return "**Tools offered:** none -- the model must answer in text."
    names = ", ".join(tool["name"] for tool in tools)
    if tools == previous:
        return f"**Tools offered:** {names} (as in the previous call)"
    return f"**Tools offered:** {names}\n\n<details><summary>Tool definitions</summary>\n\n{_json(tools)}\n\n</details>"


def _new_messages(exchange: LLMExchange, previous: LLMExchange | None) -> list[str]:
    """The messages this call added to the conversation, or all of them if it's the first."""

    sent = exchange.messages
    known: list[dict[str, Any]] = []
    if previous is not None:
        reply: dict[str, Any] = {"role": "assistant", "content": previous.response}
        if previous.thinking:
            reply["thinking"] = previous.thinking
        if previous.tool_calls:
            reply["tool_calls"] = previous.tool_calls
        known = [*previous.messages, reply]
    same = 0
    while same < min(len(sent), len(known)) and sent[same] == known[same]:
        same += 1

    out: list[str] = []
    if previous is None:
        out.append(f"**Conversation sent** ({len(sent)} message(s)):")
    elif same == len(known):
        out.append(f"**Added since the previous call** ({len(sent) - same} message(s); the earlier {same} are unchanged):")
    elif same == len(previous.messages):
        out.append(
            f"**Added since the previous call** ({len(sent) - same} message(s)). The previous reply was "
            "not kept in the conversation."
        )
    else:
        out.append(
            f"**Messages from #{same + 1} on differ from the previous call** (the agent rewrote them, "
            "e.g. after a context overflow) and are shown again:"
        )
    for index in range(same, len(sent)):
        out += _message(index + 1, sent[index])
    return out


def _message(number: int, message: dict[str, Any]) -> list[str]:
    role = message["role"]
    if role == "tool":
        label = f"#### {number}. tool result: {message.get('name', '')}"
        out = [label, f"Arguments: `{json.dumps(message.get('arguments', {}), ensure_ascii=False)}`"]
    else:
        out = [f"#### {number}. {role}"]
    if message.get("thinking"):
        out.append(_details("Reasoning", message["thinking"]))
    if message.get("content") or not message.get("tool_calls"):
        out.append(_fence(message.get("content", "")))
    if message.get("tool_calls"):
        out.append("Tool calls:\n\n" + _json(message["tool_calls"]))
    return out


def _reply(exchange: LLMExchange) -> list[str]:
    if exchange.error:
        return ["#### The call raised", _fence(exchange.error)]
    out = ["#### Response"]
    if exchange.thinking:
        out.append(_details(f"Reasoning ({len(exchange.thinking):,} chars)", exchange.thinking))
    if exchange.response or not exchange.tool_calls:
        out.append(_fence(exchange.response) if exchange.response else "_empty_")
    if exchange.tool_calls:
        out.append("Tool calls:\n\n" + _json(exchange.tool_calls))
    return out


def _details(summary: str, text: str, *, fence: bool = True) -> str:
    return f"<details><summary>{summary}</summary>\n\n{_fence(text) if fence else text}\n\n</details>"
