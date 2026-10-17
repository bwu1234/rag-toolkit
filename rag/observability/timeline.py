"""A turn's stages and LLM calls as a tree of timed intervals, for drawing it as a trace.

Shared by the OTel exporter (`rag.observability.otel`) and the MLflow export
(`rag.eval.tracking`) so both draw the same tree, and free of either SDK. The
record holds every stage's timing (`StageEvent.ended_ms`) and every LLM call's
(`LLMExchange.started_ms`) on one clock, so each sits at the time it ran, and a
node's parent is the innermost stage whose interval contains it -- a model call
inside its agent step or the pipeline's `generate`, the embed and rerank stages
inside the `rag_search` that ran them. Records stored before stages kept their
end time are laid out from the model calls' times instead (`reconstructed_ends`).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass

from rag.events import RETRIEVAL_STAGES
from rag.observability.records import AgentToolCall, LLMExchange, StageEvent, TurnRecord

logger = logging.getLogger(__name__)

#: Agent tool events, by stage, and the tool each one ran.
TOOL_STAGES = {
    "search": "rag_search",
    "search_refused": "rag_search",
    "list_documents": "rag_list_documents",
    "read_document": "rag_read_document",
    "calculate": "calculator",
}
#: Turn events that are one agent model call, made in order.
LLM_EVENTS = frozenset({"agent_step", "agent_answer"})
#: Slack when testing whether one interval contains another: float rounding in
#: a stage's derived start (`ended_ms - elapsed_ms`), nothing more.
EPSILON_MS = 1e-6


@dataclass(frozen=True)
class Node:
    """A span to draw: a stage or an LLM call, at its offset into the turn."""

    start_ms: float
    end_ms: float
    order: int
    event: StageEvent | None = None
    exchange: LLMExchange | None = None
    call: AgentToolCall | None = None


def nodes(record: TurnRecord) -> list[Node]:
    """Every stage and LLM call as an interval, outermost first among those starting together."""

    calls = calls_by_event(record)
    ends = (
        [event.ended_ms for event in record.events]
        if all(event.ended_ms is not None for event in record.events)
        else reconstructed_ends(record)
    )
    out: list[Node] = []
    for order, (event, end) in enumerate(zip(record.events, ends, strict=True)):
        assert end is not None
        out.append(Node(end - (event.elapsed_ms or 0.0), end, order, event=event, call=calls.get(order)))
    for order, exchange in enumerate(record.llm_exchanges or [], start=len(record.events)):
        start = exchange.started_ms
        out.append(Node(start, start + exchange.elapsed_ms, order, exchange=exchange))
    return sorted(out, key=lambda n: (n.start_ms, -n.end_ms, n.exchange is not None, n.order))


def reconstructed_ends(record: TurnRecord) -> list[float]:
    """When each stage ended, for a record that didn't keep it (`StageEvent.ended_ms` unset).

    What is known: events arrive in order as their stages end; each agent
    step or answer is one `chat` call, whose start and length are recorded; a
    tool's event follows the retrieval stages it ran, and its length covers
    theirs. So an agent event ends when its call does (or, if its own length
    says later, then), a tool event starts
    with its first retrieval stage, and everything else runs end to end from
    there. Gaps no event measured -- the pipeline's own model calls, which no
    event is matched to -- are not recovered, so a pipeline turn's stages can
    drift early against its calls.
    """

    chats = iter([exchange for exchange in record.llm_exchanges or [] if exchange.kind == "chat"])
    ends = [0.0] * len(record.events)
    cursor = 0.0
    pending: list[int] = []

    def lay(start: float) -> float:
        for index in pending:
            start += record.events[index].elapsed_ms or 0.0
            ends[index] = start
        pending.clear()
        return start

    for index, event in enumerate(record.events):
        elapsed = event.elapsed_ms or 0.0
        if event.stage in RETRIEVAL_STAGES:
            pending.append(index)
            continue
        if event.stage in TOOL_STAGES:
            ends[index] = max(cursor + elapsed, lay(cursor))
        elif event.stage in LLM_EVENTS and (chat := next(chats, None)) is not None:
            # No earlier than its own length allows: the event also times a
            # little work after the call returns.
            ends[index] = max(chat.started_ms + chat.elapsed_ms, lay(cursor) + elapsed)
        else:
            ends[index] = lay(cursor) + elapsed
        cursor = ends[index]
    lay(cursor)
    return ends


def calls_by_event(record: TurnRecord) -> dict[int, AgentToolCall]:
    """Each agent tool call, keyed by the index of the tool event it produced.

    A call's `step` is the agent step that asked for it, and its event follows
    that step's `agent_step` event, so the two are paired in order within a
    step. A step whose counts differ -- a call that emitted no event -- is
    left unpaired rather than guessed: its tool spans carry no call details.
    """

    events: dict[int, list[int]] = defaultdict(list)
    step = 0
    for index, event in enumerate(record.events):
        if event.stage == "agent_step":
            step += 1
        elif event.stage in TOOL_STAGES:
            events[step].append(index)
    calls: dict[int, list[AgentToolCall]] = defaultdict(list)
    for call in record.agent_calls:
        calls[call.step].append(call)
    paired: dict[int, AgentToolCall] = {}
    for step, indices in events.items():
        if len(indices) == len(calls[step]):
            paired |= dict(zip(indices, calls[step], strict=True))
        else:
            logger.debug("Turn %s step %d: %d tool events, %d calls; left unpaired",
                         record.turn_id, step, len(indices), len(calls[step]))
    return paired


def parents(ordered: list[Node]) -> list[int | None]:
    """Each node's parent, as an index into `ordered` (`nodes`' order), or None for the turn itself.

    The innermost stage still open when a node starts and that it ends inside.
    Only a stage with some length can hold others; LLM calls never do.
    """

    out: list[int | None] = []
    stack: list[int] = []
    for index, node in enumerate(ordered):
        # Close every open stage this node isn't inside: one that ended
        # before it starts, or before it ends.
        while stack and (node.start_ms >= ordered[stack[-1]].end_ms
                         or node.end_ms > ordered[stack[-1]].end_ms + EPSILON_MS):
            stack.pop()
        out.append(stack[-1] if stack else None)
        if node.event is not None and node.end_ms > node.start_ms:
            stack.append(index)
    return out
