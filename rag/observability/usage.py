"""Per-turn LLM call and token metering.

A single chat turn can call the LLM from five different places -- query
expansion, the condenser, CRAG's grader and retry rewriter, the groundedness
checker, and generation itself -- and all of them share one `LLMClient` (see
`build_chat_service`). Counting at the client is therefore the only place that
sees every call without threading a counter through each component's API.

`MeteredLLMClient` wraps that shared client. It records into whichever
`UsageMeter` is *active in the current context*, rather than into a counter on
the wrapper itself, because the API serves concurrent requests from one
`ChatService` (FastAPI runs sync routes in a thread pool): a per-instance
counter would mix two turns' calls together. A `ContextVar` is per-thread and
per-task, so each turn's meter sees only that turn's calls.

Calls made from threads the turn spawns itself would not inherit the meter --
nothing on the query path does that today; if something starts to, it must
copy the context (`contextvars.copy_context().run`) or its calls go uncounted.

The same position makes the wrapper the place to keep a transcript: a meter
opened with `metered(capture=True)` also records every call verbatim as an
`LLMExchange` -- system prompt, prompt or conversation, tools offered, reply.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, overload

from rag.generation.llm import (
    AssistantTurn,
    ChatMessage,
    LLMClient,
    LLMUsage,
    Message,
    ToolCallingLLM,
    ToolDefinition,
)
from rag.observability.records import LLMExchange


@dataclass
class UsageMeter:
    """Running totals for the LLM calls made during one turn.

    Token totals stay `None` until at least one call reports a count, so "the
    provider doesn't report tokens" reads as unknown rather than as a free turn.
    """

    calls: int = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    llm_ms: float = 0.0
    exchanges: list[LLMExchange] | None = None
    """Every call verbatim, in order -- None unless the meter was opened with `capture=True`."""
    started: float = field(default_factory=time.monotonic)

    def since_start_ms(self, at: float) -> float:
        return (at - self.started) * 1000

    def record(self, usage: LLMUsage | None, elapsed_ms: float) -> None:
        self.calls += 1
        self.llm_ms += elapsed_ms
        if usage is None:
            return
        if usage.prompt_tokens is not None:
            self.prompt_tokens = (self.prompt_tokens or 0) + usage.prompt_tokens
        if usage.completion_tokens is not None:
            self.completion_tokens = (self.completion_tokens or 0) + usage.completion_tokens


_ACTIVE_METER: ContextVar[UsageMeter | None] = ContextVar("rag_active_usage_meter", default=None)


@contextmanager
def metered(*, capture: bool = False) -> Iterator[UsageMeter]:
    """Activate a fresh `UsageMeter` for the duration of the block.

    Nested use gets its own meter and restores the outer one on exit, so a
    turn that (someday) runs a sub-turn doesn't double-count into its parent.
    With `capture`, the meter also keeps each call verbatim in `exchanges`.
    """

    meter = UsageMeter(exchanges=[] if capture else None)
    token = _ACTIVE_METER.set(meter)
    try:
        yield meter
    finally:
        _ACTIVE_METER.reset(token)


class MeteredLLMClient(LLMClient):
    """Delegates to another `LLMClient`, recording each call into the active `UsageMeter`.

    Outside a `metered()` block it's a transparent pass-through -- the index-time
    contextualizer or an eval judge can share a metered client without anything
    being recorded for them.
    """

    def __init__(self, inner: LLMClient) -> None:
        self.inner = inner

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        return self.generate_with_usage(prompt, system=system)[0]

    def generate_with_usage(self, prompt: str, *, system: str | None = None) -> tuple[str, LLMUsage | None]:
        start = time.monotonic()
        meter = _ACTIVE_METER.get()
        try:
            text, usage = self.inner.generate_with_usage(prompt, system=system)
        except Exception as exc:
            if meter is not None and meter.exchanges is not None:
                meter.exchanges.append(LLMExchange(
                    kind="generate", started_ms=meter.since_start_ms(start), elapsed_ms=_ms_since(start),
                    system=system, prompt=prompt, error=f"{type(exc).__name__}: {exc}",
                ))
            raise
        if meter is not None:
            elapsed = _ms_since(start)
            meter.record(usage, elapsed)
            if meter.exchanges is not None:
                meter.exchanges.append(LLMExchange(
                    kind="generate", started_ms=meter.since_start_ms(start), elapsed_ms=elapsed,
                    system=system, prompt=prompt, response=text,
                    prompt_tokens=usage.prompt_tokens if usage else None,
                    completion_tokens=usage.completion_tokens if usage else None,
                ))
        return text, usage


class MeteredToolCallingLLM(MeteredLLMClient, ToolCallingLLM):
    """`MeteredLLMClient` for a client that can also call tools.

    A separate class because the agent checks for `ToolCallingLLM` when it is
    built: wrapping a tool-calling client in plain `MeteredLLMClient` would hide
    the capability and fail that check, and leaving it unwrapped would drop the
    agent's calls from the turn's `llm_calls` and tokens. Each `chat()` call
    counts as one LLM call, recorded into the same active meter as `generate`.
    """

    inner: ToolCallingLLM

    def __init__(self, inner: ToolCallingLLM) -> None:
        super().__init__(inner)

    def chat(self, messages: Sequence[Message], tools: Sequence[ToolDefinition] = ()) -> AssistantTurn:
        start = time.monotonic()
        meter = _ACTIVE_METER.get()
        capture = meter is not None and meter.exchanges is not None
        # Serialized before the call: the caller appends to and (on a context
        # overflow) rewrites its message list afterwards.
        sent = [message_dict(message) for message in messages] if capture else []
        offered = [tool_dict(tool) for tool in tools] if capture else []
        try:
            turn = self.inner.chat(messages, tools)
        except Exception as exc:
            if meter is not None and meter.exchanges is not None:
                meter.exchanges.append(LLMExchange(
                    kind="chat", started_ms=meter.since_start_ms(start), elapsed_ms=_ms_since(start),
                    messages=sent, tools=offered, error=f"{type(exc).__name__}: {exc}",
                ))
            raise
        if meter is not None:
            elapsed = _ms_since(start)
            meter.record(turn.usage, elapsed)
            if meter.exchanges is not None:
                reply = message_dict(turn)
                meter.exchanges.append(LLMExchange(
                    kind="chat", started_ms=meter.since_start_ms(start), elapsed_ms=elapsed,
                    messages=sent, tools=offered, response=turn.content, thinking=turn.thinking,
                    tool_calls=reply.get("tool_calls", []), stop_reason=turn.stop_reason,
                    prompt_tokens=turn.usage.prompt_tokens if turn.usage else None,
                    completion_tokens=turn.usage.completion_tokens if turn.usage else None,
                ))
        return turn


def _ms_since(start: float) -> float:
    return (time.monotonic() - start) * 1000


def message_dict(message: Message) -> dict[str, Any]:
    """A conversation message as plain data, for an `LLMExchange`."""

    if isinstance(message, ChatMessage):
        return {"role": message.role, "content": message.content}
    if isinstance(message, AssistantTurn):
        data: dict[str, Any] = {"role": "assistant", "content": message.content}
        if message.thinking:
            data["thinking"] = message.thinking
        if message.tool_calls:
            data["tool_calls"] = [{"name": call.name, "arguments": call.arguments} for call in message.tool_calls]
        return data
    return {"role": "tool", "name": message.call.name, "arguments": message.call.arguments, "content": message.content}


def tool_dict(tool: ToolDefinition) -> dict[str, Any]:
    return {"name": tool.name, "description": tool.description, "parameters": tool.parameters}


@overload
def metered_client(inner: ToolCallingLLM) -> MeteredToolCallingLLM: ...
@overload
def metered_client(inner: LLMClient) -> MeteredLLMClient: ...
def metered_client(inner: LLMClient) -> MeteredLLMClient:
    """Wrap `inner` in the metering wrapper that keeps its capabilities.

    Builders call this rather than naming a wrapper, so a tool-calling client
    stays a `ToolCallingLLM` after metering.
    """

    if isinstance(inner, ToolCallingLLM):
        return MeteredToolCallingLLM(inner)
    return MeteredLLMClient(inner)
