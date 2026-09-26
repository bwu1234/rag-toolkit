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
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from rag.generation.llm import LLMClient, LLMUsage


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
def metered() -> Iterator[UsageMeter]:
    """Activate a fresh `UsageMeter` for the duration of the block.

    Nested use gets its own meter and restores the outer one on exit, so a
    turn that (someday) runs a sub-turn doesn't double-count into its parent.
    """

    meter = UsageMeter()
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
        text, usage = self.inner.generate_with_usage(prompt, system=system)
        meter = _ACTIVE_METER.get()
        if meter is not None:
            meter.record(usage, (time.monotonic() - start) * 1000)
        return text, usage
