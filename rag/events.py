"""A tiny, dependency-free event type for observing pipeline progress.

`Retriever` and `ChatService` accept an optional `on_event` callback and call
it once per completed stage (embed, vector search, rerank, generate, ...).
Kept as a single frozen dataclass + callable type alias -- rather than a
logging handler or a queue -- so any caller (CLI, API, UI, tests) can pass a
plain closure and get a synchronous, in-order stream of steps with no
threading or serialization concerns.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

Stage = str


@dataclass(frozen=True)
class PipelineEvent:
    """One completed step of the retrieve -> rerank -> generate pipeline."""

    stage: Stage
    message: str
    elapsed_ms: float | None = None


EventSink = Callable[[PipelineEvent], None]


def emit(on_event: EventSink | None, start: float, stage: Stage, message: str) -> None:
    """Report a stage that started at `start` (a `time.monotonic()` reading).

    Shared by `Retriever` and `ChatService` so both time and report stages the
    same way. A no-op when `on_event` is `None`, so callers can time a block
    unconditionally without an `if on_event:` guard at every call site.
    """

    if on_event is None:
        return
    elapsed_ms = (time.monotonic() - start) * 1000
    on_event(PipelineEvent(stage=stage, message=message, elapsed_ms=elapsed_ms))
