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
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Protocol

Stage = str

#: Every stage `Retriever` or `RagTools.retrieve` can emit. A trace drawn after
#: the fact reads one missing here as a tool's own event, closing the search
#: that ran it early.
RETRIEVAL_STAGES = frozenset({
    "route", "expand", "embed", "vector_search", "sparse_search", "web_search", "fusion", "rerank", "select",
})


class _Scored(Protocol):
    @property
    def chunk_id(self) -> str: ...
    @property
    def document_id(self) -> str: ...
    @property
    def score(self) -> float: ...


@dataclass(frozen=True)
class Hit:
    """One chunk in a stage's output ranking. Ids and score only -- no text, so
    a turn that keeps every stage's ranking stays small."""

    chunk_id: str
    document_id: str
    score: float


def hits_of(chunks: Iterable[_Scored]) -> tuple[Hit, ...]:
    """A stage's ranked output as `Hit`s, best first."""

    return tuple(Hit(c.chunk_id, c.document_id, c.score) for c in chunks)


@dataclass(frozen=True)
class PipelineEvent:
    """One completed step of the retrieve -> rerank -> generate pipeline."""

    stage: Stage
    message: str
    elapsed_ms: float | None = None
    hits: tuple[Hit, ...] | None = None
    """The stage's ranked output, for stages that rank chunks. Each stage scores
    on its own scale, so compare positions across stages, never scores."""
    score_scale: str | None = None
    """What `hits` scores mean, as the component that produced them describes it."""
    at: float | None = None
    """The `time.monotonic()` reading `elapsed_ms` ends at, so the stage began at
    exactly `at - elapsed_ms / 1000`. None for an event built without `emit`."""


EventSink = Callable[[PipelineEvent], None]


def emit(
    on_event: EventSink | None,
    start: float,
    stage: Stage,
    message: str,
    hits: Iterable[_Scored] | None = None,
    score_scale: str | None = None,
) -> None:
    """Report a stage that started at `start` (a `time.monotonic()` reading).

    Shared by `Retriever` and `ChatService` so both time and report stages the
    same way. A no-op when `on_event` is `None`, so callers can time a block
    unconditionally without an `if on_event:` guard at every call site.
    """

    if on_event is None:
        return
    at = time.monotonic()
    on_event(PipelineEvent(
        stage=stage, message=message, elapsed_ms=(at - start) * 1000,
        hits=None if hits is None else hits_of(hits), score_scale=score_scale, at=at,
    ))
