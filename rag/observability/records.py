"""The shapes persisted by a `TurnSink`: one record per chat turn, one per feedback click.

Plain dataclasses serialized with `dataclasses.asdict`, not pydantic models --
they're internal data (see the Conventions in CLAUDE.md), and the JSONL on disk
is meant to be read back with nothing more than `json.loads`.

Every record carries a `kind` so turns and feedback can share one store and be
joined on `turn_id` after the fact: feedback arrives later than the turn it
rates, from a different request, so it's appended rather than written into the
turn's own line.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

TurnOutcome = Literal["answered", "blank_query", "empty_index", "below_min_score", "graded_out", "error"]
"""How a turn ended. Every value but `answered` and `error` means generation was skipped."""

Rating = Literal["up", "down"]


def new_id() -> str:
    return uuid.uuid4().hex


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass(frozen=True)
class RetrievedPassage:
    """One passage a retrieval attempt returned, before CRAG grading."""

    chunk_id: str
    document_id: str
    score: float


@dataclass(frozen=True)
class RetrievalAttempt:
    """One pass through the retrieve (-> grade) loop.

    `kept` is `None` when no grader ran -- every retrieved passage went through
    ungraded -- and a list (possibly empty) when one did. Recording the split
    per attempt is what makes the grader's verdicts inspectable after the fact:
    `retrieved` minus `kept` is exactly what it rejected.
    """

    query: str
    retrieved: list[RetrievedPassage] = field(default_factory=list)
    kept: list[str] | None = None
    candidate_count: int = 0
    dropped_below_min_score: int = 0
    routed_to: list[str] = field(default_factory=list)
    """Documents `retrieval.document_routing` filtered this attempt to; empty if it didn't."""


AgentSearchStatus = Literal["searched", "refused", "error"]
"""What became of an agent's search call: run, refused by a loop guard, or rejected as invalid."""


@dataclass(frozen=True)
class AgentSearch:
    """One `rag_search` call an agent's model made, as written, and what it got back.

    The agent's trajectory in order: every call, including the ones a guard
    refused (a repeat, the budget) and the ones with bad arguments, because
    those are the decisions that explain how a turn spent its searches. A
    `RetrievalAttempt` exists only for the calls that searched, and carries
    the retrieval side (scores, candidate counts) instead.

    `passages` are the ledger numbers the result showed, in rank order --
    the `[n]` the answer cites -- and `new_passages` the ones it showed for
    the first time; the rest came back as "already shown" stubs.
    """

    step: int
    """Which model call asked for it, from 1. Calls from one step share it."""
    query: str
    status: AgentSearchStatus = "searched"
    filters: dict[str, Any] | None = None
    """The model's filter as applied: validated, dates as YYYYMMDD. None when it set none or it was invalid.
    The turn's own filter is `query_filter`."""
    filters_raw: Any = None
    """The `filters` argument exactly as the model sent it -- a JSON-encoded string, an object, or
    something invalid. What `filters` was parsed from, and the only record of a rejected one."""
    note: str | None = None
    """For a refused or rejected call, the reason the model was shown."""
    passages: list[int] = field(default_factory=list)
    new_passages: list[int] = field(default_factory=list)
    chunk_ids: list[str] = field(default_factory=list)
    """The chunk behind each of `passages`, in the same order."""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentSearch:
        return cls(**data)


@dataclass(frozen=True)
class StageEvent:
    """A persisted `PipelineEvent` -- the live UI trace, kept."""

    stage: str
    message: str
    elapsed_ms: float | None = None


@dataclass(frozen=True)
class TurnRecord:
    """Everything one `ChatResponder.ask` call did, in one JSON line.

    `shown_chunk_ids` and `cited_chunk_ids` together are an implicit relevance
    judgment logged on every turn: of the passages the model was shown (in
    prompt order), the ones it chose to cite. `metadata` carries what the
    record can't be interpreted without -- corpus, model, config fingerprint --
    since a turn logged under a different config is a different experiment.
    """

    turn_id: str
    timestamp: str
    query: str
    outcome: TurnOutcome
    answer: str | None = None
    kind: Literal["turn"] = "turn"
    history_turns: int = 0
    query_filter: dict[str, Any] | None = None
    """The metadata filter the caller restricted retrieval with, as sent; absent when none."""
    rewritten_query: str | None = None
    search_queries: list[str] = field(default_factory=list)
    retry_queries: list[str] = field(default_factory=list)
    attempts: list[RetrievalAttempt] = field(default_factory=list)
    shown_chunk_ids: list[str] = field(default_factory=list)
    cited_chunk_ids: list[str] = field(default_factory=list)
    groundedness_checks: list[bool | None] = field(default_factory=list)
    """Every groundedness verdict, in order: the first check, then one per regeneration."""
    grounded: bool | None = None
    stage_ms: dict[str, float] = field(default_factory=dict)
    total_ms: float = 0.0
    llm_calls: int = 0
    llm_ms: float = 0.0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    events: list[StageEvent] = field(default_factory=list)
    error: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)
    tool_calls: int = 0
    """Agentic turns: searches run. Each one is also an entry in `attempts`."""
    stopped_reason: str | None = None
    """Agentic turns: `answered`, `cap`, `timeout` or `context`. None for a pipeline turn."""
    agent_searches: list[AgentSearch] = field(default_factory=list)
    """Agentic turns: every search call the model made, in order, refused ones included."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FeedbackRecord:
    """A user's thumbs up/down on one turn. The latest record for a `turn_id` wins."""

    turn_id: str
    rating: Rating
    comment: str | None = None
    feedback_id: str = field(default_factory=new_id)
    timestamp: str = field(default_factory=utc_now)
    kind: Literal["feedback"] = "feedback"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
