"""Agentic retrieval (Milestone 19): the model calls search as a tool, as often as it needs.

`AgentService` is `chat.mode: agentic`'s answer to `ChatService`: same
`ask()`, same `ChatAnswer`, so every caller and both eval runners work
unchanged. What differs is who decides to search. The pipeline retrieves once
for the question as asked; here the model writes its own queries, one per
entity or period, and (under `react`) searches again on what earlier results
turned up.

Three pieces carry the design (`docs/milestone-19-plan.md`, decisions 5, 6, 9):

- **A passage ledger** (`PassageLedger`) numbers every passage the model is
  shown, in first-seen order across all searches, deduplicated by chunk id.
  It is the single source of truth for `[n] -> Citation`, the job
  `build_rag_prompt` does in the pipeline.
- **Loop guards**, each answering a failure the prototype showed: a search
  cap, a forced tool-free synthesis turn, refusing a repeated query without
  spending a search, a wall-clock budget, a per-passage character cap, and
  answering from what's already shown when the prompt outgrows the context
  window.
- **Two strategies over the same ledger and guards.** `react` lets the model
  decide after every result; `planned` takes one round of searches from a
  single planning call, then forces the answer -- exactly two model calls.

Search goes through `RagTools.retrieve`, the same path MCP's `rag_search`
uses, and the model is offered that tool's own schema, minus the arguments the
turn fixes (see `ToolSpec.definition_without`).
"""

from __future__ import annotations

import dataclasses
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from rag.config.settings import AgentStrategy
from rag.events import EventSink, emit
from rag.query_filter import QueryFilter
from rag.generation.chat_service import (
    BLANK_QUERY_ANSWER,
    ChatAnswer,
    ChatResponder,
    StoppedReason,
    TurnTrace,
    to_citation,
)
from rag.generation.crag import GroundednessChecker
from rag.generation.llm import (
    AssistantTurn,
    ChatMessage,
    ContextOverflowError,
    Message,
    ToolCall,
    ToolCallingLLM,
    ToolDefinition,
    ToolResult,
)
from rag.generation.prompts import (
    AGENT_SYNTHESIS_INSTRUCTION,
    format_passage,
    parse_cited_passages,
    strip_citation_markers,
)
from rag.generation.query_rewriter import ChatTurn
from rag.observability.records import RetrievalAttempt, RetrievedPassage
from rag.observability.sink import TurnSink
from rag.tools import RagTools, build_tool_specs
from rag.vectorstore.base import ScoredChunk

logger = logging.getLogger(__name__)

SEARCH_TOOL = "rag_search"

#: `rag_search` arguments the turn decides, not the model. `corpus` is the
#: turn's corpus selection (an eval's `--corpus`); a model free to widen it
#: would be measured against a different index. `top_k` and `max_chars` are
#: the prompt-size guards: prompts grew ~6x over the pipeline's in the
#: prototype, and a model asking for 20 full-length passages per search would
#: undo `max_tool_calls` x `max_passage_chars` as a bound.
#: `filters` is the turn's too: a `/chat` filter applies to every search, and
#: letting the model choose its own filters is Milestone 19's call to measure.
PINNED_ARGUMENTS = ("corpus", "top_k", "max_chars", "filters")


class PassageLedger:
    """Every passage one agent turn has shown the model, numbered for citation.

    Numbers are assigned in first-seen order across all searches and never
    change, and a chunk found by two searches keeps its first number -- so a
    `[3]` in the final answer means the same passage however many searches
    came after it. The text kept is the text the model was shown, capped at
    `max_chars`: a citation (and the eval's evidence recall) should reflect
    what the answer could have been based on, not the full chunk.
    """

    def __init__(self, max_chars: int) -> None:
        self._max_chars = max_chars
        self._chunks: list[ScoredChunk] = []
        self._numbers: dict[str, int] = {}

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def chunks(self) -> list[ScoredChunk]:
        """The passages in numbering order: `chunks[n - 1]` is passage `[n]`."""

        return list(self._chunks)

    def add(self, chunks: Sequence[ScoredChunk]) -> list[tuple[int, ScoredChunk, bool]]:
        """Number `chunks`, returning `(number, chunk as shown, is_new)` for each, in order."""

        entries: list[tuple[int, ScoredChunk, bool]] = []
        for chunk in chunks:
            number = self._numbers.get(chunk.chunk_id)
            if number is not None:
                entries.append((number, self._chunks[number - 1], False))
                continue
            shown = chunk
            if len(chunk.text) > self._max_chars:
                shown = dataclasses.replace(chunk, text=chunk.text[: self._max_chars] + " [...]")
            self._chunks.append(shown)
            number = len(self._chunks)
            self._numbers[chunk.chunk_id] = number
            entries.append((number, shown, True))
        return entries

    def truncate(self, size: int) -> None:
        """Forget every passage numbered above `size` -- ones the model turned out never to see."""

        for chunk in self._chunks[size:]:
            del self._numbers[chunk.chunk_id]
        del self._chunks[size:]


@dataclass
class _Run:
    """One turn's mutable loop state. Private: `ChatAnswer` is what callers see."""

    messages: list[Message]
    ledger: PassageLedger
    deadline: float
    on_event: EventSink
    trace: TurnTrace
    #: The turn's metadata filter, applied to every search.
    query_filter: QueryFilter | None = None
    queries: list[str] = field(default_factory=list)
    #: Normalized query -> the passage numbers its search returned.
    searched: dict[str, list[int]] = field(default_factory=dict)
    rounds: int = 0
    steps: int = 0
    dropped_below_min_score: int = 0
    #: A search was refused for budget or time -- the guard cut something off.
    hit_cap: bool = False
    hit_timeout: bool = False
    #: Where the latest step's tool results start in `messages`, and the ledger
    #: size before it -- what a context overflow rolls back.
    step_message_mark: int | None = None
    step_ledger_mark: int = 0
    answer: str = ""


def _normalize(query: str) -> str:
    return " ".join(query.casefold().split())


def _numbers_label(numbers: Sequence[int]) -> str:
    return ", ".join(f"[{n}]" for n in numbers)


class AgentService(ChatResponder):
    """Answers a turn by letting the model search as often as it needs, within guards.

    Collaborators are the `ToolCallingLLM` that drives the loop (built by
    `build_agent_llm`, so a provider without tool calling fails at build
    time) and the `RagTools` it searches through. `groundedness_checker`,
    when set, checks the final answer against the ledger and reports the
    verdict on `ChatAnswer.grounded` -- check only, no regeneration, so its
    effect can be measured on its own (plan decision 7).

    `clock` exists for tests: the timeout guard reads it, not `time` directly.
    """

    def __init__(
        self,
        llm: ToolCallingLLM,
        tools: RagTools,
        *,
        system_prompt: str,
        corpora: Sequence[str] | None = None,
        strategy: AgentStrategy = "react",
        max_tool_calls: int = 8,
        timeout_s: float = 600.0,
        max_passage_chars: int = 1200,
        max_history_turns: int = 6,
        groundedness_checker: GroundednessChecker | None = None,
        turn_sink: TurnSink | None = None,
        turn_metadata: Mapping[str, str] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(turn_sink=turn_sink, turn_metadata=turn_metadata)
        self._llm = llm
        self._tools = tools
        self._system_prompt = system_prompt
        self._corpora = list(corpora) if corpora is not None else None
        self.strategy: AgentStrategy = strategy
        self.max_tool_calls = max_tool_calls
        self.timeout_s = timeout_s
        self.max_passage_chars = max_passage_chars
        self.max_history_turns = max_history_turns
        self._groundedness_checker = groundedness_checker
        self._clock = clock
        spec = next(s for s in build_tool_specs(tools) if s.name == SEARCH_TOOL)
        self.search_tool = spec.definition_without(*PINNED_ARGUMENTS)

    # -- the turn -----------------------------------------------------------

    def _answer(
        self,
        query: str,
        history: list[ChatTurn] | None,
        on_event: EventSink,
        trace: TurnTrace,
        query_filter: QueryFilter | None = None,
    ) -> ChatAnswer:
        """Run the strategy, then build a `ChatAnswer` from the ledger.

        `history` goes to the model as real messages -- no condenser: the
        model sees the conversation and writes standalone queries itself
        (plan decision 8). Earlier answers lose their `[n]` markers on the way
        in, because those numbers belonged to earlier turns' passages.
        """

        if not query.strip():
            trace.outcome = "blank_query"
            return ChatAnswer(answer=BLANK_QUERY_ANSWER, citations=[], retrieval_attempts=0)

        run = _Run(
            messages=[ChatMessage("system", self._system_prompt), *self._history(history), ChatMessage("user", query)],
            ledger=PassageLedger(self.max_passage_chars),
            deadline=self._clock() + self.timeout_s,
            on_event=on_event,
            trace=trace,
            query_filter=query_filter,
        )
        stopped = self._react(run) if self.strategy == "react" else self._planned(run)

        chunks = run.ledger.chunks
        trace.shown_chunk_ids = [chunk.chunk_id for chunk in chunks]
        grounded: bool | None = None
        if self._groundedness_checker is not None and chunks and run.answer.strip():
            start = time.monotonic()
            grounded = self._groundedness_checker.check(query, chunks, run.answer)
            trace.groundedness_checks.append(grounded)
            label = "inconclusive" if grounded is None else ("grounded" if grounded else "UNGROUNDED")
            emit(on_event, start, "crag_groundedness", f"Groundedness check: {label}")

        if not run.answer.strip():
            logger.warning("Agent returned an empty answer for %r (stopped: %s)", query, stopped)
        logger.info(
            "Agent answered %r: %d search(es) in %d round(s), %d passage(s) shown, stopped=%s",
            query, len(run.queries), run.rounds, len(chunks), stopped,
        )
        return ChatAnswer(
            answer=run.answer,
            citations=[to_citation(chunk) for chunk in chunks],
            dropped_below_min_score=run.dropped_below_min_score,
            search_queries=list(run.queries),
            retrieval_attempts=run.rounds,
            grounded=grounded,
            cited_chunk_ids=[chunks[n - 1].chunk_id for n in parse_cited_passages(run.answer, len(chunks))],
            tool_calls=len(run.queries),
            stopped_reason=stopped,
        )

    def _history(self, history: list[ChatTurn] | None) -> list[Message]:
        turns = (history or [])[-self.max_history_turns :] if self.max_history_turns else []
        return [
            ChatMessage("user", turn.content)
            if turn.role == "user"
            else AssistantTurn(content=strip_citation_markers(turn.content))
            for turn in turns
        ]

    # -- strategies ---------------------------------------------------------

    def _react(self, run: _Run) -> StoppedReason:
        """The model decides after every result whether to search again.

        Bounded twice: by searches run (`max_tool_calls`) and by model steps
        (also `max_tool_calls`), so a model that only ever repeats itself --
        every call refused, no search spent -- still runs out.
        """

        while True:
            if run.hit_timeout or self._clock() >= run.deadline:
                return self._finish(run, "timeout")
            try:
                turn = self._call(run, [self.search_tool], "agent_step")
            except ContextOverflowError:
                if not self._roll_back_step(run):
                    raise
                return self._finish(run, "context")
            run.messages.append(turn)
            if not turn.tool_calls:
                run.answer = turn.content
                return "answered"

            self._run_calls(run, turn.tool_calls)
            # Out of searches means no tools next time, so the next call is the
            # forced one whether or not this step had a search refused.
            if len(run.queries) >= self.max_tool_calls or run.steps >= self.max_tool_calls:
                return self._finish(run, "cap")

    def _planned(self, run: _Run) -> StoppedReason:
        """One planning call, every planned search back to back, one answering call.

        The plan is the tool calls of a single turn: they are the model's
        structured output, and a model that "never made a malformed call" in
        the probe needs no JSON parser. The same guards apply unchanged --
        duplicates in the plan are refused, `max_tool_calls` truncates it, and
        the timeout stops it between searches. A model that answers instead
        of planning (a greeting) is taken at its word, in one call.
        """

        plan = self._call(run, [self.search_tool], "agent_plan")
        run.messages.append(plan)
        if not plan.tool_calls:
            run.answer = plan.content
            return "answered"

        self._run_calls(run, plan.tool_calls)
        if run.hit_timeout:
            return self._finish(run, "timeout")
        return self._finish(run, "cap" if run.hit_cap else "answered")

    # -- one model call, one step's tool calls ------------------------------

    def _call(self, run: _Run, tools: Sequence[ToolDefinition], stage: str) -> AssistantTurn:
        start = time.monotonic()
        turn = self._llm.chat(run.messages, tools)
        run.steps += 1 if tools else 0
        if turn.tool_calls:
            message = f"Model asked for {len(turn.tool_calls)} search(es)"
        else:
            message = "Model answered" if turn.content.strip() else "Model returned an empty answer"
        emit(run.on_event, start, stage, message)
        return turn

    def _run_calls(self, run: _Run, calls: Sequence[ToolCall]) -> None:
        """Answer every call of one step, in order, and append the results.

        Every call gets a result -- providers pair them up, and Gemini rejects
        a step with a call left unanswered -- but only a new, valid query
        within budget and time spends a search.
        """

        run.step_message_mark = len(run.messages)
        run.step_ledger_mark = len(run.ledger)
        searched_this_step = False
        for call in calls:
            content = self._refusal(run, call)
            if content is None:
                content = self._search(run, str(call.arguments["query"]))
                searched_this_step = True
            run.messages.append(ToolResult(call=call, content=content))
        run.rounds += 1 if searched_this_step else 0

    def _refusal(self, run: _Run, call: ToolCall) -> str | None:
        """Why `call` won't run, as the tool result the model sees -- or None to run it."""

        start = time.monotonic()
        query = call.arguments.get("query")
        if call.name != SEARCH_TOOL:
            reason = f"Unknown tool {call.name!r}. The only tool is {SEARCH_TOOL}."
        elif not isinstance(query, str) or not query.strip():
            reason = f"{SEARCH_TOOL} needs a non-empty 'query' string."
        elif (earlier := run.searched.get(_normalize(query))) is not None:
            found = f"its results are passages {_numbers_label(earlier)} above" if earlier else "it found nothing"
            reason = (
                f"Already searched for {query!r}; {found}. Search for something different, "
                "or answer from the passages you have."
            )
        elif len(run.queries) >= self.max_tool_calls:
            run.hit_cap = True
            reason = "Search budget for this question is used up; this search was not run."
        elif self._clock() >= run.deadline:
            run.hit_timeout = True
            reason = "Out of time for this question; this search was not run."
        else:
            return None
        emit(run.on_event, start, "search_refused", reason)
        return reason

    def _search(self, run: _Run, query: str) -> str:
        """Run one search, number its passages, and render them for the model."""

        start = time.monotonic()
        try:
            _, result = self._tools.retrieve(
                query, self._corpora, query_filter=run.query_filter, on_event=run.on_event
            )
        except ValueError as exc:
            # An argument the model got wrong: tell it, don't fail the turn.
            emit(run.on_event, start, "search_refused", f"Search error: {exc}")
            return f"Search error: {exc}"

        run.queries.append(query)
        run.dropped_below_min_score += result.dropped_below_min_score
        entries = run.ledger.add(result.chunks)
        run.searched[_normalize(query)] = [number for number, _, _ in entries]
        run.trace.attempts.append(
            RetrievalAttempt(
                query=query,
                retrieved=[RetrievedPassage(c.chunk_id, c.document_id, c.score) for c in result.chunks],
                candidate_count=result.candidate_count,
                dropped_below_min_score=result.dropped_below_min_score,
                routed_to=result.routed_to,
            )
        )

        new = [number for number, _, is_new in entries if is_new]
        emit(
            run.on_event, start, "search",
            f"Searched {query!r}: {len(entries)} passage(s), {len(new)} new"
            + (f" ({_numbers_label(new)})" if new else ""),
        )

        if not entries:
            if result.candidate_count == 0:
                return "No passages found. The index returned nothing at all; it may be empty or not built."
            return (
                f"No passages found: {result.candidate_count} candidate(s) all scored below the "
                "relevance threshold. The corpus may not cover this."
            )
        return "\n\n".join(
            format_passage(number, chunk) if is_new
            else f"Passage [{number}] (source: {chunk.document_id}): already shown above."
            for number, chunk, is_new in entries
        )

    # -- ending the turn ----------------------------------------------------

    def _finish(self, run: _Run, reason: StoppedReason) -> StoppedReason:
        """The forced synthesis turn: no tools offered, so the model must answer in text.

        Preceded by an explicit instruction (`AGENT_SYNTHESIS_INSTRUCTION`):
        without one, the 27b at its cap answered "" or "Let me try...". A
        prompt that overflows here gets its latest step rolled back and one
        more try.
        """

        run.messages.append(ChatMessage("user", AGENT_SYNTHESIS_INSTRUCTION))
        try:
            turn = self._call(run, [], "agent_answer")
        except ContextOverflowError:
            if not self._roll_back_step(run):
                raise
            turn = self._call(run, [], "agent_answer")
            reason = "context"
        run.messages.append(turn)
        run.answer = turn.content
        return reason

    def _roll_back_step(self, run: _Run) -> bool:
        """Blank the latest step's results after a context overflow; False if there is none.

        The calls keep their results (providers require one per call), but the
        passages are replaced with a note and dropped from the ledger, so the
        answer can't cite what the model never saw. Their searches did run and
        stay counted.
        """

        mark = run.step_message_mark
        if mark is None:
            return False
        note = "Results not shown: the conversation reached the model's context window."
        for index in range(mark, len(run.messages)):
            message = run.messages[index]
            if isinstance(message, ToolResult):
                run.messages[index] = dataclasses.replace(message, content=note)
        run.ledger.truncate(run.step_ledger_mark)
        run.step_message_mark = None
        emit(run.on_event, time.monotonic(), "agent_overflow", "Context window full; answering from earlier passages")
        return True
