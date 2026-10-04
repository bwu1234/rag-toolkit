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
import json
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

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
from rag.agent.calculator import (
    CALCULATOR_DEFINITION,
    CALCULATOR_TOOL,
    CalculatorError,
    evaluate,
    format_result,
)
from rag.generation.crag import GroundednessChecker
from rag.llm.base import (
    AssistantTurn,
    ChatMessage,
    ContextOverflowError,
    Message,
    ToolCall,
    ToolCallingLLM,
    ToolDefinition,
    ToolResult,
)
from rag.agent.prompts import AGENT_SYNTHESIS_INSTRUCTION
from rag.generation.prompts import (
    format_passage,
    parse_cited_passages,
    strip_citation_markers,
)
from rag.generation.query_rewriter import ChatTurn
from rag.observability.records import AgentToolCall, RetrievalAttempt, RetrievedPassage
from rag.observability.sink import TurnSink
from rag.tools import RagTools, build_tool_specs
from rag.vectorstore.base import ScoredChunk

logger = logging.getLogger(__name__)

SEARCH_TOOL = "rag_search"
LIST_TOOL = "rag_list_documents"

#: `rag_search` arguments the turn decides, not the model. `corpus` is the
#: turn's corpus selection (an eval's `--corpus`); a model free to widen it
#: would be measured against a different index. `top_k` and `max_chars` are
#: the prompt-size guards: prompts grew ~6x over the pipeline's in the
#: prototype, and a model asking for 20 full-length passages per search would
#: undo `max_tool_calls` x `max_passage_chars` as a bound.
#: `filters` is pinned unless `agent.model_filters` is on. Either way a `/chat`
#: filter applies to every search; a model filter can only narrow it.
PINNED_ARGUMENTS = ("corpus", "top_k", "max_chars")
FILTERS_ARGUMENT = "filters"
#: `rag_list_documents` arguments the turn decides: the corpus, as for search,
#: and the size and page of a listing (one page of `DEFAULT_LIST_LIMIT`, which
#: covers `edgar_md`; paging would also need repeat-refusal keyed by page).
#: Its `filters` is always the model's -- narrowing a listing is what the tool
#: is for, and it changes no search results.
LIST_PINNED_ARGUMENTS = ("corpus", "limit", "offset")


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
    #: (Normalized query, the model's filter) -> the passage numbers its search returned.
    searched: dict[tuple[str, str], list[int]] = field(default_factory=dict)
    #: The model's filter keys of the listings already run, for refusing a repeat.
    listed: set[str] = field(default_factory=set)
    #: Every tool call the model made, in order: the trajectory `ChatAnswer` reports.
    calls: list[AgentToolCall] = field(default_factory=list)
    #: Tool calls that ran -- searches and listings -- against `max_tool_calls`.
    calls_run: int = 0
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
    step_call_mark: int = 0
    answer: str = ""


def _normalize(query: str) -> str:
    return " ".join(query.casefold().split())


def _numbers_label(numbers: Sequence[int]) -> str:
    return ", ".join(f"[{n}]" for n in numbers)


def _filter_key(model_filter: QueryFilter | None) -> str:
    """A filter as part of the repeat-search key: one query under two filters is two searches.

    Canonical, so the same conditions written in another order are the same
    key: fields sorted, and `any_of` values too (membership ignores order).
    """

    if model_filter is None:
        return ""
    dumped = model_filter.model_dump(exclude_defaults=True)
    dumped["any_of"] = {name: sorted(options) for name, options in model_filter.any_of.items()}
    return json.dumps(dumped, sort_keys=True)


def _filter_label(model_filter: QueryFilter | None) -> str:
    return "" if model_filter is None else f" [filter: {model_filter.describe()}]"


def _filter_dump(model_filter: QueryFilter | None) -> dict[str, Any] | None:
    return None if model_filter is None else model_filter.model_dump(exclude_defaults=True)


def _render_listing(payload: Mapping[str, Any], model_filter: QueryFilter | None) -> str:
    """A `list_documents` payload as the model reads it: one line per document.

    Text, not the MCP JSON: a line per document costs a fraction of the
    tokens, and the agent's other results are text too.
    """

    documents = payload["documents"]
    if not documents:
        if model_filter is not None:
            return (
                f"No documents match the filter ({model_filter.describe()}). Check its values "
                "-- dates are compared exactly within a range -- or list without it."
            )
        return "No documents in the corpus."
    head = f"{payload['total']} document(s){_filter_label(model_filter)}"
    if payload["total"] > len(documents):
        head += f"; showing the first {len(documents)}. Narrow the listing with a filter to see the rest"
    lines = [
        f"- {document['document_id']}: "
        + "; ".join(
            f"{key} {value}" for key, value in document.items() if key not in ("document_id", "chars")
        )
        + f"; {document['chars']:,} chars"
        for document in documents
    ]
    return head + ":\n" + "\n".join(lines)


def _log_empty(turn: AssistantTurn, where: str) -> None:
    """Say why a model call came back with nothing, as far as the provider reports it."""

    reasoning = len(turn.thinking) if turn.thinking else 0
    hint = (
        " It stopped at max_tokens: reasoning can use the whole budget, so raise the "
        "agent model's max_tokens or set think: low."
        if turn.stop_reason == "length"
        else ""
    )
    logger.warning(
        "Model returned neither text nor a tool call at the %s (stop_reason=%s, %d reasoning chars).%s",
        where, turn.stop_reason, reasoning, hint,
    )


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
        model_filters: bool = False,
        offered_tools: Sequence[str] = (SEARCH_TOOL,),
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
        self.model_filters = model_filters
        self._groundedness_checker = groundedness_checker
        self._clock = clock
        specs = {spec.name: spec for spec in build_tool_specs(tools)}
        unknown = sorted(set(offered_tools) - {SEARCH_TOOL, LIST_TOOL, CALCULATOR_TOOL})
        if unknown or SEARCH_TOOL not in offered_tools:
            raise ValueError(
                f"The agent offers {SEARCH_TOOL} and optionally {LIST_TOOL} and {CALCULATOR_TOOL}; "
                f"got {list(offered_tools)}"
            )
        pinned = PINNED_ARGUMENTS if model_filters else (*PINNED_ARGUMENTS, FILTERS_ARGUMENT)
        # Descriptions keep only the notes that hold for what this agent offers:
        # no pointer to rag_read_document, no paging by a pinned `offset`.
        self.search_tool = specs[SEARCH_TOOL].definition_without(*pinned, offered_tools=offered_tools)
        self.list_tool = (
            specs[LIST_TOOL].definition_without(*LIST_PINNED_ARGUMENTS, offered_tools=offered_tools)
            if LIST_TOOL in offered_tools
            else None
        )
        self.calculator_tool = CALCULATOR_DEFINITION if CALCULATOR_TOOL in offered_tools else None
        #: What the model is offered on every tool-bearing call, search first.
        self.tool_definitions = [self.search_tool] + [
            tool for tool in (self.list_tool, self.calculator_tool) if tool is not None
        ]

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
            "Agent answered %r: %d tool call(s), %d search round(s), %d passage(s) shown, stopped=%s",
            query, run.calls_run, run.rounds, len(chunks), stopped,
        )
        return ChatAnswer(
            answer=run.answer,
            citations=[to_citation(chunk) for chunk in chunks],
            dropped_below_min_score=run.dropped_below_min_score,
            search_queries=list(run.queries),
            retrieval_attempts=run.rounds,
            grounded=grounded,
            cited_chunk_ids=[chunks[n - 1].chunk_id for n in parse_cited_passages(run.answer, len(chunks))],
            tool_calls=run.calls_run,
            stopped_reason=stopped,
            agent_calls=list(run.calls),
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
                turn = self._call(run, self.tool_definitions, "agent_step")
            except ContextOverflowError:
                if not self._roll_back_step(run):
                    raise
                return self._finish(run, "context")
            if not turn.tool_calls:
                if not turn.content.strip():
                    return self._recover_empty(run, turn)
                run.messages.append(turn)
                run.answer = turn.content
                return "answered"

            run.messages.append(turn)
            self._run_calls(run, turn.tool_calls)
            # Out of searches means no tools next time, so the next call is the
            # forced one whether or not this step had a search refused.
            if run.calls_run >= self.max_tool_calls or run.steps >= self.max_tool_calls:
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

        plan = self._call(run, self.tool_definitions, "agent_plan")
        if not plan.tool_calls:
            if not plan.content.strip():
                return self._recover_empty(run, plan)
            run.messages.append(plan)
            run.answer = plan.content
            return "answered"

        run.messages.append(plan)
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
            message = f"Model asked for {len(turn.tool_calls)} tool call(s)"
        else:
            message = "Model answered" if turn.content.strip() else "Model returned an empty answer"
        emit(run.on_event, start, stage, message)
        return turn

    def _run_calls(self, run: _Run, calls: Sequence[ToolCall]) -> None:
        """Answer every call of one step, in order, and append the results.

        Every call gets a result -- providers pair them up, and Gemini rejects
        a step with a call left unanswered -- but only a new, valid call within
        budget and time runs. Every call is also recorded in `run.calls`,
        refused or not.
        """

        run.step_message_mark = len(run.messages)
        run.step_ledger_mark = len(run.ledger)
        run.step_call_mark = len(run.calls)
        searched_this_step = False
        for call in calls:
            if call.name == LIST_TOOL and self.list_tool is not None:
                content = self._list(run, call)
            elif call.name == CALCULATOR_TOOL and self.calculator_tool is not None:
                content = self._calculate(run, call)
            else:
                content = self._search_call(run, call)
                searched_this_step = searched_this_step or (
                    run.calls[-1].tool == SEARCH_TOOL and run.calls[-1].status == "ran"
                )
            run.messages.append(ToolResult(call=call, content=content))
        run.rounds += 1 if searched_this_step else 0

    def _search_call(self, run: _Run, call: ToolCall) -> str:
        """One `rag_search` call (or a call to a tool not offered): run it, or say why not."""

        query = call.arguments.get("query")
        raw_filters = call.arguments.get(FILTERS_ARGUMENT)
        model_filter, filter_error = self._parse_filter(raw_filters, offered=self.model_filters, error="Search error")
        refusal = filter_error or self._refusal(run, call, model_filter)
        if refusal is None:
            return self._search(run, str(query), model_filter, raw_filters)
        run.calls.append(
            AgentToolCall(
                step=run.steps,
                tool=call.name,
                query=query if isinstance(query, str) else "",
                status="error" if filter_error else "refused",
                filters=_filter_dump(model_filter),
                filters_raw=raw_filters,
                note=refusal,
            )
        )
        return refusal

    def _parse_filter(self, raw: Any, *, offered: bool, error: str) -> tuple[QueryFilter | None, str | None]:
        """A tool's `filters` argument, validated -- or the error the model is shown instead.

        An empty filter is no filter. When the tool doesn't offer `filters`, one
        sent anyway is an error, not ignored: a silently dropped filter would
        make the result look narrower than it is. `error` prefixes the message.
        """

        if raw is None:
            return None, None
        if not offered:
            return None, f"{error}: {SEARCH_TOOL} takes no '{FILTERS_ARGUMENT}' argument here."
        if isinstance(raw, str):
            # The 27b sends the object JSON-encoded, as a string, every time it
            # filters (first live run, `ad-airline-fuel`). Decode it rather than
            # refuse a filter whose only fault is its wire encoding.
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                return None, f"{error}: invalid filters: not a JSON object."
        try:
            model_filter = QueryFilter.model_validate(raw)
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(str(part) for part in detail['loc']) or 'filters'}: {detail['msg']}"
                for detail in exc.errors()
            )
            return None, f"{error}: invalid filters: {problems}"
        return (None if model_filter.is_empty else model_filter), None

    def _refusal(self, run: _Run, call: ToolCall, model_filter: QueryFilter | None) -> str | None:
        """Why a search call won't run, as the tool result the model sees -- or None to run it."""

        start = time.monotonic()
        query = call.arguments.get("query")
        if call.name != SEARCH_TOOL:
            offered = ", ".join(tool.name for tool in self.tool_definitions)
            reason = f"Unknown tool {call.name!r}. The tools are: {offered}."
        elif not isinstance(query, str) or not query.strip():
            reason = f"{SEARCH_TOOL} needs a non-empty 'query' string."
        elif (earlier := run.searched.get((_normalize(query), _filter_key(model_filter)))) is not None:
            found = f"its results are passages {_numbers_label(earlier)} above" if earlier else "it found nothing"
            reason = (
                f"Already searched for {query!r}{_filter_label(model_filter)}; {found}. "
                "Search for something different, or answer from the passages you have."
            )
        else:
            budget = self._budget_refusal(run, "search")
            if budget is None:
                return None
            reason = budget
        emit(run.on_event, start, "search_refused", reason)
        return reason

    def _budget_refusal(self, run: _Run, what: str) -> str | None:
        """Why no tool call can run now -- the shared budget or the clock -- or None."""

        if run.calls_run >= self.max_tool_calls:
            run.hit_cap = True
            return f"Tool budget for this question is used up; this {what} was not run."
        if self._clock() >= run.deadline:
            run.hit_timeout = True
            return f"Out of time for this question; this {what} was not run."
        return None

    def _list(self, run: _Run, call: ToolCall) -> str:
        """One `rag_list_documents` call: list the turn's documents, or say why not.

        The listing goes to the model as text and into the trace; it is not a
        passage, so it takes no ledger number and can't be cited. Scope is the
        search's: the turn's corpora, and the turn's filter narrowed by the
        model's.
        """

        start = time.monotonic()
        raw_filters = call.arguments.get(FILTERS_ARGUMENT)
        model_filter, filter_error = self._parse_filter(raw_filters, offered=True, error="Listing error")
        key = _filter_key(model_filter)
        refusal = filter_error
        if refusal is None and key in run.listed:
            refusal = (
                f"Already listed documents{_filter_label(model_filter)}; the listing is above. "
                "List with a different filter, or search."
            )
        if refusal is None:
            refusal = self._budget_refusal(run, "listing")
        payload: dict[str, Any] | None = None
        if refusal is None:
            try:
                query_filter = run.query_filter
                if model_filter is not None:
                    query_filter = model_filter if query_filter is None else query_filter.intersect(model_filter)
                payload = self._tools.list_documents(self._corpora, filters=query_filter)
            except ValueError as exc:
                filter_error = refusal = f"Listing error: {exc}"
        if payload is None:
            assert refusal is not None
            emit(run.on_event, start, "search_refused", refusal)
            run.calls.append(
                AgentToolCall(
                    step=run.steps, tool=LIST_TOOL, status="error" if filter_error else "refused",
                    filters=_filter_dump(model_filter), filters_raw=raw_filters, note=refusal,
                )
            )
            return refusal

        run.calls_run += 1
        run.listed.add(key)
        documents = payload["documents"]
        run.calls.append(
            AgentToolCall(
                step=run.steps, tool=LIST_TOOL, filters=_filter_dump(model_filter), filters_raw=raw_filters,
                documents=[document["document_id"] for document in documents],
            )
        )
        emit(
            run.on_event, start, "list_documents",
            f"Listed {len(documents)} of {payload['total']} document(s){_filter_label(model_filter)}",
        )
        return _render_listing(payload, model_filter)

    def _calculate(self, run: _Run, call: ToolCall) -> str:
        """One `calculator` call: the expression's value, or what to fix.

        Not counted against `max_tool_calls` -- it adds no passages, and a
        search budget spent on arithmetic would make the tool look worse than
        it is -- and never refused as a repeat: it's deterministic and cheap.
        Each step that calculates is still a model step, which `react` caps.
        """

        start = time.monotonic()
        expression = call.arguments.get("expression")
        result: str | None = None
        if not isinstance(expression, str):
            note: str | None = f"{CALCULATOR_TOOL} needs an 'expression' string."
        else:
            try:
                result = format_result(evaluate(expression))
                note = None
            except CalculatorError as exc:
                note = f"Calculator error: {exc}."
        run.calls.append(
            AgentToolCall(
                step=run.steps, tool=CALCULATOR_TOOL, status="ran" if result is not None else "error",
                expression=expression if isinstance(expression, str) else None, result=result, note=note,
            )
        )
        if result is None:
            assert note is not None
            emit(run.on_event, start, "search_refused", note)
            return note
        emit(run.on_event, start, "calculate", f"Calculated {expression} = {result}")
        return f"{expression} = {result}"

    def _search(self, run: _Run, query: str, model_filter: QueryFilter | None, raw_filters: Any) -> str:
        """Run one search, number its passages, and render them for the model.

        The model's filter narrows the turn's, never replaces it: a `/chat`
        filter holds on every search whatever the model asks for.
        """

        start = time.monotonic()
        try:
            query_filter = run.query_filter
            if model_filter is not None:
                query_filter = model_filter if query_filter is None else query_filter.intersect(model_filter)
            _, result = self._tools.retrieve(
                query, self._corpora, query_filter=query_filter, on_event=run.on_event
            )
        except ValueError as exc:
            # An argument the model got wrong: tell it, don't fail the turn.
            note = f"Search error: {exc}"
            emit(run.on_event, start, "search_refused", note)
            run.calls.append(
                AgentToolCall(
                    step=run.steps, query=query, status="error",
                    filters=_filter_dump(model_filter), filters_raw=raw_filters, note=note,
                )
            )
            return note

        run.queries.append(query)
        run.calls_run += 1
        run.dropped_below_min_score += result.dropped_below_min_score
        entries = run.ledger.add(result.chunks)
        numbers = [number for number, _, _ in entries]
        new = [number for number, _, is_new in entries if is_new]
        run.searched[(_normalize(query), _filter_key(model_filter))] = numbers
        run.calls.append(
            AgentToolCall(
                step=run.steps,
                query=query,
                filters=_filter_dump(model_filter),
                filters_raw=raw_filters,
                passages=numbers,
                new_passages=new,
                chunk_ids=[chunk.chunk_id for _, chunk, _ in entries],
            )
        )
        run.trace.attempts.append(
            RetrievalAttempt(
                query=query,
                retrieved=[RetrievedPassage(c.chunk_id, c.document_id, c.score) for c in result.chunks],
                candidate_count=result.candidate_count,
                dropped_below_min_score=result.dropped_below_min_score,
                routed_to=result.routed_to,
            )
        )

        emit(
            run.on_event, start, "search",
            f"Searched {query!r}{_filter_label(model_filter)}: {len(entries)} passage(s), {len(new)} new"
            + (f" ({_numbers_label(new)})" if new else ""),
        )

        if not entries:
            if result.candidate_count == 0 and model_filter is not None:
                # With a filter of the model's own, nothing at all almost always
                # means the filter, not the index: say so, or the model reads it
                # as "the corpus doesn't cover this" and gives up.
                return (
                    f"No passages match the filter ({model_filter.describe()}). Check its values "
                    "-- dates are compared exactly within a range -- or search without it."
                )
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
        if not turn.content.strip():
            _log_empty(turn, "forced answer turn")
        return reason

    def _recover_empty(self, run: _Run, turn: AssistantTurn) -> StoppedReason:
        """A reply with neither text nor a tool call is not an answer: force one, once.

        The 27b with default thinking did this on a refusal question in every
        run of the Milestone 19 re-run on `edgar_md`, and the turn returned ""
        as if answered. The empty turn is left out of the conversation -- it
        carries nothing the model needs, and resending a reasoning trace that
        filled `max_tokens` would only spend more of the window -- and the
        forced synthesis turn gets one try. If that is empty too, the turn
        returns empty, logged, rather than retrying without bound.
        """

        _log_empty(turn, "agent step")
        return self._finish(run, "empty")

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
        run.calls[run.step_call_mark :] = [
            dataclasses.replace(search, passages=[], new_passages=[], chunk_ids=[], documents=[], note=note)
            if search.status == "ran" else search
            for search in run.calls[run.step_call_mark :]
        ]
        run.step_message_mark = None
        emit(run.on_event, time.monotonic(), "agent_overflow", "Context window full; answering from earlier passages")
        return True
