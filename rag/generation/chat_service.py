"""Orchestrates retrieve -> prompt -> generate -> cite for the chat API.

`ChatService` is the seam between the retrieval pipeline (Milestone 5) and
generation: it's the one place that knows how to turn a `Retriever`'s
`ScoredChunk`s into both a grounded prompt and a citation list the API can
return alongside the model's answer. Keeping this orchestration out of the
API route layer means the eval pipeline (Milestone 7) can reuse it directly
without spinning up FastAPI.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass, field

from rag.events import EventSink, PipelineEvent, emit
from rag.generation.crag import DocumentGrader, GroundednessChecker, RetryQueryRewriter
from rag.generation.llm import LLMClient
from rag.config.settings import PromptStyle
from rag.generation.prompts import (
    REGROUND_SYSTEM_PROMPT,
    build_prompt,
    parse_cited_passages,
    system_prompt_for,
)
from rag.generation.query_rewriter import ChatTurn, QueryCondenser
from rag.observability.records import (
    RetrievalAttempt,
    RetrievedPassage,
    StageEvent,
    TurnOutcome,
    TurnRecord,
    new_id,
    utc_now,
)
from rag.observability.sink import TurnSink
from rag.observability.usage import UsageMeter, metered
from rag.retrieval.retriever import RetrievalResult, Retriever
from rag.vectorstore.base import ScoredChunk

logger = logging.getLogger(__name__)

# Four distinct ways a turn can end without a generated answer. They were one
# string until the `retrieval.min_score` floor made the distinction real: with
# a floor in place, "nothing came back" and "things came back and none of them
# were relevant" are different facts about the corpus, and collapsing them
# leaves a user unable to tell "I need to run the indexer" from "this corpus
# doesn't cover my question". CRAG's document grader adds a fourth: passages
# that cleared the floor and were then judged not to answer the question.
_BLANK_QUERY_ANSWER = "I didn't get a question to answer -- try asking something."

_EMPTY_INDEX_ANSWER = (
    "Nothing came back from the index at all, which usually means it's empty "
    "or hasn't been built yet. Run `python -m rag.cli index` to build it."
)


def _no_relevant_context_answer(candidate_count: int, dropped: int) -> str:
    return (
        f"I couldn't find anything relevant to that in the indexed corpus. "
        f"{candidate_count} passage(s) were retrieved, but all {dropped} of the "
        f"best-ranked ones scored below the relevance threshold -- so rather "
        f"than guess from material the reranker judged unrelated, I'm telling "
        f"you the corpus doesn't appear to cover this."
    )


def _graded_out_answer(graded_out: int, attempts: int) -> str:
    attempt_note = (
        f" across {attempts} search attempts" if attempts > 1 else ""
    )
    return (
        f"I found passages that looked relevant but none of them actually "
        f"answer your question. {graded_out} retrieved passage(s){attempt_note} "
        f"were checked and each was judged not to bear on what you asked -- so "
        f"rather than write an answer from material that doesn't support one, "
        f"I'm telling you the corpus doesn't appear to cover this."
    )


@dataclass(frozen=True)
class Citation:
    """A single retrieved passage backing an answer, shaped for API responses.

    Deliberately a thin, flat projection of `ScoredChunk` (rather than
    re-exposing it directly) -- the API's citation contract should be able to
    evolve independently of the internal retrieval result type, and callers
    shouldn't need to know about `ScoredChunk.metadata`'s grab-bag shape.
    """

    chunk_id: str
    document_id: str
    text: str
    score: float
    page: int | None = None


@dataclass(frozen=True)
class ChatAnswer:
    """The full result of a chat turn: the model's answer, its sources, and what shaped them.

    The trailing fields exist so every caller -- not just the one UI that
    happens to pass an `on_event` sink -- can see the places this pipeline
    silently changes what the user asked for, what it answered from, or how
    much it trusts the result: the question was rewritten for a follow-up, it
    was expanded into several searches, passages were withheld by the relevance
    floor or by CRAG's grader, retrieval was retried, or the answer failed its
    own groundedness check. None of that is visible in the answer text, and all
    of it explains results a user might otherwise find baffling.
    """

    answer: str
    citations: list[Citation] = field(default_factory=list)
    rewritten_query: str | None = None
    """The standalone question retrieval actually used, when condensing changed it."""
    dropped_below_min_score: int = 0
    """Reranked passages withheld by the `retrieval.min_score` floor."""
    search_queries: list[str] = field(default_factory=list)
    """Every query searched, when HyDE / multi-query expansion generated more than one."""
    graded_out: int = 0
    """Passages CRAG's grader judged irrelevant and dropped, across all attempts."""
    retry_queries: list[str] = field(default_factory=list)
    """Rewritten queries CRAG searched with after an attempt came back empty."""
    retrieval_attempts: int = 1
    """Times retrieval ran for this turn -- more than one means CRAG retried."""
    grounded: bool | None = None
    """CRAG's groundedness verdict: True/False, or None if unchecked or inconclusive."""
    cited_chunk_ids: list[str] = field(default_factory=list)
    """Chunk ids of the passages the answer actually cites as `[n]`, in first-cited order."""
    turn_id: str | None = None
    """Id of this turn's record -- what feedback is attached to. Set by `ChatService.ask`."""
    stage_ms: dict[str, float] = field(default_factory=dict)
    """Milliseconds per pipeline stage, summed when a stage ran more than once (CRAG retries)."""
    total_ms: float | None = None
    """Wall-clock milliseconds for the whole turn."""
    llm_calls: int = 0
    """LLM calls this turn made, from every component: expansion, condensing, CRAG, generation."""
    prompt_tokens: int | None = None
    """Prompt tokens across those calls; None when the provider doesn't report them."""
    completion_tokens: int | None = None
    """Generated tokens across those calls; None when the provider doesn't report them."""


@dataclass
class _TurnTrace:
    """What a turn did that `ChatAnswer` doesn't carry, collected for its `TurnRecord`.

    Mutable and private: the pipeline methods append to it as they go, which
    keeps their return types about the answer rather than about logging.
    """

    events: list[PipelineEvent] = field(default_factory=list)
    attempts: list[RetrievalAttempt] = field(default_factory=list)
    groundedness_checks: list[bool | None] = field(default_factory=list)
    shown_chunk_ids: list[str] = field(default_factory=list)
    outcome: TurnOutcome = "answered"


def _stage_totals(events: list[PipelineEvent]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for event in events:
        if event.elapsed_ms is not None:
            totals[event.stage] = totals.get(event.stage, 0.0) + event.elapsed_ms
    return totals


@dataclass(frozen=True)
class _CorrectedRetrieval:
    """Internal result of the retrieve-grade-retry loop.

    Private to this module: it exists to keep `ask` readable by giving the loop
    somewhere to put its five outputs, not to be part of any contract. What
    callers see is the flat `ChatAnswer` projection built from it.
    """

    chunks: list[ScoredChunk]
    result: RetrievalResult
    """The last attempt's raw retrieval result -- the source of the counts and queries."""
    graded_out: int = 0
    retry_queries: list[str] = field(default_factory=list)
    attempts: int = 1


def _verdict_label(grounded: bool | None) -> str:
    if grounded is None:
        return "inconclusive"
    return "grounded" if grounded else "UNGROUNDED"


def _to_citation(chunk: ScoredChunk) -> Citation:
    page = chunk.metadata.get("page")
    return Citation(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        text=chunk.text,
        score=chunk.score,
        page=page if isinstance(page, int) else None,
    )


class ChatService:
    """Answers a question by retrieving context and asking the LLM to ground its reply in it.

    Two required collaborators, both injected as interfaces (`Retriever` is
    itself a thin orchestrator over `EmbeddingModel`/`VectorStore`/`Reranker`;
    see `rag.retrieval.retriever`), plus an optional `QueryCondenser` for
    multi-turn query rewriting: nothing here depends on Chroma, Ollama, or any
    concrete adapter, so it's trivial to test with fakes and trivial to point
    at different components via config alone.

    The CRAG collaborators (`grader`, `retry_rewriter`, `groundedness_checker`;
    see `rag.generation.crag`) are optional in the same way, and each is
    independently omittable -- `crag.enabled: false` simply passes `None` for
    all three, leaving the method body's fast path identical to what it was
    before corrective retrieval existed. They're wired in here, rather than
    inside `Retriever`, because two of the three span the retrieve/generate
    boundary this class exists to own: retrying means going back to retrieval
    *after* judging its output, and checking groundedness means comparing a
    generated answer against the passages that produced it. Neither half of the
    pipeline can see both sides; this class is the only thing that can.
    """

    def __init__(
        self,
        retriever: Retriever,
        llm_client: LLMClient,
        condenser: QueryCondenser | None = None,
        *,
        grader: DocumentGrader | None = None,
        retry_rewriter: RetryQueryRewriter | None = None,
        groundedness_checker: GroundednessChecker | None = None,
        max_retries: int = 0,
        max_regenerations: int = 0,
        turn_sink: TurnSink | None = None,
        turn_metadata: Mapping[str, str] | None = None,
        prompt_style: PromptStyle = "grounded",
    ) -> None:
        self._retriever = retriever
        self._llm_client = llm_client
        self._condenser = condenser
        self._grader = grader
        self._retry_rewriter = retry_rewriter
        self._groundedness_checker = groundedness_checker
        self.max_retries = max_retries
        self.max_regenerations = max_regenerations
        self._turn_sink = turn_sink
        self._turn_metadata = dict(turn_metadata or {})
        self.prompt_style: PromptStyle = prompt_style

    def ask(
        self,
        query: str,
        *,
        history: list[ChatTurn] | None = None,
        on_event: EventSink | None = None,
    ) -> ChatAnswer:
        """Answer `query`, grounded in (and citing) the retrieved context.

        Returns an empty-citation `ChatAnswer` without calling the LLM at all
        when the query is blank or retrieval finds nothing -- both are cases
        where a generated answer could only be a hallucination, and skipping
        the LLM call keeps the no-context path fast and free. The three ways
        that can happen get three different messages, because "you haven't
        built an index" and "your corpus doesn't cover this" are different
        problems with different fixes.

        The returned `ChatAnswer` also reports the things the pipeline did that
        the answer text can't show: the rewritten query (if condensing changed
        it), how many passages the relevance floor withheld, and -- when CRAG is
        enabled -- how many passages the grader rejected, what it retried with,
        and whether the answer passed its groundedness check.

        With `crag.enabled`, retrieval becomes a loop rather than a single call:
        each attempt's passages are graded, an attempt that leaves nothing
        triggers a reworded retry, and the generated answer is checked back
        against the passages before being returned. See `rag.generation.crag`.
        With CRAG off, none of those collaborators exist and the path through
        this method is unchanged.

        `history`, if given, is the conversation preceding `query`, oldest
        first. When a `QueryCondenser` is wired in, it's used to rewrite `query`
        into a standalone question first -- see `rag.generation.query_rewriter`
        for why that matters and what drives the rewritten query afterwards.
        Callers with no conversation (the CLI, the eval pipeline) simply omit
        it and nothing extra runs.

        `on_event`, if given, is called once per completed pipeline stage
        (condensing, retrieval's own sub-stages, prompt assembly, generation)
        with a `PipelineEvent` -- the seam the UI uses to show a
        from-prompt-to-answer trace. Purely observational: omitting it changes
        no return value.

        Every turn is also measured and, when a `TurnSink` is wired in,
        recorded: the returned `ChatAnswer` carries per-stage latency, LLM call
        and token counts, the passages the answer cited, and the `turn_id`
        feedback attaches to. A turn that raises is recorded too (outcome
        `error`) before the exception propagates -- a failing turn is exactly
        the one worth finding in the log afterwards.
        """

        turn_id = new_id()
        timestamp = utc_now()
        trace = _TurnTrace()

        # Tee every pipeline event into the trace as well as the caller's sink.
        # The events already carry each stage's timing, so latency comes from
        # the same stream the UI renders rather than from a second set of timers.
        def observe(event: PipelineEvent) -> None:
            trace.events.append(event)
            if on_event is not None:
                on_event(event)

        start = time.monotonic()
        with metered() as meter:
            try:
                answer = self._answer(query, history, observe, trace)
            except Exception as exc:
                trace.outcome = "error"
                self._record(
                    turn_id, timestamp, query, history, None, trace, meter,
                    total_ms=(time.monotonic() - start) * 1000,
                    error=f"{type(exc).__name__}: {exc}",
                )
                raise

        answer = dataclasses.replace(
            answer,
            turn_id=turn_id,
            stage_ms=_stage_totals(trace.events),
            total_ms=(time.monotonic() - start) * 1000,
            llm_calls=meter.calls,
            prompt_tokens=meter.prompt_tokens,
            completion_tokens=meter.completion_tokens,
        )
        self._record(turn_id, timestamp, query, history, answer, trace, meter, total_ms=answer.total_ms or 0.0)
        return answer

    def _record(
        self,
        turn_id: str,
        timestamp: str,
        query: str,
        history: list[ChatTurn] | None,
        answer: ChatAnswer | None,
        trace: _TurnTrace,
        meter: UsageMeter,
        *,
        total_ms: float,
        error: str | None = None,
    ) -> None:
        """Write this turn's `TurnRecord` to the sink, if there is one.

        A failed write is logged and swallowed: losing one log line is not a
        reason to fail the user's turn, which has already been answered.
        """

        if self._turn_sink is None:
            return
        record = TurnRecord(
            turn_id=turn_id,
            timestamp=timestamp,
            query=query,
            outcome=trace.outcome,
            answer=answer.answer if answer is not None else None,
            history_turns=len(history or []),
            rewritten_query=answer.rewritten_query if answer is not None else None,
            search_queries=answer.search_queries if answer is not None else [],
            retry_queries=answer.retry_queries if answer is not None else [],
            attempts=trace.attempts,
            shown_chunk_ids=trace.shown_chunk_ids,
            cited_chunk_ids=answer.cited_chunk_ids if answer is not None else [],
            groundedness_checks=trace.groundedness_checks,
            grounded=answer.grounded if answer is not None else None,
            stage_ms=_stage_totals(trace.events),
            total_ms=total_ms,
            llm_calls=meter.calls,
            llm_ms=meter.llm_ms,
            prompt_tokens=meter.prompt_tokens,
            completion_tokens=meter.completion_tokens,
            events=[StageEvent(e.stage, e.message, e.elapsed_ms) for e in trace.events],
            error=error,
            metadata=self._turn_metadata,
        )
        try:
            self._turn_sink.record_turn(record)
        except Exception:
            logger.warning("Failed to record turn %s; the answer is unaffected", turn_id, exc_info=True)

    def _answer(
        self,
        query: str,
        history: list[ChatTurn] | None,
        on_event: EventSink,
        trace: _TurnTrace,
    ) -> ChatAnswer:
        """The pipeline itself: condense, retrieve with correction, generate, cite."""

        if not query.strip():
            trace.outcome = "blank_query"
            return ChatAnswer(answer=_BLANK_QUERY_ANSWER, citations=[])

        search_query = query
        if history and self._condenser is not None:
            start = time.monotonic()
            search_query = self._condenser.condense(query, history)
            emit(on_event, start, "condense", f"Condensed follow-up into: {search_query!r}")

        # Only report a rewrite that actually changed something -- a condenser
        # that correctly leaves an already-standalone question alone shouldn't
        # make callers render "rewritten to: <the same question>".
        rewritten_query = search_query if search_query != query else None

        corrected = self._retrieve_with_correction(search_query, on_event, trace)
        chunks = corrected.chunks
        if not chunks:
            return self._no_context_answer(query, corrected, rewritten_query, on_event, trace)
        trace.shown_chunk_ids = [chunk.chunk_id for chunk in chunks]

        start = time.monotonic()
        # Generation always answers `search_query` -- the user's question, or
        # its condensed standalone form -- never a CRAG retry rewrite. A retry
        # rewrite is a search device, like a HyDE passage: it exists to find
        # passages, and answering the reworded version would quietly change the
        # question the user actually asked.
        prompt = build_prompt(self.prompt_style, search_query, chunks)
        emit(on_event, start, "prompt", f"Built prompt from {len(chunks)} passage(s)")

        answer, grounded = self._generate_grounded(search_query, chunks, prompt, on_event, trace)
        citations = [_to_citation(chunk) for chunk in chunks]
        # `citations` stays every passage shown, in prompt order -- that's the
        # numbering the answer's `[n]` markers refer to. Which of them the model
        # actually relied on is reported separately.
        cited_chunk_ids = [chunks[n - 1].chunk_id for n in parse_cited_passages(answer, len(chunks))]

        logger.info(
            "Answered query %r with %d citation(s) (%d withheld below min_score, "
            "%d graded out, %d retrieval attempt(s), grounded=%s)",
            query,
            len(citations),
            corrected.result.dropped_below_min_score,
            corrected.graded_out,
            corrected.attempts,
            grounded,
        )
        return ChatAnswer(
            answer=answer,
            citations=citations,
            rewritten_query=rewritten_query,
            dropped_below_min_score=corrected.result.dropped_below_min_score,
            search_queries=corrected.result.search_queries,
            graded_out=corrected.graded_out,
            retry_queries=corrected.retry_queries,
            retrieval_attempts=corrected.attempts,
            grounded=grounded,
            cited_chunk_ids=cited_chunk_ids,
        )

    def _retrieve_with_correction(
        self, query: str, on_event: EventSink | None, trace: _TurnTrace
    ) -> "_CorrectedRetrieval":
        """Retrieve, grade, and retry with a reworded query until something survives.

        The loop runs at least once and at most `1 + max_retries` times, exiting
        the moment an attempt yields usable passages. With no grader and no
        rewriter wired in it degenerates to exactly one `Retriever.retrieve`
        call, which is the pre-CRAG behaviour.

        Grading is always done against the *original* `query`, even on a retry
        that searched for something else: relevance means "bears on what the
        user asked", and grading a passage against the reworded query would let
        a rewrite that drifted validate the drifted results it found.
        """

        attempt_query = query
        retry_queries: list[str] = []
        graded_out = 0
        result = RetrievalResult()

        for attempt in range(1 + self.max_retries):
            if attempt > 0:
                if self._retry_rewriter is None:
                    break
                start = time.monotonic()
                attempt_query = self._retry_rewriter.rewrite(attempt_query, attempt=attempt)
                if attempt_query in retry_queries or attempt_query == query:
                    # A rewrite that reproduces a query we've already run would
                    # spend a full retrieval round trip re-deriving the same
                    # empty result. Stop instead of looping on it.
                    logger.info("Retry %d produced an already-searched query; stopping", attempt)
                    break
                retry_queries.append(attempt_query)
                emit(on_event, start, "crag_retry", f"Retrying with rewritten query: {attempt_query!r}")

            result = self._retriever.retrieve(attempt_query, on_event=on_event)
            chunks = result.chunks
            kept_ids: list[str] | None = None

            if chunks and self._grader is not None:
                start = time.monotonic()
                graded = self._grader.grade(query, chunks)
                graded_out += graded.graded_out
                chunks = graded.kept
                emit(
                    on_event,
                    start,
                    "crag_grade",
                    f"Graded {len(result.chunks)} passage(s): kept {len(chunks)}, "
                    f"dropped {graded.graded_out} as irrelevant",
                )
                kept_ids = [chunk.chunk_id for chunk in chunks]

            trace.attempts.append(
                RetrievalAttempt(
                    query=attempt_query,
                    retrieved=[RetrievedPassage(c.chunk_id, c.document_id, c.score) for c in result.chunks],
                    kept=kept_ids,
                    candidate_count=result.candidate_count,
                    dropped_below_min_score=result.dropped_below_min_score,
                )
            )

            if chunks:
                return _CorrectedRetrieval(
                    chunks=chunks,
                    result=result,
                    graded_out=graded_out,
                    retry_queries=retry_queries,
                    attempts=attempt + 1,
                )

        return _CorrectedRetrieval(
            chunks=[],
            result=result,
            graded_out=graded_out,
            retry_queries=retry_queries,
            attempts=len(retry_queries) + 1,
        )

    def _generate_grounded(
        self,
        query: str,
        chunks: list[ScoredChunk],
        prompt: str,
        on_event: EventSink | None,
        trace: _TurnTrace,
    ) -> tuple[str, bool | None]:
        """Generate an answer and, if a checker is wired in, verify it's supported.

        A rejected answer is regenerated from the same passages under a stricter
        system prompt (`REGROUND_SYSTEM_PROMPT`) -- the passages were already
        graded relevant, so the suspect is the generation, not the retrieval.

        Returns the last answer produced along with its verdict, even when that
        verdict is still `False`. Withholding an answer the checker didn't like
        would be a third kind of refusal built on a single small model's
        one-word opinion; reporting it on `ChatAnswer.grounded` lets the caller
        decide, and keeps a checker that's simply wrong from silently costing
        the user their answer.
        """

        start = time.monotonic()
        answer = self._llm_client.generate(prompt, system=system_prompt_for(self.prompt_style))
        emit(on_event, start, "generate", "Generated answer")

        if self._groundedness_checker is None:
            return answer, None

        start = time.monotonic()
        grounded = self._groundedness_checker.check(query, chunks, answer)
        trace.groundedness_checks.append(grounded)
        emit(on_event, start, "crag_groundedness", f"Groundedness check: {_verdict_label(grounded)}")

        for attempt in range(1, self.max_regenerations + 1):
            if grounded is not False:
                break

            start = time.monotonic()
            answer = self._llm_client.generate(prompt, system=REGROUND_SYSTEM_PROMPT)
            emit(on_event, start, "regenerate", f"Regenerated answer (attempt {attempt})")

            start = time.monotonic()
            grounded = self._groundedness_checker.check(query, chunks, answer)
            trace.groundedness_checks.append(grounded)
            emit(
                on_event,
                start,
                "crag_groundedness",
                f"Groundedness re-check after regeneration {attempt}: {_verdict_label(grounded)}",
            )

        if grounded is False:
            logger.warning("Returning an answer for %r that failed its groundedness check", query)
        return answer, grounded

    def _no_context_answer(
        self,
        query: str,
        corrected: _CorrectedRetrieval,
        rewritten_query: str | None,
        on_event: EventSink | None,
        trace: _TurnTrace,
    ) -> ChatAnswer:
        """Explain *which* kind of nothing retrieval came back with, and skip generation."""

        result = corrected.result

        # Order matters: grading runs last, so if it rejected everything that's
        # the most specific true statement about this turn -- the index wasn't
        # empty and the floor wasn't the obstacle, the passages just didn't
        # answer the question.
        if corrected.graded_out:
            answer = _graded_out_answer(corrected.graded_out, corrected.attempts)
            reason = (
                f"{corrected.graded_out} passage(s) graded irrelevant across "
                f"{corrected.attempts} attempt(s)"
            )
            trace.outcome = "graded_out"
        elif result.candidate_count == 0:
            answer = _EMPTY_INDEX_ANSWER
            reason = "index returned no candidates at all"
            trace.outcome = "empty_index"
        else:
            trace.outcome = "below_min_score"
            answer = _no_relevant_context_answer(result.candidate_count, result.dropped_below_min_score)
            reason = (
                f"{result.candidate_count} candidate(s) retrieved, all "
                f"{result.dropped_below_min_score} reranked result(s) below min_score"
            )

        logger.info("No usable context for query %r (%s) -- skipping generation", query, reason)
        if on_event is not None:
            on_event(PipelineEvent(stage="no_context", message=f"Skipping generation: {reason}"))

        return ChatAnswer(
            answer=answer,
            citations=[],
            rewritten_query=rewritten_query,
            dropped_below_min_score=result.dropped_below_min_score,
            search_queries=result.search_queries,
            graded_out=corrected.graded_out,
            retry_queries=corrected.retry_queries,
            retrieval_attempts=corrected.attempts,
        )
