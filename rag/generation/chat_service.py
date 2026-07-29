"""Orchestrates retrieve -> prompt -> generate -> cite for the chat API.

`ChatService` is the seam between the retrieval pipeline (Milestone 5) and
generation: it's the one place that knows how to turn a `Retriever`'s
`ScoredChunk`s into both a grounded prompt and a citation list the API can
return alongside the model's answer. Keeping this orchestration out of the
API route layer means the eval pipeline (Milestone 7) can reuse it directly
without spinning up FastAPI.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from rag.events import EventSink, PipelineEvent, emit
from rag.generation.llm import LLMClient
from rag.generation.prompts import SYSTEM_PROMPT, build_rag_prompt
from rag.generation.query_rewriter import ChatTurn, QueryCondenser
from rag.retrieval.retriever import RetrievalResult, Retriever
from rag.vectorstore.base import ScoredChunk

logger = logging.getLogger(__name__)

# Three distinct ways a turn can end without a generated answer. They were one
# string until the `retrieval.min_score` floor made the distinction real: with
# a floor in place, "nothing came back" and "things came back and none of them
# were relevant" are different facts about the corpus, and collapsing them
# leaves a user unable to tell "I need to run the indexer" from "this corpus
# doesn't cover my question".
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

    The three trailing fields exist so every caller -- not just the one UI that
    happens to pass an `on_event` sink -- can see the places this pipeline
    silently changes what the user asked for or what it answered from: the
    question was rewritten for a follow-up, it was expanded into several
    searches, or passages were withheld. None of that is visible in the answer
    text, and all of it explains results a user might otherwise find baffling.
    """

    answer: str
    citations: list[Citation] = field(default_factory=list)
    rewritten_query: str | None = None
    """The standalone question retrieval actually used, when condensing changed it."""
    dropped_below_min_score: int = 0
    """Reranked passages withheld by the `retrieval.min_score` floor."""
    search_queries: list[str] = field(default_factory=list)
    """Every query searched, when HyDE / multi-query expansion generated more than one."""


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
    """

    def __init__(
        self,
        retriever: Retriever,
        llm_client: LLMClient,
        condenser: QueryCondenser | None = None,
    ) -> None:
        self._retriever = retriever
        self._llm_client = llm_client
        self._condenser = condenser

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

        The returned `ChatAnswer` also reports the two things the pipeline did
        that the answer text can't show: the rewritten query (if condensing
        changed it) and how many passages the relevance floor withheld.

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
        """

        if not query.strip():
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

        result = self._retriever.retrieve(search_query, on_event=on_event)
        chunks = result.chunks
        if not chunks:
            return self._no_context_answer(query, result, rewritten_query, on_event)

        start = time.monotonic()
        prompt = build_rag_prompt(search_query, chunks)
        emit(on_event, start, "prompt", f"Built prompt from {len(chunks)} passage(s)")

        start = time.monotonic()
        answer = self._llm_client.generate(prompt, system=SYSTEM_PROMPT)
        emit(on_event, start, "generate", "Generated answer")

        citations = [_to_citation(chunk) for chunk in chunks]

        logger.info(
            "Answered query %r with %d citation(s) (%d withheld below min_score)",
            query,
            len(citations),
            result.dropped_below_min_score,
        )
        return ChatAnswer(
            answer=answer,
            citations=citations,
            rewritten_query=rewritten_query,
            dropped_below_min_score=result.dropped_below_min_score,
            search_queries=result.search_queries,
        )

    def _no_context_answer(
        self,
        query: str,
        result: RetrievalResult,
        rewritten_query: str | None,
        on_event: EventSink | None,
    ) -> ChatAnswer:
        """Explain *which* kind of nothing retrieval came back with, and skip generation."""

        if result.candidate_count == 0:
            answer = _EMPTY_INDEX_ANSWER
            reason = "index returned no candidates at all"
        else:
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
        )
