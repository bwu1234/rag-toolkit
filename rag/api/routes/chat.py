"""`POST /chat` route: the chat API's single endpoint.

Translates between the wire contract (`ChatRequest`/`ChatResponse`) and the
pipeline's internal types (`ChatService`/`ChatAnswer`/`Citation`), and nothing
else -- all retrieval/generation orchestration lives in `ChatService` so it
stays reusable (e.g. by the eval pipeline) without a running HTTP server.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status

from rag.api.schemas import (
    ChatRequest,
    ChatResponse,
    ChatTurnModel,
    CitationModel,
    FeedbackRequest,
    FeedbackResponse,
)
from rag.generation.chat_service import ChatAnswer, ChatResponder, Citation
from rag.generation.query_rewriter import ChatTurn
from rag.observability.records import FeedbackRecord
from rag.observability.sink import TurnSink
from rag.query_filter import UnfilterableField

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


def get_chat_service(request: Request) -> ChatResponder:
    """Fetch the `ChatService` built once at startup (see `rag.api.main`'s lifespan).

    A `Depends`-based dependency rather than a direct `request.app.state` read
    inside the handler -- this is what lets tests swap in a fake `ChatService`
    via `app.dependency_overrides[get_chat_service]` without needing a real
    Ollama daemon, Chroma index, or even a running lifespan.
    """

    return request.app.state.chat_service


def get_turn_sink(request: Request) -> TurnSink | None:
    """Fetch the `TurnSink` built at startup, or `None` when turn logging is off."""

    return getattr(request.app.state, "turn_sink", None)


def _to_chat_turn(turn: ChatTurnModel) -> ChatTurn:
    return ChatTurn(role=turn.role, content=turn.content)


def _to_citation_model(citation: Citation) -> CitationModel:
    return CitationModel(
        chunk_id=citation.chunk_id,
        document_id=citation.document_id,
        text=citation.text,
        score=citation.score,
        page=citation.page,
    )


def _to_response(answer: ChatAnswer) -> ChatResponse:
    return ChatResponse(
        answer=answer.answer,
        citations=[_to_citation_model(c) for c in answer.citations],
        rewritten_query=answer.rewritten_query,
        dropped_below_min_score=answer.dropped_below_min_score,
        search_queries=answer.search_queries,
        graded_out=answer.graded_out,
        retry_queries=answer.retry_queries,
        retrieval_attempts=answer.retrieval_attempts,
        grounded=answer.grounded,
        cited_chunk_ids=answer.cited_chunk_ids,
        invalid_citations=answer.invalid_citations,
        generation_failure=answer.generation_failure,
        stopped_reason=answer.stopped_reason,
        turn_id=answer.turn_id,
        stage_ms=answer.stage_ms,
        total_ms=answer.total_ms,
        llm_calls=answer.llm_calls,
        prompt_tokens=answer.prompt_tokens,
        completion_tokens=answer.completion_tokens,
    )


@router.post("/chat", response_model=ChatResponse, summary="Ask a question of the indexed corpus")
def chat(payload: ChatRequest, chat_service: ChatResponder = Depends(get_chat_service)) -> ChatResponse:
    logger.info("Received chat query: %r (%d prior turn(s))", payload.query, len(payload.history))
    try:
        answer = chat_service.ask(
            payload.query,
            history=[_to_chat_turn(turn) for turn in payload.history],
            query_filter=payload.filters,
        )
    except UnfilterableField as exc:
        # The request named a field chunks don't store: the caller's error,
        # and one they can fix, so 400 with the message rather than a 500.
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return _to_response(answer)


@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Rate an earlier answer thumbs up or down",
)
def feedback(payload: FeedbackRequest, sink: TurnSink | None = Depends(get_turn_sink)) -> FeedbackResponse:
    # 503 rather than a silent 201: with logging off there is no turn record to
    # attach the rating to, and accepting it would lose it without saying so.
    if sink is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Turn logging is disabled (observability.turn_log.provider: none); feedback is not stored.",
        )
    # The turn id isn't checked against the log: that would mean scanning the
    # whole file per click. A rating for an unknown id simply never joins.
    record = FeedbackRecord(turn_id=payload.turn_id, rating=payload.rating, comment=payload.comment)
    sink.record_feedback(record)
    logger.info("Recorded %s feedback for turn %s", payload.rating, payload.turn_id)
    return FeedbackResponse(feedback_id=record.feedback_id)
