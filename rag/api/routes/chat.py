"""`POST /chat` route: the chat API's single endpoint.

Translates between the wire contract (`ChatRequest`/`ChatResponse`) and the
pipeline's internal types (`ChatService`/`ChatAnswer`/`Citation`), and nothing
else -- all retrieval/generation orchestration lives in `ChatService` so it
stays reusable (e.g. by the eval pipeline) without a running HTTP server.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Request

from rag.api.schemas import ChatRequest, ChatResponse, ChatTurnModel, CitationModel
from rag.generation.chat_service import ChatAnswer, ChatService, Citation
from rag.generation.query_rewriter import ChatTurn

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


def get_chat_service(request: Request) -> ChatService:
    """Fetch the `ChatService` built once at startup (see `rag.api.main`'s lifespan).

    A `Depends`-based dependency rather than a direct `request.app.state` read
    inside the handler -- this is what lets tests swap in a fake `ChatService`
    via `app.dependency_overrides[get_chat_service]` without needing a real
    Ollama daemon, Chroma index, or even a running lifespan.
    """

    return request.app.state.chat_service


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
    )


@router.post("/chat", response_model=ChatResponse, summary="Ask a question of the indexed corpus")
def chat(payload: ChatRequest, chat_service: ChatService = Depends(get_chat_service)) -> ChatResponse:
    logger.info("Received chat query: %r (%d prior turn(s))", payload.query, len(payload.history))
    answer = chat_service.ask(payload.query, history=[_to_chat_turn(turn) for turn in payload.history])
    return _to_response(answer)
