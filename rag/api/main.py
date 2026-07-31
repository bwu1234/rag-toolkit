"""FastAPI application entrypoint.

Run with `uvicorn rag.api.main:app --reload`. All component wiring happens
once at startup (via the lifespan handler) and is stashed on `app.state` --
routes are thin translators that pull the prebuilt `ChatService` off the
request rather than re-reading config or reconnecting to Ollama/Chroma per
request.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from rag.api.routes.chat import router as chat_router
from rag.config.settings import load_config
from rag.generation.builder import build_chat_service
from rag.logging_config import configure_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    config = load_config()
    logger.info("Building chat service (embedding=%s, vector_store=%s, reranker=%s, llm=%s)",
                config.embedding.provider, config.vector_store.provider, config.reranker.provider, config.llm.provider)
    app.state.chat_service = build_chat_service(config)
    yield


app = FastAPI(
    title="RAG Chat API",
    description="Retrieval-augmented chat over a local document corpus.",
    version="0.1.0",
    lifespan=lifespan,
)
app.include_router(chat_router)


@app.get("/", summary="Redirect to docs")
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/health", summary="Liveness check")
def health() -> dict[str, str]:
    return {"status": "ok"}
