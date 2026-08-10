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
from starlette.applications import Starlette

from rag.api.routes.chat import router as chat_router
from rag.config.settings import load_config
from rag.generation.builder import build_chat_service
from rag.logging_config import configure_logging

logger = logging.getLogger(__name__)


def _build_mcp_app() -> Starlette | None:
    """Build the MCP streamable-HTTP sub-app, or None if the SDK isn't installed.

    The `mcp` extra is optional, so a checkout that installed only the core
    dependencies still serves `/chat` -- it just doesn't offer `/mcp`.
    """

    try:
        from rag.mcp.server import mcp_http_app
    except ImportError:
        logger.info("`mcp` SDK not installed; /mcp endpoint disabled (pip install -e '.[mcp]')")
        return None
    # Internal path "/" because the sub-app is mounted *under* "/mcp" below --
    # the prefix comes from the mount, so asking for "/mcp" here too would
    # serve the endpoint at "/mcp/mcp".
    return mcp_http_app(path="/")


_mcp_app = _build_mcp_app()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    config = load_config()
    logger.info("Building chat service (embedding=%s, vector_store=%s, reranker=%s, llm=%s)",
                config.embedding.provider, config.vector_store.provider, config.reranker.provider, config.llm.provider)
    app.state.chat_service = build_chat_service(config)

    if _mcp_app is None:
        yield
        return

    # Mounting a Starlette sub-app does NOT run its lifespan -- ASGI delivers
    # lifespan events to the outermost app only. The MCP session manager starts
    # its task group in that lifespan, so without this chaining every /mcp
    # request fails with "Task group is not initialized".
    async with _mcp_app.router.lifespan_context(app):
        logger.info("MCP streamable HTTP endpoint mounted at /mcp")
        yield


app = FastAPI(
    title="RAG Chat API",
    description="Retrieval-augmented chat over a local document corpus.",
    version="0.1.0",
    lifespan=lifespan,
)
app.include_router(chat_router)
if _mcp_app is not None:
    # Mounted under its own prefix rather than at "/": a Mount at the root
    # matches every path and, being registered before the routes declared
    # below, would swallow /health and /docs.
    app.mount("/mcp", _mcp_app)


@app.get("/", summary="Redirect to docs")
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/health", summary="Liveness check")
def health() -> dict[str, str]:
    return {"status": "ok"}
