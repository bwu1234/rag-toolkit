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
from rag.chat import build_chat_service
from rag.logging_config import configure_logging
from rag.observability.factory import get_turn_sink

logger = logging.getLogger(__name__)

#: The one MCP endpoint URL, exactly as documented: no trailing slash.
MCP_PATH = "/mcp"


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
    return mcp_http_app(path=MCP_PATH)


_mcp_app = _build_mcp_app()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    config = load_config()
    logger.info("Building chat service (embedding=%s, vector_store=%s, reranker=%s, llm=%s)",
                config.embedding.provider, config.vector_store.provider, config.reranker.provider, config.llm.provider)
    app.state.turn_sink = get_turn_sink(config.observability.turn_log)
    app.state.chat_service = build_chat_service(config, turn_sink=app.state.turn_sink)

    if _mcp_app is None:
        yield
        return

    # Mounting a Starlette sub-app does NOT run its lifespan -- ASGI delivers
    # lifespan events to the outermost app only. The MCP session manager starts
    # its task group in that lifespan, so without this chaining every /mcp
    # request fails with "Task group is not initialized".
    async with _mcp_app.router.lifespan_context(app):
        logger.info("MCP streamable HTTP endpoint served at %s", MCP_PATH)
        yield


app = FastAPI(
    title="RAG Chat API",
    description="Retrieval-augmented chat over a local document corpus.",
    version="0.1.0",
    lifespan=lifespan,
)
app.include_router(chat_router)
if _mcp_app is not None:
    # The sub-app's route is added to this app's router, not mounted. A Mount
    # at "/mcp" serves its inner "/" at "/mcp/" only, so a request to the
    # documented `/mcp` drew a 307 to `/mcp/` -- which a client that won't
    # re-POST across a redirect never follows. A root Mount is no better: it
    # matches every path and would swallow /health and /docs.
    #
    # Taking the routes alone drops the sub-app's middleware, so refuse to
    # start if it ever has any (the SDK adds auth middleware when auth is
    # configured) rather than silently serving MCP without it.
    if _mcp_app.user_middleware:
        raise RuntimeError("the MCP app now carries middleware; mount it instead of copying its routes")
    app.router.routes.extend(_mcp_app.routes)


@app.get("/", summary="Redirect to docs")
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/health", summary="Liveness check")
def health() -> dict[str, str]:
    return {"status": "ok"}
