"""MCP (Model Context Protocol) server exposing this RAG pipeline as agent tools.

Two transports, one tool definition:

- **stdio** -- `python -m rag.mcp`, for an agent that spawns the server as a
  subprocess. Uses the official `mcp` SDK when installed and falls back to a
  dependency-free JSON-RPC loop (`rag.mcp.fallback`) when it isn't.
- **streamable HTTP** -- mounted at `/mcp` on the existing FastAPI app
  (`rag.api.main`), for an agent that talks to one already-warm process.

Both read their tools from `rag.mcp.tools.build_tool_specs`, so the two
transports can never drift apart on names, schemas, or behavior.
"""

from __future__ import annotations

from rag.mcp.tools import RagTools, ToolSpec, build_tool_specs

__all__ = ["RagTools", "ToolSpec", "build_tool_specs"]
