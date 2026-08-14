"""SDK-backed MCP server, serving both stdio and streamable HTTP.

Imports the official `mcp` package, so it is only reachable when the optional
`mcp` extra is installed (`pip install -e '.[mcp]'`). `rag.mcp.__main__` falls
back to `rag.mcp.fallback` when it is not; nothing else imports this module at
package import time, so the fallback path never pays for a missing dependency.

Protocol revision 2026-07-28 only. The SDK would otherwise also serve the
handshake era off the same server object; `ProtocolVersionGate` turns that off
so both transports here honour exactly one contract. Requires `mcp >= 2.0`,
which is where 2026-07-28 support (and the `MCPServer` API used below) landed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from mcp.server import MCPServer
from mcp.server.context import CallNext, HandlerResult, ServerRequestContext
from mcp.server.transport_security import TransportSecuritySettings
from mcp.shared.exceptions import MCPError
from mcp.types import UNSUPPORTED_PROTOCOL_VERSION, ToolAnnotations
from starlette.applications import Starlette

from rag.mcp.tools import (
    INSTRUCTIONS,
    PROTOCOL_VERSION,
    SERVER_NAME,
    SERVER_VERSION,
    RagTools,
    build_tool_specs,
)

logger = logging.getLogger(__name__)

#: Streamable HTTP applies DNS-rebinding protection, which rejects any Host
#: header not on this list with a 421. These cover the loopback addresses a
#: local agent actually connects to; add more when serving elsewhere.
DEFAULT_ALLOWED_HOSTS = [
    "localhost",
    "127.0.0.1",
    "localhost:8000",
    "127.0.0.1:8000",
]


class ProtocolVersionGate:
    """Refuse every request that is not on `PROTOCOL_VERSION`.

    The SDK serves both protocol eras off one server object: a request
    carrying the 2026-07-28 `_meta` envelope opens a modern connection, and
    anything else -- notably an `initialize` handshake -- opens a legacy one
    and negotiates down to a 2025 revision. This project serves 2026-07-28 and
    nothing else, so the older era is turned off here rather than left as a
    quiet second contract that the fallback transport does not honour.

    Middleware is where the check belongs because it sees every inbound
    request on both transports (stdio and streamable HTTP), before validation
    or handshake commit, with the connection's negotiated version on
    `ctx.protocol_version`. `-32022` names what is served, so an
    auto-negotiating client learns the version instead of guessing from a
    closed connection.
    """

    async def __call__(self, ctx: ServerRequestContext[Any, Any], call_next: CallNext) -> HandlerResult:
        # `request_id is None` means a notification: it draws no response, so
        # refusing it would only log noise. The request that matters on a
        # legacy connection (`initialize`) is refused before it can commit.
        if ctx.request_id is not None and ctx.protocol_version != PROTOCOL_VERSION:
            raise MCPError(
                code=UNSUPPORTED_PROTOCOL_VERSION,
                message=f"this server speaks MCP {PROTOCOL_VERSION} only",
                data={"supported": [PROTOCOL_VERSION], "requested": ctx.protocol_version},
            )
        return await call_next(ctx)


def build_mcp_server(
    tools: RagTools | None = None, *, config_path: str | Path | None = None
) -> MCPServer:
    """Construct the MCP server with this project's tools registered.

    Registration hands the SDK the annotated handler and lets it derive the
    input schema, which is the same derivation `rag.mcp.tools.input_schema_for`
    performs for the fallback transport.
    """

    tools = tools if tools is not None else RagTools(config_path)
    server = MCPServer(
        name=SERVER_NAME,
        version=SERVER_VERSION,
        instructions=INSTRUCTIONS,
        middleware=[ProtocolVersionGate()],
    )
    for spec in build_tool_specs(tools):
        server.add_tool(
            spec.handler,
            name=spec.name,
            description=spec.description,
            # Both tools only read. Saying so lets a client skip the
            # confirmation prompts it would otherwise apply to tool calls.
            annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
        )
    return server


def mcp_http_app(
    server: MCPServer | None = None,
    *,
    path: str = "/mcp",
    allowed_hosts: list[str] | None = None,
) -> Starlette:
    """Return the streamable-HTTP ASGI app for mounting on the FastAPI app.

    The returned app carries a lifespan that starts the session manager; a host
    that mounts it **must** chain that lifespan in, or every request fails with
    "Task group is not initialized". See `rag.api.main`.
    """

    server = server if server is not None else build_mcp_server()
    return server.streamable_http_app(
        streamable_http_path=path,
        transport_security=TransportSecuritySettings(
            allowed_hosts=allowed_hosts if allowed_hosts is not None else DEFAULT_ALLOWED_HOSTS,
            allowed_origins=allowed_hosts if allowed_hosts is not None else DEFAULT_ALLOWED_HOSTS,
        ),
    )


def run_stdio(config_path: str | Path | None = None) -> None:
    """Serve MCP over stdio until the client disconnects.

    The SDK's stdio transport points fd 1 at stderr while serving, so log lines
    and any library chatter from Chroma or torch land on stderr instead of
    corrupting the JSON-RPC stream on stdout.
    """

    build_mcp_server(config_path=config_path).run(transport="stdio")
