"""SDK-backed MCP server, serving both stdio and streamable HTTP.

Imports the official `mcp` package, so it is only reachable when the optional
`mcp` extra is installed (`pip install -e '.[mcp]'`). `rag.mcp.__main__` falls
back to `rag.mcp.fallback` when it is not; nothing else imports this module at
package import time, so the fallback path never pays for a missing dependency.
"""

from __future__ import annotations

import logging
from pathlib import Path

from mcp.server import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations
from starlette.applications import Starlette

from rag.mcp.tools import (
    INSTRUCTIONS,
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
