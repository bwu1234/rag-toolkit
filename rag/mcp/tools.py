"""MCP server identity, plus the shared tool surface re-exported for the transports.

The tools themselves live in `rag.tools`, where the in-process agent reads them
too. What stays here is what only an MCP client sees: the server's name and
version, the protocol revision, and the instructions string. Like `rag.tools`,
this module must not import the `mcp` SDK, because the fallback transport
imports it.
"""

from __future__ import annotations

from rag.tools import (
    DEFAULT_MAX_CHARS,
    MAX_RESULTS,
    RagTools,
    ToolSpec,
    build_tool_specs,
    input_schema_for,
)

__all__ = [
    "DEFAULT_MAX_CHARS",
    "INSTRUCTIONS",
    "MAX_RESULTS",
    "PROTOCOL_VERSION",
    "SERVER_NAME",
    "SERVER_VERSION",
    "RagTools",
    "ToolSpec",
    "build_tool_specs",
    "input_schema_for",
]

#: Server identity, shared by both transports so a client sees the same server
#: whichever way it connected. Lives here rather than in `rag.mcp.server`
#: because the fallback must not import anything that pulls in the `mcp` SDK.
SERVER_NAME = "rag-toolkit"
SERVER_VERSION = "0.1.0"

#: The one MCP revision this server speaks, on both transports. 2026-07-28 has
#: no `initialize` handshake: every request is self-contained and carries the
#: protocol version and the client's capabilities in `params._meta`. Older
#: revisions are refused rather than negotiated down -- there is no wire this
#: server serves them on, so a client that offers one is told what is served
#: (`-32022` with `supported: ["2026-07-28"]`) instead of half-working.
PROTOCOL_VERSION = "2026-07-28"

INSTRUCTIONS = """\
Retrieval over a local document corpus. Call rag_list_corpora to see what is \
searchable, then rag_search to pull ranked passages with their source paths. \
Search results are raw passages, not answers -- read them and cite the \
`document_id` of whatever you use."""
