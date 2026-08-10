"""Dependency-free MCP stdio server, used when the `mcp` SDK is absent.

MCP over stdio is newline-delimited JSON-RPC 2.0, and the read-only slice this
project needs is four methods wide -- so `python -m rag.mcp` stays runnable in
a checkout that only installed the core dependencies, rather than failing at
import with a stack trace about a missing extra.

This is a deliberate subset, not a reimplementation of the SDK. It serves one
request at a time and supports no resources, prompts, sampling, progress, or
cancellation. Tool names, schemas, and results are identical to the SDK path
because both are generated from the same `ToolSpec`s. Install the extra
(`pip install -e '.[mcp]'`) to get the real transport.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, BinaryIO

from rag.mcp.tools import (
    INSTRUCTIONS,
    SERVER_NAME,
    SERVER_VERSION,
    RagTools,
    ToolSpec,
    build_tool_specs,
)

logger = logging.getLogger(__name__)

#: Advertised when the client asks for something we don't recognize. Clients
#: are expected to accept a different version in the response or disconnect.
LATEST_PROTOCOL_VERSION = "2025-06-18"
SUPPORTED_PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")

# JSON-RPC reserved codes.
_METHOD_NOT_FOUND = -32601
_INVALID_PARAMS = -32602
_INTERNAL_ERROR = -32603


def _divert_stdout() -> BinaryIO:
    """Hand back the real stdout and point fd 1 at stderr.

    Everything on this process's stdout is protocol. Chroma, torch, and our own
    `configure_logging` all write to stdout given the chance, and a single
    stray line there is a parse error that kills the session -- so the wire
    gets a private duplicate of the descriptor and every other writer, at both
    the Python and C level, is redirected to stderr. The SDK's `stdio_server`
    does the same thing; this mirrors it so both paths behave alike.
    """

    protocol_fd = os.dup(1)
    os.dup2(2, 1)
    return os.fdopen(protocol_fd, "wb", buffering=0)


class FallbackServer:
    """Minimal JSON-RPC 2.0 dispatcher over a set of `ToolSpec`s."""

    def __init__(self, specs: list[ToolSpec], instructions: str = "") -> None:
        self._specs = {spec.name: spec for spec in specs}
        self._instructions = instructions

    def handle(self, request: dict[str, Any]) -> dict[str, Any] | None:
        """Dispatch one request; returns None for notifications."""

        method = request.get("method")
        request_id = request.get("id")
        # Notifications carry no id and must never draw a response.
        is_notification = "id" not in request

        try:
            if method == "initialize":
                result = self._initialize(request.get("params") or {})
            elif method == "tools/list":
                result = {"tools": [self._describe(spec) for spec in self._specs.values()]}
            elif method == "tools/call":
                result = self._call(request.get("params") or {})
            elif method == "ping":
                result = {}
            elif is_notification:
                # `notifications/initialized` and friends: nothing to do, and
                # nothing to say back.
                return None
            else:
                return _error(request_id, _METHOD_NOT_FOUND, f"Unknown method: {method!r}")
        except _RpcError as exc:
            return _error(request_id, exc.code, str(exc))
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Unhandled error in %s", method)
            return _error(request_id, _INTERNAL_ERROR, f"{type(exc).__name__}: {exc}")

        if is_notification:
            return None
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    def _initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        requested = params.get("protocolVersion")
        version = requested if requested in SUPPORTED_PROTOCOL_VERSIONS else LATEST_PROTOCOL_VERSION
        return {
            "protocolVersion": version,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            "instructions": self._instructions,
        }

    def _describe(self, spec: ToolSpec) -> dict[str, Any]:
        return {
            "name": spec.name,
            "description": spec.description,
            "inputSchema": spec.input_schema,
            "annotations": {"readOnlyHint": True, "openWorldHint": False},
        }

    def _call(self, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        spec = self._specs.get(name or "")
        if spec is None:
            raise _RpcError(_INVALID_PARAMS, f"Unknown tool: {name!r}")

        arguments = params.get("arguments") or {}
        try:
            payload = spec.handler(**arguments)
        except TypeError as exc:
            # Bad argument names/arity is a caller error, so report it as a
            # tool error the agent can correct rather than a protocol failure.
            return _tool_error(f"Invalid arguments for {name}: {exc}")
        except Exception as exc:
            logger.warning("Tool %s failed: %s", name, exc, exc_info=True)
            return _tool_error(f"{type(exc).__name__}: {exc}")

        return {
            "content": [{"type": "text", "text": json.dumps(payload, indent=2, default=str)}],
            "isError": False,
        }


class _RpcError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code


def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def _tool_error(message: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": message}], "isError": True}


def serve(server: FallbackServer, stdin: Any = None, stdout: BinaryIO | None = None) -> None:
    """Read newline-delimited JSON-RPC from `stdin`, write responses to `stdout`.

    Streams are injectable so tests can drive a full session without touching
    the process's real descriptors.
    """

    source = stdin if stdin is not None else sys.stdin
    sink = stdout if stdout is not None else _divert_stdout()

    for line in source:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("Discarding unparseable line (%d bytes)", len(line))
            continue

        response = server.handle(request)
        if response is None:
            continue
        sink.write((json.dumps(response) + "\n").encode("utf-8"))
        sink.flush()


def run_stdio(config_path: str | Path | None = None) -> None:
    """Serve MCP over stdio with this project's tools, without the SDK."""

    specs = build_tool_specs(RagTools(config_path))
    serve(FallbackServer(specs, instructions=INSTRUCTIONS))
