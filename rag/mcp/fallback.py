"""Dependency-free MCP stdio server, used when the `mcp` SDK is absent.

MCP over stdio is newline-delimited JSON-RPC 2.0, and the read-only slice this
project needs is three methods wide -- so `python -m rag.mcp` stays runnable in
a checkout that only installed the core dependencies, rather than failing at
import with a stack trace about a missing extra.

Protocol revision 2026-07-28 only, matching the SDK path. That revision
deleted the `initialize` handshake: there is no session state, and every
request is self-contained, carrying the protocol version and the client's
capabilities in `params._meta` (`io.modelcontextprotocol/protocolVersion` and
`io.modelcontextprotocol/clientCapabilities`, both required). Servers advertise
themselves through `server/discover` instead, which a client MAY call and MAY
skip. `ping` and `notifications/initialized` are gone with the handshake.

This is a deliberate subset, not a reimplementation of the SDK. It serves one
request at a time and supports no resources, prompts, sampling, subscriptions,
progress, or cancellation. Tool names, schemas, and results are identical to
the SDK path because both are generated from the same `ToolSpec`s. Install the
extra (`pip install -e '.[mcp]'`) to get the real transport.
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
    PROTOCOL_VERSION,
    SERVER_NAME,
    SERVER_VERSION,
    RagTools,
    ToolSpec,
    build_tool_specs,
)

logger = logging.getLogger(__name__)

#: The per-request envelope keys 2026-07-28 puts in `params._meta`. The
#: `io.modelcontextprotocol/` prefix is reserved by the spec, so no other
#: traffic mints them.
_PROTOCOL_VERSION_KEY = "io.modelcontextprotocol/protocolVersion"
_CLIENT_CAPABILITIES_KEY = "io.modelcontextprotocol/clientCapabilities"
#: Stamped on every result: the spec asks servers to identify themselves on
#: each response now that there is no handshake to do it once.
_SERVER_INFO_KEY = "io.modelcontextprotocol/serverInfo"

# JSON-RPC reserved codes, plus the MCP-specific ones this server can raise.
_METHOD_NOT_FOUND = -32601
_INVALID_PARAMS = -32602
_INTERNAL_ERROR = -32603
_UNSUPPORTED_PROTOCOL_VERSION = -32022

#: Freshness hints on cacheable results (`server/discover`, `tools/list`).
#: Both are the spec's conservative reading: immediately stale, never shared
#: across authorization contexts. The tool list is cheap to recompute and the
#: corpus registry can change under a long-lived client, so nothing here is
#: worth serving from a cache.
_TTL_MS = 0
_CACHE_SCOPE = "private"


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
            if is_notification:
                # `notifications/cancelled` is the only one a 2026-07-28 client
                # sends us, and single-request-at-a-time serving means it can
                # only ever arrive after the work it would cancel is done.
                return None
            if method == "initialize":
                # Removed in 2026-07-28. Naming what is served beats a bare
                # METHOD_NOT_FOUND: a client that auto-negotiates can read the
                # supported list and retry with the envelope.
                raise _RpcError(
                    _UNSUPPORTED_PROTOCOL_VERSION,
                    "this server speaks the 2026-07-28 protocol, which has no "
                    "`initialize` handshake; send self-contained requests carrying "
                    "the per-request `_meta` envelope instead",
                    data=self._unsupported_data((request.get("params") or {}).get("protocolVersion")),
                )

            self._check_envelope(request.get("params"))

            if method == "server/discover":
                result = self._discover()
            elif method == "tools/list":
                result = {
                    "tools": [self._describe(spec) for spec in self._specs.values()],
                    "ttlMs": _TTL_MS,
                    "cacheScope": _CACHE_SCOPE,
                }
            elif method == "tools/call":
                result = self._call(request.get("params") or {})
            else:
                return _error(request_id, _METHOD_NOT_FOUND, f"Unknown method: {method!r}")
        except _RpcError as exc:
            return _error(request_id, exc.code, str(exc), data=exc.data)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Unhandled error in %s", method)
            return _error(request_id, _INTERNAL_ERROR, f"{type(exc).__name__}: {exc}")

        return {"jsonrpc": "2.0", "id": request_id, "result": self._finish(result)}

    def _check_envelope(self, params: Any) -> None:
        """Run the 2026-07-28 inbound ladder over a request's params.

        Two rungs, first failure wins: the required envelope keys are present
        in a `_meta` object, and the version they carry is the one served.
        Shape defects are INVALID_PARAMS; a well-formed request offering
        another revision is -32022, the one code an auto-negotiating client is
        required *not* to fall back from.
        """

        meta = params.get("_meta") if isinstance(params, dict) else None
        if not isinstance(meta, dict):
            raise _RpcError(
                _INVALID_PARAMS,
                f"params._meta must be an object carrying the required {_PROTOCOL_VERSION_KEY!r} "
                f"and {_CLIENT_CAPABILITIES_KEY!r} envelope keys",
            )
        missing = [key for key in (_PROTOCOL_VERSION_KEY, _CLIENT_CAPABILITIES_KEY) if key not in meta]
        if missing:
            raise _RpcError(
                _INVALID_PARAMS,
                f"params._meta is missing the required envelope key(s): {', '.join(missing)}",
            )

        version = meta[_PROTOCOL_VERSION_KEY]
        if not isinstance(version, str):
            # A non-string is a malformed envelope, not a negotiation outcome,
            # so it must not come back as -32022.
            raise _RpcError(_INVALID_PARAMS, "the protocol-version envelope value must be a string")
        if version != PROTOCOL_VERSION:
            raise _RpcError(
                _UNSUPPORTED_PROTOCOL_VERSION,
                "Unsupported protocol version",
                data=self._unsupported_data(version),
            )

    @staticmethod
    def _unsupported_data(requested: Any) -> dict[str, Any]:
        data: dict[str, Any] = {"supported": [PROTOCOL_VERSION]}
        if isinstance(requested, str):
            data["requested"] = requested
        return data

    def _discover(self) -> dict[str, Any]:
        """Answer `server/discover`: what this server is and what it serves."""

        return {
            "supportedVersions": [PROTOCOL_VERSION],
            "capabilities": {"tools": {"listChanged": False}},
            "instructions": self._instructions,
            "ttlMs": _TTL_MS,
            "cacheScope": _CACHE_SCOPE,
        }

    def _finish(self, result: dict[str, Any]) -> dict[str, Any]:
        """Apply what 2026-07-28 requires of every result.

        `resultType` is mandatory on this revision (there is no absent-means-
        complete bridge for a server that speaks it), and `serverInfo` replaces
        the identity the deleted handshake used to carry.
        """

        result.setdefault("resultType", "complete")
        meta = dict(result.get("_meta") or {})
        meta.setdefault(_SERVER_INFO_KEY, {"name": SERVER_NAME, "version": SERVER_VERSION})
        result["_meta"] = meta
        return result

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
    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.data = data


def _error(request_id: Any, code: int, message: str, data: Any = None) -> dict[str, Any]:
    error: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {"jsonrpc": "2.0", "id": request_id, "error": error}


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
