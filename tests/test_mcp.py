"""Tests for the MCP server.

Hermetic: no Ollama, no Chroma, no model weights. `build_retriever` is patched
out with a fake, which is enough because everything under test lives *above*
retrieval -- payload shaping, corpus resolution, and the JSON-RPC framing.

The parity tests are the point of the file. Two transports advertise the same
tools from the same `ToolSpec`s, and the way that breaks is silent drift, so
the SDK's derived schema is asserted equal to the fallback's rather than each
being checked against a hand-written expectation that could rot with them.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from rag.mcp.fallback import FallbackServer, serve
from rag.mcp.tools import MAX_RESULTS, PROTOCOL_VERSION, RagTools, build_tool_specs
from rag.retrieval.retriever import RetrievalResult
from rag.vectorstore.base import ScoredChunk

def _chunk(chunk_id: str = "doc.md::0", *, text: str = "hello world", score: float = 0.9) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk_id,
        text=text,
        document_id="doc.md",
        source=Path("data/corpora/baseline/documents/doc.md"),
        doc_type="markdown",
        score=score,
        metadata={"page": 3},
    )


class _FakeRetriever:
    """Stands in for a wired `Retriever`; records what it was asked."""

    def __init__(self, result: RetrievalResult) -> None:
        self.result = result
        self.queries: list[str] = []
        self.rerank_top_k = 5
        self.top_k = 20

    def retrieve(self, query: str, *, on_event: object = None) -> RetrievalResult:
        self.queries.append(query)
        return self.result


@pytest.fixture()
def fake_retriever(monkeypatch: pytest.MonkeyPatch) -> _FakeRetriever:
    """Patch `build_retriever` so `RagTools` wires up without any I/O."""

    retriever = _FakeRetriever(RetrievalResult(chunks=[_chunk()], candidate_count=7))

    def _build(config, llm_client=None, corpora=None):  # type: ignore[no-untyped-def]
        return retriever

    monkeypatch.setattr("rag.mcp.tools.build_retriever", _build)
    return retriever


@pytest.fixture()
def tools(fake_retriever: _FakeRetriever) -> RagTools:
    # A path that does not exist yields a default RagConfig, so the tests never
    # depend on whatever the checked-in config.yaml currently says.
    return RagTools("/nonexistent/config.yaml")


# --------------------------------------------------------------------------
# Tool surface
# --------------------------------------------------------------------------


def test_exposes_only_read_only_tools(tools: RagTools) -> None:
    names = [spec.name for spec in build_tool_specs(tools)]
    assert names == ["rag_search", "rag_list_corpora"]


def test_search_schema_documents_every_argument(tools: RagTools) -> None:
    spec = next(s for s in build_tool_specs(tools) if s.name == "rag_search")
    schema = spec.input_schema

    assert schema["required"] == ["query"]
    assert set(schema["properties"]) == {"query", "corpus", "top_k", "max_chars"}
    # Descriptions are what an agent reads to pick arguments; a schema that
    # loses them still validates but degrades tool use.
    for name, prop in schema["properties"].items():
        assert prop.get("description"), f"{name} has no description"


def test_search_schema_bounds_top_k(tools: RagTools) -> None:
    spec = next(s for s in build_tool_specs(tools) if s.name == "rag_search")
    top_k = spec.input_schema["properties"]["top_k"]
    bounds = json.dumps(top_k)
    assert '"maximum": 20' in bounds or f'"maximum":{MAX_RESULTS}' in bounds


# --------------------------------------------------------------------------
# search payload
# --------------------------------------------------------------------------


def test_search_returns_ranked_results_with_provenance(tools: RagTools) -> None:
    payload = tools.search("what is x?")

    assert payload["candidate_count"] == 7
    assert payload["returned"] == 1
    result = payload["results"][0]
    assert result["rank"] == 1
    assert result["chunk_id"] == "doc.md::0"
    assert result["document_id"] == "doc.md"
    assert result["page"] == 3
    # Paths are JSON-hostile and absolute ones leak the developer's home dir.
    assert result["source"] == "data/corpora/baseline/documents/doc.md"
    assert isinstance(result["source"], str)


def test_search_truncates_long_passages_and_says_so(
    tools: RagTools, fake_retriever: _FakeRetriever
) -> None:
    fake_retriever.result = RetrievalResult(
        chunks=[_chunk(text="x" * 5000)], candidate_count=1
    )

    payload = tools.search("q", max_chars=100)

    result = payload["results"][0]
    assert len(result["text"]) == 100
    assert result["truncated"] is True
    assert result["full_length"] == 5000


def test_search_accepts_a_single_corpus_name_or_a_list(tools: RagTools) -> None:
    # Agents reliably pass a bare string; the pooled case needs a list. Both
    # must work rather than one failing schema validation.
    assert tools.search("q", corpus="default")["corpora"] == ["default"]
    assert tools.search("q", corpus=["default"])["corpora"] == ["default"]


def test_search_rejects_out_of_range_top_k(tools: RagTools) -> None:
    # The SDK enforces this from the schema, but the fallback calls the handler
    # directly -- so the handler has to enforce it too.
    with pytest.raises(ValueError, match="top_k must be between"):
        tools.search("q", top_k=MAX_RESULTS + 1)
    with pytest.raises(ValueError, match="top_k must be between"):
        tools.search("q", top_k=0)


def test_search_rejects_empty_query(tools: RagTools) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        tools.search("   ")


def test_search_rejects_unknown_corpus_by_name(tools: RagTools) -> None:
    with pytest.raises(ValueError, match="Unknown corpus name"):
        tools.search("q", corpus="not-a-corpus")


def test_empty_index_and_empty_corpus_give_different_hints(
    tools: RagTools, fake_retriever: _FakeRetriever
) -> None:
    """The two kinds of "no results" need different actions from the agent."""

    fake_retriever.result = RetrievalResult(chunks=[], candidate_count=0)
    unbuilt = tools.search("q")["hint"]
    assert "python -m rag.cli index" in unbuilt

    fake_retriever.result = RetrievalResult(
        chunks=[], candidate_count=12, dropped_below_min_score=12
    )
    filtered = tools.search("q")["hint"]
    assert "min_score" in filtered
    assert "python -m rag.cli index" not in filtered


def test_retriever_is_built_once_and_reused(tools: RagTools, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}
    retriever = _FakeRetriever(RetrievalResult(chunks=[_chunk()], candidate_count=1))

    def _build(config, llm_client=None, corpora=None):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        return retriever

    monkeypatch.setattr("rag.mcp.tools.build_retriever", _build)

    tools.search("one")
    tools.search("two")

    # The cross-encoder holds hundreds of MB once loaded; rebuilding per call
    # would reload it every time.
    assert calls["n"] == 1


def test_search_widens_rerank_depth_so_top_k_can_be_served(
    tools: RagTools, fake_retriever: _FakeRetriever
) -> None:
    tools.search("q")
    assert fake_retriever.rerank_top_k == MAX_RESULTS


# --------------------------------------------------------------------------
# list_corpora
# --------------------------------------------------------------------------


def test_list_corpora_reports_registry_and_index_state(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        f"""
paths:
  index_dir: {tmp_path / "index"}
corpora:
  active: [alpha]
  registry:
    alpha:
      documents_dir: {tmp_path / "alpha"}
      description: The alpha corpus
    beta:
      documents_dir: {tmp_path / "beta"}
""",
        encoding="utf-8",
    )
    (tmp_path / "alpha").mkdir()

    payload = RagTools(config_file).list_corpora()

    assert payload["active"] == ["alpha"]
    by_name = {entry["name"]: entry for entry in payload["corpora"]}
    assert by_name["alpha"]["active"] is True
    assert by_name["alpha"]["description"] == "The alpha corpus"
    assert by_name["alpha"]["documents_present"] is True
    assert by_name["beta"]["active"] is False
    # beta's directory was never created.
    assert by_name["beta"]["documents_present"] is False
    # Nothing indexed, and listing must not have created anything either.
    assert by_name["alpha"]["indexed"] is False
    assert by_name["alpha"]["indexed_chunks"] == 0


def test_list_corpora_does_not_create_an_index(tmp_path: Path) -> None:
    """Listing is read-only.

    `get_vector_store` uses get_or_create, so probing collections to count them
    would litter the store with empty ones as a side effect of a list call.
    """

    index_dir = tmp_path / "index"
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        f"paths:\n  index_dir: {index_dir}\ncorpora:\n  active: [alpha]\n"
        f"  registry:\n    alpha:\n      documents_dir: {tmp_path / 'alpha'}\n",
        encoding="utf-8",
    )

    RagTools(config_file).list_corpora()

    assert not index_dir.exists()


# --------------------------------------------------------------------------
# Fallback JSON-RPC transport
# --------------------------------------------------------------------------


@pytest.fixture()
def rpc(tools: RagTools) -> FallbackServer:
    return FallbackServer(build_tool_specs(tools), instructions="hi")


_ENVELOPE = {
    "io.modelcontextprotocol/protocolVersion": PROTOCOL_VERSION,
    "io.modelcontextprotocol/clientCapabilities": {},
}


def _request(method: str, params: dict | None = None, request_id: int | None = 1) -> dict:
    """A well-formed 2026-07-28 request: the `_meta` envelope is not optional."""

    payload: dict = {"jsonrpc": "2.0", "method": method}
    if request_id is not None:
        payload["id"] = request_id
    params = dict(params or {})
    params["_meta"] = {**_ENVELOPE, **params.get("_meta", {})}
    payload["params"] = params
    return payload


def test_discover_advertises_the_one_served_version(rpc: FallbackServer) -> None:
    response = rpc.handle(_request("server/discover"))
    assert response is not None
    result = response["result"]
    assert result["supportedVersions"] == [PROTOCOL_VERSION]
    assert result["capabilities"]["tools"] == {"listChanged": False}
    assert result["instructions"] == "hi"


def test_every_result_carries_result_type_and_server_info(rpc: FallbackServer) -> None:
    """Both are MUSTs on 2026-07-28, which has no handshake to carry identity."""

    for method in ("server/discover", "tools/list"):
        result = rpc.handle(_request(method))["result"]  # type: ignore[index]
        assert result["resultType"] == "complete"
        info = result["_meta"]["io.modelcontextprotocol/serverInfo"]
        assert info == {"name": "rag-toolkit", "version": "0.1.0"}


def test_cacheable_results_carry_freshness_hints(rpc: FallbackServer) -> None:
    for method in ("server/discover", "tools/list"):
        result = rpc.handle(_request(method))["result"]  # type: ignore[index]
        assert result["ttlMs"] == 0
        assert result["cacheScope"] == "private"


def test_initialize_is_refused_with_the_version_that_is_served(rpc: FallbackServer) -> None:
    """The handshake is gone at 2026-07-28.

    -32022 rather than METHOD_NOT_FOUND: an auto-negotiating client reads the
    supported list off this error and can retry with the envelope.
    """

    response = rpc.handle(
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}}
    )
    assert response is not None
    assert response["error"]["code"] == -32022
    assert response["error"]["data"] == {"supported": [PROTOCOL_VERSION], "requested": "2025-06-18"}


def test_an_older_protocol_version_is_refused_not_negotiated_down(rpc: FallbackServer) -> None:
    request = _request("tools/list")
    request["params"]["_meta"]["io.modelcontextprotocol/protocolVersion"] = "2025-11-25"

    response = rpc.handle(request)
    assert response is not None
    assert response["error"]["code"] == -32022
    assert response["error"]["data"]["supported"] == [PROTOCOL_VERSION]


def test_a_missing_envelope_is_a_params_error(rpc: FallbackServer) -> None:
    """A shape defect, not a negotiation outcome -- so not -32022."""

    bare = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
    assert rpc.handle(bare)["error"]["code"] == -32602  # type: ignore[index]

    partial = _request("tools/list")
    del partial["params"]["_meta"]["io.modelcontextprotocol/clientCapabilities"]
    response = rpc.handle(partial)
    assert response is not None
    assert response["error"]["code"] == -32602
    assert "clientCapabilities" in response["error"]["message"]


def test_a_non_string_protocol_version_is_a_params_error(rpc: FallbackServer) -> None:
    request = _request("tools/list")
    request["params"]["_meta"]["io.modelcontextprotocol/protocolVersion"] = 20260728
    assert rpc.handle(request)["error"]["code"] == -32602  # type: ignore[index]


def test_tools_list_advertises_both_tools_as_read_only(rpc: FallbackServer) -> None:
    response = rpc.handle(_request("tools/list"))
    assert response is not None
    listed = response["result"]["tools"]
    assert [t["name"] for t in listed] == ["rag_search", "rag_list_corpora"]
    for tool in listed:
        assert tool["annotations"]["readOnlyHint"] is True
        assert tool["inputSchema"]["type"] == "object"


def test_tools_call_returns_the_payload_as_json_text(rpc: FallbackServer) -> None:
    response = rpc.handle(
        _request("tools/call", {"name": "rag_search", "arguments": {"query": "q"}})
    )
    assert response is not None
    assert response["result"]["isError"] is False
    payload = json.loads(response["result"]["content"][0]["text"])
    assert payload["results"][0]["chunk_id"] == "doc.md::0"


def test_tool_failures_come_back_as_tool_errors_not_protocol_errors(rpc: FallbackServer) -> None:
    """A bad argument is the agent's problem to fix, not a broken session."""

    response = rpc.handle(
        _request("tools/call", {"name": "rag_search", "arguments": {"query": "q", "top_k": 999}})
    )
    assert response is not None
    assert "error" not in response
    assert response["result"]["isError"] is True
    assert "top_k must be between" in response["result"]["content"][0]["text"]


def test_unknown_tool_is_reported(rpc: FallbackServer) -> None:
    response = rpc.handle(_request("tools/call", {"name": "nope", "arguments": {}}))
    assert response is not None
    assert response["error"]["code"] == -32602


def test_unknown_method_is_reported(rpc: FallbackServer) -> None:
    response = rpc.handle(_request("resources/list"))
    assert response is not None
    assert response["error"]["code"] == -32601


def test_notifications_never_draw_a_response(rpc: FallbackServer) -> None:
    # A response to a notification is a protocol violation -- an id-less
    # message must be answered with silence.
    assert rpc.handle(_request("notifications/cancelled", request_id=None)) is None


def test_serve_round_trips_a_session_and_skips_malformed_lines(rpc: FallbackServer) -> None:
    stdin = io.StringIO(
        "\n".join(
            [
                json.dumps(_request("server/discover")),
                "{ this is not json",
                "",
                json.dumps(_request("notifications/cancelled", request_id=None)),
                json.dumps(_request("tools/list", request_id=2)),
            ]
        )
        + "\n"
    )
    stdout = io.BytesIO()

    serve(rpc, stdin=stdin, stdout=stdout)

    responses = [json.loads(line) for line in stdout.getvalue().decode().splitlines() if line]
    # Two responses: the garbage line and the notification produce none.
    assert [r["id"] for r in responses] == [1, 2]


# --------------------------------------------------------------------------
# Parity between the two transports
# --------------------------------------------------------------------------


def test_sdk_and_fallback_publish_identical_schemas(tools: RagTools) -> None:
    """The SDK derives schemas itself; the fallback derives its own.

    Both start from the same annotated handler, and this is the assertion that
    keeps them from drifting apart as arguments are added.
    """

    pytest.importorskip("mcp")
    import anyio

    from rag.mcp.server import build_mcp_server

    server = build_mcp_server(tools)
    sdk_tools = anyio.run(server.list_tools)
    sdk_schemas = {t.name: t.input_schema for t in sdk_tools}
    own_schemas = {s.name: s.input_schema for s in build_tool_specs(tools)}

    assert set(sdk_schemas) == set(own_schemas)
    for name, sdk_schema in sdk_schemas.items():
        own = own_schemas[name]
        assert sdk_schema["properties"] == own["properties"], name
        assert set(sdk_schema.get("required", [])) == set(own.get("required", [])), name


def test_mount_does_not_shadow_the_existing_api_routes() -> None:
    """Regression: mounting the MCP sub-app at "/" swallowed /health and /docs.

    A Starlette Mount matches every path beneath its prefix, and mounts are
    matched in registration order -- so a root mount registered before the
    route declarations below it wins every request. It belongs under "/mcp".
    """

    pytest.importorskip("mcp")

    from rag.api.main import app

    mounts = [route for route in app.routes if type(route).__name__ == "Mount"]
    assert [mount.path for mount in mounts] == ["/mcp"]

    paths = {getattr(route, "path", None) for route in app.routes}
    assert {"/health", "/docs"} <= paths


def test_sdk_serves_only_the_pinned_protocol_version() -> None:
    """The SDK ships every revision it knows; this project serves one."""

    pytest.importorskip("mcp")
    from mcp_types.version import MODERN_PROTOCOL_VERSIONS

    assert MODERN_PROTOCOL_VERSIONS == (PROTOCOL_VERSION,)


def test_sdk_gate_refuses_a_handshake_era_request(tools: RagTools) -> None:
    """A 2025 connection reaches the gate with its negotiated version.

    Driving a full legacy session would need a client; the gate is the whole
    behaviour, so it is called directly with the context the runner builds.
    """

    pytest.importorskip("mcp")
    import anyio
    from mcp.shared.exceptions import MCPError

    from rag.mcp.server import ProtocolVersionGate

    class _Ctx:
        protocol_version = "2025-06-18"
        request_id = 1

    async def _call_next(_ctx):  # type: ignore[no-untyped-def]
        raise AssertionError("the handler must not run for a refused version")

    with pytest.raises(MCPError) as excinfo:
        anyio.run(lambda: ProtocolVersionGate()(_Ctx(), _call_next))  # type: ignore[arg-type]

    assert excinfo.value.code == -32022
    assert excinfo.value.data == {"supported": [PROTOCOL_VERSION], "requested": "2025-06-18"}


def test_sdk_marks_both_tools_read_only(tools: RagTools) -> None:
    pytest.importorskip("mcp")
    import anyio

    from rag.mcp.server import build_mcp_server

    sdk_tools = anyio.run(build_mcp_server(tools).list_tools)
    assert all(t.annotations and t.annotations.read_only_hint for t in sdk_tools)
