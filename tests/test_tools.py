"""The shared tool surface (`rag.tools`): what the agent and MCP both advertise.

Hermetic like `test_mcp.py`: `build_retriever` is patched wherever a search
would otherwise build one, so nothing touches Ollama, Chroma or model weights.
"""

from __future__ import annotations

from typing import Any

import pytest

import rag.mcp.tools as mcp_tools
from rag.config.settings import RagConfig
from rag.generation.gemini_llm import _declaration
from rag.generation.ollama_llm import _tool_to_wire
from rag.mcp.fallback import FallbackServer
from rag.retrieval.retriever import RetrievalResult
from rag.tools import RagTools, ToolSpec, build_tool_specs


@pytest.fixture()
def specs() -> list[ToolSpec]:
    # Building specs is free: RagTools is lazy, and nothing here searches.
    return build_tool_specs(RagTools("/nonexistent/config.yaml"))


def _spec(specs: list[ToolSpec], name: str) -> ToolSpec:
    return next(s for s in specs if s.name == name)


def _mcp_listing(specs: list[ToolSpec]) -> dict[str, dict[str, Any]]:
    """What an MCP client's `tools/list` returns, keyed by tool name."""

    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/list",
        "params": {
            "_meta": {
                "io.modelcontextprotocol/protocolVersion": mcp_tools.PROTOCOL_VERSION,
                "io.modelcontextprotocol/clientCapabilities": {},
            }
        },
    }
    response = FallbackServer(specs).handle(request)
    assert response is not None
    return {tool["name"]: tool for tool in response["result"]["tools"]}


def test_agent_and_mcp_advertise_the_same_rag_search(specs: list[ToolSpec]) -> None:
    """Phase 2's exit: one `rag_search`, described identically to both callers."""

    definition = _spec(specs, "rag_search").definition
    served = _mcp_listing(specs)["rag_search"]

    assert definition.name == served["name"]
    assert definition.description == served["description"]
    assert definition.parameters == served["inputSchema"]


def test_the_sdk_transport_serves_the_agents_schema_too() -> None:
    # `test_mcp.py` already pins SDK == fallback; this closes the triangle
    # directly, so a drift shows up as "the agent disagrees with MCP".
    pytest.importorskip("mcp")
    import anyio

    from rag.mcp.server import build_mcp_server

    tools = RagTools("/nonexistent/config.yaml")
    sdk = {t.name: t for t in anyio.run(build_mcp_server(tools).list_tools)}
    definition = _spec(build_tool_specs(tools), "rag_search").definition

    assert sdk["rag_search"].description == definition.description
    assert sdk["rag_search"].input_schema["properties"] == definition.parameters["properties"]


@pytest.mark.parametrize(
    "to_wire",
    [
        lambda d: _tool_to_wire(d)["function"]["parameters"],
        lambda d: _declaration(d)["parametersJsonSchema"],
    ],
    ids=["ollama", "gemini"],
)
def test_adapters_send_the_mcp_schema_unmodified(specs: list[ToolSpec], to_wire: Any) -> None:
    # The converter is only worth having if no adapter rewrites the schema on
    # the way out; if one ever has to, it should fail here, not in an eval.
    definition = _spec(specs, "rag_search").definition
    assert to_wire(definition) == _mcp_listing(specs)["rag_search"]["inputSchema"]


def test_mcp_module_re_exports_the_shared_objects() -> None:
    # Same objects, not copies: an MCP-side import must never see a stale fork.
    assert mcp_tools.RagTools is RagTools
    assert mcp_tools.build_tool_specs is build_tool_specs
    assert mcp_tools.ToolSpec is ToolSpec


def test_rag_tools_uses_a_given_config_without_reading_disk(monkeypatch: pytest.MonkeyPatch) -> None:
    config = RagConfig()
    seen: list[RagConfig] = []

    class _Retriever:
        rerank_top_k = 5

        def retrieve(self, query: str, *, query_filter: object = None, on_event: object = None) -> RetrievalResult:
            return RetrievalResult(chunks=[], candidate_count=0)

    def _build(cfg, llm_client=None, corpora=None):  # type: ignore[no-untyped-def]
        seen.append(cfg)
        return _Retriever()

    def _no_disk(*args: object, **kwargs: object) -> RagConfig:
        raise AssertionError("RagTools(config=...) must not load config from disk")

    monkeypatch.setattr("rag.tools.build_retriever", _build)
    monkeypatch.setattr("rag.tools.load_config", _no_disk)

    tools = RagTools(config=config)
    tools.search("anything")

    assert tools.config is config
    assert seen == [config]


def test_rag_tools_refuses_both_a_path_and_a_config() -> None:
    with pytest.raises(ValueError, match="not both"):
        RagTools("config.yaml", config=RagConfig())
