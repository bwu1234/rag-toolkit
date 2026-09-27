"""Live check of Gemini tool calling. Spends 3 free-tier requests, so it is opt-in.

Runs only with `RAG_GEMINI_LIVE=1` and the key in `GEMINI_API_KEY`; a key in the
environment alone is not consent to spend quota on every `pytest`. It pins the
two things mocked tests can't: that a Gemini 3 model's `thoughtSignature` goes
back in a form the API accepts, and that a follow-up turn with no tools
declared is allowed after function calls (the agent's forced-synthesis turn).
"""

from __future__ import annotations

import os

import pytest

from rag.config.settings import DEFAULT_CONFIG_PATH, load_config
from rag.generation.factory import get_llm_client
from rag.generation.llm import ChatMessage, ToolCallingLLM, ToolDefinition, ToolResult

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.environ.get("RAG_GEMINI_LIVE") != "1" or not os.environ.get("GEMINI_API_KEY"),
        reason="spends Gemini quota; set RAG_GEMINI_LIVE=1 and GEMINI_API_KEY to run",
    ),
]

SEARCH = ToolDefinition(
    name="rag_search",
    description="Search SEC filings. Returns numbered passages.",
    parameters={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
)


def test_signed_tool_call_round_trip_and_tool_free_follow_up() -> None:
    client = get_llm_client(load_config(DEFAULT_CONFIG_PATH.parent / "gemini-3.5-flash-lite.yaml").llm)
    assert isinstance(client, ToolCallingLLM)
    messages: list = [
        ChatMessage("system", "Always search with rag_search before answering; cite passages as [n]."),
        ChatMessage("user", "What were Apple's total net sales in fiscal 2025?"),
    ]

    turn = client.chat(messages, [SEARCH])
    assert turn.tool_calls and turn.tool_calls[0].name == "rag_search"
    assert turn.tool_calls[0].signature, "Gemini 3 signs a step's first call; without it the next request is a 400"

    messages += [turn, ToolResult(turn.tool_calls[0], "Passage [1]: Total net sales were $416.2 billion in 2025.")]
    assert client.chat(messages, [SEARCH]).content.strip()
    assert client.chat(messages).content.strip()  # no tools declared after function calls
