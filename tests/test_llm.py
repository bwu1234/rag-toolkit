"""Tests for the LLMClient interface, Ollama adapter, and factory.

Mirrors `test_embedding.py`: the Ollama adapter is tested against a mocked
HTTP transport (`httpx.MockTransport`) rather than a real daemon, exercising
the real request/response handling (message construction, option translation,
error wrapping, response-shape validation) hermetically and fast.
"""

from __future__ import annotations

import json
import logging

import httpx
import pytest

from rag.config.settings import LLMConfig
from rag.generation.factory import get_llm_client
from rag.generation.llm import (
    AssistantTurn,
    ChatMessage,
    LLMClient,
    ToolCall,
    ToolCallingLLM,
    ToolDefinition,
    ToolResult,
)
from rag.generation.ollama_llm import ContextOverflowError, OllamaLLMClient


def _client_with_handler(handler, **kwargs) -> OllamaLLMClient:
    """Build an `OllamaLLMClient` whose internal client routes through a mock transport."""

    client = OllamaLLMClient(model="test-chat", base_url="http://fake-ollama:11434", **kwargs)
    client._client = httpx.Client(base_url=client.base_url, transport=httpx.MockTransport(handler))
    return client


def _echo_handler(request: httpx.Request) -> httpx.Response:
    """Echoes back the messages it received as the assistant's reply, for assertions."""

    payload = json.loads(request.read())
    return httpx.Response(200, json={"message": {"role": "assistant", "content": json.dumps(payload)}})


# ---------------------------------------------------------------------------
# OllamaLLMClient.generate
# ---------------------------------------------------------------------------


def test_generate_returns_message_content() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "The answer is 42."}})

    client = _client_with_handler(handler)

    assert client.generate("What is the answer?") == "The answer is 42."


def test_generate_sends_user_message_without_system_by_default() -> None:
    client = _client_with_handler(_echo_handler)

    reply = json.loads(client.generate("hello there"))

    assert reply["messages"] == [{"role": "user", "content": "hello there"}]


def test_generate_prepends_system_message_when_given() -> None:
    client = _client_with_handler(_echo_handler)

    reply = json.loads(client.generate("hello there", system="Be terse."))

    assert reply["messages"] == [
        {"role": "system", "content": "Be terse."},
        {"role": "user", "content": "hello there"},
    ]


def test_generate_translates_temperature_and_max_tokens_into_ollama_options() -> None:
    client = _client_with_handler(_echo_handler, temperature=0.7, max_tokens=256)

    reply = json.loads(client.generate("prompt"))

    assert reply["model"] == "test-chat"
    assert reply["stream"] is False
    assert reply["options"] == {"temperature": 0.7, "num_predict": 256}


def test_generate_wraps_http_errors_in_runtime_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    client = _client_with_handler(handler)

    with pytest.raises(RuntimeError, match="Failed to get a chat completion from Ollama"):
        client.generate("prompt")


def test_generate_rejects_unexpected_response_shape() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": "shape"})

    client = _client_with_handler(handler)

    with pytest.raises(RuntimeError, match="Unexpected response shape"):
        client.generate("prompt")


# ---------------------------------------------------------------------------
# get_llm_client factory
# ---------------------------------------------------------------------------


def test_get_llm_client_factory_selects_ollama() -> None:
    client = get_llm_client(LLMConfig(provider="ollama", model="m", base_url="http://localhost:11434"))

    assert isinstance(client, LLMClient)
    assert isinstance(client, OllamaLLMClient)
    assert client.model == "m"


def test_get_llm_client_factory_rejects_known_but_unimplemented_providers() -> None:
    config = LLMConfig.model_construct(provider="anthropic", model="claude", base_url="unused")

    with pytest.raises(ValueError, match="recognized config value but has no adapter yet"):
        get_llm_client(config)


def test_get_llm_client_factory_rejects_unknown_provider() -> None:
    config = LLMConfig.model_construct(provider="cohere", model="m", base_url="unused")

    with pytest.raises(ValueError, match="Unknown LLM provider"):
        get_llm_client(config)


def test_get_llm_client_factory_passes_the_configured_timeout() -> None:
    # A judge grading a long multi-hop answer needs far more than the 120s
    # default; the setting must actually reach the HTTP client.
    client = get_llm_client(LLMConfig(model="m", timeout_s=900))

    assert isinstance(client, OllamaLLMClient)
    assert client._client.timeout.read == 900



def test_get_llm_client_factory_passes_num_ctx_to_ollama_only_when_asked() -> None:
    default = get_llm_client(LLMConfig(model="m"))
    assert isinstance(default, OllamaLLMClient) and default.num_ctx is None
    client = get_llm_client(LLMConfig(model="m"), num_ctx=32768)

    assert isinstance(client, ToolCallingLLM)
    assert isinstance(client, OllamaLLMClient) and client.num_ctx == 32768


# ---------------------------------------------------------------------------
# OllamaLLMClient.chat (tool calling)
# ---------------------------------------------------------------------------

SEARCH = ToolDefinition(
    name="rag_search",
    description="Search the filings.",
    parameters={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
)

#: The shape Ollama 0.34 returned for a real tool call: arguments arrive as an
#: object, not a JSON string, and the call carries an id.
TOOL_CALL_RESPONSE = {
    "model": "test-chat",
    "message": {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": "call_dt0ma30h", "function": {"index": 0, "name": "rag_search", "arguments": {"query": "Apple revenue 2025"}}}
        ],
    },
    "done": True,
    "prompt_eval_count": 280,
    "eval_count": 31,
}


def _recording_handler(sent: list[dict], reply: dict):
    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.read()))
        return httpx.Response(200, json=reply)

    return handler


def test_chat_parses_tool_calls_and_usage() -> None:
    client = _client_with_handler(_recording_handler([], TOOL_CALL_RESPONSE))

    turn = client.chat([ChatMessage("user", "Apple revenue?")], [SEARCH])

    assert turn.content == ""
    assert turn.tool_calls == (ToolCall(name="rag_search", arguments={"query": "Apple revenue 2025"}, id="call_dt0ma30h"),)
    assert turn.usage is not None and (turn.usage.prompt_tokens, turn.usage.completion_tokens) == (280, 31)
    assert turn.thinking is None


def test_chat_sends_tools_in_ollama_function_format() -> None:
    sent: list[dict] = []
    client = _client_with_handler(_recording_handler(sent, TOOL_CALL_RESPONSE))

    client.chat([ChatMessage("user", "q")], [SEARCH])

    assert sent[0]["tools"] == [
        {
            "type": "function",
            "function": {"name": "rag_search", "description": "Search the filings.", "parameters": SEARCH.parameters},
        }
    ]


def test_chat_without_tools_omits_the_key_so_the_model_must_answer_in_text() -> None:
    sent: list[dict] = []
    client = _client_with_handler(_recording_handler(sent, {"message": {"role": "assistant", "content": "Done."}}))

    turn = client.chat([ChatMessage("user", "q")])

    assert "tools" not in sent[0]
    assert turn == AssistantTurn(content="Done.", usage=turn.usage)
    assert turn.tool_calls == ()


def test_chat_round_trips_a_tool_conversation_in_ollama_message_format() -> None:
    sent: list[dict] = []
    client = _client_with_handler(_recording_handler(sent, {"message": {"role": "assistant", "content": "Answer [1]."}}))
    call = ToolCall(name="rag_search", arguments={"query": "Apple revenue 2025"}, id="call_1")

    client.chat(
        [
            ChatMessage("system", "Search first."),
            ChatMessage("user", "Apple revenue?"),
            AssistantTurn(content="", tool_calls=(call,), thinking="I should search."),
            ToolResult(call=call, content="Passage [1] ..."),
        ],
        [SEARCH],
    )

    assert sent[0]["messages"] == [
        {"role": "system", "content": "Search first."},
        {"role": "user", "content": "Apple revenue?"},
        {
            "role": "assistant",
            "content": "",
            "thinking": "I should search.",
            "tool_calls": [{"id": "call_1", "function": {"name": "rag_search", "arguments": {"query": "Apple revenue 2025"}}}],
        },
        {"role": "tool", "tool_name": "rag_search", "content": "Passage [1] ..."},
    ]


def test_chat_returns_the_reasoning_trace_when_thinking() -> None:
    reply = {"message": {"role": "assistant", "content": "Answer.", "thinking": "Let me compare."}}
    sent: list[dict] = []
    client = _client_with_handler(_recording_handler(sent, reply), think="low")

    turn = client.chat([ChatMessage("user", "q")])

    assert turn.thinking == "Let me compare."
    assert sent[0]["think"] == "low"


def test_chat_sends_num_ctx_and_turns_off_truncation_when_set() -> None:
    sent: list[dict] = []
    client = _client_with_handler(_recording_handler(sent, TOOL_CALL_RESPONSE), num_ctx=32768, max_tokens=256)

    client.chat([ChatMessage("user", "q")], [SEARCH])

    assert sent[0]["options"] == {"temperature": 0.2, "num_predict": 256, "num_ctx": 32768}
    assert sent[0]["truncate"] is False


def test_requests_without_num_ctx_are_unchanged() -> None:
    # The pipeline's measured results were produced with neither key.
    sent: list[dict] = []
    client = _client_with_handler(_recording_handler(sent, {"message": {"role": "assistant", "content": "ok"}}))

    client.generate("q")
    client.chat([ChatMessage("user", "q")])

    for body in sent:
        assert "num_ctx" not in body["options"] and "truncate" not in body


def test_context_overflow_raises_a_distinct_error() -> None:
    # Ollama's reply with `truncate: false`, verbatim from 0.34.4.
    overflow = (
        '{"error":"{\\"error\\":{\\"code\\":400,\\"message\\":\\"request (6921 tokens) exceeds the available '
        'context size (4096 tokens), try increasing it\\",\\"type\\":\\"exceed_context_size_error\\"}}"}'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text=overflow)

    client = _client_with_handler(handler, num_ctx=4096)

    with pytest.raises(ContextOverflowError, match="4096-token context window"):
        client.chat([ChatMessage("user", "q")])


def test_other_bad_requests_stay_generic_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "model 'x' not found"})

    client = _client_with_handler(handler, num_ctx=4096)

    with pytest.raises(RuntimeError, match="Failed to get a chat completion") as info:
        client.chat([ChatMessage("user", "q")])
    assert not isinstance(info.value, ContextOverflowError)


@pytest.mark.parametrize(
    "tool_call",
    [
        {"function": {"name": "rag_search", "arguments": '{"query": "x"}'}},  # arguments as a string
        {"function": {"arguments": {"query": "x"}}},  # no name
        {"name": "rag_search"},  # no function object
    ],
)
def test_chat_rejects_a_malformed_tool_call(tool_call: dict) -> None:
    reply = {"message": {"role": "assistant", "content": "", "tool_calls": [tool_call]}}
    client = _client_with_handler(_recording_handler([], reply))

    with pytest.raises(RuntimeError, match="Unexpected tool call shape"):
        client.chat([ChatMessage("user", "q")], [SEARCH])


def test_chat_rejects_a_response_without_content() -> None:
    client = _client_with_handler(_recording_handler([], {"unexpected": "shape"}))

    with pytest.raises(RuntimeError, match="Unexpected response shape"):
        client.chat([ChatMessage("user", "q")])


@pytest.mark.parametrize(("prompt_tokens", "warns"), [(29_491, False), (29_492, True), (32_768, True)])
def test_warns_when_a_prompt_nears_num_ctx(prompt_tokens: int, warns: bool, caplog: pytest.LogCaptureFixture) -> None:
    # An early warning before the conversation's next prompt overflows.
    # 90% of 32768 is 29491.2.
    reply = {"message": {"role": "assistant", "content": "ok"}, "prompt_eval_count": prompt_tokens, "eval_count": 1}
    client = _client_with_handler(_recording_handler([], reply), num_ctx=32768)

    with caplog.at_level(logging.WARNING, logger="rag.generation.ollama_llm"):
        client.chat([ChatMessage("user", "q")])

    assert ("will raise ContextOverflowError" in caplog.text) is warns


def test_no_context_warning_without_num_ctx(caplog: pytest.LogCaptureFixture) -> None:
    reply = {"message": {"role": "assistant", "content": "ok"}, "prompt_eval_count": 10**6}
    client = _client_with_handler(_recording_handler([], reply))

    with caplog.at_level(logging.WARNING, logger="rag.generation.ollama_llm"):
        client.generate("q")

    assert caplog.text == ""
