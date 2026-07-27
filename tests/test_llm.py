"""Tests for the LLMClient interface, Ollama adapter, and factory.

Mirrors `test_embedding.py`: the Ollama adapter is tested against a mocked
HTTP transport (`httpx.MockTransport`) rather than a real daemon, exercising
the real request/response handling (message construction, option translation,
error wrapping, response-shape validation) hermetically and fast.
"""

from __future__ import annotations

import json

import httpx
import pytest

from rag.config.settings import LLMConfig
from rag.generation.factory import get_llm_client
from rag.generation.llm import LLMClient
from rag.generation.ollama_llm import OllamaLLMClient


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
