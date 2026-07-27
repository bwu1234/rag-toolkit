"""Tests for the embedding interface, Ollama adapter, and factory.

The Ollama adapter is tested against a mocked HTTP transport (`httpx.MockTransport`)
rather than a real daemon -- this keeps the suite hermetic and fast while still
exercising the real request/response handling code (batching, error wrapping,
lazy dimension discovery).
"""

from __future__ import annotations

import httpx
import pytest

from rag.config.settings import EmbeddingConfig
from rag.embedding.base import EmbeddingModel
from rag.embedding.factory import get_embedder
from rag.embedding.ollama_embedder import OllamaEmbedder


def _embedder_with_handler(handler, **kwargs) -> OllamaEmbedder:
    """Build an `OllamaEmbedder` whose internal client routes through a mock transport."""

    embedder = OllamaEmbedder(model="test-embed", base_url="http://fake-ollama:11434", **kwargs)
    embedder._client = httpx.Client(base_url=embedder.base_url, transport=httpx.MockTransport(handler))
    return embedder


def _vector_for(text: str, length: int = 4) -> list[float]:
    """Deterministic, text-dependent fake vector so we can assert on order/content."""

    return [float(ord(ch)) for ch in (text + "\0" * length)[:length]]


def _ok_handler(request: httpx.Request) -> httpx.Response:
    body = request.read()
    import json

    payload = json.loads(body)
    inputs = payload["input"]
    return httpx.Response(200, json={"embeddings": [_vector_for(t) for t in inputs]})


def test_embed_documents_returns_one_vector_per_text_in_order() -> None:
    embedder = _embedder_with_handler(_ok_handler)

    vectors = embedder.embed_documents(["alpha", "beta", "gamma"])

    assert vectors == [_vector_for("alpha"), _vector_for("beta"), _vector_for("gamma")]


def test_embed_documents_batches_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    seen_batches: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        payload = json.loads(request.read())
        seen_batches.append(payload["input"])
        return httpx.Response(200, json={"embeddings": [_vector_for(t) for t in payload["input"]]})

    embedder = _embedder_with_handler(handler, batch_size=2)

    vectors = embedder.embed_documents(["a", "b", "c", "d", "e"])

    assert seen_batches == [["a", "b"], ["c", "d"], ["e"]]
    assert vectors == [_vector_for(t) for t in "abcde"]


def test_embed_documents_on_empty_list_makes_no_request() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("should not be called for an empty batch")

    embedder = _embedder_with_handler(handler)

    assert embedder.embed_documents([]) == []


def test_embed_query_returns_single_vector() -> None:
    embedder = _embedder_with_handler(_ok_handler)

    assert embedder.embed_query("hello") == _vector_for("hello")


def test_dimensions_are_discovered_lazily_and_cached() -> None:
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return _ok_handler(request)

    embedder = _embedder_with_handler(handler)

    assert embedder.dimensions == 4
    assert embedder.dimensions == 4
    assert call_count == 1, "dimension probe should only hit the network once"


def test_configured_dimensions_skip_the_probe() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("should not probe when dimensions are configured")

    embedder = _embedder_with_handler(handler, dimensions=1024)

    assert embedder.dimensions == 1024


def test_http_error_is_wrapped_in_runtime_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "model not found"})

    embedder = _embedder_with_handler(handler)

    with pytest.raises(RuntimeError, match="Failed to get embeddings from Ollama"):
        embedder.embed_documents(["x"])


def test_malformed_response_is_wrapped_in_runtime_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"embeddings": [[1.0, 2.0]]})  # only 1, asked for 2

    embedder = _embedder_with_handler(handler)

    with pytest.raises(RuntimeError, match="Unexpected response shape"):
        embedder.embed_documents(["x", "y"])


def test_get_embedder_factory_selects_ollama() -> None:
    embedder = get_embedder(EmbeddingConfig(provider="ollama", model="m", base_url="http://localhost:11434"))

    assert isinstance(embedder, EmbeddingModel)
    assert isinstance(embedder, OllamaEmbedder)


def test_get_embedder_factory_rejects_unknown_provider() -> None:
    config = EmbeddingConfig.model_construct(provider="sentence_transformers", model="m", base_url="http://x", dimensions=None)

    with pytest.raises(ValueError, match="Unknown embedding provider"):
        get_embedder(config)
