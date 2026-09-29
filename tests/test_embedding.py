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
from rag.embedding.sentence_transformers_embedder import SentenceTransformersEmbedder
from rag.index_manifest import IndexManifest
from rag.config.settings import RagConfig


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



def _recording_handler(seen: list[str]):
    def handler(request: httpx.Request) -> httpx.Response:
        import json

        inputs = json.loads(request.read())["input"]
        seen.extend(inputs)
        return httpx.Response(200, json={"embeddings": [_vector_for(t) for t in inputs]})

    return handler


def test_query_instruction_prefixes_queries_in_the_model_card_format() -> None:
    seen: list[str] = []
    embedder = _embedder_with_handler(_recording_handler(seen), query_instruction="Find passages")

    embedder.embed_query("revenue in 2025")

    # No space after "Query:" -- Qwen3-Embedding's documented format.
    assert seen == ["Instruct: Find passages\nQuery:revenue in 2025"]


def test_query_instruction_never_reaches_documents() -> None:
    seen: list[str] = []
    embedder = _embedder_with_handler(_recording_handler(seen), query_instruction="Find passages")

    embedder.embed_documents(["chunk one", "chunk two"])

    assert seen == ["chunk one", "chunk two"]


def test_no_query_instruction_embeds_the_bare_query() -> None:
    seen: list[str] = []
    embedder = _embedder_with_handler(_recording_handler(seen))

    embedder.embed_query("revenue in 2025")

    assert seen == ["revenue in 2025"]

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


def test_get_embedder_factory_passes_the_query_instruction() -> None:
    config = EmbeddingConfig(model="m", query_instruction="Find passages")

    embedder = get_embedder(config)

    assert isinstance(embedder, OllamaEmbedder)
    assert embedder.query_instruction == "Find passages"


def test_get_embedder_factory_rejects_unknown_provider() -> None:
    config = EmbeddingConfig.model_construct(provider="no_such_provider", model="m", base_url="http://x", dimensions=None)

    with pytest.raises(ValueError, match="Unknown embedding provider"):
        get_embedder(config)


# --- sentence-transformers adapter ------------------------------------------
#
# A stub module stands in for `sentence_transformers`, as in test_reranker.py,
# so the suite never downloads a model.


class _StubSentenceTransformer:
    constructed: list[tuple[str, str | None]] = []

    def __init__(self, model_name: str, revision: str | None = None) -> None:
        type(self).constructed.append((model_name, revision))
        self.device = "cpu"
        self.max_seq_length = 512
        self.encoded: list[list[str]] = []
        self.normalize_flags: list[bool] = []

    def encode(self, texts, *, batch_size, normalize_embeddings, convert_to_numpy, show_progress_bar):
        import numpy as np

        self.encoded.append(list(texts))
        self.normalize_flags.append(normalize_embeddings)
        return np.array([_vector_for(t) for t in texts], dtype=np.float32)

    def get_embedding_dimension(self) -> int:
        return 768


@pytest.fixture
def stub_sentence_transformers(monkeypatch: pytest.MonkeyPatch) -> type[_StubSentenceTransformer]:
    import sys
    import types

    _StubSentenceTransformer.constructed = []
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = _StubSentenceTransformer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return _StubSentenceTransformer


def test_sentence_transformers_loads_lazily_once_at_the_pinned_revision(stub_sentence_transformers) -> None:
    embedder = SentenceTransformersEmbedder("BAAI/bge-base-en-v1.5", revision="abc123")
    assert stub_sentence_transformers.constructed == [], "constructing must not load the model"

    embedder.embed_documents(["a"])
    embedder.embed_query("b")

    assert stub_sentence_transformers.constructed == [("BAAI/bge-base-en-v1.5", "abc123")]


def test_sentence_transformers_returns_normalised_vectors_in_order(stub_sentence_transformers) -> None:
    embedder = SentenceTransformersEmbedder("m")

    vectors = embedder.embed_documents(["alpha", "beta"])

    assert vectors == [_vector_for("alpha"), _vector_for("beta")]
    assert all(isinstance(x, float) for x in vectors[0])
    assert embedder._model.normalize_flags == [True]


def test_sentence_transformers_query_instruction_is_a_plain_prefix(stub_sentence_transformers) -> None:
    instruction = "Represent this sentence for searching relevant passages:"
    embedder = SentenceTransformersEmbedder("m", query_instruction=instruction)

    embedder.embed_query("what is BM25")
    embedder.embed_documents(["a passage"])

    # BGE's format (and Pyserini's --query-prefix): one space, no template.
    # Documents never get it.
    assert embedder._model.encoded == [[f"{instruction} what is BM25"], ["a passage"]]


def test_sentence_transformers_empty_batch_loads_nothing(stub_sentence_transformers) -> None:
    assert SentenceTransformersEmbedder("m").embed_documents([]) == []
    assert stub_sentence_transformers.constructed == []


def test_sentence_transformers_dimensions_come_from_config_or_the_model(stub_sentence_transformers) -> None:
    assert SentenceTransformersEmbedder("m", dimensions=384).dimensions == 384
    assert stub_sentence_transformers.constructed == []
    assert SentenceTransformersEmbedder("m").dimensions == 768


def test_get_embedder_factory_selects_sentence_transformers() -> None:
    config = EmbeddingConfig(
        provider="sentence_transformers", model="BAAI/bge-base-en-v1.5", revision="abc123", query_instruction="Q:"
    )

    embedder = get_embedder(config)

    assert isinstance(embedder, SentenceTransformersEmbedder)
    assert (embedder.model_name, embedder.revision, embedder.query_instruction) == ("BAAI/bge-base-en-v1.5", "abc123", "Q:")


def test_revision_is_refused_for_ollama() -> None:
    with pytest.raises(ValueError, match="revision pins a Hugging Face model"):
        EmbeddingConfig(provider="ollama", revision="abc123")


def test_revision_is_part_of_the_index_manifest_only_when_set() -> None:
    unpinned = IndexManifest.from_config(RagConfig())
    pinned = IndexManifest.from_config(
        RagConfig(embedding=EmbeddingConfig(provider="sentence_transformers", model="m", revision="abc123"))
    )
    repinned = IndexManifest.from_config(
        RagConfig(embedding=EmbeddingConfig(provider="sentence_transformers", model="m", revision="def456"))
    )

    # Manifests written before the field existed carry no key, and still match.
    assert "revision" not in unpinned.embedding
    assert pinned.differences(repinned, sections=("embedding",)) == ["embedding.revision: 'abc123' -> 'def456'"]
