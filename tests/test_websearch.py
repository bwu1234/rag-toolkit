"""Tests for `SearxNGWebSearch`.

Mocked against `httpx.MockTransport` (mirroring `test_embedding.py`) so the
suite never needs a running SearxNG instance -- and against a fake
`EmbeddingModel` so similarity scores are deterministic and assertable.
"""

from __future__ import annotations

import httpx
import pytest

from rag.config.settings import WebSearchConfig
from rag.retrieval.websearch import SearxNGWebSearch


class _FakeEmbedder:
    """Maps specific texts to hand-picked vectors so cosine similarity is predictable."""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self._vectors = vectors

    def embed_query(self, text: str) -> list[float]:
        return self._vectors[text]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vectors[t] for t in texts]

    @property
    def dimensions(self) -> int:
        return 2


def _search_with(handler, embedder, **config_kwargs) -> SearxNGWebSearch:
    web_search = SearxNGWebSearch(embedder, WebSearchConfig(**config_kwargs))
    web_search._client = httpx.Client(transport=httpx.MockTransport(handler))
    return web_search


def _results_handler(results: list[dict]):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": results})

    return handler


def test_search_scores_results_by_query_similarity() -> None:
    embedder = _FakeEmbedder(
        {
            "query": [1.0, 0.0],
            "relevant snippet": [1.0, 0.0],  # cosine 1.0 vs query
            "irrelevant snippet": [0.0, 1.0],  # cosine 0.0 vs query
        }
    )
    handler = _results_handler(
        [
            {"url": "https://a.example", "title": "A", "content": "relevant snippet"},
            {"url": "https://b.example", "title": "B", "content": "irrelevant snippet"},
        ]
    )
    web_search = _search_with(handler, embedder, min_similarity=0.0)

    results = web_search.search("query")

    assert [r.chunk_id for r in results] == ["https://a.example", "https://b.example"]
    assert results[0].score == pytest.approx(1.0)
    assert results[1].score == pytest.approx(0.0)


def test_search_drops_results_below_min_similarity() -> None:
    embedder = _FakeEmbedder(
        {
            "query": [1.0, 0.0],
            "relevant snippet": [1.0, 0.0],
            "irrelevant snippet": [0.0, 1.0],
        }
    )
    handler = _results_handler(
        [
            {"url": "https://a.example", "title": "A", "content": "relevant snippet"},
            {"url": "https://b.example", "title": "B", "content": "irrelevant snippet"},
        ]
    )
    web_search = _search_with(handler, embedder, min_similarity=0.5)

    results = web_search.search("query")

    assert [r.chunk_id for r in results] == ["https://a.example"]


def test_search_dedups_near_identical_snippets() -> None:
    embedder = _FakeEmbedder(
        {
            "query": [1.0, 0.0],
            "snippet one": [1.0, 0.0],
            "snippet two": [0.99, 0.14107],  # cosine ~0.99 vs "snippet one" -- a near-duplicate
        }
    )
    handler = _results_handler(
        [
            {"url": "https://a.example", "title": "A", "content": "snippet one"},
            {"url": "https://b.example", "title": "B", "content": "snippet two"},
        ]
    )
    web_search = _search_with(handler, embedder, min_similarity=0.0, dedup_threshold=0.75)

    results = web_search.search("query")

    assert [r.chunk_id for r in results] == ["https://a.example"]


def test_search_returns_empty_list_on_no_results() -> None:
    embedder = _FakeEmbedder({})
    handler = _results_handler([])
    web_search = _search_with(handler, embedder)

    assert web_search.search("query") == []


def test_search_returns_empty_list_on_http_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    embedder = _FakeEmbedder({})
    web_search = _search_with(handler, embedder)

    assert web_search.search("query") == []


def test_search_falls_back_to_title_when_content_missing() -> None:
    embedder = _FakeEmbedder({"query": [1.0, 0.0], "Title Only": [1.0, 0.0]})
    handler = _results_handler([{"url": "https://a.example", "title": "Title Only"}])
    web_search = _search_with(handler, embedder, min_similarity=0.0)

    results = web_search.search("query")

    assert results[0].text == "Title Only"
