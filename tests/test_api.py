"""Tests for the FastAPI chat API.

The app's `lifespan` builds a real `ChatService` (Ollama + Chroma + reranker)
on startup -- far too heavy for a test. Instead these tests override the
`get_chat_service` dependency with a fake, so requests never touch the
lifespan-built service, and the suite stays hermetic and fast while still
exercising real routing, validation, and request/response (de)serialization.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag.api.main import app
from rag.api.routes.chat import get_chat_service
from rag.generation.chat_service import ChatAnswer, ChatService, Citation


class _FakeChatService:
    def __init__(self, answer: ChatAnswer) -> None:
        self.answer = answer
        self.queries: list[str] = []

    def ask(self, query: str) -> ChatAnswer:
        self.queries.append(query)
        return self.answer


_SAMPLE_ANSWER = ChatAnswer(
    answer="Refunds are accepted within 30 days [1].",
    citations=[
        Citation(
            chunk_id="handbook.pdf#page=2::chunk0",
            document_id="handbook.pdf#page=2",
            text="Refunds are accepted within 30 days of purchase.",
            score=0.93,
            page=2,
        )
    ],
)


@pytest.fixture()
def client() -> TestClient:
    """A `TestClient` with `get_chat_service` overridden by a fake -- no lifespan, no real I/O."""

    fake = _FakeChatService(_SAMPLE_ANSWER)
    app.dependency_overrides[get_chat_service] = lambda: fake
    test_client = TestClient(app)
    test_client.fake_chat_service = fake  # type: ignore[attr-defined]
    try:
        yield test_client
    finally:
        app.dependency_overrides.pop(get_chat_service, None)


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_returns_answer_and_citations(client: TestClient) -> None:
    response = client.post("/chat", json={"query": "what is the refund policy"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Refunds are accepted within 30 days [1]."
    assert body["citations"] == [
        {
            "chunk_id": "handbook.pdf#page=2::chunk0",
            "document_id": "handbook.pdf#page=2",
            "text": "Refunds are accepted within 30 days of purchase.",
            "score": pytest.approx(0.93),
            "page": 2,
        }
    ]
    assert client.fake_chat_service.queries == ["what is the refund policy"]  # type: ignore[attr-defined]


def test_chat_rejects_blank_query_with_422(client: TestClient) -> None:
    response = client.post("/chat", json={"query": ""})

    assert response.status_code == 422
    assert client.fake_chat_service.queries == []  # type: ignore[attr-defined]


def test_chat_rejects_missing_query_with_422(client: TestClient) -> None:
    response = client.post("/chat", json={})

    assert response.status_code == 422


def test_chat_response_omits_page_when_none() -> None:
    fake = _FakeChatService(ChatAnswer(
        answer="No page info available.",
        citations=[Citation(chunk_id="c", document_id="readme.md", text="t", score=0.4, page=None)],
    ))
    app.dependency_overrides[get_chat_service] = lambda: fake
    try:
        response = TestClient(app).post("/chat", json={"query": "q"})
    finally:
        app.dependency_overrides.pop(get_chat_service, None)

    assert response.json()["citations"][0]["page"] is None
