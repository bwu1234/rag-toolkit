"""Tests for `ChatService` orchestration.

`ChatService` is tested against fakes for `Retriever` and `LLMClient` --
mirroring how `test_retriever.py` fakes `EmbeddingModel`/`VectorStore`/
`Reranker` -- so the orchestration logic (when to skip generation, how chunks
become citations, what gets passed to the LLM) is verified without a running
Ollama daemon, Chroma index, or cross-encoder model.
"""

from __future__ import annotations

from pathlib import Path

from rag.generation.chat_service import ChatService, Citation
from rag.vectorstore.base import ScoredChunk


def _scored(chunk_id: str, text: str = "text", score: float = 0.5, page: int | None = None) -> ScoredChunk:
    metadata = {"page": page} if page is not None else {}
    return ScoredChunk(
        chunk_id=chunk_id,
        text=text,
        document_id=f"{chunk_id}-doc",
        source=Path(f"{chunk_id}.md"),
        doc_type="markdown",
        score=score,
        metadata=metadata,
    )


class _FakeRetriever:
    def __init__(self, results: list[ScoredChunk]) -> None:
        self.results = results
        self.queries: list[str] = []

    def retrieve(self, query: str) -> list[ScoredChunk]:
        self.queries.append(query)
        return self.results


class _FakeLLMClient:
    def __init__(self, reply: str = "a generated answer") -> None:
        self.reply = reply
        self.calls: list[tuple[str, str | None]] = []

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        self.calls.append((prompt, system))
        return self.reply


def _service(results: list[ScoredChunk] | None = None, reply: str = "a generated answer"):
    results = results if results is not None else [_scored("a", "Refunds within 30 days.", page=2)]
    retriever = _FakeRetriever(results)
    llm_client = _FakeLLMClient(reply)
    service = ChatService(retriever=retriever, llm_client=llm_client)
    return service, retriever, llm_client


def test_ask_returns_generated_answer_with_citations_from_retrieved_chunks() -> None:
    chunks = [_scored("a", "Refunds within 30 days.", score=0.9, page=2), _scored("b", "Exchanges allowed.", score=0.7)]
    service, retriever, llm_client = _service(chunks, reply="You can get a refund within 30 days [1].")

    result = service.ask("what is the refund policy")

    assert result.answer == "You can get a refund within 30 days [1]."
    assert result.citations == [
        Citation(chunk_id="a", document_id="a-doc", text="Refunds within 30 days.", score=0.9, page=2),
        Citation(chunk_id="b", document_id="b-doc", text="Exchanges allowed.", score=0.7, page=None),
    ]
    assert retriever.queries == ["what is the refund policy"]


def test_ask_passes_a_grounded_prompt_and_system_prompt_to_the_llm() -> None:
    service, _retriever, llm_client = _service([_scored("a", "Refunds within 30 days.")])

    service.ask("what is the refund policy")

    [(prompt, system)] = llm_client.calls
    assert "Refunds within 30 days." in prompt
    assert "what is the refund policy" in prompt
    assert system is not None and "cite" in system.lower()


def test_ask_on_blank_query_skips_retrieval_and_generation() -> None:
    service, retriever, llm_client = _service()

    result = service.ask("   ")

    assert result.citations == []
    assert result.answer  # some explanatory message
    assert retriever.queries == []
    assert llm_client.calls == []


def test_ask_on_empty_retrieval_skips_generation_and_explains() -> None:
    service, retriever, llm_client = _service(results=[])

    result = service.ask("an unanswerable question")

    assert result.citations == []
    assert result.answer
    assert retriever.queries == ["an unanswerable question"]
    assert llm_client.calls == [], "the LLM should never be called when there's no context to ground it in"


def test_ask_drops_non_integer_page_metadata() -> None:
    chunk = ScoredChunk(
        chunk_id="a", text="t", document_id="a-doc", source=Path("a.md"),
        doc_type="markdown", score=0.5, metadata={"page": "not-a-number"},
    )
    service, *_ = _service(results=[chunk])

    [citation] = service.ask("query").citations

    assert citation.page is None
