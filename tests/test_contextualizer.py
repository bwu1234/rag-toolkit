"""Tests for contextual chunking (`rag.chunking.contextualizer`).

Hermetic like every other LLM-backed component's tests: `ChunkContextualizer`
is driven by a fake `LLMClient` that records its prompts and returns canned
replies, so prompt assembly, truncation, and every fail-open path are verified
with no Ollama daemon involved.

The load-bearing property under test is the separation `Chunk` maintains:
generated context enriches what gets *indexed* (`contextual_text`) and never
touches what gets *cited* (`text`).
"""

from __future__ import annotations

from pathlib import Path

from rag.chunking.contextualizer import ChunkContextualizer, build_context_prompt
from rag.chunking.models import Chunk
from rag.ingestion.models import Document


def _document(doc_id: str = "api_reference.md", text: str = "ACS API reference. Rate limits apply.") -> Document:
    return Document(
        id=doc_id,
        text=text,
        source=Path(doc_id),
        doc_type="markdown",
        metadata={"title": "API Reference"},
    )


def _chunk(chunk_id: str = "api_reference.md::chunk0", *, document_id: str = "api_reference.md") -> Chunk:
    return Chunk(
        id=chunk_id,
        text="The limit is 1,000 requests per minute.",
        document_id=document_id,
        source=Path(document_id),
        doc_type="markdown",
        metadata={"chunk_index": 0, "title": "API Reference"},
    )


class _FakeLLMClient:
    """Returns canned replies in order (repeating the last), recording prompts."""

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies) or ["Context sentence."]
        self.calls: list[tuple[str, str | None]] = []

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        self.calls.append((prompt, system))
        index = min(len(self.calls) - 1, len(self.replies) - 1)
        return self.replies[index]


class _RaisingLLMClient:
    def generate(self, prompt: str, *, system: str | None = None) -> str:
        raise RuntimeError("daemon unreachable")


# ---------------------------------------------------------------------------
# The core contract: context enriches the index, never the citation
# ---------------------------------------------------------------------------


def test_contextualize_attaches_context_without_touching_chunk_text() -> None:
    client = _FakeLLMClient("This excerpt is from the ACS API rate limiting section.")
    contextualizer = ChunkContextualizer(client)  # type: ignore[arg-type]

    [result] = contextualizer.contextualize([_chunk()], [_document()])

    assert result.context == "This excerpt is from the ACS API rate limiting section."
    assert result.text == "The limit is 1,000 requests per minute.", "the cited span must stay verbatim"


def test_contextual_text_joins_context_and_text_for_indexing() -> None:
    client = _FakeLLMClient("From the ACS rate limiting section.")
    contextualizer = ChunkContextualizer(client)  # type: ignore[arg-type]

    [result] = contextualizer.contextualize([_chunk()], [_document()])

    assert result.contextual_text == (
        "From the ACS rate limiting section.\n\nThe limit is 1,000 requests per minute."
    )


def test_contextual_text_without_context_is_exactly_the_chunk_text() -> None:
    chunk = _chunk()

    assert chunk.context is None
    assert chunk.contextual_text == chunk.text


def test_contextualize_preserves_id_provenance_and_metadata() -> None:
    contextualizer = ChunkContextualizer(_FakeLLMClient())  # type: ignore[arg-type]
    original = _chunk()

    [result] = contextualizer.contextualize([original], [_document()])

    assert result.id == original.id
    assert result.document_id == original.document_id
    assert result.source == original.source
    assert result.doc_type == original.doc_type
    assert result.metadata == original.metadata


def test_contextualize_returns_chunks_in_order() -> None:
    client = _FakeLLMClient("first context", "second context")
    contextualizer = ChunkContextualizer(client)  # type: ignore[arg-type]
    chunks = [_chunk("doc.md::chunk0", document_id="doc.md"), _chunk("doc.md::chunk1", document_id="doc.md")]

    results = contextualizer.contextualize(chunks, [_document("doc.md")])

    assert [c.id for c in results] == ["doc.md::chunk0", "doc.md::chunk1"]
    assert [c.context for c in results] == ["first context", "second context"]


# ---------------------------------------------------------------------------
# Prompt assembly and bounds
# ---------------------------------------------------------------------------


def test_prompt_carries_both_the_document_and_the_excerpt() -> None:
    client = _FakeLLMClient()
    contextualizer = ChunkContextualizer(client)  # type: ignore[arg-type]

    contextualizer.contextualize([_chunk()], [_document(text="ACS API reference. Rate limits apply.")])

    [(prompt, system)] = client.calls
    assert "ACS API reference. Rate limits apply." in prompt
    assert "The limit is 1,000 requests per minute." in prompt
    assert system is not None and "situate" in system.lower()


def test_document_text_is_truncated_to_max_document_chars() -> None:
    client = _FakeLLMClient()
    contextualizer = ChunkContextualizer(client, max_document_chars=10)  # type: ignore[arg-type]

    contextualizer.contextualize([_chunk()], [_document(text="A" * 500)])

    [(prompt, _system)] = client.calls
    assert "A" * 10 in prompt
    assert "A" * 11 not in prompt


def test_generated_context_is_truncated_to_max_context_chars() -> None:
    client = _FakeLLMClient("B" * 500)
    contextualizer = ChunkContextualizer(client, max_context_chars=20)  # type: ignore[arg-type]

    [result] = contextualizer.contextualize([_chunk()], [_document()])

    assert result.context == "B" * 20


def test_context_is_collapsed_to_one_line_and_unquoted() -> None:
    client = _FakeLLMClient('  "From the\n  rate limiting   section."  ')
    contextualizer = ChunkContextualizer(client)  # type: ignore[arg-type]

    [result] = contextualizer.contextualize([_chunk()], [_document()])

    assert result.context == "From the rate limiting section."


def test_build_context_prompt_delimits_document_and_excerpt() -> None:
    prompt = build_context_prompt("the document", "the excerpt")

    assert "<document>" in prompt and "the document" in prompt
    assert "<excerpt>" in prompt and "the excerpt" in prompt


# ---------------------------------------------------------------------------
# Fail-open paths: a broken contextualizer must never break an index run
# ---------------------------------------------------------------------------


def test_raising_client_leaves_the_chunk_unchanged() -> None:
    contextualizer = ChunkContextualizer(_RaisingLLMClient())  # type: ignore[arg-type]

    [result] = contextualizer.contextualize([_chunk()], [_document()])

    assert result.context is None
    assert result.contextual_text == result.text


def test_empty_reply_leaves_the_chunk_unchanged() -> None:
    contextualizer = ChunkContextualizer(_FakeLLMClient("   "))  # type: ignore[arg-type]

    [result] = contextualizer.contextualize([_chunk()], [_document()])

    assert result.context is None


def test_chunk_with_no_matching_document_is_left_unchanged() -> None:
    client = _FakeLLMClient()
    contextualizer = ChunkContextualizer(client)  # type: ignore[arg-type]

    [result] = contextualizer.contextualize([_chunk(document_id="missing.md")], [_document("other.md")])

    assert result.context is None
    assert client.calls == [], "no parent document means nothing to generate context from"


def test_one_failure_does_not_stop_the_rest_of_the_batch() -> None:
    # First chunk gets an empty (unusable) reply, second gets a real one.
    client = _FakeLLMClient("", "a real context")
    contextualizer = ChunkContextualizer(client)  # type: ignore[arg-type]
    chunks = [_chunk("doc.md::chunk0", document_id="doc.md"), _chunk("doc.md::chunk1", document_id="doc.md")]

    results = contextualizer.contextualize(chunks, [_document("doc.md")])

    assert [c.context for c in results] == [None, "a real context"]


def test_empty_chunk_list_makes_no_llm_calls() -> None:
    client = _FakeLLMClient()
    contextualizer = ChunkContextualizer(client)  # type: ignore[arg-type]

    assert contextualizer.contextualize([], [_document()]) == []
    assert client.calls == []
