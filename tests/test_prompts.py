"""Tests for RAG prompt construction."""

from __future__ import annotations

from pathlib import Path

from rag.generation.prompts import REGROUND_SYSTEM_PROMPT, SYSTEM_PROMPT, build_rag_prompt
from rag.vectorstore.base import ScoredChunk


def _scored(chunk_id: str, text: str, *, document_id: str = "doc.md", page: int | None = None) -> ScoredChunk:
    metadata = {"page": page} if page is not None else {}
    return ScoredChunk(
        chunk_id=chunk_id,
        text=text,
        document_id=document_id,
        source=Path(document_id),
        doc_type="markdown",
        score=0.5,
        metadata=metadata,
    )


def test_system_prompt_instructs_grounding_and_citation() -> None:
    assert "context" in SYSTEM_PROMPT.lower() or "passages" in SYSTEM_PROMPT.lower()
    assert "cite" in SYSTEM_PROMPT.lower()


def test_build_rag_prompt_numbers_passages_and_includes_question() -> None:
    chunks = [_scored("a", "Refunds are accepted within 30 days."), _scored("b", "Shipping takes 3-5 business days.")]

    prompt = build_rag_prompt("What is the refund policy?", chunks)

    assert "Passage [1]" in prompt
    assert "Passage [2]" in prompt
    assert "Refunds are accepted within 30 days." in prompt
    assert "Shipping takes 3-5 business days." in prompt
    assert "Question: What is the refund policy?" in prompt
    # Passage 1 must precede passage 2 -- numbering is the source of truth for citations.
    assert prompt.index("Passage [1]") < prompt.index("Passage [2]")


def test_build_rag_prompt_labels_sources_with_page_when_known() -> None:
    chunks = [_scored("a", "text", document_id="handbook.pdf#page=3", page=3)]

    prompt = build_rag_prompt("query", chunks)

    assert "handbook.pdf#page=3 (p.3)" in prompt


def test_build_rag_prompt_omits_page_label_when_unknown() -> None:
    chunks = [_scored("a", "text", document_id="readme.md")]

    prompt = build_rag_prompt("query", chunks)

    assert "readme.md" in prompt
    assert "(p." not in prompt


def test_build_rag_prompt_labels_chunk_context_separately_from_its_text() -> None:
    chunk = _scored("a", "The limit is 1,000 requests per minute.")
    contextual = ScoredChunk(
        chunk_id=chunk.chunk_id,
        text=chunk.text,
        document_id=chunk.document_id,
        source=chunk.source,
        doc_type=chunk.doc_type,
        score=chunk.score,
        context="ACS API rate limiting.",
    )

    prompt = build_rag_prompt("What is the rate limit?", [contextual])

    # Labeled rather than run together: the context is model-generated, and the
    # answering model shouldn't be able to cite it as though it were corpus text.
    assert "Context: ACS API rate limiting." in prompt
    assert prompt.index("Context: ") < prompt.index("The limit is 1,000")


def test_build_rag_prompt_omits_the_context_line_when_there_is_no_context() -> None:
    prompt = build_rag_prompt("q", [_scored("a", "Refunds are accepted within 30 days.")])

    assert "Context:" not in prompt


def test_reground_system_prompt_states_the_previous_attempt_failed() -> None:
    assert "previous attempt" in REGROUND_SYSTEM_PROMPT
    assert REGROUND_SYSTEM_PROMPT != SYSTEM_PROMPT
    assert REGROUND_SYSTEM_PROMPT.startswith(SYSTEM_PROMPT), "it must still carry the base grounding rules"
