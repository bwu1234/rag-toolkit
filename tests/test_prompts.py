"""Tests for RAG prompt construction."""

from __future__ import annotations

from pathlib import Path

from rag.generation.prompts import SYSTEM_PROMPT, build_rag_prompt
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
