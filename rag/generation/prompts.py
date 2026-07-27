"""Prompt templates for retrieval-augmented generation.

Kept as plain functions returning strings (not a templating-library
dependency) -- the format is simple enough that string formatting is more
readable than it would be wrapped in a template DSL, and it keeps this module
trivially testable.
"""

from __future__ import annotations

from rag.vectorstore.base import ScoredChunk

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using only the "
    "numbered context passages provided below. "
    "Cite the passages you rely on inline using their bracketed numbers, "
    "e.g. [1] or [2][3]. "
    "If the passages don't contain enough information to answer, say so "
    "plainly instead of guessing or using outside knowledge."
)


def _citation_label(chunk: ScoredChunk) -> str:
    """Build a human-readable source label, including a page number if known."""

    page = chunk.metadata.get("page")
    return f"{chunk.document_id} (p.{page})" if page is not None else chunk.document_id


def build_rag_prompt(query: str, chunks: list[ScoredChunk]) -> str:
    """Render the retrieved chunks and question into a single user-turn prompt.

    Each chunk becomes a numbered "Passage [n]" block labeled with its source
    (and page, if known) so the model can ground its answer in -- and cite --
    specific passages. Numbering here is the single source of truth for the
    `[n]` markers the system prompt asks the model to use, and for mapping a
    cited number back to a `Citation` in `ChatService`.
    """

    passages = "\n\n".join(
        f"Passage [{index}] (source: {_citation_label(chunk)}):\n{chunk.text}"
        for index, chunk in enumerate(chunks, start=1)
    )

    return (
        f"Context passages:\n\n{passages}\n\n"
        f"Question: {query}\n\n"
        "Answer the question using only the passages above, citing them by "
        "number as you go."
    )
