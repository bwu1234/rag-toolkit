"""Typed data model for chunks, plus the Chunker interface.

A `Chunk` is the atomic unit that gets embedded and retrieved. It carries
enough provenance (document id, source path, character offsets, and copies of
useful document metadata like title/page) that a citation can always be traced
back to an exact span of an exact source file.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from rag.ingestion.models import Document


@dataclass(frozen=True)
class Chunk:
    """A contiguous span of text from one `Document`, ready for embedding.

    Attributes:
        id: Stable, human-readable identifier — `"<document-id>::chunk<index>"`,
            e.g. `"architecture_overview.md::chunk2"`. Readable ids make it easy
            to trace a retrieved chunk back to its source while debugging or
            authoring an eval set.
        text: The chunk's text (chunking operates on already-cleaned text —
            see `rag.ingestion.cleaners.clean_documents`).
        document_id: The `Document.id` this chunk was extracted from.
        source: Absolute path to the originating file (copied from the document
            for convenience — avoids a join back to the document at query time).
        doc_type: Copied from the originating `Document` (e.g. "pdf", "markdown").
        metadata: Chunk-specific extras (`chunk_index`, `char_start`, `char_end`)
            merged with relevant document metadata (`title`, `page`, ...), so a
            chunk is self-describing without needing its parent `Document`.
        context: A generated sentence or two situating this chunk in its parent
            document, or `None` when contextual chunking is off (the default).
            See `rag.chunking.contextualizer` for what it's for; `text` is never
            modified, so a chunk with context still cites exactly the span the
            chunker produced.
        header: A deterministic line naming the chunk's document, rendered from
            its metadata by `chunking.header.template`, or `None` when that is
            off or the document lacks a field it names. Like `context`, it is
            indexed but never part of `text`.
    """

    id: str
    text: str
    document_id: str
    source: Path
    doc_type: str
    metadata: dict[str, Any] = field(default_factory=dict)
    context: str | None = None
    header: str | None = None

    @property
    def index_text(self) -> str:
        """The text that should be *indexed* for this chunk: header, context, then the chunk.

        Kept as a derived property rather than folded into `text` so the two
        stay separable at every layer: retrieval matches against the enriched
        string, while citations, previews, and character offsets keep pointing
        at the real span of the real document. Falls back to `text` verbatim
        when there's neither, so nothing downstream needs to branch.
        """

        return join_index_text(self.header, self.context, self.text)


def join_index_text(header: str | None, context: str | None, text: str) -> str:
    """Header, context and text as the one string an index sees, skipping empty parts.

    Shared by `Chunk`, `ScoredChunk` and the BM25 index's persisted records, so
    the embedded and keyword-indexed strings can't drift apart.
    """

    return "\n\n".join(part for part in (header, context, text) if part)


def content_hash(text: str) -> str:
    """The hash stored as a chunk's `content_hash`, which incremental indexing compares."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_chunk_id(document_id: str, index: int) -> str:
    """Build a stable, human-readable chunk id from a document id and chunk index."""

    return f"{document_id}::chunk{index}"


class Chunker(ABC):
    """Interface for splitting documents into retrievable chunks.

    To add a new strategy: subclass `Chunker`, implement `chunk`, and register
    it in `rag.chunking.chunkers.get_chunker`'s factory based on
    `ChunkingConfig.strategy`. Pipeline code only depends on this interface.
    """

    @abstractmethod
    def chunk(self, documents: Iterable[Document]) -> list[Chunk]:
        """Split `documents` into `Chunk`s, preserving order within each document.

        Implementations should assume `Document.text` is already cleaned —
        cleaning is a separate ingestion-time concern (see
        `rag.ingestion.cleaners.clean_documents`) so chunkers can focus purely
        on splitting strategy.
        """
        raise NotImplementedError
