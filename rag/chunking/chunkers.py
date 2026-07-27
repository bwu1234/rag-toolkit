"""Chunking strategies plus a config-driven factory.

Adding a new strategy: write a `Chunker` subclass and add a branch to
`get_chunker`. Pipeline code calls `get_chunker(config.chunking)` and never
references a concrete class.
"""

from __future__ import annotations

import logging
from typing import Iterable

from rag.chunking.models import Chunk, Chunker, make_chunk_id
from rag.config.settings import ChunkingConfig
from rag.ingestion.models import Document

logger = logging.getLogger(__name__)

# When a chunk boundary lands mid-word, search backwards up to this many
# characters for a whitespace character to break on instead. Small enough to
# keep chunk sizes close to the configured target, large enough to find a
# break in normal prose (average English word length is ~5 chars).
_BOUNDARY_SEARCH_WINDOW = 50


class FixedSizeChunker(Chunker):
    """Splits each document's text into overlapping, roughly-fixed-size windows.

    Character-based rather than token-based: it needs no tokenizer dependency
    and is good enough to get the end-to-end pipeline working. The tradeoff is
    that chunk sizes don't map precisely onto an embedding/LLM's token budget —
    a token-aware chunker is a natural future upgrade behind the same
    `Chunker` interface, selected via `chunking.strategy` in config.

    Boundaries snap to the nearest preceding whitespace (within
    `_BOUNDARY_SEARCH_WINDOW` characters) so words aren't split mid-token,
    which would otherwise hurt embedding quality.
    """

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be smaller than chunk_size ({chunk_size}), "
                "otherwise chunking never advances"
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, documents: Iterable[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        document_count = 0
        for document in documents:
            document_count += 1
            chunks.extend(self._chunk_one(document))
        logger.info("Chunked %d document(s) into %d chunk(s)", document_count, len(chunks))
        return chunks

    def _chunk_one(self, document: Document) -> list[Chunk]:
        text = document.text
        if not text:
            return []

        # Carry forward metadata that helps a citation describe its source
        # without a join back to the parent Document (title, page, ...).
        inherited_metadata = {
            key: value for key, value in document.metadata.items() if key in ("title", "page", "page_count")
        }

        spans = list(self._spans(text))
        result: list[Chunk] = []
        for index, (start, end) in enumerate(spans):
            chunk_text = text[start:end].strip()
            if not chunk_text:
                continue
            result.append(
                Chunk(
                    id=make_chunk_id(document.id, index),
                    text=chunk_text,
                    document_id=document.id,
                    source=document.source,
                    doc_type=document.doc_type,
                    metadata={
                        **inherited_metadata,
                        "chunk_index": index,
                        "char_start": start,
                        "char_end": end,
                    },
                )
            )
        return result

    def _spans(self, text: str) -> Iterable[tuple[int, int]]:
        """Yield (start, end) character spans covering `text` with overlap.

        Short-circuits to a single span when the whole text already fits —
        avoids pointless overlap bookkeeping for short documents (e.g. FAQ
        entries, short PDF pages).
        """

        n = len(text)
        if n <= self.chunk_size:
            yield (0, n)
            return

        start = 0
        while start < n:
            end = min(start + self.chunk_size, n)
            if end < n:
                end = self._snap_backward_to_boundary(text, end)
            yield (start, end)

            if end >= n:
                return

            # The overlap means the next chunk starts before this one ends.
            # That start point can land mid-word -- snap it FORWARD to the next
            # whitespace (bounded by `end`) so the next chunk also begins on a
            # word boundary. Without this, overlap silently produces chunks
            # that start with a fragment like "8" from "word008".
            next_start = end - self.chunk_overlap
            if next_start <= start:
                start = end
            else:
                snapped = self._snap_forward_to_boundary(text, next_start, limit=end)
                start = snapped if snapped > start else end

    @staticmethod
    def _snap_backward_to_boundary(text: str, target: int) -> int:
        """Move `target` back to just after the nearest preceding whitespace, if one is nearby."""

        window_start = max(0, target - _BOUNDARY_SEARCH_WINDOW)
        boundary = max(text.rfind(" ", window_start, target), text.rfind("\n", window_start, target))
        return boundary + 1 if boundary != -1 else target

    @staticmethod
    def _snap_forward_to_boundary(text: str, target: int, *, limit: int) -> int:
        """Move `target` forward to just after the nearest following whitespace, if one is nearby.

        Search is bounded by both `_BOUNDARY_SEARCH_WINDOW` and `limit` (the
        current chunk's end) so we never snap past the content we're about to
        emit, which would silently skip text.
        """

        window_end = min(limit, target + _BOUNDARY_SEARCH_WINDOW)
        space = text.find(" ", target, window_end)
        newline = text.find("\n", target, window_end)
        candidates = [c for c in (space, newline) if c != -1]
        return (min(candidates) + 1) if candidates else target


def get_chunker(config: ChunkingConfig) -> Chunker:
    """Instantiate the `Chunker` selected by `config.strategy`.

    Single point of truth for "string in config" -> "concrete implementation",
    so adding a strategy means adding one branch here, not hunting through
    pipeline code for hardcoded chunker construction.
    """

    if config.strategy == "fixed":
        return FixedSizeChunker(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)

    raise ValueError(f"Unknown chunking strategy: {config.strategy!r}")
