"""Chunking strategies plus a config-driven factory.

Adding a new strategy: write a `Chunker` subclass and add a branch to
`get_chunker`. Pipeline code calls `get_chunker(config.chunking)` and never
references a concrete class.
"""

from __future__ import annotations

import logging
from datetime import date
from string import Formatter
from typing import Any, Iterable

from rag.chunking.models import Chunk, Chunker, make_chunk_id
from rag.config.settings import DEFAULT_CARRY_METADATA, ChunkingConfig
from rag.ingestion.models import Document

logger = logging.getLogger(__name__)

# When a chunk boundary lands mid-word, search backwards up to this many
# characters for a whitespace character to break on instead. Small enough to
# keep chunk sizes close to the configured target, large enough to find a
# break in normal prose (average English word length is ~5 chars).
_BOUNDARY_SEARCH_WINDOW = 50


def carried_metadata(document: Document, keys: Iterable[str]) -> dict[str, Any]:
    """The document metadata a chunk keeps, in a form every index can store.

    Dates become `YYYYMMDD` integers: Chroma accepts only primitive values, and
    an integer keeps the ordering a period range filter needs, which a string
    like "2024-09-28" would only keep by accident of formatting.
    """

    carried: dict[str, Any] = {}
    for key in keys:
        if key not in document.metadata:
            continue
        value = document.metadata[key]
        carried[key] = int(value.strftime("%Y%m%d")) if isinstance(value, date) else value
    return carried


def template_fields(template: str) -> list[str]:
    """The metadata keys a header template names, in order.

    Raises:
        ValueError: the template is malformed, or uses positional (`{}`) or
            nested fields, which have no metadata key to fill them.
    """

    fields = [name for _literal, name, _spec, _conv in Formatter().parse(template) if name is not None]
    for name in fields:
        if not name.isidentifier():
            raise ValueError(f"chunking.header.template field {{{name}}} must be a metadata key name")
    return fields


def render_header(template: str, document: Document) -> str | None:
    """`template` filled from `document.metadata`, or `None` if any named field is missing.

    All or nothing, because a header like "Apple Inc. (None) 10-K" would be
    indexed as though "None" were part of the document's identity.
    """

    values = {}
    for name in template_fields(template):
        value = document.metadata.get(name)
        if value is None or value == "":
            return None
        values[name] = value.isoformat() if isinstance(value, date) else value
    return template.format_map(values)


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

    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        *,
        carry_metadata: Iterable[str] = DEFAULT_CARRY_METADATA,
        header_template: str | None = None,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be smaller than chunk_size ({chunk_size}), "
                "otherwise chunking never advances"
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.carry_metadata = tuple(carry_metadata)
        if header_template is not None:
            template_fields(header_template)  # fail at construction, not per document
        self.header_template = header_template

    def chunk(self, documents: Iterable[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        document_count = 0
        headerless = 0
        for document in documents:
            document_count += 1
            header = render_header(self.header_template, document) if self.header_template else None
            if self.header_template and header is None:
                headerless += 1
            chunks.extend(self._chunk_one(document, header))
        logger.info("Chunked %d document(s) into %d chunk(s)", document_count, len(chunks))
        if headerless:
            # Info, not a warning: it's expected for a corpus without the
            # metadata (`baseline`, the default active corpus). `index-report`
            # lists which documents, for when it's a corpus that should have it.
            logger.info(
                "%d of %d document(s) lack a field chunking.header.template names; "
                "their chunks have no header",
                headerless,
                document_count,
            )
        return chunks

    def _chunk_one(self, document: Document, header: str | None) -> list[Chunk]:
        text = document.text
        if not text:
            return []

        # Carry forward metadata that describes the source without a join back
        # to the parent Document (title, page, company, period, ...).
        inherited_metadata = carried_metadata(document, self.carry_metadata)

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
                    header=header,
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


class WholeDocumentChunker(Chunker):
    """One chunk per document, text unchanged (`chunking.strategy: none`).

    For corpora that arrive already split into retrieval units, such as BEIR
    passages scored by published relevance labels. Splitting one would let a
    single document occupy several ranks, which the benchmark's scoring does
    not allow. The text is not stripped, so the indexed string is exactly the
    loader's serialisation.

    A document whose text is empty or whitespace yields no chunk, as with
    `FixedSizeChunker`: there is nothing to embed or match. (FiQA has 38 such
    passages; Lucene's reference index skips them the same way.)
    """

    def __init__(
        self,
        *,
        carry_metadata: Iterable[str] = DEFAULT_CARRY_METADATA,
        header_template: str | None = None,
    ) -> None:
        self.carry_metadata = tuple(carry_metadata)
        if header_template is not None:
            template_fields(header_template)
        self.header_template = header_template

    def chunk(self, documents: Iterable[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        empty = 0
        for document in documents:
            if not document.text.strip():
                empty += 1
                continue
            header = render_header(self.header_template, document) if self.header_template else None
            chunks.append(
                Chunk(
                    id=make_chunk_id(document.id, 0),
                    text=document.text,
                    document_id=document.id,
                    source=document.source,
                    doc_type=document.doc_type,
                    metadata={
                        **carried_metadata(document, self.carry_metadata),
                        "chunk_index": 0,
                        "char_start": 0,
                        "char_end": len(document.text),
                    },
                    header=header,
                )
            )
        logger.info("Kept %d document(s) whole as chunks", len(chunks))
        if empty:
            logger.info("%d document(s) have no text and produced no chunk", empty)
        return chunks


def get_chunker(config: ChunkingConfig) -> Chunker:
    """Instantiate the `Chunker` selected by `config.strategy`.

    Single point of truth for "string in config" -> "concrete implementation",
    so adding a strategy means adding one branch here, not hunting through
    pipeline code for hardcoded chunker construction.
    """

    if config.strategy == "fixed":
        return FixedSizeChunker(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            carry_metadata=config.carry_metadata,
            header_template=config.header.template,
        )

    if config.strategy == "none":
        return WholeDocumentChunker(
            carry_metadata=config.carry_metadata,
            header_template=config.header.template,
        )

    raise ValueError(
        f"Unknown chunking strategy: {config.strategy!r}. "
        "Add a Chunker and register it here to support a new strategy."
    )
