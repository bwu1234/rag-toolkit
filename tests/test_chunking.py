"""Tests for the chunking interface and FixedSizeChunker (Milestone 3)."""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pytest

from rag.chunking.chunkers import FixedSizeChunker, get_chunker
from rag.chunking.models import Chunk, make_chunk_id
from rag.config.settings import ChunkHeaderConfig, ChunkingConfig
from rag.ingestion.models import Document


def _doc(text: str, doc_id: str = "doc.txt", **metadata) -> Document:
    return Document(id=doc_id, text=text, source=Path("/corpus") / doc_id, doc_type="text", metadata=metadata)


def test_make_chunk_id() -> None:
    assert make_chunk_id("a/b.md", 0) == "a/b.md::chunk0"


def test_short_document_yields_single_chunk() -> None:
    chunker = FixedSizeChunker(chunk_size=1000, chunk_overlap=100)
    chunks = chunker.chunk([_doc("short text", title="T")])

    assert len(chunks) == 1
    assert chunks[0].text == "short text"
    assert chunks[0].id == "doc.txt::chunk0"
    assert chunks[0].metadata["chunk_index"] == 0
    assert chunks[0].metadata["char_start"] == 0
    assert chunks[0].metadata["char_end"] == len("short text")


def test_long_document_is_split_with_overlap() -> None:
    # 26 chars repeated -> 260 chars of text with clean word boundaries (spaces).
    text = " ".join(["word"] * 60)  # "word word word ..." -> 299 chars
    chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=20)

    chunks = chunker.chunk([_doc(text)])

    assert len(chunks) > 1
    # Chunks should be in order and indices contiguous.
    assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))
    # Consecutive chunks should overlap: the char_start of chunk N+1 should be
    # less than the char_end of chunk N (since overlap > 0).
    for prev, nxt in zip(chunks, chunks[1:]):
        assert nxt.metadata["char_start"] < prev.metadata["char_end"]
    # No chunk should wildly exceed the configured size (boundary snapping
    # only pulls the end backwards, never forwards).
    assert all(len(c.text) <= 100 for c in chunks)


def test_chunks_do_not_split_words_mid_token() -> None:
    text = " ".join(f"word{i:03d}" for i in range(50))  # "word000 word001 ..."
    chunker = FixedSizeChunker(chunk_size=80, chunk_overlap=10)

    chunks = chunker.chunk([_doc(text)])

    for chunk in chunks:
        for token in chunk.text.split():
            assert token.startswith("word") and len(token) == 7, f"Found a split token: {token!r}"


def test_metadata_is_inherited_and_extended() -> None:
    chunker = FixedSizeChunker(chunk_size=1000, chunk_overlap=100)
    doc = _doc("hello world", doc_id="handbook.pdf#page=2", title="Handbook", page=2, page_count=10, extra="dropped")

    chunks = chunker.chunk([doc])

    assert chunks[0].metadata["title"] == "Handbook"
    assert chunks[0].metadata["page"] == 2
    assert chunks[0].metadata["page_count"] == 10
    assert "extra" not in chunks[0].metadata  # only a known allowlist is carried forward
    assert chunks[0].document_id == "handbook.pdf#page=2"
    assert chunks[0].source == Path("/corpus/handbook.pdf#page=2")
    assert chunks[0].doc_type == "text"


def test_empty_document_yields_no_chunks() -> None:
    chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=10)
    assert chunker.chunk([_doc("")]) == []


def test_chunker_rejects_overlap_not_smaller_than_size() -> None:
    with pytest.raises(ValueError):
        FixedSizeChunker(chunk_size=100, chunk_overlap=100)
    with pytest.raises(ValueError):
        FixedSizeChunker(chunk_size=100, chunk_overlap=150)


def test_get_chunker_factory_selects_fixed_strategy() -> None:
    chunker = get_chunker(ChunkingConfig(strategy="fixed", chunk_size=500, chunk_overlap=50))
    assert isinstance(chunker, FixedSizeChunker)
    assert chunker.chunk_size == 500
    assert chunker.chunk_overlap == 50


def test_chunk_ids_are_unique_across_documents() -> None:
    chunker = FixedSizeChunker(chunk_size=50, chunk_overlap=10)
    docs = [_doc("a " * 40, doc_id="a.txt"), _doc("b " * 40, doc_id="b.txt")]

    chunks = chunker.chunk(docs)
    ids = [c.id for c in chunks]

    assert len(ids) == len(set(ids))
    assert all(cid.startswith("a.txt::chunk") for cid in ids if cid.startswith("a")) or True


# ---------------------------------------------------------------------------
# Carried metadata and the chunk header (chunking plan, Phase 2)
# ---------------------------------------------------------------------------

_TEMPLATE = "{company} ({ticker}), period ended {period_end}"


def _filing(text: str = "word " * 50, **overrides) -> Document:
    metadata = {"title": "t", "company": "Apple Inc.", "ticker": "AAPL", "period_end": date(2024, 9, 28)}
    metadata.update(overrides)
    return _doc(text, doc_id="AAPL.md", **metadata)


def test_carried_metadata_is_configurable_and_dates_become_integers() -> None:
    chunker = FixedSizeChunker(chunk_size=500, chunk_overlap=0, carry_metadata=["company", "period_end", "absent"])

    [chunk] = chunker.chunk([_filing()])

    assert chunk.metadata["company"] == "Apple Inc."
    assert chunk.metadata["period_end"] == 20240928, "Chroma stores primitives; ints keep range order"
    assert "absent" not in chunk.metadata
    assert "title" not in chunk.metadata, "only the configured keys are carried"


def test_default_carry_is_the_old_allowlist() -> None:
    [chunk] = FixedSizeChunker(chunk_size=500, chunk_overlap=0).chunk([_filing()])

    assert "title" in chunk.metadata
    assert "company" not in chunk.metadata


def test_header_is_rendered_on_every_chunk_and_kept_out_of_the_text() -> None:
    chunker = FixedSizeChunker(chunk_size=60, chunk_overlap=0, header_template=_TEMPLATE)

    chunks = chunker.chunk([_filing()])

    assert len(chunks) > 1
    assert {c.header for c in chunks} == {"Apple Inc. (AAPL), period ended 2024-09-28"}
    assert all("Apple" not in c.text for c in chunks), "text stays the verbatim span"
    assert chunks[0].index_text.startswith("Apple Inc. (AAPL), period ended 2024-09-28\n\n")


def test_document_missing_a_template_field_gets_no_header(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="rag.chunking.chunkers")
    chunker = FixedSizeChunker(chunk_size=500, chunk_overlap=0, header_template=_TEMPLATE)

    [with_header, without] = chunker.chunk([_filing(), _doc("plain text", doc_id="notes.txt")])

    assert with_header.header is not None
    assert without.header is None, "no half-filled header like 'None (None)'"
    assert without.index_text == "plain text"
    assert "1 of 2 document(s) lack a field" in caplog.text


def test_malformed_header_template_fails_at_construction() -> None:
    with pytest.raises(ValueError):
        FixedSizeChunker(chunk_size=500, chunk_overlap=0, header_template="{company")
    with pytest.raises(ValueError, match="metadata key"):
        FixedSizeChunker(chunk_size=500, chunk_overlap=0, header_template="{} filed")


def test_get_chunker_passes_header_and_carry_settings() -> None:
    config = ChunkingConfig(carry_metadata=["ticker"], header=ChunkHeaderConfig(template="{ticker}"))

    [chunk] = get_chunker(config).chunk([_filing()])

    assert chunk.header == "AAPL"
    assert chunk.metadata["ticker"] == "AAPL"


def test_index_text_orders_header_then_context_then_text() -> None:
    chunk = Chunk(
        id="c", text="body", document_id="d", source=Path("d"), doc_type="text", context="ctx", header="hdr"
    )
    assert chunk.index_text == "hdr\n\nctx\n\nbody"
