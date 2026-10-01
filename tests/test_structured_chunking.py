"""Tests for `StructuredChunker` (chunking plan Phase 5)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from rag.chunking.chunkers import get_chunker
from rag.chunking.models import Chunk
from rag.chunking.structured import StructuredChunker, parse_blocks
from rag.config.settings import ChunkingConfig, StructuredChunkingConfig
from rag.index_report import build_report, chunk_table_start
from rag.ingestion.models import Document


def _doc(text: str, doc_id: str = "d.md", **metadata) -> Document:
    return Document(id=doc_id, text=text, source=Path(doc_id), doc_type="markdown", metadata=metadata)


def _chunker(chunk_size: int = 300, chunk_overlap: int = 0, **kwargs) -> StructuredChunker:
    kwargs.setdefault("min_chars", 0)
    return StructuredChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)


def _assert_lossless(document: Document, chunks: list[Chunk]) -> None:
    """Every chunk is a verbatim slice (after any repeated header), and no text is dropped."""

    covered = [False] * len(document.text)
    for chunk in chunks:
        start, end = chunk.metadata["char_start"], chunk.metadata["char_end"]
        assert chunk.text.endswith(document.text[start:end])
        covered[start:end] = [True] * (end - start)
    lost = [ch for ch, hit in zip(document.text, covered) if not hit and not ch.isspace()]
    assert not lost, "".join(lost)


def _table(rows: int, *, label: str = "Item") -> str:
    lines = ["| | 2025 | 2024 |", "|---|---|---|"]
    lines += [f"| {label} {i} | {i},000 | {i},500 |" for i in range(rows)]
    return "\n".join(lines)


# -- parsing -----------------------------------------------------------------


def test_parse_blocks_finds_headings_tables_and_paragraphs() -> None:
    text = "# Title\n\nIntro line one\nline two.\n\n## Section\n\n" + _table(2) + "\n\nAfter."
    blocks = parse_blocks(text)

    assert [b.kind for b in blocks] == ["heading", "paragraph", "heading", "table", "paragraph"]
    assert [b.level for b in blocks if b.kind == "heading"] == [1, 2]
    table = blocks[3]
    assert text[table.start : table.header_end] == "| | 2025 | 2024 |\n|---|---|---|"
    assert [text[s:e] for s, e in table.rows] == ["| Item 0 | 0,000 | 0,500 |", "| Item 1 | 1,000 | 1,500 |"]
    assert text[blocks[1].start : blocks[1].end] == "Intro line one\nline two."


def test_blank_lines_between_rows_do_not_end_a_table() -> None:
    # The `edgar` fetcher's format: no separator, a blank line after every row.
    text = "| Segment | 2024\n\n| Americas | 167\n\n| Europe | 101\n\nAfter."
    blocks = parse_blocks(text)

    assert [b.kind for b in blocks] == ["table", "paragraph"]
    assert text[blocks[0].start : blocks[0].header_end] == "| Segment | 2024"
    assert len(blocks[0].rows) == 2


# -- packing -----------------------------------------------------------------


def test_short_document_is_one_chunk_with_offsets_and_metadata() -> None:
    text = "# Apple 10-K\n\n## Revenue\n\nSales rose."
    [chunk] = _chunker().chunk([_doc(text, title="T")])

    assert chunk.text == text
    assert chunk.metadata["char_start"] == 0 and chunk.metadata["char_end"] == len(text)
    assert chunk.metadata["title"] == "T"
    assert chunk.metadata["section_path"] == "Revenue"  # the `#` document title is left out
    assert chunk.metadata["block_types"] == "heading,paragraph"


def test_a_split_level_heading_starts_a_chunk_and_deeper_ones_pack() -> None:
    text = "## A\n\nAlpha text.\n\n### A1\n\nMore alpha.\n\n## B\n\nBeta text."
    chunks = _chunker(split_level=2).chunk([_doc(text)])

    assert [c.text for c in chunks] == ["## A\n\nAlpha text.\n\n### A1\n\nMore alpha.", "## B\n\nBeta text."]
    assert [c.metadata["section_path"] for c in chunks] == ["A", "B"]

    deeper = _chunker(split_level=3).chunk([_doc(text)])
    assert len(deeper) == 3
    assert deeper[1].metadata["section_path"] == "A > A1"


def test_a_stub_section_joins_its_next_sibling() -> None:
    text = "## Tiny\n\nShort.\n\n## Real\n\n" + "Words here. " * 10
    chunks = _chunker(min_chars=50).chunk([_doc(text)])

    assert len(chunks) == 1
    assert chunks[0].text.startswith("## Tiny")


def test_a_heading_never_ends_a_chunk() -> None:
    first = "x " * 120
    second = "y " * 120
    text = f"## A\n\n{first.strip()}\n\n### Sub\n\n{second.strip()}"
    chunks = _chunker(chunk_size=300, split_level=2).chunk([_doc(text)])

    assert len(chunks) == 2
    assert not chunks[0].text.rstrip().endswith("### Sub")
    assert chunks[1].text.startswith("### Sub")
    _assert_lossless(_doc(text), chunks)


def test_a_short_lead_in_moves_with_the_table_it_introduces() -> None:
    body = "Prose. " * 30
    text = f"## Income\n\n{body.strip()}\n\n## Statement\n\n(in millions)\n\n{_table(12)}"
    chunks = _chunker(chunk_size=400, min_chars=60).chunk([_doc(text)])

    lead = next(c for c in chunks if "(in millions)" in c.text)
    assert "| Item 0 |" in lead.text  # not a chunk of its own
    _assert_lossless(_doc(text), chunks)


def test_a_table_that_fits_stays_whole() -> None:
    text = "Before.\n\n" + _table(3) + "\n\nAfter."
    chunks = _chunker(chunk_size=len(_table(3)) + 5).chunk([_doc(text)])

    assert any(c.text == _table(3) for c in chunks)


def test_an_oversized_table_splits_by_rows_and_repeats_its_header() -> None:
    table = _table(30)
    document = _doc("## Results\n\n" + table)
    chunks = _chunker(chunk_size=250).chunk([document])

    header = "| | 2025 | 2024 |\n|---|---|---|"
    assert len(chunks) > 2
    for chunk in chunks:
        assert len(chunk.text) <= 250
        rows = [line for line in chunk.text.split("\n") if line.startswith("|")]
        assert "\n".join(rows[:2]) == header
        # Rows are never cut: every table line is complete.
        assert all(line.endswith("|") for line in rows)
    assert all(f"| Item {i} |" in "".join(c.text for c in chunks) for i in range(30))
    _assert_lossless(document, chunks)
    report = build_report(
        corpus="t", chunking={}, documents=[document], chunks=chunks, min_chars=0, max_chars=250
    )
    assert report.mid_table_starts == 0


def test_an_oversized_row_is_split_by_words_under_the_header() -> None:
    long_row = "| Pipeline " + " ".join(f"drug{i}" for i in range(80)) + " |"
    table = "| Phase 2 | |\n|---|---|\n" + long_row + "\n| Short | row |"
    document = _doc(table)
    chunks = _chunker(chunk_size=200).chunk([document])

    assert len(chunks) > 2
    assert all(c.text.startswith("| Phase 2 | |\n|---|---|") for c in chunks)
    _assert_lossless(document, chunks)


def test_an_oversized_paragraph_splits_at_sentences_with_overlap() -> None:
    paragraph = " ".join(f"Sentence number {i} says something." for i in range(20))
    document = _doc(paragraph)
    chunks = _chunker(chunk_size=200, chunk_overlap=40).chunk([document])

    assert len(chunks) > 2
    assert all(len(c.text) <= 200 for c in chunks)
    assert all(c.text.endswith(".") for c in chunks)  # cut at sentence ends
    for prev, nxt in zip(chunks, chunks[1:]):
        assert nxt.metadata["char_start"] < prev.metadata["char_end"]  # overlap
        assert nxt.metadata["char_end"] > prev.metadata["char_end"]  # and progress
    _assert_lossless(document, chunks)


def test_a_sentence_longer_than_a_chunk_falls_back_to_words() -> None:
    document = _doc(" ".join(f"w{i}" for i in range(200)))
    chunks = _chunker(chunk_size=100, chunk_overlap=20).chunk([document])

    assert all(len(c.text) <= 100 for c in chunks)
    assert all(not c.text.startswith(" ") for c in chunks)
    _assert_lossless(document, chunks)


def test_prose_overlap_is_off_by_default_and_optional() -> None:
    paragraphs = [f"Paragraph {i} " + "text " * 25 for i in range(6)]
    document = _doc("## S\n\n" + "\n\n".join(p.strip() for p in paragraphs))

    plain = _chunker(chunk_size=300, chunk_overlap=60).chunk([document])
    for prev, nxt in zip(plain, plain[1:]):
        assert nxt.metadata["char_start"] >= prev.metadata["char_end"]

    overlapped = _chunker(chunk_size=300, chunk_overlap=60, prose_overlap=True).chunk([document])
    assert all(len(c.text) <= 300 for c in overlapped)  # overlap only fills spare room
    assert any(
        nxt.metadata["char_start"] < prev.metadata["char_end"] for prev, nxt in zip(overlapped, overlapped[1:])
    )
    _assert_lossless(document, overlapped)


def test_no_overlap_across_a_table_edge() -> None:
    text = "## S\n\n" + ("Lead text. " * 20).strip() + "\n\n" + _table(8)
    chunks = _chunker(chunk_size=260, chunk_overlap=60, prose_overlap=True).chunk([_doc(text)])

    for prev, nxt in zip(chunks, chunks[1:]):
        assert nxt.metadata["char_start"] >= prev.metadata["char_end"]


def test_a_final_stub_folds_into_its_predecessor() -> None:
    text = "## A\n\n" + ("Body text. " * 15).strip() + "\n\n## B\n\nEnd."
    chunks = _chunker(chunk_size=400, min_chars=40).chunk([_doc(text)])

    assert len(chunks) == 1


def test_length_function_is_injectable() -> None:
    words = lambda s: len(s.split())  # noqa: E731
    document = _doc(" ".join(f"Word{i}." for i in range(50)))
    chunks = StructuredChunker(chunk_size=10, chunk_overlap=0, min_chars=0, length=words).chunk([document])

    assert all(len(c.text.split()) <= 10 for c in chunks)
    assert len(chunks) == 5


def test_header_is_document_level_and_empty_documents_yield_nothing() -> None:
    chunker = _chunker(chunk_size=100, header_template="{company} {form}")
    chunks = chunker.chunk([_doc("## A\n\n" + "x " * 80, company="Apple", form="10-K"), _doc("  \n", "e.md")])

    assert len(chunks) > 1
    assert {c.header for c in chunks} == {"Apple 10-K"}
    assert all(c.document_id == "d.md" for c in chunks)


# -- config and factory --------------------------------------------------------


def test_factory_builds_the_structured_chunker_from_config() -> None:
    config = ChunkingConfig(
        strategy="structured",
        chunk_size=800,
        chunk_overlap=100,
        structured=StructuredChunkingConfig(split_level=3, min_chars=150, prose_overlap=True),
    )
    chunker = get_chunker(config)

    assert isinstance(chunker, StructuredChunker)
    assert (chunker.chunk_size, chunker.chunk_overlap) == (800, 100)
    assert (chunker.split_level, chunker.min_chars, chunker.prose_overlap) == (3, 150, True)


def test_config_rejects_min_chars_at_or_above_chunk_size() -> None:
    with pytest.raises(ValidationError, match="min_chars"):
        ChunkingConfig(strategy="structured", chunk_size=100, structured=StructuredChunkingConfig(min_chars=100))
    # Irrelevant to other strategies, so not checked for them.
    ChunkingConfig(strategy="fixed", chunk_size=100, structured=StructuredChunkingConfig(min_chars=100))


# -- index-report --------------------------------------------------------------


def test_index_report_credits_a_repeated_header() -> None:
    text = _table(4)
    later_row = text.index("| Item 2")
    header = "| | 2025 | 2024 |\n|---|---|---|"

    def chunk(body: str) -> Chunk:
        return Chunk(id="c", text=body, document_id="d.md", source=Path("d.md"), doc_type="markdown",
                     metadata={"char_start": later_row})

    assert chunk_table_start(text, chunk(text[later_row:])) == "later_row"
    assert chunk_table_start(text, chunk(f"{header}\n{text[later_row:]}")) is None


def test_index_report_credits_a_repeated_header_in_the_edgar_format() -> None:
    text = "Intro.\n\n| Segment | 2024\n\n| Americas | 167\n\n| Europe | 101"
    later_row = text.index("| Europe")
    chunk = Chunk(id="c", text="| Segment | 2024\n| Europe | 101", document_id="d.md", source=Path("d.md"),
                  doc_type="markdown", metadata={"char_start": later_row})

    assert chunk_table_start(text, chunk) is None
