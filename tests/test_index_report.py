"""Tests for `rag.index_report`: the chunk statistics behind `rag.cli index-report`."""

from __future__ import annotations

from pathlib import Path

from rag.chunking.chunkers import FixedSizeChunker
from rag.chunking.models import Chunk
from rag.index_report import IndexState, build_report, format_report, table_start
from rag.ingestion.models import Document

# Rows separated by blank lines, as `scripts/fetch_edgar.py` writes them.
TABLE_DOC = (
    "Revenue by segment follows.\n\n"
    "| Segment | 2024 | 2023\n\n"
    "| Americas | 167,045 | 162,560\n\n"
    "| Europe | 101,328 | 94,294\n\n"
    "Revenue rose in both."
)


def _doc(text: str, doc_id: str = "d.md") -> Document:
    return Document(id=doc_id, text=text, source=Path(doc_id), doc_type="markdown")


def _chunk(chunk_id: str, text: str, *, document_id: str = "d.md", char_start: int = 0) -> Chunk:
    return Chunk(
        id=chunk_id, text=text, document_id=document_id, source=Path(document_id),
        doc_type="markdown", metadata={"char_start": char_start},
    )


def _report(documents: list[Document], chunks: list[Chunk], **overrides):
    kwargs = dict(corpus="test", chunking={"strategy": "fixed"}, documents=documents,
                  chunks=chunks, min_chars=10, max_chars=1000)
    kwargs.update(overrides)
    return build_report(**kwargs)


def test_prose_is_not_a_table_start() -> None:
    assert table_start(TABLE_DOC, 0) is None


def test_a_tables_first_row_is_not_mid_table() -> None:
    assert table_start(TABLE_DOC, TABLE_DOC.index("| Segment")) is None


def test_a_later_row_across_a_blank_line_is_mid_table() -> None:
    assert table_start(TABLE_DOC, TABLE_DOC.index("| Americas")) == "later_row"


def test_partway_through_a_row_is_mid_row() -> None:
    assert table_start(TABLE_DOC, TABLE_DOC.index("167,045")) == "mid_row"
    # Even partway through the header row: the chunk still lacks its start.
    assert table_start(TABLE_DOC, TABLE_DOC.index("2024")) == "mid_row"


def test_leading_whitespace_is_skipped_before_classifying() -> None:
    assert table_start(TABLE_DOC, TABLE_DOC.index("| Americas") - 2) == "later_row"


def test_the_line_after_a_table_is_not_mid_table() -> None:
    assert table_start(TABLE_DOC, TABLE_DOC.index("Revenue rose")) is None


def test_report_counts_mid_table_starts_from_real_chunker_output() -> None:
    document = _doc(TABLE_DOC)
    chunks = FixedSizeChunker(chunk_size=60, chunk_overlap=10).chunk([document])
    expected = [c for c in chunks if table_start(TABLE_DOC, c.metadata["char_start"])]

    report = _report([document], chunks)

    assert report.mid_table_starts == len(expected) > 0
    assert report.table_chunks == sum(1 for c in chunks if "|" in c.text)


def test_report_counts_sizes_floors_caps_empties_and_duplicates() -> None:
    documents = [_doc("x" * 50, "a.md"), _doc("y" * 50, "b.md"), _doc("", "empty.md")]
    chunks = [
        _chunk("a::0", "short", document_id="a.md"),
        _chunk("a::1", "same text here", document_id="a.md"),
        _chunk("b::0", "same text here", document_id="b.md"),
        _chunk("b::1", "z" * 30, document_id="b.md"),
    ]

    report = _report(documents, chunks, min_chars=10, max_chars=20)

    assert (report.documents, report.chunks) == (3, 4)
    assert report.under_floor == 1
    assert report.over_cap == 1
    assert report.empty_documents == ["empty.md"]
    assert (report.duplicate_groups, report.duplicate_chunks) == (1, 2)
    assert report.size_percentiles["min"] == 5
    assert report.size_percentiles["max"] == 30


def test_report_on_an_empty_corpus_does_not_fail() -> None:
    report = _report([], [])
    assert report.size_percentiles == {}
    assert "Chunks             0" in format_report(report)


def test_format_says_when_the_index_is_out_of_sync() -> None:
    state = IndexState(manifest=None, manifest_differences=[], vector_chunks=3, sparse_chunks=3,
                       missing=1, stale=0, changed=0)
    out = format_report(_report([], [], index=state))
    assert "1 missing" in out
    assert "built before manifests" in out
    assert "NO -- re-run" in out


def test_format_says_when_the_index_is_not_built() -> None:
    assert "Index: not built" in format_report(_report([], []))
