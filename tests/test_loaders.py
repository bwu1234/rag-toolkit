"""Tests for format-specific loaders and the corpus directory walker (Milestone 2)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pypdf import PdfWriter

from rag.ingestion.loaders import (
    MarkdownLoader,
    PdfLoader,
    TextLoader,
    get_loader_for,
    load_corpus,
)
from rag.ingestion.models import make_document_id


def test_make_document_id_with_and_without_page() -> None:
    rel = Path("sub/dir/file.pdf")
    assert make_document_id(rel) == "sub/dir/file.pdf"
    assert make_document_id(rel, page=3) == "sub/dir/file.pdf#page=3"


def test_get_loader_for_dispatches_by_extension(tmp_path: Path) -> None:
    assert isinstance(get_loader_for(tmp_path / "a.PDF"), PdfLoader)  # case-insensitive
    assert isinstance(get_loader_for(tmp_path / "a.md"), MarkdownLoader)
    assert isinstance(get_loader_for(tmp_path / "a.markdown"), MarkdownLoader)
    assert isinstance(get_loader_for(tmp_path / "a.txt"), TextLoader)
    assert get_loader_for(tmp_path / "a.docx") is None


def test_text_loader_loads_single_document(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("hello world", encoding="utf-8")

    docs = TextLoader().load(path, corpus_root=tmp_path)

    assert len(docs) == 1
    assert docs[0].id == "notes.txt"
    assert docs[0].text == "hello world"
    assert docs[0].doc_type == "text"
    assert docs[0].metadata["title"] == "notes"


def test_markdown_loader_extracts_first_heading_as_title(tmp_path: Path) -> None:
    path = tmp_path / "guide.md"
    path.write_text("Intro line\n\n# My Guide Title\n\nBody text.", encoding="utf-8")

    docs = MarkdownLoader().load(path, corpus_root=tmp_path)

    assert len(docs) == 1
    assert docs[0].metadata["title"] == "My Guide Title"
    assert docs[0].doc_type == "markdown"


def test_markdown_loader_falls_back_to_filename_when_no_heading(tmp_path: Path) -> None:
    path = tmp_path / "untitled.md"
    path.write_text("Just a paragraph, no heading.", encoding="utf-8")

    docs = MarkdownLoader().load(path, corpus_root=tmp_path)

    assert docs[0].metadata["title"] == "untitled"


def _write_minimal_pdf(path: Path, num_pages: int = 2) -> None:
    """Write a minimal valid PDF with blank pages (no extractable text)."""
    writer = PdfWriter()
    for _ in range(num_pages):
        writer.add_blank_page(width=200, height=200)
    with path.open("wb") as f:
        writer.write(f)


def test_pdf_loader_emits_one_document_per_page(tmp_path: Path) -> None:
    path = tmp_path / "doc.pdf"
    _write_minimal_pdf(path, num_pages=3)

    docs = PdfLoader().load(path, corpus_root=tmp_path)

    assert len(docs) == 3
    assert [d.id for d in docs] == ["doc.pdf#page=1", "doc.pdf#page=2", "doc.pdf#page=3"]
    assert all(d.doc_type == "pdf" for d in docs)
    assert all(d.metadata["page_count"] == 3 for d in docs)
    assert [d.metadata["page"] for d in docs] == [1, 2, 3]


def test_load_corpus_dispatches_across_formats_and_skips_unsupported(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("# A\n\nbody", encoding="utf-8")
    (tmp_path / "b.txt").write_text("plain text", encoding="utf-8")
    (tmp_path / "c.unsupported").write_text("ignore me", encoding="utf-8")
    (tmp_path / ".hidden.txt").write_text("ignore me too", encoding="utf-8")
    _write_minimal_pdf(tmp_path / "d.pdf", num_pages=2)

    docs = load_corpus(tmp_path)

    ids = {d.id for d in docs}
    assert ids == {"a.md", "b.txt", "d.pdf#page=1", "d.pdf#page=2"}


def test_load_corpus_raises_for_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_corpus(tmp_path / "does_not_exist")
