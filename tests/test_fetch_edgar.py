"""Tests for the EDGAR fetcher's front matter and its offline backfill (chunking plan, Phase 2).

Network-free: only the rendering and file-rewriting helpers are exercised.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from fetch_edgar import FilingRef, backfill_front_matter, render_document  # noqa: E402

from rag.ingestion.loaders import MarkdownLoader  # noqa: E402

_FILING = FilingRef(
    ticker="AAPL",
    company="Apple Inc.",
    cik=320193,
    form="10-K",
    filing_date="2024-11-01",
    report_date="2024-09-28",
    accession="0000320193-24-000123",
    document="aapl-20240928.htm",
)
_METADATA = {
    "company": "Apple Inc.",
    "ticker": "AAPL",
    "form": "10-K",
    "period_end": date(2024, 9, 28),
    "filed": date(2024, 11, 1),
    "accession": "0000320193-24-000123",
}


def _load(path: Path) -> tuple[str, dict]:
    [doc] = MarkdownLoader().load(path, corpus_root=path.parent)
    return doc.text, doc.metadata


def _without_front_matter(rendered: str) -> str:
    return rendered.split("---\n", 2)[2]


def test_rendered_document_loads_with_typed_metadata(tmp_path: Path) -> None:
    path = tmp_path / _FILING.filename
    path.write_text(render_document(_FILING, "Revenue grew."), encoding="utf-8")

    text, metadata = _load(path)

    assert {k: metadata[k] for k in _METADATA} == _METADATA
    assert text.startswith("# Apple Inc. (AAPL) 10-K -- period ended 2024-09-28\n")


def test_backfill_adds_the_same_front_matter_a_fresh_fetch_writes(tmp_path: Path) -> None:
    rendered = render_document(_FILING, "Revenue grew.")
    old = tmp_path / _FILING.filename
    old.write_text(_without_front_matter(rendered), encoding="utf-8")
    text_before, _ = _load(old)

    assert backfill_front_matter(tmp_path) == 1

    assert old.read_text(encoding="utf-8") == rendered
    text_after, metadata = _load(old)
    assert text_after == text_before, "backfilling must not move any offset or eval span"
    assert metadata["period_end"] == date(2024, 9, 28)


def test_backfill_is_idempotent(tmp_path: Path) -> None:
    (tmp_path / _FILING.filename).write_text(render_document(_FILING, "Body."), encoding="utf-8")

    assert backfill_front_matter(tmp_path) == 0


def test_backfill_refuses_a_file_it_did_not_write(tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text("# Notes\n\nNothing about filings.\n", encoding="utf-8")

    with pytest.raises(ValueError, match="notes.md"):
        backfill_front_matter(tmp_path)
