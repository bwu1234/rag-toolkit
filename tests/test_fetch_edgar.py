"""Tests for the EDGAR fetcher's front matter, offline backfill (chunking plan,
Phase 2) and pinned raw-HTML cache (Phase 4).

Network-free: the cache tests swap ``EdgarClient`` for a fake.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import fetch_edgar  # noqa: E402
from fetch_edgar import (  # noqa: E402
    FilingRef,
    backfill_front_matter,
    cache_raw_filings,
    extract_mda,
    html_to_text,
    load_raw_index,
    pinned_accessions,
    raw_html_path,
    render_document,
    selection_drift,
    verify_raw_cache,
)

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


# Enough numeric prose for `extract_mda` to accept as a 10-K MD&A section.
_HTML = (
    "<html><body><p>Item 7. Management's Discussion and Analysis of Financial Condition</p>"
    + "<p>Net sales rose 8% to $94 billion, driven by strong demand across regions.</p>" * 250
    + "<p>Item 7A. Quantitative and Qualitative Disclosures About Market Risk</p></body></html>"
)


class _FakeClient:
    """Stands in for `EdgarClient`, recording every request it would have made."""

    requests: list[str] = []

    def __init__(self, user_agent: str) -> None:
        pass

    def __enter__(self) -> "_FakeClient":
        return self

    def __exit__(self, *exc: object) -> None:
        pass

    def ticker_to_cik(self) -> dict[str, tuple[int, str]]:
        self.requests.append("tickers")
        return {"AAPL": (_FILING.cik, _FILING.company)}

    def recent_submissions(self, cik: int) -> dict[str, list]:
        self.requests.append(f"submissions {cik}")
        # A newer filing listed first: the pin must pick by accession, not recency.
        return {
            "accessionNumber": ["0000320193-25-000999", _FILING.accession],
            "form": ["10-K", _FILING.form],
            "filingDate": ["2025-10-31", _FILING.filing_date],
            "reportDate": ["2025-09-27", _FILING.report_date],
            "primaryDocument": ["aapl-20250927.htm", _FILING.document],
        }

    def fetch_document(self, filing: FilingRef) -> str:
        self.requests.append(filing.document)
        return _HTML


@pytest.fixture
def fake_client(monkeypatch: pytest.MonkeyPatch) -> type[_FakeClient]:
    _FakeClient.requests = []
    monkeypatch.setattr(fetch_edgar, "EdgarClient", _FakeClient)
    return _FakeClient


def _corpus_with_one_filing(tmp_path: Path) -> Path:
    out_dir = tmp_path / "documents"
    out_dir.mkdir()
    body = extract_mda(html_to_text(_HTML), _FILING.form)
    assert body is not None
    (out_dir / _FILING.filename).write_text(render_document(_FILING, body), encoding="utf-8")
    return out_dir


def test_pinned_accessions_come_from_front_matter(tmp_path: Path) -> None:
    out_dir = _corpus_with_one_filing(tmp_path)

    assert pinned_accessions(out_dir) == {"AAPL": {_FILING.accession: _FILING.filename}}


def test_pinned_accessions_refuse_a_file_without_front_matter(tmp_path: Path) -> None:
    (tmp_path / "old.md").write_text("# Apple Inc. (AAPL) 10-K -- period ended 2024-09-28\n", encoding="utf-8")

    with pytest.raises(ValueError, match="backfill-front-matter"):
        pinned_accessions(tmp_path)


def test_cache_fetches_the_pinned_filing_not_the_newest(tmp_path: Path, fake_client: type[_FakeClient]) -> None:
    out_dir = _corpus_with_one_filing(tmp_path)
    raw_dir = tmp_path / "raw"

    assert cache_raw_filings(out_dir, raw_dir, "ua") == 1

    assert fake_client.requests == ["tickers", f"submissions {_FILING.cik}", _FILING.document]
    assert load_raw_index(raw_dir) == {_FILING.accession: _FILING}
    assert raw_html_path(raw_dir, _FILING.accession).read_text(encoding="utf-8") == _HTML


def test_cache_makes_no_request_once_everything_is_cached(tmp_path: Path, fake_client: type[_FakeClient]) -> None:
    out_dir = _corpus_with_one_filing(tmp_path)
    raw_dir = tmp_path / "raw"
    cache_raw_filings(out_dir, raw_dir, "ua")
    fake_client.requests.clear()

    assert cache_raw_filings(out_dir, raw_dir, "ua") == 0
    assert fake_client.requests == []


def test_cache_refuses_an_accession_that_resolves_to_another_period(
    tmp_path: Path, fake_client: type[_FakeClient]
) -> None:
    out_dir = _corpus_with_one_filing(tmp_path)
    (out_dir / _FILING.filename).rename(out_dir / "AAPL_10-K_2023-09-30.md")

    with pytest.raises(ValueError, match="resolves to"):
        cache_raw_filings(out_dir, tmp_path / "raw", "ua")


def test_verify_finds_the_document_body_in_the_cached_filing(tmp_path: Path, fake_client: type[_FakeClient]) -> None:
    out_dir = _corpus_with_one_filing(tmp_path)
    raw_dir = tmp_path / "raw"
    cache_raw_filings(out_dir, raw_dir, "ua")

    assert verify_raw_cache(out_dir, raw_dir) == []
    assert selection_drift(out_dir, raw_dir) == []

    document = out_dir / _FILING.filename
    document.write_text(document.read_text(encoding="utf-8").replace("8%", "9%", 1), encoding="utf-8")
    assert verify_raw_cache(out_dir, raw_dir) == [_FILING.filename]


def test_verify_reports_an_uncached_filing(tmp_path: Path) -> None:
    out_dir = _corpus_with_one_filing(tmp_path)

    assert verify_raw_cache(out_dir, tmp_path / "raw") == [_FILING.filename]


def test_a_body_todays_selection_would_not_pick_still_verifies_and_is_reported_as_drift(
    tmp_path: Path, fake_client: type[_FakeClient]
) -> None:
    # The corpus as fetched: a section today's rules reject (it's a subspan
    # with no "Item 7" heading), which is what 10 of the 61 EDGAR documents are.
    out_dir = tmp_path / "documents"
    out_dir.mkdir()
    body = extract_mda(html_to_text(_HTML), _FILING.form)
    assert body is not None
    older_selection = body.split("\n", 1)[1]
    (out_dir / _FILING.filename).write_text(render_document(_FILING, older_selection), encoding="utf-8")
    raw_dir = tmp_path / "raw"
    cache_raw_filings(out_dir, raw_dir, "ua")

    assert verify_raw_cache(out_dir, raw_dir) == []
    assert selection_drift(out_dir, raw_dir) == [_FILING.filename]
