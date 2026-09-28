"""Tests for document-level routing (chunking plan, Phase 3b).

The router's one job is to refuse to route unless BM25 and dense agree, since
a wrong route filters the answer out. Most tests here are about when it falls
back. The embedder is a bag-of-words fake over a fixed vocabulary, so the
dense ranking is predictable and can be made to disagree with BM25 on purpose.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from rag.chunking.models import Chunk
from rag.retrieval.document_router import (
    DocumentRouter,
    check_record_template,
    render_record,
    spell_date,
)
from rag.retrieval.sparse import BM25Index, IndexedDocument, tokenize

TEMPLATE = "{header}; period ended {period_end:date}"


class _BagOfWordsEmbedder:
    """One dimension per vocabulary word; counts how often a text uses it."""

    def __init__(self, vocabulary: list[str], query_as: str | None = None) -> None:
        self.vocabulary = vocabulary
        self.query_as = query_as
        self.document_batches = 0

    def _embed(self, text: str) -> list[float]:
        tokens = tokenize(text)
        return [float(tokens.count(word)) + 0.01 for word in self.vocabulary]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(self.query_as if self.query_as is not None else text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_batches += 1
        return [self._embed(text) for text in texts]

    @property
    def dimensions(self) -> int:
        return len(self.vocabulary)


def _filing(company: str, period_end: int, *, header: bool = True) -> Chunk:
    return Chunk(
        id=f"{company}_{period_end}#0",
        text="Revenue grew.",
        document_id=f"{company}_{period_end}.md",
        source=Path(f"/tmp/{company}.md"),
        doc_type="markdown",
        metadata={"company": company, "period_end": period_end},
        header=f"{company} 10-Q, period ended {period_end}" if header else None,
    )


def _router(
    tmp_path: Path, chunks: list[Chunk], vocabulary: list[str], top_m: int = 1, query_as: str | None = None
) -> DocumentRouter:
    index = BM25Index(tmp_path / "bm25.json")
    index.upsert(chunks)
    embedder = _BagOfWordsEmbedder(vocabulary, query_as)
    return DocumentRouter(embedder, index, top_m=top_m, record_template=TEMPLATE)


FILINGS = [
    _filing("Apple", 20260328),
    _filing("Apple", 20260627),
    _filing("Costco", 20260215),
    _filing("Target", 20260502),
    _filing("Walmart", 20260430),
]
VOCABULARY = ["apple", "costco", "march", "june", "february"]


def test_spell_date_reads_every_stored_form() -> None:
    assert spell_date(20260215) == "February 15, 2026"
    assert spell_date(date(2025, 6, 5)) == "June 5, 2025"
    assert spell_date("2024-09-28") == "September 28, 2024"
    with pytest.raises(ValueError):
        spell_date(3.5)


def test_render_record_spells_dates_and_needs_every_field() -> None:
    document = IndexedDocument("A.md", "Apple 10-Q", {"period_end": 20260627})

    assert render_record(TEMPLATE, document) == "Apple 10-Q; period ended June 27, 2026"
    assert render_record(TEMPLATE, IndexedDocument("A.md", None, {"period_end": 20260627})) is None
    assert render_record(TEMPLATE, IndexedDocument("A.md", "Apple 10-Q", {})) is None


def test_check_record_template_rejects_a_field_chunks_do_not_store() -> None:
    check_record_template(TEMPLATE, ["period_end"])
    with pytest.raises(ValueError, match="period_end"):
        check_record_template(TEMPLATE, ["ticker"])


def test_routes_when_bm25_and_dense_agree(tmp_path: Path) -> None:
    router = _router(tmp_path, FILINGS, VOCABULARY)

    decision = router.route("What did Apple say in the quarter ended June 27, 2026?")

    assert decision.document_ids == ["Apple_20260627.md"]
    assert decision.to_filter().any_of == {"document_id": ["Apple_20260627.md"]}


def test_top_m_widens_the_route_with_the_agreed_document_first(tmp_path: Path) -> None:
    router = _router(tmp_path, FILINGS, VOCABULARY, top_m=2)

    decision = router.route("What did Apple say in the quarter ended June 27, 2026?")

    assert decision.document_ids[0] == "Apple_20260627.md"
    assert len(decision.document_ids) == 2


def test_falls_back_when_dense_disagrees(tmp_path: Path) -> None:
    # BM25 picks June from the date; the dense fake reads every query as
    # "Apple March", so it picks the other Apple filing.
    router = _router(tmp_path, FILINGS, VOCABULARY, query_as="Apple March")

    decision = router.route("Apple quarter ended June 27, 2026")

    assert not decision.routed
    assert "dense picks" in decision.reason


def test_falls_back_on_a_bm25_tie(tmp_path: Path) -> None:
    # Names the company but no period: both Apple filings score the same.
    router = _router(tmp_path, FILINGS, VOCABULARY)

    decision = router.route("What does Apple say about services?")

    assert not decision.routed
    assert "tie" in decision.reason


def test_falls_back_when_no_record_shares_a_term(tmp_path: Path) -> None:
    router = _router(tmp_path, FILINGS, VOCABULARY)

    assert not router.route("warehouse club memberships").routed


def test_documents_without_a_record_are_never_routed_to(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    router = _router(tmp_path, [*FILINGS[:2], _filing("Costco", 20260215, header=False), *FILINGS[3:]], VOCABULARY)

    decision = router.route("Costco quarter ended February 15, 2026")

    assert "Costco_20260215.md" not in decision.document_ids
    assert "1 of 5 document(s) have no routing record" in caplog.text


def test_no_records_at_all_is_a_fallback(tmp_path: Path) -> None:
    router = _router(tmp_path, [_filing("Apple", 20260627, header=False)], VOCABULARY)

    assert router.route("Apple June 2026").reason == "no document has a routing record"


def test_records_are_embedded_once(tmp_path: Path) -> None:
    router = _router(tmp_path, FILINGS, VOCABULARY)
    router.route("Apple June 27, 2026")
    router.route("Costco February 15, 2026")

    assert router._embedder.document_batches == 1  # type: ignore[attr-defined]
