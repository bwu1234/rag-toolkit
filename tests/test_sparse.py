"""Tests for BM25 sparse index: ranking, persistence, reset, tokenization."""

from __future__ import annotations

import json
from pathlib import Path

from rag.chunking.models import Chunk
from rag.retrieval.sparse import BM25Index, tokenize


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        id=chunk_id,
        text=text,
        document_id="doc.md",
        source=Path("/tmp/doc.md"),
        doc_type="markdown",
        metadata={"title": "Doc"},
    )


def test_tokenize_lowercases_and_splits() -> None:
    assert tokenize("Hello, SKU-42! Don't") == ["hello", "sku", "42", "don't"]


def test_bm25_ranks_exact_keyword_match_highest(tmp_path: Path) -> None:
    index = BM25Index(tmp_path / "bm25_index.json")
    index.upsert(
        [
            _chunk("a", "The refund policy allows returns within thirty days."),
            _chunk("b", "Shipping takes three to five business days."),
            _chunk("c", "Contact support at help@example.com for account issues."),
        ]
    )

    results = index.query("refund policy", top_k=3)

    assert results
    assert results[0].chunk_id == "a"
    assert results[0].score == 1.0  # top hit min-max normalized to 1
    assert all(0.0 <= r.score <= 1.0 for r in results)
    assert results[0].score >= results[-1].score


def test_bm25_query_respects_top_k(tmp_path: Path) -> None:
    index = BM25Index(tmp_path / "bm25_index.json")
    index.upsert(
        [
            _chunk("a", "alpha beta gamma"),
            _chunk("b", "alpha beta"),
            _chunk("c", "alpha"),
        ]
    )
    assert len(index.query("alpha", top_k=2)) == 2


def test_bm25_empty_or_blank_query_returns_empty(tmp_path: Path) -> None:
    index = BM25Index(tmp_path / "bm25_index.json")
    index.upsert([_chunk("a", "some text about widgets")])
    assert index.query("", top_k=5) == []
    assert index.query("   ", top_k=5) == []
    assert index.query("!!!", top_k=5) == []  # no alphanumeric tokens


def test_bm25_persists_and_reloads(tmp_path: Path) -> None:
    path = tmp_path / "bm25_index.json"
    index = BM25Index(path)
    index.upsert(
        [
            _chunk("sku", "Model number WIDGET-9000 is discontinued."),
            _chunk("other", "General onboarding steps for new hires."),
        ]
    )
    index.flush()
    assert path.exists()

    reloaded = BM25Index(path)
    assert reloaded.count() == 2
    results = reloaded.query("WIDGET-9000", top_k=1)
    assert len(results) == 1
    assert results[0].chunk_id == "sku"
    assert results[0].document_id == "doc.md"
    assert results[0].metadata.get("title") == "Doc"


def test_bm25_upsert_is_idempotent_by_chunk_id(tmp_path: Path) -> None:
    index = BM25Index(tmp_path / "bm25_index.json")
    index.upsert([_chunk("a", "old text about cats")])
    index.upsert([_chunk("a", "new text about dogs and kennels")])
    assert index.count() == 1
    results = index.query("kennels", top_k=1)
    assert results[0].text == "new text about dogs and kennels"


def test_bm25_reset_clears_memory_and_disk(tmp_path: Path) -> None:
    path = tmp_path / "bm25_index.json"
    index = BM25Index(path)
    index.upsert([_chunk("a", "text")])
    index.flush()
    assert path.exists()

    index.reset()
    assert index.count() == 0
    assert not path.exists()
    assert index.query("text", top_k=5) == []


def test_bm25_scores_prefer_rarer_exact_terms(tmp_path: Path) -> None:
    """A rare exact token should rank its document above a common-word match."""

    index = BM25Index(tmp_path / "bm25_index.json")
    index.upsert(
        [
            _chunk("common", "The system uses a standard API for all services."),
            _chunk("rare", "Configure endpoint ZX9Q-INTERNAL before deployment."),
        ]
    )
    results = index.query("ZX9Q-INTERNAL", top_k=2)
    assert results[0].chunk_id == "rare"


# ---------------------------------------------------------------------------
# Contextual chunking: BM25 matches on context, returns verbatim text
# ---------------------------------------------------------------------------


def _contextual_chunk(chunk_id: str, text: str, context: str) -> Chunk:
    return Chunk(
        id=chunk_id,
        text=text,
        document_id="doc.md",
        source=Path("/tmp/doc.md"),
        doc_type="markdown",
        metadata={"title": "Doc"},
        context=context,
    )


def test_bm25_matches_terms_that_only_appear_in_a_chunks_context(tmp_path: Path) -> None:
    # The exact case contextual retrieval exists for: the chunk never says
    # "ACS", so without its context this query cannot reach it at all.
    index = BM25Index(tmp_path / "bm25_index.json")
    index.upsert(
        [
            _contextual_chunk("a", "The limit is 1,000 requests per minute.", "ACS API rate limiting."),
            _chunk("b", "Shipping takes three to five business days."),
        ]
    )

    results = index.query("ACS rate limiting", top_k=3)

    assert [r.chunk_id for r in results] == ["a"]


def test_bm25_returns_verbatim_chunk_text_not_the_contextualized_string(tmp_path: Path) -> None:
    index = BM25Index(tmp_path / "bm25_index.json")
    index.upsert([_contextual_chunk("a", "The limit is 1,000 requests per minute.", "ACS API rate limiting.")])

    [result] = index.query("ACS", top_k=3)

    assert result.text == "The limit is 1,000 requests per minute.", "citations must quote the source span"
    assert result.context == "ACS API rate limiting."


def test_bm25_context_survives_persistence(tmp_path: Path) -> None:
    path = tmp_path / "bm25_index.json"
    index = BM25Index(path)
    index.upsert([_contextual_chunk("a", "The limit is 1,000 requests per minute.", "ACS API rate limiting.")])
    index.flush()

    reloaded = BM25Index(path)
    [result] = reloaded.query("ACS", top_k=3)

    assert result.context == "ACS API rate limiting."


def test_bm25_handles_records_written_before_contextual_chunking_existed(tmp_path: Path) -> None:
    # An index file from an older build has no "context" key at all; loading it
    # must keep working rather than raising a KeyError on every query.
    path = tmp_path / "bm25_index.json"
    index = BM25Index(path)
    index.upsert([_chunk("a", "The refund policy allows returns within thirty days.")])
    index.flush()
    raw = json.loads(path.read_text())
    for record in (raw["records"] if isinstance(raw, dict) and "records" in raw else raw).values():
        record.pop("context", None)
    path.write_text(json.dumps(raw))

    [result] = BM25Index(path).query("refund policy", top_k=3)

    assert result.context is None
    assert result.text.startswith("The refund policy")


def test_bm25_matches_terms_that_only_appear_in_a_chunks_header(tmp_path: Path) -> None:
    path = tmp_path / "bm25_index.json"
    index = BM25Index(path)
    headed = Chunk(
        id="a",
        text="Revenue grew 2% on services.",
        document_id="AAPL.md",
        source=Path("/tmp/AAPL.md"),
        doc_type="markdown",
        header="Apple Inc. (AAPL) 10-K, period ended 2024-09-28",
    )
    index.upsert([headed, _chunk("b", "Revenue grew 3% on memberships.")])
    index.flush()

    [result] = BM25Index(path).query("AAPL", top_k=3)

    assert result.chunk_id == "a"
    assert result.header == "Apple Inc. (AAPL) 10-K, period ended 2024-09-28", "and it survives persistence"
    assert result.text == "Revenue grew 2% on services."
