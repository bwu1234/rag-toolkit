"""Tests for the SQLite FTS5 sparse index.

The contract tests run against both `SparseIndex` backends, so the two can't
drift apart on behaviour the retriever and indexer rely on. The FTS5-only
tests cover what its SQL adds: filter translation, write-through persistence
and query escaping.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from rag.chunking.models import Chunk
from rag.config.settings import SparseIndexConfig
from rag.query_filter import QueryFilter
from rag.retrieval.factory import get_sparse_index, sparse_index_path
from rag.retrieval.sparse import BM25Index, SparseIndex, tokenize
from rag.retrieval.sqlite_fts5 import LAYOUT_VERSION, SqliteFts5Index, check_sqlite_version

OpenIndex = Callable[[], SparseIndex]


@pytest.fixture(params=["bm25", "sqlite_fts5"])
def open_index(request: pytest.FixtureRequest, tmp_path: Path) -> OpenIndex:
    """Opens the same on-disk index each call, so a second call is a reload."""
    config = SparseIndexConfig(provider=request.param)
    return lambda: get_sparse_index(config, tmp_path)


def _chunk(chunk_id: str, text: str, document_id: str = "doc.md", **metadata: Any) -> Chunk:
    return Chunk(
        id=chunk_id,
        text=text,
        document_id=document_id,
        source=Path(f"/tmp/{document_id}"),
        doc_type="markdown",
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Contract: both backends
# ---------------------------------------------------------------------------


def test_ranks_exact_keyword_match_first_with_normalized_scores(open_index: OpenIndex) -> None:
    index = open_index()
    index.upsert(
        [
            _chunk("a", "The refund policy allows returns within thirty days."),
            _chunk("b", "Shipping takes three to five business days."),
            _chunk("c", "Contact support for account issues."),
        ]
    )

    results = index.query("refund policy", top_k=3)

    assert results[0].chunk_id == "a"
    assert results[0].score == 1.0
    assert all(0.0 <= r.score <= 1.0 for r in results)


def test_respects_top_k_and_ignores_queries_without_tokens(open_index: OpenIndex) -> None:
    index = open_index()
    index.upsert([_chunk("a", "alpha beta gamma"), _chunk("b", "alpha beta"), _chunk("c", "alpha")])

    assert len(index.query("alpha", top_k=2)) == 2
    assert index.query("alpha", top_k=0) == []
    assert index.query("!!!", top_k=5) == []
    assert index.query("absent", top_k=5) == []


def test_survives_reopening(open_index: OpenIndex) -> None:
    index = open_index()
    index.upsert([_chunk("sku", "Model WIDGET-9000 is discontinued.", title="Doc"), _chunk("x", "Onboarding steps.")])
    index.flush()

    [result] = open_index().query("WIDGET-9000", top_k=1)

    assert (result.chunk_id, result.document_id, result.metadata) == ("sku", "doc.md", {"title": "Doc"})
    assert result.source == Path("/tmp/doc.md")


def test_upsert_overwrites_by_chunk_id(open_index: OpenIndex) -> None:
    index = open_index()
    index.upsert([_chunk("a", "old text about cats")])
    index.upsert([_chunk("a", "new text about dogs and kennels")])

    assert index.count() == 1
    assert index.query("cats", top_k=5) == []
    assert index.query("kennels", top_k=1)[0].text == "new text about dogs and kennels"


def test_ids_has_chunk_and_delete(open_index: OpenIndex) -> None:
    index = open_index()
    index.upsert([_chunk("a", "apples"), _chunk("b", "bananas")])

    index.delete(["a", "never-indexed"])
    index.flush()

    reopened = open_index()
    assert reopened.ids() == {"b"}
    assert reopened.has_chunk("b") and not reopened.has_chunk("a")
    assert reopened.query("apples", top_k=5) == []


def test_reset_empties_the_index(open_index: OpenIndex) -> None:
    index = open_index()
    index.upsert([_chunk("a", "text")])
    index.flush()

    index.reset()

    assert index.count() == 0
    assert index.query("text", top_k=5) == []
    assert open_index().count() == 0


def test_matches_header_and_context_but_returns_verbatim_text(open_index: OpenIndex) -> None:
    index = open_index()
    chunk = Chunk(
        id="a",
        text="The limit is 1,000 requests per minute.",
        document_id="AAPL.md",
        source=Path("/tmp/AAPL.md"),
        doc_type="markdown",
        context="ACS API rate limiting.",
        header="Apple Inc. (AAPL) 10-K",
    )
    index.upsert([chunk, _chunk("b", "Shipping takes three days.")])
    index.flush()

    [by_context] = open_index().query("ACS", top_k=3)
    [by_header] = open_index().query("AAPL", top_k=3)

    assert by_context.chunk_id == by_header.chunk_id == "a"
    assert by_context.text == "The limit is 1,000 requests per minute.", "citations must quote the source span"
    assert (by_context.context, by_context.header) == ("ACS API rate limiting.", "Apple Inc. (AAPL) 10-K")


def test_documents_lists_each_document_once_in_first_indexed_order(open_index: OpenIndex) -> None:
    index = open_index()
    index.upsert(
        [
            _chunk("m1", "x", "MSFT.md", ticker="MSFT"),
            _chunk("a1", "x", "AAPL.md", ticker="AAPL"),
            _chunk("m2", "x", "MSFT.md", ticker="MSFT"),
        ]
    )
    index.upsert([_chunk("m1", "updated", "MSFT.md", ticker="MSFT")])  # an overwrite keeps its place
    index.flush()

    documents = open_index().documents()

    assert [(d.document_id, d.metadata) for d in documents] == [
        ("MSFT.md", {"ticker": "MSFT"}),
        ("AAPL.md", {"ticker": "AAPL"}),
    ]


def _filings(index: SparseIndex) -> SparseIndex:
    # MSFT's chunks repeat the query term, so they outrank every AAPL chunk.
    index.upsert(
        [_chunk(f"m{i}", "revenue revenue revenue grew", "MSFT.md", ticker="MSFT", period_end=20250630) for i in range(5)]
        + [
            _chunk("a24", "revenue grew", "AAPL_2024.md", ticker="AAPL", period_end=20240928),
            _chunk("a25", "revenue grew", "AAPL_2025.md", ticker="AAPL", period_end=20250927),
        ]
    )
    return index


def test_filter_applies_before_top_k(open_index: OpenIndex) -> None:
    results = _filings(open_index()).query("revenue", top_k=2, query_filter=QueryFilter(equals={"ticker": "AAPL"}))

    assert {r.chunk_id for r in results} == {"a24", "a25"}


def test_filter_combines_conditions_and_can_name_the_document_id(open_index: OpenIndex) -> None:
    index = _filings(open_index())

    ranged = index.query(
        "revenue", top_k=10, query_filter=QueryFilter(equals={"ticker": "AAPL"}, range={"period_end": {"gte": 20250101}})
    )
    by_document = index.query("revenue", top_k=10, query_filter=QueryFilter(any_of={"document_id": ["AAPL_2024.md"]}))

    assert [r.chunk_id for r in ranged] == ["a25"]
    assert [r.chunk_id for r in by_document] == ["a24"]


# ---------------------------------------------------------------------------
# FTS5 only
# ---------------------------------------------------------------------------


def test_sql_filter_agrees_with_query_filter_matches_on_mixed_value_types(tmp_path: Path) -> None:
    # The SQL translation must reproduce `QueryFilter.matches` exactly,
    # including its type strictness: a bool is not an int, an int is not a
    # string, and a missing field matches nothing.
    values: list[dict[str, Any]] = [
        {"n": 5, "s": "a"},
        {"n": True, "s": "b"},
        {"n": 5.0, "s": 5},
        {"n": "5", "s": "a"},
        {"s": "a"},
        {"n": 20250101, "s": "c"},
    ]
    chunks = [_chunk(f"c{i}", "common term", f"doc{i}.md", **meta) for i, meta in enumerate(values)]
    index = SqliteFts5Index(tmp_path / "fts.sqlite3")
    index.upsert(chunks)
    filters = [
        QueryFilter(range={"n": {"gte": 1}}),
        QueryFilter(range={"n": {"lte": 5}}),
        QueryFilter(equals={"s": "a"}),
        QueryFilter(equals={"n": "5"}),
        QueryFilter(any_of={"s": ["a", "c"]}),
        QueryFilter(any_of={"document_id": ["doc1.md", "doc3.md"]}),
        QueryFilter(range={"document_id": {"gte": 0}}),
        QueryFilter(equals={"s": "a"}, range={"n": {"gte": 5, "lte": 5}}),
        QueryFilter(equals={"missing": "a"}),
    ]

    for query_filter in filters:
        expected = {c.id for c in chunks if query_filter.matches({**c.metadata, "document_id": c.document_id})}
        got = {r.chunk_id for r in index.query("common", top_k=100, query_filter=query_filter)}
        assert got == expected, query_filter


def test_writes_are_durable_without_flush(tmp_path: Path) -> None:
    # A run killed before `flush` must not leave the vector store ahead of
    # this index (see `SparseIndex.has_chunk`).
    path = tmp_path / "fts.sqlite3"
    SqliteFts5Index(path).upsert([_chunk("a", "persisted immediately")])

    assert SqliteFts5Index(path).has_chunk("a")


@pytest.mark.parametrize("query", ['or not near', 'AND "quoted" (paren) col:on * -minus ^caret', "don't"])
def test_fts5_syntax_in_queries_is_matched_as_plain_terms(tmp_path: Path, query: str) -> None:
    index = SqliteFts5Index(tmp_path / "fts.sqlite3")
    index.upsert([_chunk("a", "or not near and quoted paren col on minus caret don't")])

    assert [r.chunk_id for r in index.query(query, top_k=5)] == ["a"]


def test_apostrophe_tokens_match_whole_as_in_bm25_index(tmp_path: Path) -> None:
    index = SqliteFts5Index(tmp_path / "fts.sqlite3")
    index.upsert([_chunk("a", "we don't ship"), _chunk("b", "don and t are names")])

    assert [r.chunk_id for r in index.query("don't", top_k=5)] == ["a"]


def test_factory_selects_the_backend_and_gives_each_its_own_file(tmp_path: Path) -> None:
    bm25, fts5 = SparseIndexConfig(provider="bm25"), SparseIndexConfig(provider="sqlite_fts5")

    assert isinstance(get_sparse_index(bm25, tmp_path, "edgar"), BM25Index)
    assert isinstance(get_sparse_index(fts5, tmp_path, "edgar"), SqliteFts5Index)
    assert sparse_index_path(fts5, tmp_path, "edgar").name == "fts5_index__edgar.sqlite3"
    assert sparse_index_path(bm25, tmp_path, "edgar") != sparse_index_path(fts5, tmp_path, "edgar")
    assert sparse_index_path(fts5, tmp_path, "edgar") != sparse_index_path(fts5, tmp_path, "baseline+edgar")


# ---------------------------------------------------------------------------
# Contentless layout, its SQLite floor, and migration from layout 1
# ---------------------------------------------------------------------------

# Layout 1 as shipped in PR #46: the FTS table kept its own copy of the tokens.
_LAYOUT_1 = """
CREATE TABLE chunks (
    rowid INTEGER PRIMARY KEY, chunk_id TEXT NOT NULL UNIQUE, document_id TEXT NOT NULL,
    source TEXT NOT NULL, doc_type TEXT NOT NULL, metadata TEXT NOT NULL,
    context TEXT, header TEXT, text TEXT NOT NULL
);
CREATE VIRTUAL TABLE chunks_fts USING fts5(tokens, tokenize = "unicode61 tokenchars ''''");
"""


def _write_layout_1(path: Path, rows: list[tuple[str, str, str]]) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(_LAYOUT_1)
    with conn:
        for rowid, (chunk_id, document_id, text) in enumerate(rows, start=1):
            conn.execute(
                "INSERT INTO chunks VALUES (?, ?, ?, ?, 'markdown', '{}', NULL, NULL, ?)",
                (rowid, chunk_id, document_id, f"/tmp/{document_id}", text),
            )
            conn.execute("INSERT INTO chunks_fts (rowid, tokens) VALUES (?, ?)", (rowid, " ".join(tokenize(text))))
    conn.close()


def test_fts_table_stores_no_copy_of_the_tokens(tmp_path: Path) -> None:
    path = tmp_path / "fts.sqlite3"
    SqliteFts5Index(path).upsert([_chunk("a", "revenue grew")])

    conn = sqlite3.connect(path)
    assert conn.execute("SELECT tokens FROM chunks_fts").fetchall() == [(None,)], "contentless: nothing to read back"
    assert conn.execute("PRAGMA user_version").fetchone()[0] == LAYOUT_VERSION


def test_layout_1_file_is_migrated_in_place_without_losing_chunks(tmp_path: Path) -> None:
    path = tmp_path / "fts.sqlite3"
    _write_layout_1(path, [("a", "AAPL.md", "revenue grew on services"), ("b", "MSFT.md", "cloud margins")])

    index = SqliteFts5Index(path)

    assert index.ids() == {"a", "b"}
    assert [r.chunk_id for r in index.query("services revenue", top_k=5)] == ["a"]
    # And the migrated table accepts the writes layout 1 needed content for.
    index.upsert([_chunk("a", "rewritten entirely", "AAPL.md")])
    index.delete(["b"])
    assert index.query("services", top_k=5) == [] and index.query("cloud", top_k=5) == []
    assert [r.chunk_id for r in SqliteFts5Index(path).query("rewritten", top_k=5)] == ["a"], "reopening is a no-op"


def test_refuses_a_layout_newer_than_it_reads(tmp_path: Path) -> None:
    path = tmp_path / "fts.sqlite3"
    SqliteFts5Index(path).close()
    conn = sqlite3.connect(path)
    conn.execute(f"PRAGMA user_version = {LAYOUT_VERSION + 1}")
    conn.close()

    with pytest.raises(RuntimeError, match="--reset"):
        SqliteFts5Index(path)


def test_sqlite_older_than_3_43_is_refused_with_what_to_do() -> None:
    check_sqlite_version((3, 43, 0))

    with pytest.raises(RuntimeError, match=r"needs SQLite >= 3\.43\.0.*3\.40\.1.*bm25"):
        check_sqlite_version((3, 40, 1))
