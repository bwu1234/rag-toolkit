"""SQLite FTS5 sparse index: BM25 keyword search that updates in place.

`BM25Index` holds every chunk in a Python dict, rewrites one JSON file per
flush and rebuilds the whole Okapi model after any change -- fine to ~10^4
chunks, not beyond. This backend keeps the same `SparseIndex` contract on an
SQLite file: rows are written per batch inside a transaction, FTS5 maintains
the inverted index incrementally, and ranking runs in SQLite's C code via its
built-in `bm25()` function. `sqlite3` is in the standard library, so it adds no
dependency.

Tokens are identical to `BM25Index`'s: the text is pre-tokenized with
`tokenize()` and stored space-joined, and FTS5's `unicode61` tokenizer is told
to keep `'` inside tokens so "don't" stays one term. Scores are *not*
identical: FTS5 fixes k1 = 1.2 (rank_bm25's `BM25Okapi` defaults to 1.5), and
floors a non-positive IDF at 1e-6 where rank_bm25 substitutes a quarter of the
mean IDF. Rankings therefore differ on terms common to most chunks, so treat a
switch as a retrieval change and measure it.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from rag.chunking.models import Chunk, join_index_text
from rag.query_filter import DOCUMENT_ID, QueryFilter
from rag.retrieval.sparse import IndexedDocument, SparseIndex, min_max_normalize, tokenize
from rag.vectorstore.base import ScoredChunk

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    rowid INTEGER PRIMARY KEY,
    chunk_id TEXT NOT NULL UNIQUE,
    document_id TEXT NOT NULL,
    source TEXT NOT NULL,
    doc_type TEXT NOT NULL,
    metadata TEXT NOT NULL,
    context TEXT,
    header TEXT,
    text TEXT NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    tokens,
    tokenize = "unicode61 tokenchars ''''"
);
"""

_RECORD_COLUMNS = "c.chunk_id, c.document_id, c.source, c.doc_type, c.metadata, c.context, c.header, c.text"


class SqliteFts5Index(SparseIndex):
    """`SparseIndex` on an SQLite FTS5 table, persisted as one file under the index dir.

    Every `upsert` and `delete` commits before returning, so the file is
    always current and `flush` has nothing left to do. That also closes the
    window `SparseIndex.has_chunk` describes: a run killed mid-way leaves this
    index holding exactly the batches the vector store holds.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        # The API serves requests from a thread pool; one connection shared
        # under a lock is simpler than a per-thread pool and plenty for a
        # local index.
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        with self._lock, self._conn:
            # WAL lets a running API read while `cli index` writes.
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(_SCHEMA)
        logger.info("Opened FTS5 sparse index at %s (%d chunks)", path, self.count())

    def upsert(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        with self._lock, self._conn:
            for chunk in chunks:
                row = self._conn.execute("SELECT rowid FROM chunks WHERE chunk_id = ?", (chunk.id,)).fetchone()
                values = (
                    chunk.document_id,
                    str(chunk.source),
                    chunk.doc_type,
                    json.dumps(chunk.metadata, ensure_ascii=False),
                    chunk.context,
                    chunk.header,
                    chunk.text,
                )
                if row is None:
                    cursor = self._conn.execute(
                        "INSERT INTO chunks (document_id, source, doc_type, metadata, context, header, text, chunk_id)"
                        " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (*values, chunk.id),
                    )
                    rowid = cursor.lastrowid
                else:
                    # Update in place, keeping the rowid, so `documents()` keeps
                    # first-indexed order the way `BM25Index`'s dict does.
                    rowid = row[0]
                    self._conn.execute(
                        "UPDATE chunks SET document_id = ?, source = ?, doc_type = ?, metadata = ?,"
                        " context = ?, header = ?, text = ? WHERE rowid = ?",
                        (*values, rowid),
                    )
                    self._conn.execute("DELETE FROM chunks_fts WHERE rowid = ?", (rowid,))
                # Index the header and context too, as `Chunk.index_text` does,
                # while `chunks.text` keeps the verbatim span for citations.
                tokens = " ".join(tokenize(join_index_text(chunk.header, chunk.context, chunk.text)))
                self._conn.execute("INSERT INTO chunks_fts (rowid, tokens) VALUES (?, ?)", (rowid, tokens))
            total = self._count_locked()
        logger.info("Upserted %d chunk(s) into FTS5 index (%d total)", len(chunks), total)

    def query(self, query: str, top_k: int, query_filter: QueryFilter | None = None) -> list[ScoredChunk]:
        tokens = tokenize(query)
        if top_k <= 0 or not tokens:
            return []

        # Quote every token: it is then a phrase of one term, never an FTS5
        # operator ("or", "near") or a syntax error. Repeated query tokens stay
        # repeated, so they weigh twice, as in rank_bm25's `get_scores`.
        match = " OR ".join(f'"{token}"' for token in tokens)
        # The filter is part of the WHERE clause, so it decides which chunks
        # compete before the LIMIT, and IDF stays corpus-wide -- the same
        # semantics as `BM25Index.query`.
        where, params = _filter_sql(query_filter)
        sql = (
            f"SELECT {_RECORD_COLUMNS}, chunks_fts.rank FROM chunks_fts"
            " JOIN chunks AS c ON c.rowid = chunks_fts.rowid"
            f" WHERE chunks_fts MATCH ?{where} ORDER BY chunks_fts.rank LIMIT ?"
        )
        with self._lock:
            rows = self._conn.execute(sql, (match, *params, top_k)).fetchall()
        if not rows:
            return []

        # FTS5's bm25() is negated so that ascending order is best-first.
        scores = min_max_normalize([-row[-1] for row in rows])
        return [_scored_chunk(row, score) for row, score in zip(rows, scores, strict=True)]

    def count(self) -> int:
        with self._lock:
            return self._count_locked()

    def has_chunk(self, chunk_id: str) -> bool:
        with self._lock:
            return self._conn.execute("SELECT 1 FROM chunks WHERE chunk_id = ?", (chunk_id,)).fetchone() is not None

    def ids(self) -> set[str]:
        with self._lock:
            return {row[0] for row in self._conn.execute("SELECT chunk_id FROM chunks")}

    def documents(self) -> list[IndexedDocument]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT document_id, header, metadata FROM chunks"
                " WHERE rowid IN (SELECT MIN(rowid) FROM chunks GROUP BY document_id) ORDER BY rowid"
            ).fetchall()
        return [IndexedDocument(document_id=row[0], header=row[1], metadata=json.loads(row[2])) for row in rows]

    def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        with self._lock, self._conn:
            removed = 0
            for chunk_id in ids:
                row = self._conn.execute("SELECT rowid FROM chunks WHERE chunk_id = ?", (chunk_id,)).fetchone()
                if row is None:
                    continue
                self._conn.execute("DELETE FROM chunks_fts WHERE rowid = ?", row)
                self._conn.execute("DELETE FROM chunks WHERE rowid = ?", row)
                removed += 1
            left = self._count_locked()
        if removed:
            logger.info("Deleted %d chunk(s) from FTS5 index (%d left)", removed, left)

    def reset(self) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM chunks_fts")
            self._conn.execute("DELETE FROM chunks")
        logger.info("Reset FTS5 index at %s", self._path)

    def close(self) -> None:
        """Close the connection. Optional; the index is committed after every write."""
        with self._lock:
            self._conn.close()

    def _count_locked(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])


def _scored_chunk(row: tuple[Any, ...], score: float) -> ScoredChunk:
    chunk_id, document_id, source, doc_type, metadata, context, header, text, _rank = row
    return ScoredChunk(
        chunk_id=chunk_id,
        text=text,
        document_id=document_id,
        source=Path(source),
        doc_type=doc_type,
        # RRF only uses rank; the score is for display / NoOp rerank paths.
        score=score,
        metadata=json.loads(metadata),
        context=context,
        header=header,
    )


def _filter_sql(query_filter: QueryFilter | None) -> tuple[str, list[Any]]:
    """`query_filter` as ` AND ...` SQL over the `chunks AS c` row, with its parameters.

    Must agree with `QueryFilter.matches`, which compares Python values:
    `equals`/`any_of` hold only for string values and `range` only for
    integers (never booleans), so each clause also checks the JSON type --
    `json_extract` alone would read `true` as 1 and let it into a range.
    """

    if query_filter is None or query_filter.is_empty:
        return "", []

    clauses: list[str] = []
    params: list[Any] = []

    def field(name: str) -> tuple[str, str, list[Any]]:
        """(value expression, type expression, their parameters) for `name`."""
        if name == DOCUMENT_ID:
            return "c.document_id", "'text'", []
        if '"' in name:
            raise ValueError(f"metadata field name {name!r} can't be filtered on: it contains a double quote")
        path = f'$."{name}"'
        return "json_extract(c.metadata, ?)", "json_type(c.metadata, ?)", [path]

    def add(name: str, json_type: str, condition: str, values: Iterable[Any]) -> None:
        value_sql, type_sql, path_params = field(name)
        clauses.append(f"({type_sql} = '{json_type}' AND {value_sql} {condition})")
        params.extend([*path_params, *path_params, *values])

    for name, value in query_filter.equals.items():
        add(name, "text", "= ?", [value])
    for name, options in query_filter.any_of.items():
        add(name, "text", f"IN ({', '.join('?' * len(options))})", options)
    for name, bounds in query_filter.range.items():
        if bounds.gte is not None:
            add(name, "integer", ">= ?", [bounds.gte])
        if bounds.lte is not None:
            add(name, "integer", "<= ?", [bounds.lte])

    return "".join(f" AND {clause}" for clause in clauses), params
