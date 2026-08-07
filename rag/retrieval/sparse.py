"""Sparse (keyword) index interface and a BM25 implementation.

Complements dense vector search: BM25 excels at exact-term matches (product
IDs, acronyms, proper names) that embedding similarity often ranks poorly.
Used by hybrid retrieval and fused with dense results via RRF.
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from rag.chunking.models import Chunk
from rag.vectorstore.base import ScoredChunk

logger = logging.getLogger(__name__)

# Filename under paths.index_dir — sits beside Chroma's sqlite, rebuildable.
BM25_INDEX_FILENAME = "bm25_index.json"

# Lowercase alphanumeric tokens; keeps simple contractions like "don't".
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenization shared by index and query paths."""

    return _TOKEN_RE.findall(text.lower())


def bm25_index_path(index_dir: Path) -> Path:
    """Return the on-disk path for the BM25 index under ``index_dir``."""

    return index_dir / BM25_INDEX_FILENAME


class SparseIndex(ABC):
    """Interface for keyword/sparse retrieval over chunks.

    To add a new backend: subclass, implement these methods, and wire it in
    ``rag.retrieval.builder`` (or a future factory) based on config.
    """

    @abstractmethod
    def upsert(self, chunks: list[Chunk]) -> None:
        """Insert or overwrite chunks keyed by ``chunk.id``."""
        raise NotImplementedError

    @abstractmethod
    def query(self, query: str, top_k: int) -> list[ScoredChunk]:
        """Return up to ``top_k`` chunks most relevant to ``query``, best first.

        ``score`` is normalized to ``[0, 1]`` within the returned result set
        (relative BM25 strength), matching the ``ScoredChunk`` convention.
        """
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        """Number of chunks currently indexed."""
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Delete all indexed chunks."""
        raise NotImplementedError


class BM25Index(SparseIndex):
    """In-memory BM25Okapi index with JSON persistence under the vector index dir.

    Persistence stores chunk text + provenance (not the BM25 model itself);
    the Okapi index is rebuilt on load and after upserts. That keeps the
    on-disk format simple and version-stable across ``rank_bm25`` upgrades.

    Rebuilds are lazy: ``upsert`` marks the index dirty and only retokenizes
    / rebuilds on the next ``query`` or explicit ``flush``, so batched
    indexing does not pay O(n) per batch.
    """

    def __init__(self, persist_path: Path) -> None:
        self._persist_path = persist_path
        # chunk_id -> serializable record used both for BM25 docs and ScoredChunk rebuild
        self._records: dict[str, dict[str, Any]] = {}
        self._bm25: BM25Okapi | None = None
        # Parallel arrays aligned with the current BM25 corpus order.
        self._ordered_ids: list[str] = []
        self._dirty = False
        self._load()

    def upsert(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        for chunk in chunks:
            self._records[chunk.id] = {
                "text": chunk.text,
                "document_id": chunk.document_id,
                "source": str(chunk.source),
                "doc_type": chunk.doc_type,
                "metadata": dict(chunk.metadata),
                "context": chunk.context,
            }
        self._dirty = True
        logger.info("Upserted %d chunk(s) into BM25 index (%d total)", len(chunks), len(self._records))

    def query(self, query: str, top_k: int) -> list[ScoredChunk]:
        if top_k <= 0 or not self._records:
            return []

        self._ensure_index()
        assert self._bm25 is not None  # set by _ensure_index when records exist

        tokens = tokenize(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)
        # BM25Okapi IDF can go negative when a term appears in most/all docs,
        # so "relevant" is "non-zero", not "positive". Zero means no overlap.
        ranked = sorted(
            ((float(score), idx) for idx, score in enumerate(scores) if score != 0.0),
            key=lambda pair: pair[0],
            reverse=True,
        )[:top_k]

        if not ranked:
            return []

        # Min-max normalize raw BM25 into [0, 1] within this result set.
        # (Division by max alone fails when all scores are negative.)
        raw_scores = [raw for raw, _ in ranked]
        lo, hi = min(raw_scores), max(raw_scores)
        span = hi - lo

        results: list[ScoredChunk] = []
        for raw_score, idx in ranked:
            chunk_id = self._ordered_ids[idx]
            record = self._records[chunk_id]
            if span > 0:
                norm = (raw_score - lo) / span
            else:
                # Every hit scored identically — still a match; give them 1.0.
                norm = 1.0
            results.append(
                ScoredChunk(
                    chunk_id=chunk_id,
                    text=record["text"],
                    document_id=record["document_id"],
                    source=Path(record["source"]),
                    doc_type=record["doc_type"],
                    # RRF only uses rank; norm is for display / NoOp rerank paths.
                    score=norm,
                    metadata=dict(record["metadata"]),
                    context=record.get("context"),
                )
            )
        return results

    def count(self) -> int:
        return len(self._records)

    def reset(self) -> None:
        self._records.clear()
        self._bm25 = None
        self._ordered_ids = []
        self._dirty = False
        if self._persist_path.exists():
            self._persist_path.unlink()
        logger.info("Reset BM25 index at %s", self._persist_path)

    def flush(self) -> None:
        """Rebuild (if dirty) and persist to disk. Call after a batch index run."""

        self._ensure_index()
        self._persist()

    @staticmethod
    def _index_text(record: dict[str, Any]) -> list[str]:
        """Tokens for one record: its context (if any) plus its text.

        Mirrors `Chunk.contextual_text`, but reads the persisted record rather
        than a `Chunk` -- the on-disk index is the only place BM25 sees chunks
        after indexing, and older index files predate the `context` key.
        """

        context = record.get("context")
        text = record["text"]
        return tokenize(f"{context}\n\n{text}" if context else text)

    def _ensure_index(self) -> None:
        if not self._dirty and self._bm25 is not None:
            return
        if not self._records:
            self._bm25 = None
            self._ordered_ids = []
            self._dirty = False
            return

        self._ordered_ids = list(self._records.keys())
        # Tokenize the *contextualized* text so keyword search benefits from
        # contextual chunking too -- a chunk whose own words never say "ACS"
        # is unreachable by BM25 until its context supplies the term. Records
        # written before contextual chunking existed simply have no context.
        corpus = [self._index_text(self._records[cid]) for cid in self._ordered_ids]
        # BM25Okapi requires a non-empty corpus; empty token docs are fine.
        self._bm25 = BM25Okapi(corpus)
        # Prevent rank_bm25 IDF <= 0 bug on small corpora or high-doc-frequency terms
        # by ensuring all indexed terms have a small positive IDF floor.
        for token, idf in list(self._bm25.idf.items()):
            if idf <= 0:
                self._bm25.idf[token] = 1e-6
        self._dirty = False

    def _persist(self) -> None:
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"chunks": self._records}
        self._persist_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        logger.debug("Persisted BM25 index (%d chunks) to %s", len(self._records), self._persist_path)

    def _load(self) -> None:
        if not self._persist_path.exists():
            return
        try:
            raw = json.loads(self._persist_path.read_text(encoding="utf-8"))
            chunks = raw.get("chunks", {})
            if not isinstance(chunks, dict):
                raise ValueError("bm25 index 'chunks' must be an object")
            self._records = {str(k): v for k, v in chunks.items()}
            self._dirty = True  # rebuild BM25 lazily on first query
            logger.info("Loaded %d chunk(s) from BM25 index at %s", len(self._records), self._persist_path)
        except (json.JSONDecodeError, OSError, ValueError, TypeError) as exc:
            logger.warning("Failed to load BM25 index from %s (%s); starting empty", self._persist_path, exc)
            self._records = {}
            self._dirty = False
