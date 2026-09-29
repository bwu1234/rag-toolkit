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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from rag.chunking.models import Chunk, join_index_text
from rag.query_filter import QueryFilter
from rag.vectorstore.base import ScoredChunk

logger = logging.getLogger(__name__)

# Filename under paths.index_dir — sits beside Chroma's sqlite, rebuildable.
BM25_INDEX_FILENAME = "bm25_index.json"

# Lowercase alphanumeric tokens; keeps simple contractions like "don't".
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenization shared by index and query paths."""

    return _TOKEN_RE.findall(text.lower())


def build_bm25(corpus: list[list[str]]) -> BM25Okapi:
    """BM25Okapi over tokenized texts, with every term's IDF kept positive.

    rank_bm25's IDF goes to zero or below for a term in most documents of a
    small corpus, which would make a match score no better than a miss.
    """

    bm25 = BM25Okapi(corpus)
    for token, idf in list(bm25.idf.items()):
        if idf <= 0:
            bm25.idf[token] = 1e-6
    return bm25


def min_max_normalize(raw_scores: list[float]) -> list[float]:
    """Min-max scale raw scores into ``[0, 1]`` within one result set.

    Division by the max alone fails when every score is negative. When all
    scores are equal, every hit is still a match, so each gets 1.0.
    """

    lo, hi = min(raw_scores), max(raw_scores)
    span = hi - lo
    return [(raw - lo) / span if span > 0 else 1.0 for raw in raw_scores]


def bm25_index_path(index_dir: Path, slug: str | None = None) -> Path:
    """Return the on-disk path for the BM25 index under ``index_dir``.

    ``slug`` names the corpus selection this index was built from (see
    :class:`~rag.config.settings.CorpusSelection`), so an isolated and a pooled
    index sit side by side instead of overwriting each other. Omitting it gives
    the unqualified filename, which is what direct callers and tests use.
    """

    if slug is None:
        return index_dir / BM25_INDEX_FILENAME
    return index_dir / f"bm25_index__{slug}.json"


@dataclass(frozen=True)
class IndexedDocument:
    """One document as the index stores it: its chunk header and carried metadata.

    Every chunk of a document carries the same header and metadata, so one
    entry per document is lossless.
    """

    document_id: str
    header: str | None
    metadata: dict[str, Any]


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
    def query(self, query: str, top_k: int, query_filter: QueryFilter | None = None) -> list[ScoredChunk]:
        """Return up to ``top_k`` chunks most relevant to ``query``, best first.

        With ``query_filter``, only chunks whose metadata matches it are
        ranked, so the filter narrows what competes before the top-k cut.

        ``score`` is normalized to ``[0, 1]`` within the returned result set
        (relative BM25 strength), matching the ``ScoredChunk`` convention.
        """
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        """Number of chunks currently indexed."""
        raise NotImplementedError

    @abstractmethod
    def has_chunk(self, chunk_id: str) -> bool:
        """Whether ``chunk_id`` is present.

        Exists so the indexer can require a chunk to be current in *both* the
        vector store and here before skipping it. The two persist differently --
        Chroma writes immediately, this index is flushed once at the end of a run
        -- so a run killed mid-way leaves chunks in the vector store that never
        reached the sparse index. Change detection keyed on the vector store
        alone then treats them as up to date forever, and hybrid retrieval
        silently searches a smaller keyword index than it thinks it has.
        """
        raise NotImplementedError

    @abstractmethod
    def ids(self) -> set[str]:
        """Every chunk id currently indexed -- mirrors `VectorStore.ids`."""
        raise NotImplementedError

    @abstractmethod
    def documents(self) -> list[IndexedDocument]:
        """One entry per indexed document, in first-indexed order.

        What document-level routing ranks: read from the index rather than the
        corpus on disk, so it covers exactly the documents retrieval can return.
        """
        raise NotImplementedError

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        """Remove the chunks with these ids; ids not present are ignored."""
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Delete all indexed chunks."""
        raise NotImplementedError

    def flush(self) -> None:
        """Persist pending writes; the indexer calls it once at the end of a run.

        A no-op by default, for backends that commit on every write.
        """


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
                "header": chunk.header,
            }
        self._dirty = True
        logger.info("Upserted %d chunk(s) into BM25 index (%d total)", len(chunks), len(self._records))

    def query(self, query: str, top_k: int, query_filter: QueryFilter | None = None) -> list[ScoredChunk]:
        if top_k <= 0 or not self._records:
            return []

        self._ensure_index()
        assert self._bm25 is not None  # set by _ensure_index when records exist

        tokens = tokenize(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)
        # IDF stays corpus-wide under a filter: the filter decides which chunks
        # compete, not how rare a term is, so a chunk scores the same filtered
        # or not and only its rank among the survivors changes.
        allowed = (
            None
            if query_filter is None or query_filter.is_empty
            else {idx for idx, cid in enumerate(self._ordered_ids) if query_filter.matches(self._flat_metadata(cid))}
        )
        # BM25Okapi IDF can go negative when a term appears in most/all docs,
        # so "relevant" is "non-zero", not "positive". Zero means no overlap.
        ranked = sorted(
            (
                (float(score), idx)
                for idx, score in enumerate(scores)
                if score != 0.0 and (allowed is None or idx in allowed)
            ),
            key=lambda pair: pair[0],
            reverse=True,
        )[:top_k]

        if not ranked:
            return []

        results: list[ScoredChunk] = []
        norms = min_max_normalize([raw for raw, _ in ranked])
        for (_raw, idx), norm in zip(ranked, norms, strict=True):
            chunk_id = self._ordered_ids[idx]
            record = self._records[chunk_id]
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
                    header=record.get("header"),
                )
            )
        return results

    def count(self) -> int:
        return len(self._records)

    def _flat_metadata(self, chunk_id: str) -> dict[str, Any]:
        """A record's metadata plus `document_id`, the shape a `QueryFilter` matches against."""
        record = self._records[chunk_id]
        return {**record["metadata"], "document_id": record["document_id"]}

    def has_chunk(self, chunk_id: str) -> bool:
        return chunk_id in self._records

    def ids(self) -> set[str]:
        return set(self._records)

    def documents(self) -> list[IndexedDocument]:
        seen: dict[str, IndexedDocument] = {}
        for record in self._records.values():
            if record["document_id"] not in seen:
                seen[record["document_id"]] = IndexedDocument(
                    document_id=record["document_id"],
                    header=record.get("header"),
                    metadata=dict(record["metadata"]),
                )
        return list(seen.values())

    def delete(self, ids: list[str]) -> None:
        removed = sum(self._records.pop(chunk_id, None) is not None for chunk_id in ids)
        if removed:
            # Same lazy rebuild as `upsert`; persisted on the next `flush`.
            self._dirty = True
            logger.info("Deleted %d chunk(s) from BM25 index (%d left)", removed, len(self._records))

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
        """Tokens for one record: its header and context (if any) plus its text.

        Mirrors `Chunk.index_text`, but reads the persisted record rather than
        a `Chunk` -- the on-disk index is the only place BM25 sees chunks after
        indexing, and older index files predate the `context` and `header` keys.
        """

        return tokenize(join_index_text(record.get("header"), record.get("context"), record["text"]))

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
        self._bm25 = build_bm25(corpus)
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
