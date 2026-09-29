"""Chroma-backed `VectorStore` adapter.

Uses Chroma's persistent local mode -- a single on-disk directory, no server
process -- which fits the local-first default and keeps the dependency surface
to one library that bundles vector search with metadata storage.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import chromadb

from rag.chunking.models import Chunk
from rag.query_filter import QueryFilter
from rag.vectorstore.base import ScoredChunk, VectorStore

logger = logging.getLogger(__name__)

# Fields stored on every chunk's metadata that describe *where it came from*
# rather than being chunk-specific extras. Pulled back out into `ScoredChunk`'s
# dedicated fields on query, so callers don't have to know they were ever
# stashed in Chroma's flat metadata dict.
_PROVENANCE_FIELDS = ("document_id", "source", "doc_type")

#: Chroma's default `ef_search`, kept in step with `DEFAULT_HNSW_EF_SEARCH`.
_CHROMA_DEFAULT_EF_SEARCH = 100

# `ef_search` each collection was opened with in this process, keyed by
# (index directory, collection name). Chroma reads the setting once, when a
# process first loads the collection's HNSW segment, and ignores a later
# change until the next process -- so a second, different value in the same
# process would be silently ignored. See `_apply_ef_search`.
_EF_SEARCH_IN_PROCESS: dict[tuple[str, str], int] = {}


class ChromaVectorStore(VectorStore):
    """Wraps a single persistent Chroma collection.

    Chroma's metadata values must be `str | int | float | bool` (no `None`,
    `Path`, or nested structures), so this adapter is responsible for
    flattening `Chunk` provenance + metadata into that shape on the way in,
    and reconstructing typed `ScoredChunk`s on the way out.
    """

    def __init__(
        self, persist_dir: Path, collection_name: str, *, hnsw_ef_search: int = _CHROMA_DEFAULT_EF_SEARCH
    ) -> None:
        persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection_name = collection_name
        self._ef_key = (str(persist_dir.resolve()), collection_name)
        self._ef_search = hnsw_ef_search
        # Cosine similarity is the natural match for normalized text
        # embeddings and is what most embedding models (including Ollama's)
        # are tuned/evaluated against.
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._apply_ef_search()

    def _apply_ef_search(self) -> None:
        """Make the collection search with the configured `ef_search`.

        Chroma persists `ef_search` with the collection, so a value is written
        whenever it differs from the configured one -- otherwise the setting
        would depend on whichever process last changed it. It must happen
        before this process's first query or write, the point at which Chroma
        loads the HNSW segment and fixes the value for the process's lifetime.

        Raises:
            ValueError: this process already opened the collection with a
                different value, which Chroma would silently ignore (e.g. a
                `run_matrix` variant sweeping `hnsw_ef_search`; run one
                process per value instead).
        """

        opened_with = _EF_SEARCH_IN_PROCESS.setdefault(self._ef_key, self._ef_search)
        if opened_with != self._ef_search:
            raise ValueError(
                f"Collection {self._collection_name!r} was already opened in this process with "
                f"hnsw_ef_search={opened_with}; Chroma ignores a change to {self._ef_search} until the "
                "process restarts. Use one process per hnsw_ef_search value."
            )
        hnsw = (self._collection.configuration or {}).get("hnsw") or {}
        if hnsw.get("ef_search") != self._ef_search:
            self._collection.modify(configuration={"hnsw": {"ef_search": self._ef_search}})
            logger.info("Set hnsw ef_search=%d on collection %r", self._ef_search, self._collection_name)

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) must be the same length"
            )
        if not chunks:
            return

        # Flatten chunk metadata (including any index-time fields like
        # `content_hash`) and upsert into Chroma. The caller is responsible
        # for computing and attaching content hashes to `chunk.metadata`.
        self._collection.upsert(
            ids=[chunk.id for chunk in chunks],
            # Chroma's stub types an invariant `List[Sequence[float] | ...]`
            # param, which structurally accepts our `list[list[float]]` at
            # runtime but mypy rejects due to list invariance.
            embeddings=embeddings,  # type: ignore[arg-type]
            documents=[chunk.text for chunk in chunks],
            metadatas=[self._to_chroma_metadata(chunk) for chunk in chunks],
        )
        logger.info("Upserted %d chunk(s) into collection %r", len(chunks), self._collection_name)

    def get_metadatas(self, ids: list[str]) -> dict[str, dict[str, Any]]:
        """Return stored metadata for the given chunk ids.

        The returned mapping includes only ids that exist in the collection.
        Useful for incremental indexing: callers can compare a stored
        `content_hash` (if present) against a freshly computed hash to detect
        unchanged chunks.
        """
        if not ids:
            return {}

        result = self._collection.get(ids=ids, include=["metadatas", "documents"])
        out: dict[str, dict[str, Any]] = {}
        # `result` contains parallel arrays under keys 'ids', 'metadatas', 'documents'.
        # Chroma types each as optional even though we always requested them;
        # `or []` narrows away the `None` case for mypy.
        result_ids = result.get("ids") or []
        result_metadatas = result.get("metadatas") or []
        result_documents = result.get("documents") or []
        for cid, meta, doc in zip(result_ids, result_metadatas, result_documents):
            # Chroma stores only primitive metadata values; return as-is.
            out[cid] = dict(meta or {})
            # also expose stored document text under a well-known key for callers
            # that might want to sanity-check or diff text (optional).
            out[cid]["_document_text"] = doc
        return out

    def query(
        self, embedding: list[float], top_k: int, query_filter: QueryFilter | None = None
    ) -> list[ScoredChunk]:
        if self.count() == 0:
            return []

        # A native `where` pre-filters inside Chroma, so `n_results` counts
        # matching chunks only -- never an unfiltered top-k cut down afterwards.
        where = query_filter.to_chroma_where() if query_filter is not None else None
        result = self._collection.query(
            # See the `upsert` comment above -- same invariant-List mismatch.
            query_embeddings=[embedding],  # type: ignore[arg-type]
            n_results=top_k,
            where=where,  # type: ignore[arg-type]
            include=["documents", "metadatas", "distances"],
        )

        # Each field is typed as optional even though we always requested it
        # via `include`; `or [[]]` narrows away the `None` case for mypy.
        ids = (result["ids"] or [[]])[0]
        documents = (result["documents"] or [[]])[0]
        metadatas = (result["metadatas"] or [[]])[0]
        distances = (result["distances"] or [[]])[0]

        return [
            self._to_scored_chunk(chunk_id, text, metadata, distance)
            for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances)
        ]

    def count(self) -> int:
        return self._collection.count()

    def ids(self) -> set[str]:
        # `include=[]` fetches ids alone, not every stored document and metadata.
        return set(self._collection.get(include=[])["ids"])

    def delete(self, ids: list[str]) -> None:
        # Chroma caps how many records one call may touch; batch to stay under it.
        batch_size = self._client.get_max_batch_size()
        for start in range(0, len(ids), batch_size):
            self._collection.delete(ids=ids[start : start + batch_size])
        if ids:
            logger.info("Deleted %d chunk(s) from collection %r", len(ids), self._collection_name)

    def reset(self) -> None:
        self._client.delete_collection(name=self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        # A new collection starts at Chroma's default, and has no segment
        # loaded yet, so the configured value applies afresh.
        _EF_SEARCH_IN_PROCESS.pop(self._ef_key, None)
        self._apply_ef_search()
        logger.info("Reset collection %r", self._collection_name)

    @staticmethod
    def _to_chroma_metadata(chunk: Chunk) -> dict[str, Any]:
        """Flatten a `Chunk`'s provenance + metadata into Chroma-compatible primitives.

        `None` values are dropped (Chroma rejects them) and `Path`s are
        stringified -- both are reversed/handled symmetrically in
        `_to_scored_chunk`.
        """

        flat: dict[str, Any] = {
            "document_id": chunk.document_id,
            "source": str(chunk.source),
            "doc_type": chunk.doc_type,
        }
        # Stored alongside provenance rather than inside `documents`: the
        # document text must stay the verbatim chunk so citations quote the
        # source, but the context has to survive the round trip for the prompt.
        if chunk.context:
            flat["context"] = chunk.context
        if chunk.header:
            flat["header"] = chunk.header
        for key, value in chunk.metadata.items():
            if value is None:
                continue
            flat[key] = str(value) if isinstance(value, Path) else value
        return flat

    @staticmethod
    def _to_scored_chunk(chunk_id: str, text: str, metadata: Mapping[str, Any], distance: float) -> ScoredChunk:
        """Reconstruct a typed `ScoredChunk`, splitting provenance out of the flat metadata dict."""

        metadata = dict(metadata)
        document_id = metadata.pop("document_id", "")
        source = Path(metadata.pop("source", ""))
        doc_type = metadata.pop("doc_type", "")
        context = metadata.pop("context", None)
        header = metadata.pop("header", None)

        return ScoredChunk(
            chunk_id=chunk_id,
            text=text,
            document_id=document_id,
            source=source,
            doc_type=doc_type,
            context=context,
            header=header,
            # Chroma reports cosine *distance* (1 - cosine similarity) when
            # the collection is configured with `hnsw:space: cosine`;
            # convert back to the `[0, 1]`-similarity convention `ScoredChunk`
            # documents so callers never need to know which metric was used.
            score=1.0 - distance,
            metadata=metadata,
        )
