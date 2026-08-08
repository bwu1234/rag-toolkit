"""Small integration test that validates index-time content-hash detection.

This ensures the vector store can be queried for stored metadata and that a
freshly-computed `content_hash` can be compared to decide whether a chunk
needs re-embedding/upsert.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from rag.chunking.models import Chunk
from rag.retrieval.sparse import BM25Index
from rag.vectorstore.chroma_store import ChromaVectorStore

_AXIS = [1.0, 0.0, 0.0]


def _chunk(chunk_id: str, text: str = "text", *, source: str = "doc.md", **metadata: object) -> Chunk:
    return Chunk(
        id=chunk_id,
        text=text,
        document_id=source,
        source=Path(source),
        doc_type="markdown",
        metadata=dict(metadata),
    )


def test_incremental_detection(tmp_path: Path) -> None:
    store = ChromaVectorStore(persist_dir=tmp_path / "index", collection_name="inc-test")

    a = _chunk("a", "first")
    b = _chunk("b", "second")

    # Initial upsert -- include a synthetic content_hash in metadata like the
    # indexing pipeline will do.
    a_meta = dict(a.metadata)
    b_meta = dict(b.metadata)
    a_meta["content_hash"] = "h1"
    b_meta["content_hash"] = "h2"

    store.upsert([Chunk(a.id, a.text, a.document_id, a.source, a.doc_type, a_meta),
                  Chunk(b.id, b.text, b.document_id, b.source, b.doc_type, b_meta)],
                 [_AXIS, _AXIS])

    # Read back stored metadata and confirm hashes round-trip
    meta = store.get_metadatas(["a", "b"])  # mapping id -> metadata
    assert meta["a"]["content_hash"] == "h1"
    assert meta["b"]["content_hash"] == "h2"

    # Simulate an edit: 'a' changed, 'b' unchanged
    new_hashes = {"a": "h3", "b": "h2"}

    changed = [cid for cid in new_hashes.keys() if meta.get(cid, {}).get("content_hash") != new_hashes[cid]]
    assert changed == ["a"]


def test_bm25_membership_is_reported() -> None:
    """`has_chunk` is what lets the indexer notice a chunk missing from the
    sparse index even though the vector store has it."""
    index = BM25Index(Path(tempfile.mkdtemp()) / "bm25.json")
    assert not index.has_chunk("d.md::chunk0")

    index.upsert([_chunk("d.md::chunk0", "some text", source="d.md")])
    assert index.has_chunk("d.md::chunk0")
    assert not index.has_chunk("d.md::chunk1")


def test_interrupted_run_leaves_a_chunk_the_indexer_must_re_add() -> None:
    """Regression for a real divergence.

    Chroma persists on write; the BM25 index is flushed once at the end of a
    run. A run killed in between leaves the chunk in the vector store and absent
    from the sparse one. Change detection keyed on the vector store alone would
    call it unchanged forever, so hybrid retrieval would quietly search a
    smaller keyword index than the vector count implies.
    """
    index = BM25Index(Path(tempfile.mkdtemp()) / "bm25.json")
    index.upsert([_chunk("d.md::chunk0", "kept", source="d.md")])
    index.flush()

    # chunk1 is what the killed run wrote to Chroma but never flushed here.
    stored_hashes = {"d.md::chunk0": "h0", "d.md::chunk1": "h1"}
    fresh_hashes = {"d.md::chunk0": "h0", "d.md::chunk1": "h1"}

    needs_reindex = [
        cid for cid in fresh_hashes
        if stored_hashes.get(cid) != fresh_hashes[cid] or not index.has_chunk(cid)
    ]
    assert needs_reindex == ["d.md::chunk1"], "the sparse-only gap must force a re-index"
