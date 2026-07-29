"""Small integration test that validates index-time content-hash detection.

This ensures the vector store can be queried for stored metadata and that a
freshly-computed `content_hash` can be compared to decide whether a chunk
needs re-embedding/upsert.
"""

from __future__ import annotations

from pathlib import Path

from rag.chunking.models import Chunk
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
