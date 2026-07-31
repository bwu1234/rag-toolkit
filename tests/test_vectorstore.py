"""Tests for the VectorStore interface, Chroma adapter, and factory.

Uses a real (but temp-directory-backed) Chroma instance rather than mocking it
-- Chroma's persistent client is fast and self-contained, and the adapter's
whole job is translating to/from Chroma's metadata and distance conventions,
which is best verified end to end.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.chunking.models import Chunk
from rag.config.settings import VectorStoreConfig
from rag.vectorstore.base import ScoredChunk, VectorStore
from rag.vectorstore.chroma_store import ChromaVectorStore
from rag.vectorstore.factory import get_vector_store

# Orthogonal unit vectors give exactly-predictable cosine similarity:
# a query equal to one axis is perfectly similar to that axis (score 1.0)
# and equally (in)similar to the other two (score 0.0).
_AXIS_X = [1.0, 0.0, 0.0]
_AXIS_Y = [0.0, 1.0, 0.0]
_AXIS_Z = [0.0, 0.0, 1.0]


def _chunk(chunk_id: str, text: str = "text", *, source: str = "doc.md", **metadata: object) -> Chunk:
    return Chunk(
        id=chunk_id,
        text=text,
        document_id=source,
        source=Path(source),
        doc_type="markdown",
        metadata=dict(metadata),
    )


def _store(tmp_path: Path, name: str = "test-collection") -> ChromaVectorStore:
    return ChromaVectorStore(persist_dir=tmp_path / "index", collection_name=name)


def test_upsert_and_count(tmp_path: Path) -> None:
    store = _store(tmp_path)

    store.upsert([_chunk("a"), _chunk("b")], [_AXIS_X, _AXIS_Y])

    assert store.count() == 2


def test_upsert_rejects_mismatched_lengths(tmp_path: Path) -> None:
    store = _store(tmp_path)

    with pytest.raises(ValueError, match="must be the same length"):
        store.upsert([_chunk("a"), _chunk("b")], [_AXIS_X])


def test_upsert_empty_list_is_a_noop(tmp_path: Path) -> None:
    store = _store(tmp_path)

    store.upsert([], [])

    assert store.count() == 0


def test_upsert_is_idempotent_on_chunk_id(tmp_path: Path) -> None:
    store = _store(tmp_path)

    store.upsert([_chunk("a", "first version")], [_AXIS_X])
    store.upsert([_chunk("a", "second version")], [_AXIS_Y])

    assert store.count() == 1
    [result] = store.query(_AXIS_Y, top_k=1)
    assert result.text == "second version"


def test_query_ranks_by_similarity_best_first(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.upsert(
        [_chunk("x-aligned", "x"), _chunk("y-aligned", "y"), _chunk("z-aligned", "z")],
        [_AXIS_X, _AXIS_Y, _AXIS_Z],
    )

    results = store.query(_AXIS_X, top_k=3)

    assert [r.chunk_id for r in results] == ["x-aligned", "y-aligned", "z-aligned"]
    assert results[0].score == pytest.approx(1.0)
    assert results[1].score == pytest.approx(0.0, abs=1e-6)
    assert all(isinstance(r, ScoredChunk) for r in results)


def test_query_respects_top_k(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.upsert(
        [_chunk("x", "x"), _chunk("y", "y"), _chunk("z", "z")],
        [_AXIS_X, _AXIS_Y, _AXIS_Z],
    )

    assert len(store.query(_AXIS_X, top_k=1)) == 1
    assert len(store.query(_AXIS_X, top_k=10)) == 3


def test_query_on_empty_store_returns_empty_list(tmp_path: Path) -> None:
    store = _store(tmp_path)

    assert store.query(_AXIS_X, top_k=5) == []


def test_reset_empties_the_collection(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.upsert([_chunk("a"), _chunk("b")], [_AXIS_X, _AXIS_Y])
    assert store.count() == 2

    store.reset()

    assert store.count() == 0
    assert store.query(_AXIS_X, top_k=5) == []


def test_metadata_round_trips_with_provenance_split_out(tmp_path: Path) -> None:
    store = _store(tmp_path)
    chunk = Chunk(
        id="c",
        text="hello",
        document_id="handbook.pdf#page=2",
        source=Path("/corpus/handbook.pdf"),
        doc_type="pdf",
        metadata={"title": "Handbook", "page": 2, "page_count": None, "chunk_index": 0},
    )

    store.upsert([chunk], [_AXIS_X])
    [result] = store.query(_AXIS_X, top_k=1)

    assert result.chunk_id == "c"
    assert result.text == "hello"
    assert result.document_id == "handbook.pdf#page=2"
    assert result.source == Path("/corpus/handbook.pdf")
    assert result.doc_type == "pdf"
    # `None` values are dropped on the way in -- they never round-trip.
    assert result.metadata == {"title": "Handbook", "page": 2, "chunk_index": 0}


def test_get_vector_store_factory_selects_chroma(tmp_path: Path) -> None:
    store = get_vector_store(
        VectorStoreConfig(provider="chroma", collection_name="c-collection"),
        index_dir=tmp_path / "index",
    )

    assert isinstance(store, VectorStore)
    assert isinstance(store, ChromaVectorStore)


def test_get_vector_store_factory_rejects_unknown_provider(tmp_path: Path) -> None:
    config = VectorStoreConfig.model_construct(provider="pinecone", collection_name="c-collection")

    with pytest.raises(ValueError, match="Unknown vector store provider"):
        get_vector_store(config, index_dir=tmp_path / "index")
