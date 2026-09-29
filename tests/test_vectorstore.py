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
from rag.query_filter import QueryFilter
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


def test_get_metadatas_batches_ids_past_the_sql_variable_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)
    store.upsert([_chunk(f"c{i}") for i in range(5)], [[1.0, float(i)] for i in range(5)])
    monkeypatch.setattr("rag.vectorstore.chroma_store.GET_BATCH_SIZE", 2)

    stored = store.get_metadatas([f"c{i}" for i in range(5)] + ["missing"])

    assert sorted(stored) == [f"c{i}" for i in range(5)]


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


# ---------------------------------------------------------------------------
# Contextual chunking: context round-trips, stored text stays verbatim
# ---------------------------------------------------------------------------


def test_chunk_context_survives_the_round_trip(tmp_path: Path) -> None:
    store = _store(tmp_path)
    chunk = Chunk(
        id="a",
        text="The limit is 1,000 requests per minute.",
        document_id="api.md",
        source=Path("api.md"),
        doc_type="markdown",
        context="ACS API rate limiting.",
    )
    store.upsert([chunk], [_AXIS_X])

    [result] = store.query(_AXIS_X, top_k=1)

    assert result.context == "ACS API rate limiting."
    assert result.text == "The limit is 1,000 requests per minute.", "stored text must stay the verbatim span"
    assert result.index_text == "ACS API rate limiting.\n\nThe limit is 1,000 requests per minute."


def test_chunk_without_context_round_trips_as_none(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.upsert([_chunk("a")], [_AXIS_X])

    [result] = store.query(_AXIS_X, top_k=1)

    assert result.context is None
    assert "context" not in result.metadata, "an absent context must not leak into the metadata grab-bag"


def test_chunk_header_survives_the_round_trip(tmp_path: Path) -> None:
    store = _store(tmp_path)
    chunk = Chunk(
        id="a",
        text="Revenue grew 2%.",
        document_id="AAPL.md",
        source=Path("AAPL.md"),
        doc_type="markdown",
        metadata={"period_end": 20240928},
        header="Apple Inc. (AAPL) 10-K, period ended 2024-09-28",
    )
    store.upsert([chunk], [_AXIS_X])

    [result] = store.query(_AXIS_X, top_k=1)

    assert result.header == "Apple Inc. (AAPL) 10-K, period ended 2024-09-28"
    assert "header" not in result.metadata, "provenance fields are split back out of the flat metadata"
    assert result.metadata["period_end"] == 20240928
    assert result.text == "Revenue grew 2%."


# ---------------------------------------------------------------------------
# Metadata filters (chunking plan, Phase 3)
# ---------------------------------------------------------------------------


def _filing_chunks() -> tuple[list[Chunk], list[list[float]]]:
    """Ten chunks, two tickers and two periods; AAPL's are the *least* similar to _AXIS_X."""
    chunks, vectors = [], []
    for i in range(10):
        ticker = "MSFT" if i < 6 else "AAPL"
        period = 20240928 if i % 2 else 20250927
        chunks.append(_chunk(f"c{i}", f"text {i}", source=f"{ticker}.md", ticker=ticker, period_end=period))
        vectors.append([1.0, i / 10, 0.0])
    return chunks, vectors


def test_filtered_query_never_returns_a_chunk_outside_the_filter(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.upsert(*_filing_chunks())
    only_2024_aapl = QueryFilter(equals={"ticker": "AAPL"}, range={"period_end": {"lte": 20241231}})

    results = store.query(_AXIS_X, top_k=10, query_filter=only_2024_aapl)

    assert results, "matching chunks exist"
    assert all(r.metadata["ticker"] == "AAPL" and r.metadata["period_end"] <= 20241231 for r in results)


def test_filter_applies_before_top_k_so_results_are_not_short(tmp_path: Path) -> None:
    # The 4 AAPL chunks rank below all 6 MSFT ones, so filtering an unfiltered
    # top-3 afterwards would return none. Pre-filtering returns 3.
    store = _store(tmp_path)
    store.upsert(*_filing_chunks())

    results = store.query(_AXIS_X, top_k=3, query_filter=QueryFilter(equals={"ticker": "AAPL"}))

    assert [r.chunk_id for r in results] == ["c6", "c7", "c8"]


# ---------------------------------------------------------------------------
# HNSW `ef_search`: applied before Chroma loads the segment, fixed per process
# ---------------------------------------------------------------------------


def _ef_search(store: ChromaVectorStore) -> int:
    return store._collection.configuration["hnsw"]["ef_search"]


def test_configured_ef_search_is_written_to_the_collection(tmp_path: Path) -> None:
    assert _ef_search(ChromaVectorStore(tmp_path / "index", "ef-default")) == 100
    assert _ef_search(ChromaVectorStore(tmp_path / "index", "ef-wide", hnsw_ef_search=400)) == 400


def test_the_factory_passes_ef_search(tmp_path: Path) -> None:
    store = get_vector_store(VectorStoreConfig(collection_name="ef-factory", hnsw_ef_search=300), tmp_path / "index")

    assert isinstance(store, ChromaVectorStore)
    assert _ef_search(store) == 300


def test_a_new_process_takes_the_configured_value_over_the_stored_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import rag.vectorstore.chroma_store as chroma_store

    ChromaVectorStore(tmp_path / "index", "ef-persisted", hnsw_ef_search=400)
    # A later process has no record of what this one opened.
    monkeypatch.setattr(chroma_store, "_EF_SEARCH_IN_PROCESS", {})

    # Chroma persisted 400, but the config now says 100, and config wins.
    assert _ef_search(ChromaVectorStore(tmp_path / "index", "ef-persisted")) == 100


def test_a_second_value_in_one_process_is_refused(tmp_path: Path) -> None:
    ChromaVectorStore(tmp_path / "index", "ef-once", hnsw_ef_search=200)
    ChromaVectorStore(tmp_path / "index", "ef-once", hnsw_ef_search=200)  # same value: fine

    # Chroma would keep searching with 200 until the process restarts.
    with pytest.raises(ValueError, match="already opened in this process"):
        ChromaVectorStore(tmp_path / "index", "ef-once", hnsw_ef_search=800)


def test_reset_keeps_the_configured_ef_search(tmp_path: Path) -> None:
    store = ChromaVectorStore(tmp_path / "index", "ef-reset", hnsw_ef_search=250)
    store.reset()

    assert _ef_search(store) == 250


def test_a_wider_beam_finds_more_of_the_exact_top_k(tmp_path: Path) -> None:
    """The value takes effect: set after Chroma loads the segment, it would be silently ignored."""
    import random

    rng = random.Random(0)
    dims, n, k = 32, 3000, 10
    vectors = [[rng.gauss(0, 1) for _ in range(dims)] for _ in range(n)]
    queries = [[rng.gauss(0, 1) for _ in range(dims)] for _ in range(40)]
    chunks = [_chunk(f"c{i}") for i in range(n)]

    def cosine(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b)) / (sum(x * x for x in a) * sum(y * y for y in b)) ** 0.5

    def recall(ef: int) -> float:
        store = ChromaVectorStore(tmp_path / "index", f"ef-recall-{ef}", hnsw_ef_search=ef)
        store.upsert(chunks, vectors)
        found = 0
        for q in queries:
            exact = sorted(range(n), key=lambda i: -cosine(q, vectors[i]))[:k]
            got = {r.chunk_id for r in store.query(q, top_k=k)}
            found += len(got & {f"c{i}" for i in exact})
        return found / (k * len(queries))

    narrow, wide = recall(1), recall(500)
    assert wide > narrow
    assert wide > 0.99
