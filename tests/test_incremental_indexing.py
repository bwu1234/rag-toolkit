"""Index-time incremental behaviour: change detection, stale-chunk removal, and
the manifest guard against mixing incompatible settings in one index.

The later tests drive the real `rag.cli index` command end to end against real
Chroma and BM25 stores in a temp dir; only the embedder is faked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import pytest

import rag.cli
from rag.chunking.models import Chunk
from rag.config.settings import (
    ChunkingConfig,
    CorporaConfig,
    CorpusConfig,
    PathsConfig,
    RagConfig,
    VectorStoreConfig,
)
from rag.index_manifest import IndexManifestMismatch, index_manifest_path, read_index_manifest
from rag.retrieval.builder import build_retriever
from rag.retrieval.sparse import BM25Index, bm25_index_path
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


# ---------------------------------------------------------------------------
# `rag.cli index`, end to end
# ---------------------------------------------------------------------------


class _CountingEmbedder:
    """Deterministic 4-d vectors from a text hash; counts texts embedded."""

    def __init__(self) -> None:
        self.embedded = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.embedded += len(texts)
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()
        return [b / 255 + 0.01 for b in digest[:4]]

    @property
    def dimensions(self) -> int:
        return 4


def _words(n: int) -> str:
    return " ".join(f"word{i}" for i in range(n))


@pytest.fixture()
def corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[RagConfig, _CountingEmbedder]:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "long.txt").write_text(_words(60), encoding="utf-8")
    (docs / "other.txt").write_text(_words(20), encoding="utf-8")

    config = RagConfig(
        paths=PathsConfig(index_dir=tmp_path / "index"),
        corpora=CorporaConfig(active=["c"], registry={"c": CorpusConfig(documents_dir=docs)}),
        chunking=ChunkingConfig(chunk_size=40, chunk_overlap=0),
        vector_store=VectorStoreConfig(provider="chroma", collection_name="rag_test"),
    )
    embedder = _CountingEmbedder()
    monkeypatch.setattr(rag.cli, "load_config", lambda _path: config)
    monkeypatch.setattr(rag.cli, "get_embedder", lambda _cfg: embedder)
    return config, embedder


def _index(*, reset: bool = False) -> None:
    rag.cli._cmd_index(argparse.Namespace(config=None, corpus=None, reset=reset, clear_context_cache=False))


def _stored_ids(config: RagConfig) -> tuple[set[str], set[str]]:
    """Chunk ids in (Chroma, BM25-on-disk), read back through fresh handles."""
    selection = config.corpus_selection()
    store = ChromaVectorStore(selection.index_dir, collection_name=selection.collection_name)
    sparse = BM25Index(bm25_index_path(selection.index_dir, selection.slug))
    return store.ids(), sparse.ids()


def _docs(config: RagConfig) -> Path:
    return config.corpora.registry["c"].documents_dir


def test_rerun_on_an_unchanged_corpus_embeds_and_removes_nothing(corpus) -> None:
    config, embedder = corpus
    _index()
    first = _stored_ids(config)
    embedder.embedded = 0

    _index()

    assert embedder.embedded == 0
    assert _stored_ids(config) == first


def test_deleted_document_is_removed_from_both_indexes(corpus) -> None:
    config, _ = corpus
    _index()
    vectors, sparse = _stored_ids(config)
    assert any(cid.startswith("other.txt::") for cid in vectors)
    assert vectors == sparse

    (_docs(config) / "other.txt").unlink()
    _index()

    vectors, sparse = _stored_ids(config)
    assert vectors and not any(cid.startswith("other.txt::") for cid in vectors)
    assert vectors == sparse, "BM25 on disk must match Chroma after the purge"


def test_shortened_document_loses_its_tail_chunks(corpus) -> None:
    config, _ = corpus
    _index()
    before = {cid for cid in _stored_ids(config)[0] if cid.startswith("long.txt::")}

    (_docs(config) / "long.txt").write_text(_words(10), encoding="utf-8")
    _index()

    vectors, sparse = _stored_ids(config)
    after = {cid for cid in vectors if cid.startswith("long.txt::")}
    assert 0 < len(after) < len(before)
    assert vectors == sparse


def test_first_build_records_a_manifest(corpus) -> None:
    config, _ = corpus
    _index()

    selection = config.corpus_selection()
    manifest = read_index_manifest(index_manifest_path(selection.index_dir, selection.slug))
    assert manifest is not None
    assert manifest.embedding["model"] == config.embedding.model
    assert manifest.contextual == {"enabled": False}


def test_changing_the_embedding_model_refuses_to_extend_the_index(corpus) -> None:
    config, embedder = corpus
    _index()
    before = _stored_ids(config)
    embedder.embedded = 0

    config.embedding.model = "some-other-embedder"
    (_docs(config) / "new.txt").write_text(_words(5), encoding="utf-8")
    with pytest.raises(IndexManifestMismatch, match="embedding.model"):
        _index()

    assert embedder.embedded == 0, "nothing may be embedded before the guard runs"
    assert _stored_ids(config) == before

    _index(reset=True)  # the documented way out
    assert any(cid.startswith("new.txt::") for cid in _stored_ids(config)[0])


def test_toggling_contextual_chunking_refuses_to_extend_the_index(corpus) -> None:
    config, _ = corpus
    _index()

    config.chunking.contextual.enabled = True
    with pytest.raises(IndexManifestMismatch, match="contextual.enabled"):
        _index()


def test_changing_the_header_template_refuses_to_extend_the_index(corpus) -> None:
    """The content hash covers `chunk.text` only, so old and new headers would mix."""
    config, _ = corpus
    _index()

    config.chunking.header.template = "{title}"
    with pytest.raises(IndexManifestMismatch, match="chunk_fields.header_template"):
        _index()


def test_changing_carried_metadata_refuses_to_extend_the_index(corpus) -> None:
    config, _ = corpus
    _index()

    config.chunking.carry_metadata = ["title", "ticker"]
    with pytest.raises(IndexManifestMismatch, match="chunk_fields.carry_metadata"):
        _index()


def test_a_manifest_from_before_chunk_fields_matches_the_defaults(corpus) -> None:
    """An index built before chunk fields were recorded carried the default keys, no header."""
    config, _ = corpus
    _index()
    selection = config.corpus_selection()
    path = index_manifest_path(selection.index_dir, selection.slug)
    raw = json.loads(path.read_text())
    del raw["chunk_fields"]
    path.write_text(json.dumps(raw))

    _index()  # not refused

    config.chunking.header.template = "{title}"
    with pytest.raises(IndexManifestMismatch, match="header_template: None -> '{title}'"):
        _index()


def test_the_header_is_embedded_and_indexed_but_not_stored_as_text(corpus) -> None:
    config, embedder = corpus
    (_docs(config) / "filing.md").write_text(
        "---\nticker: AAPL\n---\n" + _words(5), encoding="utf-8"
    )
    config.chunking.header.template = "{ticker} filing"
    seen: list[str] = []
    original = embedder.embed_documents
    embedder.embed_documents = lambda texts: seen.extend(texts) or original(texts)  # type: ignore[method-assign]

    _index()

    assert "AAPL filing\n\n" + _words(5) in seen
    selection = config.corpus_selection()
    [hit] = BM25Index(bm25_index_path(selection.index_dir, selection.slug)).query("AAPL", top_k=5)
    assert hit.document_id == "filing.md"
    assert hit.text == _words(5)
    assert hit.header == "AAPL filing"


def test_an_index_without_a_manifest_is_adopted(corpus, caplog: pytest.LogCaptureFixture) -> None:
    """Indexes built before manifests existed keep working, with a warning."""
    config, _ = corpus
    _index()
    selection = config.corpus_selection()
    path = index_manifest_path(selection.index_dir, selection.slug)
    path.unlink()

    _index()

    assert "has no manifest" in caplog.text
    assert read_index_manifest(path) is not None


def test_querying_with_a_different_embedder_is_refused(corpus) -> None:
    config, _ = corpus
    _index()

    config.embedding.model = "some-other-embedder"
    with pytest.raises(IndexManifestMismatch, match="embedding.model"):
        build_retriever(config)


def test_contextual_settings_do_not_matter_at_query_time(corpus) -> None:
    """A contextual index is queried with the same embedder, whatever the flag says."""
    config, _ = corpus
    _index()

    config.chunking.contextual.enabled = True
    build_retriever(config)


def test_cli_reports_a_mismatch_without_a_traceback(
    corpus, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config, _ = corpus
    _index()
    config.embedding.model = "some-other-embedder"
    monkeypatch.setattr(sys, "argv", ["rag.cli", "index"])

    with pytest.raises(SystemExit) as exit_info:
        rag.cli.main()

    assert exit_info.value.code == 2
    assert "--reset" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Store-level deletion
# ---------------------------------------------------------------------------


def test_chroma_delete_removes_only_the_named_ids(tmp_path: Path) -> None:
    store = ChromaVectorStore(persist_dir=tmp_path / "index", collection_name="del-test")
    store.upsert([_chunk("a"), _chunk("b"), _chunk("c")], [_AXIS, _AXIS, _AXIS])

    store.delete(["a", "c", "not-there"])

    assert store.ids() == {"b"}


def test_bm25_delete_is_persisted_by_flush(tmp_path: Path) -> None:
    path = tmp_path / "bm25.json"
    index = BM25Index(path)
    index.upsert([_chunk("a", "alpha"), _chunk("b", "beta")])
    index.flush()

    index.delete(["a", "not-there"])
    assert index.query("alpha", top_k=5) == []
    index.flush()

    assert BM25Index(path).ids() == {"b"}
