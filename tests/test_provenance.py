"""Eval provenance: config classification, dataset identity, and index build verification.

The index tests drive the real `rag.cli index` command against real Chroma and
BM25 stores in a temp dir, with only the embedder faked, so a build record is
checked against what the indexer actually wrote.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest

import rag.cli
from rag.config.settings import (
    AgentConfig,
    ChunkingConfig,
    CorporaConfig,
    CorpusConfig,
    EvalConfig,
    LLMConfig,
    PathsConfig,
    RagConfig,
    VectorStoreConfig,
)
from rag.eval.dataset import EvalDataset
from rag.eval.provenance import (
    CONFIG_HASH_DENYLIST,
    IndexConflict,
    checked_index,
    config_hash,
    config_leaves,
    dataset_hash,
    hashed_config,
    provenance_record,
    selection_hash,
    verify_index,
    write_dataset_snapshot,
)
from rag.index_manifest import index_manifest_path, read_index_build
from rag.ingestion.corpora import chunk_selected_corpora

# ---------------------------------------------------------------------------
# Config classification
# ---------------------------------------------------------------------------

#: Every config leaf `config_hash` covers. A new `RagConfig` field fails
#: `test_every_config_leaf_is_classified` until it is added here or to
#: `CONFIG_HASH_DENYLIST` with a reason -- the explicit decision an allowlist
#: fingerprint never forced, and the reason the old one drifted.
HASHED = {
    'paths.corpus_dir', 'paths.index_dir',
    'corpora.active', 'corpora.registry.*.documents_dir', 'corpora.registry.*.clean',
    'embedding.provider', 'embedding.model', 'embedding.base_url', 'embedding.dimensions',
    'embedding.query_instruction', 'embedding.revision',
    'llm.provider', 'llm.model', 'llm.base_url', 'llm.temperature', 'llm.max_tokens', 'llm.think',
    'llm.timeout_s', 'llm.requests_per_minute', 'llm.tokens_per_minute', 'llm.requests_per_day',
    'llm.requests_per_day_reserve', 'llm.thinking_level', 'llm.raw',
    'chunking.strategy', 'chunking.chunk_size', 'chunking.chunk_overlap',
    'chunking.contextual.enabled', 'chunking.contextual.max_document_chars',
    'chunking.contextual.max_context_chars', 'chunking.carry_metadata', 'chunking.header.template',
    'chunking.structured.split_level', 'chunking.structured.min_chars',
    'chunking.structured.prose_overlap',
    'vector_store.provider', 'vector_store.collection_name', 'vector_store.hnsw_ef_search',
    'sparse_index.provider',
    'reranker.provider', 'reranker.model', 'reranker.aggregate', 'reranker.query_prefix',
    'reranker.document_prefix', 'reranker.include_header', 'reranker.max_length',
    'retrieval.top_k', 'retrieval.rerank_top_k', 'retrieval.mode', 'retrieval.rrf_k',
    'retrieval.min_score', 'retrieval.expansion.provider', 'retrieval.expansion.num_queries',
    'retrieval.expansion.num_documents', 'retrieval.expansion.include_original',
    'retrieval.web_search.enabled', 'retrieval.web_search.searxng_url',
    'retrieval.web_search.top_k', 'retrieval.web_search.min_similarity',
    'retrieval.web_search.dedup_threshold', 'retrieval.web_search.timeout_s',
    'retrieval.document_routing.top_m', 'retrieval.document_routing.record_template',
    'chat.mode', 'chat.condense_history', 'chat.max_history_turns', 'chat.prompt', 'chat.passage_delimiters',
    'crag.enabled', 'crag.grade_documents', 'crag.max_retries', 'crag.check_groundedness',
    'crag.max_regenerations',
    'eval.judge.provider', 'eval.judge.model', 'eval.judge.base_url', 'eval.judge.temperature',
    'eval.judge.max_tokens', 'eval.judge.think', 'eval.judge.timeout_s',
    'eval.judge.requests_per_minute', 'eval.judge.tokens_per_minute',
    'eval.judge.requests_per_day', 'eval.judge.requests_per_day_reserve',
    'eval.judge.thinking_level', 'eval.judge.raw',
    'agent.llm.provider', 'agent.llm.model', 'agent.llm.base_url', 'agent.llm.temperature',
    'agent.llm.max_tokens', 'agent.llm.think', 'agent.llm.timeout_s',
    'agent.llm.requests_per_minute', 'agent.llm.tokens_per_minute', 'agent.llm.requests_per_day',
    'agent.llm.requests_per_day_reserve', 'agent.llm.thinking_level', 'agent.llm.raw',
    'agent.strategy', 'agent.max_tool_calls', 'agent.timeout_s', 'agent.turn_deadline_s',
    'agent.synthesis_reserve_s', 'agent.max_turn_tokens', 'agent.max_passage_chars',
    'agent.max_read_chars', 'agent.num_ctx', 'agent.model_filters', 'agent.tools',
}


def test_every_config_leaf_is_classified() -> None:
    leaves = set(config_leaves())
    assert not HASHED & CONFIG_HASH_DENYLIST.keys(), "a leaf is both hashed and denylisted"
    unclassified = leaves - HASHED - CONFIG_HASH_DENYLIST.keys()
    assert not unclassified, (
        f"New config field(s) {sorted(unclassified)}: add each to HASHED, or to "
        "CONFIG_HASH_DENYLIST with the reason it cannot change an eval result"
    )
    assert not (HASHED | CONFIG_HASH_DENYLIST.keys()) - leaves, "a classified leaf no longer exists"


def test_every_denylist_entry_has_a_reason() -> None:
    assert all(reason.strip() for reason in CONFIG_HASH_DENYLIST.values())


def _fully_populated() -> RagConfig:
    """A config where every optional model and mapping is set, so every leaf is present."""
    return RagConfig(
        corpora=CorporaConfig(active=["c"], registry={"c": CorpusConfig(documents_dir=Path("d"))}),
        eval=EvalConfig(judge=LLMConfig()),
        agent=AgentConfig(llm=LLMConfig()),
    )


def _present_leaves(tree: object, prefix: str = "") -> set[str]:
    """Dotted paths of `tree`'s leaves, mapping keys under `corpora.registry` as ``*``."""
    if not isinstance(tree, dict):
        return {prefix.rstrip(".")}
    paths: set[str] = set()
    for key, value in tree.items():
        segment = "*" if prefix == "corpora.registry." else key
        paths |= _present_leaves(value, f"{prefix}{segment}.")
    return paths


def test_config_hash_covers_exactly_the_hashed_leaves() -> None:
    assert _present_leaves(hashed_config(_fully_populated())) == HASHED


def test_a_field_the_old_fingerprint_missed_changes_the_hash() -> None:
    base = RagConfig()
    for change in (
        {"chunking": {"strategy": "structured"}},
        {"reranker": {"query_prefix": "q: "}},
        {"embedding": {"provider": "sentence_transformers"}},
        {"agent": {"max_read_chars": 1}},
    ):
        changed = RagConfig.model_validate({**base.model_dump(), **change})
        assert config_hash(changed) != config_hash(base), change


def test_a_denylisted_field_leaves_the_hash_alone() -> None:
    base = _fully_populated()
    changed = base.model_copy(deep=True)
    changed.observability.turn_log.path = Path("elsewhere.jsonl")
    changed.corpora.registry["c"].description = "now described"
    assert changed.agent.llm is not None
    changed.agent.llm.daily_request_log = "other.json"
    assert config_hash(changed) == config_hash(base)


# ---------------------------------------------------------------------------
# Dataset identity
# ---------------------------------------------------------------------------


def _dataset(**extra: object) -> EvalDataset:
    return EvalDataset.from_dicts([
        {"id": "a", "query": "q1", "expected_spans": [{"text": "x", "grade": 2}], **extra},
        {"id": "b", "query": "q2", "expected_doc_ids": ["d.md"]},
    ])


def test_dataset_hash_covers_extra_fields_and_grades() -> None:
    base = dataset_hash(_dataset())
    assert dataset_hash(_dataset(tier="period")) != base
    regraded = _dataset()
    regraded.samples[0].expected_spans[0] = type(regraded.samples[0].expected_spans[0])(text="x", grade=3)
    assert dataset_hash(regraded) != base


def test_dataset_hash_ignores_formatting(tmp_path: Path) -> None:
    path = tmp_path / "set.json"
    _dataset().save(path)
    compact = tmp_path / "compact.json"
    compact.write_text(json.dumps(json.loads(path.read_text())), encoding="utf-8")
    assert dataset_hash(EvalDataset.load(compact)) == dataset_hash(EvalDataset.load(path))


def test_duplicate_sample_ids_are_refused() -> None:
    twice = EvalDataset.from_dicts([{"id": "a", "query": "q"}, {"id": "a", "query": "q again"}])
    with pytest.raises(ValueError, match="duplicate sample ids: a"):
        dataset_hash(twice)


def test_selection_hash_tells_a_limited_run_from_a_full_one() -> None:
    assert selection_hash(["a", "b"]) != selection_hash(["a"])
    assert selection_hash(["a", "b"]) != selection_hash(["b", "a"])


def test_snapshot_is_named_by_content_and_written_once(tmp_path: Path) -> None:
    first = write_dataset_snapshot(tmp_path, _dataset())
    assert first.name == f"{dataset_hash(_dataset())}.json"
    assert write_dataset_snapshot(tmp_path, _dataset()) == first
    assert write_dataset_snapshot(tmp_path, _dataset(tier="period")) != first
    assert len(list(tmp_path.iterdir())) == 2


# ---------------------------------------------------------------------------
# Index builds, end to end through `rag.cli index`
# ---------------------------------------------------------------------------


class _Embedder:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return [b / 255 + 0.01 for b in hashlib.sha256(text.encode()).digest()[:4]]

    @property
    def dimensions(self) -> int:
        return 4


def _words(n: int) -> str:
    return " ".join(f"word{i}" for i in range(n))


@pytest.fixture()
def indexed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, RagConfig]:
    """A small corpus and the config `rag.cli index` will build it with (``holder["config"]``)."""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "long.txt").write_text(_words(60), encoding="utf-8")
    (docs / "other.txt").write_text(_words(20), encoding="utf-8")
    holder = {"config": RagConfig(
        paths=PathsConfig(index_dir=tmp_path / "index"),
        corpora=CorporaConfig(active=["c"], registry={"c": CorpusConfig(documents_dir=docs)}),
        chunking=ChunkingConfig(chunk_size=40, chunk_overlap=0),
        vector_store=VectorStoreConfig(provider="chroma", collection_name="rag_test"),
    )}
    monkeypatch.setattr(rag.cli, "load_config", lambda _path: holder["config"])
    monkeypatch.setattr(rag.cli, "get_embedder", lambda _cfg: _Embedder())
    return holder


def _index(*, reset: bool = False) -> None:
    rag.cli._cmd_index(argparse.Namespace(config=None, corpus=None, reset=reset, clear_context_cache=False))


def _manifest(config: RagConfig) -> Path:
    selection = config.corpus_selection()
    return index_manifest_path(selection.index_dir, selection.slug)


def _verify(config: RagConfig):
    selection, documents, chunks = chunk_selected_corpora(config, None)
    return verify_index(config, selection, documents, chunks)


def test_a_fresh_build_verifies(indexed) -> None:
    _index()
    build = read_index_build(_manifest(indexed["config"]))
    assert build is not None and build.status == "complete" and build.build_id
    provenance = _verify(indexed["config"])
    assert provenance.verified
    assert provenance.build_id == build.build_id


def test_a_rerun_that_writes_nothing_keeps_the_build_id(indexed) -> None:
    _index()
    first = read_index_build(_manifest(indexed["config"]))
    _index()
    second = read_index_build(_manifest(indexed["config"]))
    assert first is not None and second is not None
    assert second.build_id == first.build_id


def test_reset_and_corpus_edits_get_a_new_build_id(indexed) -> None:
    config = indexed["config"]
    _index()
    first = read_index_build(_manifest(config))
    _index(reset=True)
    after_reset = read_index_build(_manifest(config))
    (config.corpora.registry["c"].documents_dir / "other.txt").write_text(_words(5), encoding="utf-8")
    _index()
    after_edit = read_index_build(_manifest(config))
    assert first and after_reset and after_edit
    assert len({first.build_id, after_reset.build_id, after_edit.build_id}) == 3
    assert _verify(config).verified


def test_an_edited_corpus_not_yet_reindexed_conflicts(indexed) -> None:
    config = indexed["config"]
    _index()
    (config.corpora.registry["c"].documents_dir / "other.txt").write_text(_words(5), encoding="utf-8")
    conflicts = " ".join(_verify(config).conflicts)
    assert "corpus has changed" in conflicts
    assert "stale" in conflicts


def test_a_chunking_change_not_yet_reindexed_conflicts(indexed) -> None:
    _index()
    rechunked = indexed["config"].model_copy(deep=True)
    rechunked.chunking.chunk_size = 25
    provenance = _verify(rechunked)
    assert any("chunker.chunk_size: 40 -> 25" in c for c in provenance.conflicts)
    selection, documents, chunks = chunk_selected_corpora(rechunked, None)
    with pytest.raises(IndexConflict, match="chunk_size"):
        checked_index(rechunked, selection, documents, chunks, label="rechunked")


def test_a_query_time_change_still_verifies(indexed) -> None:
    _index()
    tuned = indexed["config"].model_copy(deep=True)
    tuned.retrieval.rerank_top_k = 2
    tuned.reranker.include_header = True
    tuned.embedding.query_instruction = "Find passages"
    assert _verify(tuned).verified


def test_an_interrupted_build_is_unverified_and_forfeits_its_id(indexed) -> None:
    config = indexed["config"]
    _index()
    path = _manifest(config)
    first = read_index_build(path)
    # What a killed run leaves: the marker `prepare_for_indexing` writes, never finished.
    raw = json.loads(path.read_text())
    raw["build"] = {"status": "building", "previous": raw["build"]}
    path.write_text(json.dumps(raw))

    provenance = _verify(config)
    assert not provenance.verified and not provenance.conflicts
    assert "building" in (provenance.unverified or "")

    _index()
    rebuilt = read_index_build(path)
    assert first and rebuilt and rebuilt.status == "complete"
    assert rebuilt.build_id != first.build_id, "nothing says what the killed run changed"


def test_an_index_from_before_build_records_is_unverified_not_refused(indexed) -> None:
    config = indexed["config"]
    _index()
    path = _manifest(config)
    raw = json.loads(path.read_text())
    del raw["build"]
    raw["version"] = 1
    path.write_text(json.dumps(raw))

    selection, documents, chunks = chunk_selected_corpora(config, None)
    provenance = checked_index(config, selection, documents, chunks, label="old")
    assert not provenance.verified and provenance.build_id is None
    assert "rag.cli index" in (provenance.unverified or "")


def test_no_index_is_a_conflict(indexed) -> None:
    assert "no index is built" in " ".join(_verify(indexed["config"]).conflicts)


def test_provenance_record_carries_every_identity(indexed) -> None:
    config = indexed["config"]
    _index()
    full = _dataset()
    selected = EvalDataset(samples=full.samples[:1])
    record = provenance_record(
        config=config, dataset_path=Path("set.json"), dataset=full, selected=selected,
        selection="evenly spaced, --limit 1", code="abc", index=_verify(config), judge=LLMConfig(),
    )
    assert record["config_hash"] == config_hash(config)
    assert record["code_version"] == "abc"
    assert record["judge_hash"]
    assert record["dataset"]["dataset_hash"] == dataset_hash(full)
    assert record["dataset"]["selection_hash"] == selection_hash(["a"])
    assert (record["dataset"]["num_samples"], record["dataset"]["num_selected"]) == (2, 1)
    assert record["index"]["verified"] is True
    assert record["index"]["index_build_id"] == read_index_build(_manifest(config)).build_id  # type: ignore[union-attr]
    json.dumps(record)  # it is written into a results file
