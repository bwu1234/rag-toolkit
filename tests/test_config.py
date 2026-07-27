"""Sanity checks for the config-loading system (Milestone 1 scaffold)."""

from __future__ import annotations

from pathlib import Path

from rag.config.settings import RagConfig, load_config


def test_default_config_has_expected_models(default_config: RagConfig) -> None:
    assert default_config.embedding.model == "qwen3-embedding:0.6b"
    assert default_config.embedding.provider == "ollama"
    assert default_config.llm.model == "qwen3.5:4b"
    assert default_config.llm.provider == "ollama"


def test_default_config_chunking_is_positive(default_config: RagConfig) -> None:
    assert default_config.chunking.chunk_size > 0
    assert default_config.chunking.chunk_overlap >= 0
    assert default_config.chunking.chunk_overlap < default_config.chunking.chunk_size


def test_load_config_reads_repo_yaml() -> None:
    cfg = load_config()
    assert cfg.embedding.model == "qwen3-embedding:0.6b"
    assert cfg.vector_store.collection_name == "rag_corpus"


def test_load_config_missing_file_falls_back_to_defaults(tmp_path: Path) -> None:
    cfg = load_config(tmp_path / "does_not_exist.yaml")
    assert cfg == RagConfig()


def test_paths_resolve_relative_to_repo_root(default_config: RagConfig) -> None:
    resolved = default_config.paths.resolved()
    assert resolved.corpus_dir.is_absolute()
    assert resolved.index_dir.is_absolute()
    assert resolved.corpus_dir.name == "corpus"
