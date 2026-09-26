"""Sanity checks for the config-loading system (Milestone 1 scaffold)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from rag.config.settings import RagConfig, load_config


def test_default_config_has_expected_models(default_config: RagConfig) -> None:
    assert default_config.embedding.model == "qwen3-embedding:0.6b"
    assert default_config.embedding.provider == "ollama"
    assert default_config.llm.model == "qwen3.5:9b-mlx"
    assert default_config.llm.provider == "ollama"


def test_default_config_chunking_is_positive(default_config: RagConfig) -> None:
    assert default_config.chunking.chunk_size > 0
    assert default_config.chunking.chunk_overlap >= 0
    assert default_config.chunking.chunk_overlap < default_config.chunking.chunk_size


def test_load_config_reads_repo_yaml() -> None:
    cfg = load_config()
    assert cfg.embedding.model == "qwen3-embedding:0.6b"
    assert cfg.vector_store.collection_name == "rag_corpus"
    assert cfg.retrieval.mode == "hybrid"
    assert cfg.retrieval.rrf_k == 60


def test_default_retrieval_mode_is_dense() -> None:
    """Pydantic defaults stay dense so missing keys don't flip behavior."""

    cfg = RagConfig()
    assert cfg.retrieval.mode == "dense"
    assert cfg.retrieval.rrf_k == 60


def test_load_config_missing_file_falls_back_to_defaults(tmp_path: Path) -> None:
    cfg = load_config(tmp_path / "does_not_exist.yaml")
    assert cfg == RagConfig()


def test_paths_resolve_relative_to_repo_root(default_config: RagConfig) -> None:
    resolved = default_config.paths.resolved()
    assert resolved.corpus_dir.is_absolute()
    assert resolved.index_dir.is_absolute()
    assert resolved.corpus_dir.name == "documents"


# ---------------------------------------------------------------------------
# Contextual chunking and CRAG defaults
# ---------------------------------------------------------------------------


def test_contextual_chunking_is_off_by_default() -> None:
    # Both features cost LLM calls on paths that previously had none (per chunk
    # at index time, per passage at query time), so neither may turn itself on.
    assert RagConfig().chunking.contextual.enabled is False


def test_crag_is_off_by_default() -> None:
    assert RagConfig().crag.enabled is False


def test_shipped_config_leaves_both_new_features_off() -> None:
    config = load_config()

    assert config.chunking.contextual.enabled is False
    assert config.crag.enabled is False


def test_crag_sub_flags_are_configurable(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "crag:\n"
        "  enabled: true\n"
        "  grade_documents: false\n"
        "  max_retries: 2\n"
        "  check_groundedness: true\n"
        "  max_regenerations: 0\n"
    )

    config = load_config(path)

    assert config.crag.enabled is True
    assert config.crag.grade_documents is False
    assert config.crag.max_retries == 2
    assert config.crag.max_regenerations == 0


def test_crag_rejects_an_out_of_range_retry_count(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("crag:\n  max_retries: 99\n")

    with pytest.raises(ValidationError):
        load_config(path)


# ---------------------------------------------------------------------------
# RAG__SECTION__KEY environment overrides
# ---------------------------------------------------------------------------


def test_env_overrides_ollama_urls_over_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    # The docker-compose case: the shared YAML says localhost, the container
    # needs the `ollama` service name.
    monkeypatch.setenv("RAG__LLM__BASE_URL", "http://ollama:11434")
    monkeypatch.setenv("RAG__EMBEDDING__BASE_URL", "http://ollama:11434")

    config = load_config()

    assert config.llm.base_url == "http://ollama:11434"
    assert config.embedding.base_url == "http://ollama:11434"
    # Untouched keys in the same section still come from the YAML.
    assert config.llm.model == "qwen3.5:9b-mlx"


def test_env_override_values_are_coerced_by_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG__RETRIEVAL__TOP_K", "7")
    monkeypatch.setenv("RAG__CHAT__CONDENSE_HISTORY", "false")

    config = load_config()

    assert config.retrieval.top_k == 7
    assert config.chat.condense_history is False


def test_env_override_applies_without_a_config_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAG__LLM__BASE_URL", "http://ollama:11434")

    config = load_config(tmp_path / "does_not_exist.yaml")

    assert config.llm.base_url == "http://ollama:11434"


def test_env_override_is_still_validated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG__CRAG__MAX_RETRIES", "99")

    with pytest.raises(ValidationError):
        load_config()


@pytest.mark.parametrize("name", ["RAG__", "RAG__LLM__", "RAG__LLM____BASE_URL"])
def test_malformed_env_override_raises(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(name, "x")

    with pytest.raises(ValueError, match="Malformed"):
        load_config()


def test_env_override_cannot_descend_into_a_scalar(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG__LLM__MODEL__NAME", "x")

    with pytest.raises(ValueError, match="non-mapping"):
        load_config()
