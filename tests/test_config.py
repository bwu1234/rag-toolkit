"""Sanity checks for the config-loading system (Milestone 1 scaffold)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from rag.config.settings import (
    DEFAULT_CONFIG_PATH,
    GEMINI_BASE_URL,
    AgentConfig,
    LLMConfig,
    RagConfig,
    _deep_merge,
    load_config,
)
from rag.mcp.tools import DEFAULT_MAX_CHARS


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


# ---------------------------------------------------------------------------
# Vanilla baseline config
# ---------------------------------------------------------------------------

VANILLA_CONFIG_PATH = DEFAULT_CONFIG_PATH.parent / "vanilla.yaml"


def test_vanilla_config_disables_every_extra() -> None:
    cfg = load_config(VANILLA_CONFIG_PATH)

    assert cfg.retrieval.mode == "dense"
    assert cfg.reranker.provider == "none"
    assert cfg.retrieval.expansion.provider == "none"
    assert not cfg.retrieval.web_search.enabled
    assert cfg.retrieval.min_score == 0.0
    assert not cfg.chat.condense_history
    assert cfg.chat.prompt == "plain"
    assert not cfg.crag.enabled
    assert not cfg.chunking.contextual.enabled
    assert cfg.retrieval.top_k >= cfg.retrieval.rerank_top_k


def test_vanilla_config_inherits_everything_it_does_not_change() -> None:
    default = load_config()
    vanilla = load_config(VANILLA_CONFIG_PATH)

    changed = {
        "retrieval": {"mode": "dense", "top_k": 5},
        "reranker": {"provider": "none"},
        "chat": {"condense_history": False, "prompt": "plain"},
    }
    expected = RagConfig.model_validate(_deep_merge(default.model_dump(), changed))
    assert vanilla == expected


@pytest.mark.parametrize("version", ["3.1", "3.5"])
def test_flash_lite_configs_change_only_the_generator(version: str) -> None:
    default = load_config()
    cfg = load_config(DEFAULT_CONFIG_PATH.parent / f"gemini-{version}-flash-lite.yaml")

    assert cfg.llm.provider == "gemini"
    assert cfg.llm.model == f"gemini-{version}-flash-lite"
    # The inherited Ollama URL must not survive the provider switch.
    assert cfg.llm.base_url == GEMINI_BASE_URL
    assert (cfg.llm.requests_per_minute, cfg.llm.tokens_per_minute) == (15, 250_000)
    assert cfg.llm.thinking_level == "minimal"
    # Graded by the fixed local judge, not by Flash-Lite itself.
    assert cfg.eval.judge is not None
    assert (cfg.eval.judge.provider, cfg.eval.judge.model, cfg.eval.judge.temperature) == (
        "ollama", "gemma4:31b-mlx", 0.0,
    )
    assert cfg.model_dump(exclude={"llm", "eval"}) == default.model_dump(exclude={"llm", "eval"})


def test_flash_lite_configs_differ_only_in_the_model() -> None:
    old, new = (
        load_config(DEFAULT_CONFIG_PATH.parent / f"gemini-{v}-flash-lite.yaml") for v in ("3.1", "3.5")
    )
    assert old.model_copy(update={"llm": old.llm.model_copy(update={"model": new.llm.model})}) == new


# ---------------------------------------------------------------------------
# `base:` inheritance
# ---------------------------------------------------------------------------


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_base_merges_nested_mappings_and_child_wins(tmp_path: Path) -> None:
    _write(tmp_path / "base.yaml", "llm:\n  model: base-model\n  temperature: 0.7\nretrieval:\n  top_k: 9\n")
    child = _write(tmp_path / "child.yaml", "base: base.yaml\nllm:\n  model: child-model\n")

    cfg = load_config(child)

    assert cfg.llm.model == "child-model"
    assert cfg.llm.temperature == 0.7  # sibling key kept from the base
    assert cfg.retrieval.top_k == 9  # untouched section kept from the base


def test_base_lists_replace_rather_than_concatenate(tmp_path: Path) -> None:
    _write(
        tmp_path / "base.yaml",
        "corpora:\n  active: [a]\n  registry:\n    a: {documents_dir: a}\n    b: {documents_dir: b}\n",
    )
    child = _write(tmp_path / "child.yaml", "base: base.yaml\ncorpora:\n  active: [b]\n")

    # Concatenating would silently select the pooled a+b index.
    assert load_config(child).corpora.active == ["b"]


def test_base_null_overrides_the_base_value(tmp_path: Path) -> None:
    _write(tmp_path / "base.yaml", "eval:\n  judge:\n    model: some-judge\n")
    child = _write(tmp_path / "child.yaml", "base: base.yaml\neval:\n  judge: null\n")

    assert load_config(child).eval.judge is None


def test_base_resolves_relative_to_the_naming_file_and_chains(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(tmp_path / "configs" / "root.yaml", "llm:\n  model: root\n  temperature: 0.1\n")
    _write(tmp_path / "configs" / "mid" / "mid.yaml", "base: ../root.yaml\nllm:\n  temperature: 0.5\n")
    leaf = _write(tmp_path / "configs" / "mid" / "leaf.yaml", "base: mid.yaml\nretrieval:\n  top_k: 3\n")
    monkeypatch.chdir(tmp_path)  # not the files' directory

    cfg = load_config(leaf)

    assert (cfg.llm.model, cfg.llm.temperature, cfg.retrieval.top_k) == ("root", 0.5, 3)


def test_base_naming_a_missing_file_raises(tmp_path: Path) -> None:
    child = _write(tmp_path / "child.yaml", "base: nope.yaml\n")

    with pytest.raises(FileNotFoundError, match="nope.yaml"):
        load_config(child)


def test_base_cycle_raises(tmp_path: Path) -> None:
    _write(tmp_path / "a.yaml", "base: b.yaml\n")
    _write(tmp_path / "b.yaml", "base: a.yaml\n")

    with pytest.raises(ValueError, match="loops"):
        load_config(tmp_path / "a.yaml")


def test_base_must_be_a_path(tmp_path: Path) -> None:
    child = _write(tmp_path / "child.yaml", "base: [a.yaml]\n")

    with pytest.raises(ValueError, match="must be a file path"):
        load_config(child)


def test_env_override_applies_on_top_of_the_merged_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(tmp_path / "base.yaml", "llm:\n  model: base-model\n")
    child = _write(tmp_path / "child.yaml", "base: base.yaml\nllm:\n  model: child-model\n")
    monkeypatch.setenv("RAG__LLM__MODEL", "env-model")

    assert load_config(child).llm.model == "env-model"


# ---------------------------------------------------------------------------
# chat.prompt
# ---------------------------------------------------------------------------


def test_prompt_style_defaults_to_grounded() -> None:
    assert RagConfig().chat.prompt == "grounded"
    assert load_config().chat.prompt == "grounded"


def test_plain_prompt_rejects_crag_regeneration() -> None:
    with pytest.raises(ValidationError, match="max_regenerations"):
        RagConfig.model_validate({"chat": {"prompt": "plain"}, "crag": {"enabled": True}})


def test_plain_prompt_allows_crag_without_regeneration() -> None:
    cfg = RagConfig.model_validate(
        {"chat": {"prompt": "plain"}, "crag": {"enabled": True, "max_regenerations": 0}}
    )
    assert cfg.crag.check_groundedness



# ---------------------------------------------------------------------------
# agent (Milestone 19)
# ---------------------------------------------------------------------------


def test_agent_defaults_match_the_prototype() -> None:
    agent = RagConfig().agent
    assert agent.llm is None  # falls back to `llm`
    assert (agent.strategy, agent.max_tool_calls, agent.num_ctx) == ("react", 8, 32768)


def test_agent_passage_cap_matches_the_mcp_server() -> None:
    # One number, two homes until phase 2 moves the tool surface out of rag/mcp.
    assert AgentConfig().max_passage_chars == DEFAULT_MAX_CHARS


def test_shipped_config_agent_section_matches_the_model_defaults() -> None:
    assert load_config(DEFAULT_CONFIG_PATH).agent == AgentConfig()


@pytest.mark.parametrize("bad", [{"strategy": "tree"}, {"max_tool_calls": 0}, {"timeout_s": 0}, {"num_ctx": 512}])
def test_agent_rejects_invalid_values(bad: dict) -> None:
    with pytest.raises(ValidationError):
        AgentConfig(**bad)


@pytest.mark.parametrize("think", [True, False, "low", "medium", "xhigh"])
def test_llm_think_accepts_on_off_or_a_level(think: bool | str) -> None:
    assert LLMConfig(think=think).think == think


def test_llm_think_rejects_an_unknown_level() -> None:
    with pytest.raises(ValidationError):
        LLMConfig(think="extreme")
