"""Config-driven factory for `LLMClient` implementations.

Mirrors `rag.embedding.factory.get_embedder`. `LLMConfig.provider` already
admits `"anthropic"` / `"openai"` as valid *values* (so config can name an
intended future provider without failing validation), but only `"ollama"` and
`"gemini"` have adapters today -- selecting one of the others raises a clear
not-yet-implemented error rather than `Unknown provider`.
"""

from __future__ import annotations

from rag.config.settings import LLMConfig
from rag.generation.gemini_llm import GeminiLLMClient, api_key_from_env
from rag.generation.llm import LLMClient
from rag.generation.ollama_llm import OllamaLLMClient

_KNOWN_BUT_UNIMPLEMENTED = {"anthropic", "openai"}


def get_llm_client(config: LLMConfig) -> LLMClient:
    """Instantiate the `LLMClient` selected by `config.provider`."""

    if config.provider == "ollama":
        return OllamaLLMClient(
            model=config.model,
            base_url=config.base_url,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            think=config.think,
            timeout=config.timeout_s,
        )

    if config.provider == "gemini":
        return GeminiLLMClient(
            model=config.model,
            base_url=config.base_url,
            api_key=api_key_from_env(config.api_key_env),
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout=config.timeout_s,
            requests_per_minute=config.requests_per_minute,
            tokens_per_minute=config.tokens_per_minute,
            thinking_level=config.thinking_level,
        )

    if config.provider in _KNOWN_BUT_UNIMPLEMENTED:
        raise ValueError(
            f"LLM provider {config.provider!r} is a recognized config value but has no "
            "adapter yet -- only 'ollama' and 'gemini' are implemented. Add one in rag/generation "
            "and register it in this factory to enable it."
        )

    raise ValueError(
        f"Unknown LLM provider: {config.provider!r}. "
        "Add an adapter and register it here to support a new provider."
    )
