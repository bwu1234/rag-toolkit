"""Config-driven factory for `LLMClient` implementations.

Mirrors `rag.embedding.factory.get_embedder`. `LLMConfig.provider` already
admits `"anthropic"` / `"openai"` as valid *values* (so config can name an
intended future provider without failing validation), but only `"ollama"` and
`"gemini"` have adapters today -- selecting one of the others raises a clear
not-yet-implemented error rather than `Unknown provider`.
"""

from __future__ import annotations

from rag.config.settings import REPO_ROOT, LLMConfig
from rag.llm.daily_budget import DailyRequestCounter
from rag.llm.gemini_llm import GeminiLLMClient, api_key_from_env
from rag.llm.base import LLMClient
from rag.llm.ollama_llm import OllamaLLMClient

_KNOWN_BUT_UNIMPLEMENTED = {"anthropic", "openai"}


def get_llm_client(config: LLMConfig, *, num_ctx: int | None = None) -> LLMClient:
    """Instantiate the `LLMClient` selected by `config.provider`.

    `num_ctx` requests a context window from providers that size it per
    request (Ollama). It is an argument rather than an `LLMConfig` field
    because only the agent sets it (`agent.num_ctx`): the pipeline's calls keep
    the daemon default they were measured with. Hosted providers have a fixed
    window and ignore it.
    """

    if config.provider == "ollama":
        return OllamaLLMClient(
            model=config.model,
            base_url=config.base_url,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            think=config.think,
            num_ctx=num_ctx,
            timeout=config.timeout_s,
            raw=config.raw,
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
            daily_counter=(
                DailyRequestCounter(
                    (REPO_ROOT / config.daily_request_log).resolve(),
                    key=config.model,
                    limit=config.requests_per_day - config.requests_per_day_reserve,
                )
                if config.requests_per_day is not None
                else None
            ),
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
