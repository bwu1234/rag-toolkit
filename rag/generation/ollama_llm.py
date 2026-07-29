"""Ollama-backed `LLMClient` adapter.

Talks to a local Ollama daemon's `/api/chat` endpoint over HTTP -- the chat
counterpart to `OllamaEmbedder`'s use of `/api/embed`. Using `/api/chat`
(messages array) rather than `/api/generate` (raw prompt string) lets the
system prompt and user prompt stay cleanly separated, matching how
instruction-tuned chat models are actually trained to be steered.
"""

from __future__ import annotations

import logging

import httpx

from rag.generation.llm import LLMClient

logger = logging.getLogger(__name__)


class OllamaLLMClient(LLMClient):
    """Generates chat responses via a local Ollama daemon's `/api/chat` endpoint.

    `temperature`/`max_tokens` are bound at construction (from `LLMConfig`)
    rather than threaded through `generate` -- the chat service shouldn't need
    to know provider-specific option names (`num_predict` vs `max_tokens`,
    etc.); that translation is this adapter's job.
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        think: bool = False,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.think = think
        # `trust_env=False`: see `OllamaEmbedder` -- a loopback connection to
        # Ollama should never go through the system proxy.
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout, trust_env=False)

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        messages: list[dict[str, str]] = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self._client.post(
                "/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "think": self.think,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens,
                    },
                },
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"Failed to get a chat completion from Ollama at {self.base_url} "
                f"(model={self.model!r}): {exc}"
            ) from exc

        payload = response.json()
        message = payload.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise RuntimeError(
                f"Unexpected response shape from Ollama /api/chat: expected a "
                f"{{'message': {{'content': str}}}} object, got {payload!r}"
            )
        return message["content"]

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "OllamaLLMClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
