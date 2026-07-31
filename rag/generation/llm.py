"""Interface for chat/text-generation models.

Mirrors `EmbeddingModel`: pipeline code (the chat service, eval pipeline)
depends only on this ABC, never on a concrete provider's SDK or HTTP API --
swapping the local Ollama-served `qwen3.5:9b-mlx` for a hosted Anthropic/OpenAI
model is a config change plus one adapter.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class LLMClient(ABC):
    """Interface for generating a text response from a prompt.

    A single `generate` method (rather than a richer chat-with-history API)
    is deliberately the smallest surface that the RAG chat service needs: one
    user-turn prompt (already containing the retrieved context) plus an
    optional system prompt steering the model's behavior. Multi-turn
    conversation memory, if added later, can be layered on top of this
    interface without changing it.
    """

    @abstractmethod
    def generate(self, prompt: str, *, system: str | None = None) -> str:
        """Generate a response to `prompt`, optionally steered by `system`.

        Returns the model's full response text. Implementations are
        responsible for translating `temperature`/`max_tokens`-style
        parameters from their constructor config into whatever the underlying
        provider expects.
        """
        raise NotImplementedError
