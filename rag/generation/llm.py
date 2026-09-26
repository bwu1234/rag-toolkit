"""Interface for chat/text-generation models.

Mirrors `EmbeddingModel`: pipeline code (the chat service, eval pipeline)
depends only on this ABC, never on a concrete provider's SDK or HTTP API --
swapping the local Ollama-served `qwen3.5:9b-mlx` for a hosted Anthropic/OpenAI
model is a config change plus one adapter.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class LLMUsage:
    """Token counts one completion cost, as reported by the provider.

    Either count is `None` when the provider didn't report it -- which is not
    the same as zero, so it's kept distinct rather than defaulted away.
    """

    prompt_tokens: int | None = None
    completion_tokens: int | None = None


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

    def generate_with_usage(self, prompt: str, *, system: str | None = None) -> tuple[str, LLMUsage | None]:
        """`generate`, plus the token counts the call cost if the provider reports them.

        A concrete default (usage `None`) rather than a second abstract method,
        so an adapter -- or a test fake -- that has no usage to report needs no
        extra code. Adapters whose provider does report counts override this
        and implement `generate` on top of it. Only `MeteredLLMClient` calls
        it; pipeline components keep calling plain `generate`.
        """

        return self.generate(prompt, system=system), None
