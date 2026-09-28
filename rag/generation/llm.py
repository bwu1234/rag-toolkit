"""Interface for chat/text-generation models.

Mirrors `EmbeddingModel`: pipeline code (the chat service, eval pipeline)
depends only on this ABC, never on a concrete provider's SDK or HTTP API --
swapping the local Ollama-served `qwen3.5:9b-mlx` for a hosted Anthropic/OpenAI
model is a config change plus one adapter.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal


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


# ---------------------------------------------------------------------------
# Tool calling (Milestone 19)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolDefinition:
    """A tool as the model sees it: a name, what it does, and a JSON Schema for its arguments.

    Provider-neutral on purpose. Each adapter translates it into its own wire
    format (Ollama's `{"type": "function", "function": {...}}`, Anthropic's
    `input_schema`, ...), so the agent never builds a provider payload itself.
    """

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    """One tool invocation the model asked for.

    `id` is whatever the provider assigned (Ollama and Anthropic both do). It
    is carried back on the matching `ToolResult`, because a provider that pairs
    results with calls by id rejects a result without one.

    `signature` is provider state that must go back on the call, unread and
    unchanged. Gemini 3 attaches a `thoughtSignature` to a step's first function
    call and rejects the next request with a 400 if it's missing. Adapters
    whose provider has no such thing leave it `None` and ignore it.
    """

    name: str
    arguments: dict[str, Any]
    id: str | None = None
    signature: str | None = None


@dataclass(frozen=True)
class ChatMessage:
    """A system or user message in a tool-calling conversation."""

    role: Literal["system", "user"]
    content: str


@dataclass(frozen=True)
class AssistantTurn:
    """What the model returned for one `chat()` call.

    It is also the assistant message the caller appends to the conversation
    before the next call, so a model's tool calls and reasoning go back to it
    exactly as it produced them. `usage` travels with the turn rather than as a
    second return value, as `generate_with_usage` has to, because every
    tool-calling adapter can report something here.
    """

    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    # The reasoning trace, when the provider returns one (Ollama with `think`
    # on). Sent back with the turn so the model sees its own earlier reasoning.
    thinking: str | None = None
    usage: LLMUsage | None = None


@dataclass(frozen=True)
class ToolResult:
    """The output of one `ToolCall`, sent back to the model as a tool message."""

    call: ToolCall
    content: str


Message = ChatMessage | AssistantTurn | ToolResult


class ContextOverflowError(RuntimeError):
    """The prompt is longer than the context window the request asked for.

    Raised instead of letting the provider truncate (Ollama's llama.cpp engine
    otherwise drops the start of the prompt silently). A caller holding a
    growing conversation (the agent) catches it to stop adding and answer with
    what it has, rather than carrying on with a prompt that lost its beginning.
    Defined here rather than in an adapter, so the agent can catch it without
    importing a concrete provider.
    """


class ToolCallingLLM(LLMClient):
    """An `LLMClient` that can also hold a multi-turn conversation with tools.

    A capability subclass rather than a second method on `LLMClient`: the
    fourteen `generate` call sites (HyDE, the condenser, CRAG, the judges, ...)
    need no tools, and widening the base ABC would make every test fake and
    adapter implement tool calling to answer a one-shot prompt. The agent takes
    a `ToolCallingLLM`, so selecting a provider without tool support fails when
    the agent is built, not mid-conversation.
    """

    @abstractmethod
    def chat(self, messages: Sequence[Message], tools: Sequence[ToolDefinition] = ()) -> AssistantTurn:
        """Send the conversation so far and return the model's next turn.

        With `tools` empty the model is offered none and must answer in text;
        the agent relies on that for its forced-synthesis turn at the search
        cap. The caller owns the message list: this method does not append the
        returned turn to it.
        """

        raise NotImplementedError
