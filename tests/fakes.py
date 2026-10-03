"""Test doubles shared across test modules."""

from __future__ import annotations

from collections.abc import Sequence

from rag.llm.base import AssistantTurn, Message, ToolCallingLLM, ToolDefinition


class ScriptedToolLLM(ToolCallingLLM):
    """A `ToolCallingLLM` that replays a fixed list of turns, one per `chat()` call.

    Records what each call was sent, copying the message list because callers
    append to theirs after the call returns. Running past the script fails the
    test instead of inventing a reply.
    """

    def __init__(self, turns: Sequence[AssistantTurn], *, generate_reply: str = "") -> None:
        self._turns = list(turns)
        self.generate_reply = generate_reply
        self.calls: list[tuple[list[Message], list[ToolDefinition]]] = []

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        return self.generate_reply

    def chat(self, messages: Sequence[Message], tools: Sequence[ToolDefinition] = ()) -> AssistantTurn:
        if len(self.calls) >= len(self._turns):
            raise AssertionError(f"ScriptedToolLLM ran out of turns after {len(self._turns)} chat() calls")
        self.calls.append((list(messages), list(tools)))
        return self._turns[len(self.calls) - 1]
