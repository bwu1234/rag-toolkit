"""Tests for `QueryCondenser` and its prompt construction.

Driven entirely by a fake `LLMClient` (mirroring `test_chat_service.py`) --
condensing is prompt assembly plus response handling, and both are verified
without a running Ollama daemon.
"""

from __future__ import annotations

from rag.generation.query_rewriter import (
    ChatTurn,
    QueryCondenser,
    build_condense_prompt,
    format_history,
)


class _FakeLLMClient:
    def __init__(self, reply: str = "a standalone question") -> None:
        self.reply = reply
        self.calls: list[tuple[str, str | None]] = []

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        self.calls.append((prompt, system))
        return self.reply


class _ExplodingLLMClient:
    def generate(self, prompt: str, *, system: str | None = None) -> str:
        raise RuntimeError("ollama is down")


_HISTORY = [
    ChatTurn(role="user", content="what is the refund policy?"),
    ChatTurn(role="assistant", content="Refunds are accepted within 30 days."),
]


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def test_format_history_labels_each_turn_by_role() -> None:
    assert format_history(_HISTORY) == (
        "User: what is the refund policy?\nAssistant: Refunds are accepted within 30 days."
    )


def test_build_condense_prompt_includes_history_and_latest_message() -> None:
    prompt = build_condense_prompt("what about part-time staff?", _HISTORY)

    assert "what is the refund policy?" in prompt
    assert "Refunds are accepted within 30 days." in prompt
    assert "what about part-time staff?" in prompt


# ---------------------------------------------------------------------------
# Condensing
# ---------------------------------------------------------------------------


def test_condense_returns_the_models_rewrite() -> None:
    llm = _FakeLLMClient("What is the refund policy for part-time staff?")
    condenser = QueryCondenser(llm)  # type: ignore[arg-type]

    result = condenser.condense("what about part-time staff?", _HISTORY)

    assert result == "What is the refund policy for part-time staff?"


def test_condense_without_history_returns_the_query_without_calling_the_llm() -> None:
    llm = _FakeLLMClient()
    condenser = QueryCondenser(llm)  # type: ignore[arg-type]

    assert condenser.condense("a first question", []) == "a first question"
    assert llm.calls == [], "the first turn of a conversation shouldn't pay for a rewrite"


def test_condense_truncates_history_to_max_history_turns() -> None:
    llm = _FakeLLMClient()
    history = [ChatTurn(role="user", content=f"turn {i}") for i in range(10)]
    condenser = QueryCondenser(llm, max_history_turns=2)  # type: ignore[arg-type]

    condenser.condense("follow-up", history)

    [(prompt, _system)] = llm.calls
    assert "turn 9" in prompt and "turn 8" in prompt
    assert "turn 7" not in prompt


def test_condense_with_zero_max_history_turns_never_calls_the_llm() -> None:
    llm = _FakeLLMClient()
    condenser = QueryCondenser(llm, max_history_turns=0)  # type: ignore[arg-type]

    assert condenser.condense("follow-up", _HISTORY) == "follow-up"
    assert llm.calls == []


def test_condense_strips_surrounding_quotes_from_the_rewrite() -> None:
    llm = _FakeLLMClient('"What is the refund policy for part-time staff?"')
    condenser = QueryCondenser(llm)  # type: ignore[arg-type]

    result = condenser.condense("what about part-time staff?", _HISTORY)

    assert result == "What is the refund policy for part-time staff?"


def test_condense_falls_back_to_the_original_query_on_an_empty_rewrite() -> None:
    llm = _FakeLLMClient("   ")
    condenser = QueryCondenser(llm)  # type: ignore[arg-type]

    assert condenser.condense("what about part-time staff?", _HISTORY) == "what about part-time staff?"


def test_condense_falls_back_to_the_original_query_when_the_llm_raises() -> None:
    # Fail open: a degraded retrieval beats a failed turn, and the original
    # query is exactly what the pipeline used before condensing existed.
    condenser = QueryCondenser(_ExplodingLLMClient())  # type: ignore[arg-type]

    assert condenser.condense("what about part-time staff?", _HISTORY) == "what about part-time staff?"


def test_condense_uses_the_condense_system_prompt_not_the_rag_one() -> None:
    llm = _FakeLLMClient()
    condenser = QueryCondenser(llm)  # type: ignore[arg-type]

    condenser.condense("follow-up", _HISTORY)

    [(_prompt, system)] = llm.calls
    assert system is not None and "standalone" in system.lower()
