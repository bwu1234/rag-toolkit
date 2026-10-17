"""Test doubles shared across test modules."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from rag.eval.provenance import IndexProvenance
from rag.llm.base import AssistantTurn, Message, ToolCallingLLM, ToolDefinition
from rag.vectorstore.base import ScoredChunk


def scored_chunk(chunk_id: str, score: float = 0.9, doc: str = "doc.md") -> ScoredChunk:
    """A minimal retrieved chunk, for tests that only follow ids and scores."""
    return ScoredChunk(
        chunk_id=chunk_id, text=f"text of {chunk_id}", document_id=doc, source=Path(f"data/{doc}"),
        doc_type="markdown", score=score,
    )


def skip_index_checks(monkeypatch: pytest.MonkeyPatch, matrix: ModuleType, *, verified: bool = False) -> None:
    """Let a matrix script run over a faked chat service with no index on disk.

    The script verifies every variant's index before running; a test that
    fakes retrieval has none, so the check reports it unverified instead --
    or, with ``verified``, as a fixed build, which is what lets a run resume.
    """

    def chunk(config: Any, corpora: Any) -> tuple[Any, list[Any], list[Any]]:
        return config.corpus_selection(corpora), [], []

    def check(config: Any, selection: Any, documents: Any, chunks: Any, *, label: str) -> IndexProvenance:
        if verified:
            return IndexProvenance("index", "corpus", "build-1")
        return IndexProvenance("index", "corpus", None, unverified="faked in a test")

    monkeypatch.setattr(matrix, "chunk_selected_corpora", chunk)
    monkeypatch.setattr(matrix, "checked_index", check)


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
