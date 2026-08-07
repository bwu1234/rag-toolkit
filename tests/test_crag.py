"""Tests for the corrective-RAG checks (`rag.generation.crag`).

Each of the three components is a thin wrapper around one LLM call with a
strict reply protocol, so the tests concentrate on the two things that can
actually go wrong: parsing a small model's reply, and what happens when that
reply is unusable. Every component fails *open* -- the asymmetry is deliberate
and is asserted here explicitly, because a checker that silently started
failing closed would quietly turn working turns into refusals.

Driven by fake `LLMClient`s, mirroring `test_query_rewriter.py`: no daemon, no
index, no model weights.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.generation.crag import (
    DocumentGrader,
    GroundednessChecker,
    RetryQueryRewriter,
    format_passages,
)
from rag.vectorstore.base import ScoredChunk


def _scored(chunk_id: str = "a", text: str = "Refunds within 30 days.", *, context: str | None = None) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk_id,
        text=text,
        document_id=f"{chunk_id}-doc",
        source=Path(f"{chunk_id}.md"),
        doc_type="markdown",
        score=0.8,
        context=context,
    )


class _FakeLLMClient:
    """Returns canned replies in order (repeating the last), recording prompts."""

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies) or ["YES"]
        self.calls: list[tuple[str, str | None]] = []

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        self.calls.append((prompt, system))
        index = min(len(self.calls) - 1, len(self.replies) - 1)
        return self.replies[index]


class _RaisingLLMClient:
    def generate(self, prompt: str, *, system: str | None = None) -> str:
        raise RuntimeError("daemon unreachable")


# ---------------------------------------------------------------------------
# DocumentGrader
# ---------------------------------------------------------------------------


def test_grader_keeps_yes_and_drops_no() -> None:
    grader = DocumentGrader(_FakeLLMClient("YES", "NO", "YES"))  # type: ignore[arg-type]
    chunks = [_scored("a"), _scored("b"), _scored("c")]

    graded = grader.grade("what is the refund policy", chunks)

    assert [chunk.chunk_id for chunk in graded.kept] == ["a", "c"]
    assert graded.graded_out == 1


def test_grader_makes_one_call_per_chunk() -> None:
    client = _FakeLLMClient("YES")
    grader = DocumentGrader(client)  # type: ignore[arg-type]

    grader.grade("a question", [_scored("a"), _scored("b"), _scored("c")])

    assert len(client.calls) == 3


def test_grader_prompt_carries_the_question_and_the_passage() -> None:
    client = _FakeLLMClient("YES")
    grader = DocumentGrader(client)  # type: ignore[arg-type]

    grader.grade("what is the refund policy", [_scored("a", "Refunds within 30 days.")])

    [(prompt, system)] = client.calls
    assert "what is the refund policy" in prompt
    assert "Refunds within 30 days." in prompt
    assert system is not None and "YES" in system


def test_grader_grades_against_contextualized_text() -> None:
    # The whole point of contextual chunking is that a chunk's own words may not
    # name its subject; the grader has to see the context too or it will reject
    # exactly the chunks contextual retrieval exists to rescue.
    client = _FakeLLMClient("YES")
    grader = DocumentGrader(client)  # type: ignore[arg-type]

    grader.grade("what is the ACS rate limit", [_scored("a", "The limit is 1,000/min.", context="ACS rate limits.")])

    [(prompt, _system)] = client.calls
    assert "ACS rate limits." in prompt


@pytest.mark.parametrize("reply", ["yes", "  YES  ", "YES.", "Yes, it does"])
def test_grader_accepts_yes_in_various_shapes(reply: str) -> None:
    grader = DocumentGrader(_FakeLLMClient(reply))  # type: ignore[arg-type]

    assert len(grader.grade("q", [_scored()]).kept) == 1


@pytest.mark.parametrize("reply", ["no", "NO", "No."])
def test_grader_rejects_no_in_various_shapes(reply: str) -> None:
    grader = DocumentGrader(_FakeLLMClient(reply))  # type: ignore[arg-type]

    assert grader.grade("q", [_scored()]).kept == []


def test_grader_keeps_the_chunk_when_the_verdict_is_unparseable() -> None:
    grader = DocumentGrader(_FakeLLMClient("I'm not sure about this one"))  # type: ignore[arg-type]

    graded = grader.grade("q", [_scored()])

    assert len(graded.kept) == 1, "an unreadable verdict must not silently drop evidence"
    assert graded.graded_out == 0


def test_grader_keeps_the_chunk_when_the_client_raises() -> None:
    grader = DocumentGrader(_RaisingLLMClient())  # type: ignore[arg-type]

    graded = grader.grade("q", [_scored("a"), _scored("b")])

    assert len(graded.kept) == 2, "a broken grader must degrade to plain RAG, not to a refusal"
    assert graded.graded_out == 0


def test_grader_on_no_chunks_makes_no_calls() -> None:
    client = _FakeLLMClient()
    grader = DocumentGrader(client)  # type: ignore[arg-type]

    graded = grader.grade("q", [])

    assert graded.kept == []
    assert graded.graded_out == 0
    assert client.calls == []


# ---------------------------------------------------------------------------
# RetryQueryRewriter
# ---------------------------------------------------------------------------


def test_rewriter_returns_the_rewritten_query() -> None:
    rewriter = RetryQueryRewriter(_FakeLLMClient("ACS API request throttling limits"))  # type: ignore[arg-type]

    assert rewriter.rewrite("how do I stop getting 429s") == "ACS API request throttling limits"


def test_rewriter_strips_quotes_and_collapses_whitespace() -> None:
    rewriter = RetryQueryRewriter(_FakeLLMClient('  "rate   limit\n policy"  '))  # type: ignore[arg-type]

    assert rewriter.rewrite("q") == "rate limit policy"


def test_rewriter_prompt_includes_the_failed_query() -> None:
    client = _FakeLLMClient("a rewrite")
    rewriter = RetryQueryRewriter(client)  # type: ignore[arg-type]

    rewriter.rewrite("the failed query")

    [(prompt, system)] = client.calls
    assert "the failed query" in prompt
    assert system is not None and "rewrite" in system.lower()


def test_rewriter_falls_back_to_the_original_when_the_client_raises() -> None:
    rewriter = RetryQueryRewriter(_RaisingLLMClient())  # type: ignore[arg-type]

    assert rewriter.rewrite("the original query") == "the original query"


def test_rewriter_falls_back_to_the_original_on_an_empty_reply() -> None:
    rewriter = RetryQueryRewriter(_FakeLLMClient("   "))  # type: ignore[arg-type]

    assert rewriter.rewrite("the original query") == "the original query"


# ---------------------------------------------------------------------------
# GroundednessChecker
# ---------------------------------------------------------------------------


def test_checker_returns_true_for_grounded() -> None:
    checker = GroundednessChecker(_FakeLLMClient("GROUNDED"))  # type: ignore[arg-type]

    assert checker.check("q", [_scored()], "an answer") is True


def test_checker_returns_false_for_ungrounded() -> None:
    checker = GroundednessChecker(_FakeLLMClient("UNGROUNDED"))  # type: ignore[arg-type]

    assert checker.check("q", [_scored()], "an answer") is False


def test_checker_returns_none_when_the_verdict_is_unparseable() -> None:
    checker = GroundednessChecker(_FakeLLMClient("hard to say"))  # type: ignore[arg-type]

    verdict = checker.check("q", [_scored()], "an answer")

    assert verdict is None, "'inconclusive' must stay distinct from 'unsupported'"


def test_checker_returns_none_when_the_client_raises() -> None:
    checker = GroundednessChecker(_RaisingLLMClient())  # type: ignore[arg-type]

    assert checker.check("q", [_scored()], "an answer") is None


def test_checker_returns_none_with_nothing_to_check() -> None:
    client = _FakeLLMClient("GROUNDED")
    checker = GroundednessChecker(client)  # type: ignore[arg-type]

    assert checker.check("q", [], "an answer") is None
    assert checker.check("q", [_scored()], "   ") is None
    assert client.calls == []


def test_checker_prompt_carries_passages_question_and_answer() -> None:
    client = _FakeLLMClient("GROUNDED")
    checker = GroundednessChecker(client)  # type: ignore[arg-type]

    checker.check("the question", [_scored("a", "Refunds within 30 days.")], "the answer")

    [(prompt, system)] = client.calls
    assert "Refunds within 30 days." in prompt
    assert "the question" in prompt
    assert "the answer" in prompt
    assert system is not None and "GROUNDED" in system


# ---------------------------------------------------------------------------
# Shared passage rendering
# ---------------------------------------------------------------------------


def test_format_passages_numbers_from_one_like_the_rag_prompt() -> None:
    rendered = format_passages([_scored("a", "first"), _scored("b", "second")])

    assert "Passage [1]:\nfirst" in rendered
    assert "Passage [2]:\nsecond" in rendered


def test_format_passages_includes_chunk_context() -> None:
    rendered = format_passages([_scored("a", "The limit is 1,000/min.", context="ACS rate limits.")])

    assert "ACS rate limits." in rendered
