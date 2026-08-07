"""Tests for the Streamlit UI helper functions.

The Streamlit app itself (``rag/ui/app.py``) is not exercised here — it
requires a running Streamlit server.  Instead, the pure helper functions in
``rag/ui/helpers.py`` are tested directly: citation label/preview formatting
and the sidebar config summary.  These functions contain all the logic worth
verifying; everything else in ``app.py`` is thin Streamlit wiring.
"""

from __future__ import annotations

from rag.config.settings import RagConfig
from rag.events import PipelineEvent
from rag.generation.chat_service import ChatAnswer, Citation
from rag.generation.query_rewriter import ChatTurn
from rag.ui.helpers import (
    format_answer_notices,
    format_citation_label,
    format_citation_preview,
    format_event_line,
    history_from_messages,
    sidebar_config_summary,
)


def _citation(
    document_id: str = "guide.pdf",
    text: str = "Some context text.",
    score: float = 0.85,
    page: int | None = None,
) -> Citation:
    return Citation(
        chunk_id=f"{document_id}::chunk0",
        document_id=document_id,
        text=text,
        score=score,
        page=page,
    )


# ---------------------------------------------------------------------------
# format_citation_label
# ---------------------------------------------------------------------------


def test_format_citation_label_with_page() -> None:
    c = _citation(document_id="handbook.pdf", page=7)
    assert format_citation_label(c, rank=1) == "[1] handbook.pdf (p.7)"


def test_format_citation_label_without_page() -> None:
    c = _citation(document_id="faq.md")
    assert format_citation_label(c, rank=3) == "[3] faq.md"


def test_format_citation_label_rank_is_included() -> None:
    c = _citation()
    assert "[2]" in format_citation_label(c, rank=2)
    assert "[5]" in format_citation_label(c, rank=5)


# ---------------------------------------------------------------------------
# format_citation_preview
# ---------------------------------------------------------------------------


def test_format_citation_preview_short_text_unchanged() -> None:
    c = _citation(text="Short text.")
    assert format_citation_preview(c) == "Short text."


def test_format_citation_preview_truncates_long_text() -> None:
    long_text = "x" * 400
    c = _citation(text=long_text)
    preview = format_citation_preview(c, max_chars=300)
    assert len(preview) == 301  # 300 chars + ellipsis character
    assert preview.endswith("…")


def test_format_citation_preview_collapses_newlines() -> None:
    c = _citation(text="line one\nline two\nline three")
    preview = format_citation_preview(c)
    assert "\n" not in preview
    assert "line one line two line three" == preview


def test_format_citation_preview_strips_whitespace() -> None:
    c = _citation(text="  padded  ")
    assert format_citation_preview(c) == "padded"


def test_format_citation_preview_custom_max_chars() -> None:
    c = _citation(text="abcdefghij")
    preview = format_citation_preview(c, max_chars=5)
    assert preview == "abcde…"


# ---------------------------------------------------------------------------
# sidebar_config_summary
# ---------------------------------------------------------------------------


def test_sidebar_config_summary_contains_all_sections() -> None:
    config = RagConfig()
    summary = sidebar_config_summary(config)
    assert "Embedder" in summary
    assert "LLM" in summary
    assert "Reranker" in summary
    assert "top-k" in summary
    assert "Vector store" in summary


def test_sidebar_config_summary_reflects_config_values() -> None:
    config = RagConfig()
    summary = sidebar_config_summary(config)
    assert config.embedding.model in summary
    assert config.llm.model in summary
    assert config.reranker.provider in summary
    assert str(config.retrieval.top_k) in summary
    assert str(config.retrieval.rerank_top_k) in summary
    assert config.retrieval.mode in summary
    assert config.vector_store.collection_name in summary


# ---------------------------------------------------------------------------
# format_event_line
# ---------------------------------------------------------------------------


def test_format_event_line_includes_stage_message_and_timing() -> None:
    event = PipelineEvent(stage="embed", message="Embedded query into a 1024-dim vector", elapsed_ms=12.3)
    line = format_event_line(event)
    assert "embed" in line
    assert "Embedded query into a 1024-dim vector" in line
    assert "12 ms" in line


def test_format_event_line_omits_timing_when_absent() -> None:
    event = PipelineEvent(stage="no_context", message="No relevant chunks found")
    line = format_event_line(event)
    assert "ms" not in line


# ---------------------------------------------------------------------------
# history_from_messages
# ---------------------------------------------------------------------------


def _answer(text: str = "an answer") -> ChatAnswer:
    return ChatAnswer(answer=text, citations=[])


def test_history_from_messages_converts_completed_turns() -> None:
    messages = [
        {"role": "user", "content": "what is the refund policy?", "answer": None},
        {"role": "assistant", "content": "Refunds within 30 days.", "answer": _answer(), "events": []},
    ]

    assert history_from_messages(messages) == [
        ChatTurn(role="user", content="what is the refund policy?"),
        ChatTurn(role="assistant", content="Refunds within 30 days."),
    ]


def test_history_from_messages_skips_failed_assistant_turns() -> None:
    # A failed turn stores an error banner as `content` with no ChatAnswer --
    # feeding that to the condenser would have it rewrite around the error.
    messages = [
        {"role": "user", "content": "what is the refund policy?", "answer": None},
        {"role": "assistant", "content": "⚠️ Error: connection refused", "answer": None, "events": []},
    ]

    assert history_from_messages(messages) == [
        ChatTurn(role="user", content="what is the refund policy?")
    ]


def test_history_from_messages_on_an_empty_conversation() -> None:
    assert history_from_messages([]) == []


def test_history_from_messages_keeps_no_context_answers() -> None:
    # "I don't have any indexed information" is a real turn, not a failure.
    messages = [
        {"role": "user", "content": "unrelated question", "answer": None},
        {"role": "assistant", "content": "I don't have any indexed information…", "answer": _answer(), "events": []},
    ]

    assert len(history_from_messages(messages)) == 2


# ---------------------------------------------------------------------------
# format_answer_notices
# ---------------------------------------------------------------------------


def test_format_answer_notices_reports_a_rewritten_query() -> None:
    answer = ChatAnswer(answer="a", rewritten_query="What is the refund policy for part-time staff?")

    [notice] = format_answer_notices(answer)

    assert "What is the refund policy for part-time staff?" in notice


def test_format_answer_notices_reports_withheld_passages() -> None:
    answer = ChatAnswer(answer="a", dropped_below_min_score=3)

    [notice] = format_answer_notices(answer)

    assert "3" in notice and "withheld" in notice.lower()


def test_format_answer_notices_reports_both_when_both_apply() -> None:
    answer = ChatAnswer(answer="a", rewritten_query="a standalone question", dropped_below_min_score=2)

    assert len(format_answer_notices(answer)) == 2


def test_format_answer_notices_is_empty_for_a_plain_answer() -> None:
    assert format_answer_notices(ChatAnswer(answer="a")) == []


def test_format_answer_notices_reports_query_expansion() -> None:
    answer = ChatAnswer(answer="a", search_queries=["hypothetical passage", "original"])

    [notice] = format_answer_notices(answer)

    assert "2" in notice and "expanded" in notice.lower()


def test_format_answer_notices_reports_all_three_interventions() -> None:
    answer = ChatAnswer(
        answer="a",
        rewritten_query="a standalone question",
        search_queries=["q1", "q2"],
        dropped_below_min_score=2,
    )

    assert len(format_answer_notices(answer)) == 3


def test_sidebar_config_summary_includes_query_expansion() -> None:
    summary = sidebar_config_summary(RagConfig())
    assert "Query expansion" in summary


def test_format_answer_notices_reports_graded_out_passages() -> None:
    answer = ChatAnswer(answer="a", graded_out=2)

    [notice] = format_answer_notices(answer)

    assert "2" in notice


def test_format_answer_notices_reports_a_crag_retry() -> None:
    answer = ChatAnswer(answer="a", retry_queries=["a reworded query"], retrieval_attempts=2)

    [notice] = format_answer_notices(answer)

    assert "a reworded query" in notice


def test_format_answer_notices_warns_about_an_ungrounded_answer() -> None:
    answer = ChatAnswer(answer="a", grounded=False)

    [notice] = format_answer_notices(answer)

    assert "unverified" in notice.lower()


def test_format_answer_notices_stays_silent_for_a_grounded_answer() -> None:
    # Captioning the expected outcome would train users to skim past the one
    # state that needs attention.
    assert format_answer_notices(ChatAnswer(answer="a", grounded=True)) == []
    assert format_answer_notices(ChatAnswer(answer="a", grounded=None)) == []


def test_sidebar_summary_reports_contextual_chunking_and_crag() -> None:
    summary = sidebar_config_summary(RagConfig())

    assert "Contextual chunks" in summary
    assert "CRAG" in summary
