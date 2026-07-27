"""Tests for the Streamlit UI helper functions.

The Streamlit app itself (``rag/ui/app.py``) is not exercised here — it
requires a running Streamlit server.  Instead, the pure helper functions in
``rag/ui/helpers.py`` are tested directly: citation label/preview formatting
and the sidebar config summary.  These functions contain all the logic worth
verifying; everything else in ``app.py`` is thin Streamlit wiring.
"""

from __future__ import annotations

from rag.config.settings import RagConfig
from rag.generation.chat_service import Citation
from rag.ui.helpers import (
    format_citation_label,
    format_citation_preview,
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
    assert config.vector_store.collection_name in summary
