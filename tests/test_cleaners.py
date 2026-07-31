"""Unit tests for text-cleaning utilities (Milestone 2)."""

from __future__ import annotations

from rag.ingestion.cleaners import (
    clean_text,
    collapse_blank_lines,
    dehyphenate,
    normalize_whitespace,
    strip_control_characters,
)


def test_dehyphenate_joins_wrapped_words() -> None:
    assert dehyphenate("the docu-\nment store") == "the document store"


def test_dehyphenate_leaves_real_hyphens_alone() -> None:
    # Capitalized continuation -> likely a proper noun / real compound, not a wrap artifact.
    text = "well-\nKnown issue"
    assert dehyphenate(text) == text


def test_normalize_whitespace_collapses_runs_and_trims() -> None:
    assert normalize_whitespace("a   b\tc   \nx  y  ") == "a b c\nx y"


def test_collapse_blank_lines_limits_to_one_blank_line() -> None:
    assert collapse_blank_lines("a\n\n\n\n\nb") == "a\n\nb"


def test_strip_control_characters_keeps_newlines_and_tabs() -> None:
    text = "a\nb\tc\x00d\x0be"
    cleaned = strip_control_characters(text)
    assert "\n" in cleaned and "\t" in cleaned
    assert "\x00" not in cleaned and "\x0b" not in cleaned


def test_clean_text_pipeline_end_to_end() -> None:
    raw = "Title\n\n\n\nThis is a hyphen-\nated word with   extra   spaces.   \n\n\n"
    cleaned = clean_text(raw)
    assert "hyphenated word" in cleaned
    assert "   " not in cleaned
    assert "\n\n\n" not in cleaned
    assert cleaned == cleaned.strip()
