"""Text-cleaning utilities applied after loading, before chunking.

Kept as small, independently testable functions composed by `clean_text`,
rather than one monolithic regex blob -- each step targets one specific
extraction artifact (control characters, hyphenation, whitespace) and can be
reordered, dropped, or unit-tested on its own.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import replace
from typing import Iterable

from rag.ingestion.models import Document

# Matches a hyphen at end-of-line joining two lowercase letters -- the classic
# PDF line-wrap artifact: "exam-\nple" -> "example". Restricted to
# lowercase-to-lowercase joins so we don't accidentally merge real hyphenated
# compounds or proper nouns that happen to fall at a line break
# (e.g. "well-\nKnown" stays put because "K" is uppercase).
_HYPHEN_LINEBREAK = re.compile(r"([a-z])-\n([a-z])")

# Three or more consecutive newlines (with optional whitespace between them)
# collapse to a single blank line (i.e. two newlines).
_EXCESS_BLANK_LINES = re.compile(r"\n[ \t]*\n(?:[ \t]*\n)+")

# Any run of horizontal whitespace (spaces/tabs, including a lone tab)
# collapses to a single space -- extraction artifacts mix both freely.
_REPEATED_SPACES = re.compile(r"[ \t]+")


def strip_control_characters(text: str) -> str:
    """Remove non-printable Unicode control characters, keeping newlines and tabs.

    PDF extraction occasionally emits stray control bytes (form feeds, null
    bytes) that are invisible but pollute token counts and embeddings.
    """

    return "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch) != "Cc")


def dehyphenate(text: str) -> str:
    """Join words that were hyphenated across a line break by the PDF layout.

    Example: "the docu-\nment store" -> "the document store". Only applied
    between two lowercase word characters to avoid merging genuine hyphenated
    terms or proper nouns that happen to wrap.
    """

    return _HYPHEN_LINEBREAK.sub(r"\1\2", text)


def normalize_whitespace(text: str) -> str:
    """Collapse repeated horizontal whitespace and trim trailing spaces per line."""

    lines = (
        _REPEATED_SPACES.sub(" ", line).rstrip()
        for line in text.splitlines()
    )
    return "\n".join(lines)


def collapse_blank_lines(text: str) -> str:
    """Collapse 3+ consecutive newlines down to a single blank line (2 newlines)."""

    return _EXCESS_BLANK_LINES.sub("\n\n", text)


def clean_text(text: str) -> str:
    """Run the default cleaning pipeline: control chars -> dehyphenate -> whitespace -> blank lines.

    Order matters: dehyphenation must run before whitespace normalization
    (which would otherwise destroy the line breaks it depends on), and blank
    line collapsing runs last so it sees the final line structure.
    """

    text = strip_control_characters(text)
    text = dehyphenate(text)
    text = normalize_whitespace(text)
    text = collapse_blank_lines(text)
    return text.strip()


def clean_documents(documents: Iterable[Document]) -> list[Document]:
    """Return copies of `documents` with `clean_text` applied to each one's text.

    `Document` is an immutable dataclass, so this produces new instances via
    `dataclasses.replace` rather than mutating in place. Keeping cleaning as an
    explicit pipeline step (rather than baking it into loaders) means loaders
    stay focused on extraction and the cleaning policy can evolve -- or be
    skipped entirely for debugging -- without touching loader code.
    """

    return [replace(document, text=clean_text(document.text)) for document in documents]
