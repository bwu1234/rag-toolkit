"""Tests for scripts/inventory_musique.py.

What they protect: the paragraph id is a pure function of title and text (the
converter and the manifest's collision count depend on it), and the pooling
check counts a withheld paragraph as restored only when another question's
context really carries it -- the finding that keeps MuSiQue-Full out.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import inventory_musique as inv  # noqa: E402


def _p(idx: int, title: str, text: str, supporting: bool = False) -> dict:
    return {"idx": idx, "title": title, "paragraph_text": text, "is_supporting": supporting}


def test_paragraph_id_depends_on_title_and_text_only() -> None:
    a = inv.paragraph_id(_p(0, "T", "body"))
    assert a == inv.paragraph_id(_p(7, "T", "body", supporting=True))
    assert a != inv.paragraph_id(_p(0, "T2", "body"))
    assert a.startswith("musique-") and len(a) == len("musique-") + 16


def test_a_withheld_paragraph_is_restored_only_if_the_pool_carries_it() -> None:
    gold = _p(0, "G", "the answer", supporting=True)
    answerable = {"id": "q", "answerable": True, "paragraphs": [gold, _p(1, "D", "x")]}
    unanswerable = {"id": "q", "answerable": False, "paragraphs": [_p(0, "D", "x")]}
    assert inv.full_pooling_leak([answerable, unanswerable]) == {
        "pairs": 1, "unanswerable_missing_supporting": 1, "all_withheld_present_in_pool": 1,
    }
    # The same pair without its answerable twin in the split: nothing to restore from.
    lone = {"id": "r", "answerable": False, "paragraphs": [_p(0, "D", "x")]}
    assert inv.full_pooling_leak([lone])["pairs"] == 0
