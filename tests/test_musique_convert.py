"""Tests for scripts/musique_to_eval_set.py, on synthetic MuSiQue rows.

What they protect: each decomposition step maps to its own supporting
paragraph (per-hop evidence depends on it), malformed rows fail rather than
being dropped, a repeated question text is kept once, and the stratified
sample hits its hop quotas.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import musique_to_eval_set as conv  # noqa: E402
from inventory_musique import paragraph_id  # noqa: E402


def _row(qid: str = "2hop__1_2", question: str = "Who is the spouse of the Green performer?") -> dict:
    paragraphs = [
        {"idx": 0, "title": "Distractor", "paragraph_text": "noise", "is_supporting": False},
        {"idx": 1, "title": "Green", "paragraph_text": "performed by Steve Hillage", "is_supporting": True},
        {"idx": 2, "title": "Steve Hillage", "paragraph_text": "married Miquette Giraudy", "is_supporting": True},
    ]
    return {
        "id": qid, "question": question, "paragraphs": paragraphs, "answer": "Miquette Giraudy",
        "answer_aliases": [], "answerable": True,
        "question_decomposition": [
            {"id": 1, "question": "Green >> performer", "answer": "Steve Hillage", "paragraph_support_idx": 1},
            {"id": 2, "question": "#1 >> spouse", "answer": "Miquette Giraudy", "paragraph_support_idx": 2},
        ],
    }


def test_each_step_carries_its_own_supporting_paragraph_as_indexed() -> None:
    sample = conv.to_sample(_row())
    parts = sample.extra["parts"]
    assert [p["spans"] for p in parts] == [["Green performed by Steve Hillage"],
                                           ["Steve Hillage married Miquette Giraudy"]]
    assert sample.expected_doc_ids == sorted(p["source_id"] for p in parts)
    assert (sample.extra["kind"], sample.extra["hops"], sample.extra["hop_type"]) == ("2hop", 2, "2hop")
    assert parts[0]["source_id"] == paragraph_id(_row()["paragraphs"][1])


def test_malformed_rows_fail_rather_than_being_dropped() -> None:
    wrong_step = _row()
    wrong_step["question_decomposition"][1]["paragraph_support_idx"] = 0  # a distractor
    with pytest.raises(conv.ConversionError, match="no supporting paragraph"):
        conv.to_sample(wrong_step)
    too_few = _row("3hop1__1_2_3")
    with pytest.raises(conv.ConversionError, match="2 decomposition steps for a 3-hop"):
        conv.to_sample(too_few)


def test_a_repeated_question_text_is_kept_once() -> None:
    rows = [_row("2hop__a"), _row("2hop__b"), _row("2hop__c", question="Another question?")]
    kept, dropped = conv.drop_repeated_questions(rows)
    assert [r["id"] for r in kept] == ["2hop__a", "2hop__c"] and dropped == ["2hop__b"]


def test_the_stratified_sample_meets_its_quotas_or_fails() -> None:
    rows = [_row(f"{h}hop__{i}") for h, n in ((2, 50), (3, 30), (4, 20)) for i in range(n)]
    shares = {2: 0.5, 3: 0.3, 4: 0.2}
    picked = conv.stratified_sample(rows, 10, random.Random(1), shares)
    assert [conv.hops_of(r) for r in picked].count(4) == 2 and len(picked) == 10
    with pytest.raises(conv.ConversionError, match="Not enough"):
        conv.stratified_sample(rows, 100, random.Random(1), {2: 0.2, 3: 0.2, 4: 0.6})
