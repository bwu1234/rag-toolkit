"""Tests for rag/eval/answer_match.py, the port of MuSiQue's official EM/F1.

What they protect: agreement with the reference scorer
(StonyBrookNLP/musique ``metrics/answer.py`` at 922ac98). Every expected value
below was computed by that code, not by hand, including its quirks: hyphens
and curly quotes are not ASCII-stripped the same way, accents are kept, and
an answer that normalizes to nothing matches another that does. Synthetic
strings, so no dataset text is committed. Agreement on all 21,753 perturbed
dev answers was checked locally when the port was written.
"""

from __future__ import annotations

import pytest

from rag.eval.answer_match import best_exact_match, best_f1, contains_answer

REFERENCE = [
    ("Paris", ["Paris"], 1, 1.0),
    ("paris.", ["Paris"], 1, 1.0),
    ("The Beatles", ["Beatles"], 1, 1.0),
    ("an apple a day", ["apple day"], 1, 1.0),
    ("New York City", ["New York"], 0, 0.8),
    ("1945", ["September 1945", "1945"], 1, 1.0),
    ("", ["Paris"], 0, 0.0),
    ("Paris", [""], 0, 0.0),
    ("", [""], 1, 1.0),
    ("U.S.A.", ["USA"], 1, 1.0),
    ("Rock-and-roll", ["rock and roll"], 0, 0.0),
    ("Saint Petersburg, Russia", ["Saint Petersburg", "Leningrad"], 0, 0.8),
    ("Leningrad", ["Saint Petersburg", "Leningrad"], 1, 1.0),
    ("the the the", ["a"], 1, 1.0),
    ("Mozart Mozart", ["Mozart"], 0, 0.666667),
    ("Café Müller", ["cafe muller"], 0, 0.0),
    ("John “Jack” Smith", ["John Jack Smith"], 0, 0.666667),
    ("3.5 million", ["3,5 million"], 1, 1.0),
]


@pytest.mark.parametrize(("prediction", "golds", "em", "f1"), REFERENCE)
def test_scores_match_the_reference_scorer(prediction: str, golds: list[str], em: int, f1: float) -> None:
    assert best_exact_match(prediction, golds) == em
    assert best_f1(prediction, golds) == pytest.approx(f1, abs=5e-5)


def test_containment_needs_whole_tokens_of_some_gold_answer() -> None:
    response = "Based on the passages, it was founded by Jean-Paul in Saint Petersburg [2]."
    assert contains_answer(response, ["Leningrad", "Saint Petersburg"])
    assert not contains_answer(response, ["Peter"])  # inside a token run, not a whole token
    assert not contains_answer(response, ["", "Moscow"])
