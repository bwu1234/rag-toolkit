"""Short-answer exact match and token F1, as MuSiQue's official evaluation scores them.

Ported from ``metrics/answer.py`` of
https://github.com/StonyBrookNLP/musique at commit
``922ac98f19a201998dbdae6d7f2887a5258dbdeb`` (itself SQuAD's scorer, via
AllenNLP). The rules are the reference's, unchanged, so a score here is the
score the official script gives: lower-case, strip ASCII punctuation, drop
the articles "a", "an", "the", collapse whitespace; EM is equality of what
remains, F1 is token-overlap F1; both take the maximum over the gold answer
and its aliases. ``tests/test_answer_match.py`` checks agreement with values
computed by the reference code.

`contains_answer` is this repo's addition, not part of the official metric:
whether a normalized gold answer occurs as a whole-token run inside a long
answer. It needs no extraction step, so it cross-checks one: a row whose EM
moves while its containment rate doesn't points at the extractor, not the
answers.

Pure functions, stdlib only.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Sequence

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCTUATION = set(string.punctuation)


def normalize_answer(text: str) -> str:
    """Lower-case, strip punctuation, drop articles, collapse whitespace (the reference's order)."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in _PUNCTUATION)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def _tokens(text: str) -> list[str]:
    return normalize_answer(text).split() if text else []


def exact_match(prediction: str, gold: str) -> int:
    return int(normalize_answer(prediction) == normalize_answer(gold))


def token_f1(prediction: str, gold: str) -> float:
    pred, ref = _tokens(prediction), _tokens(gold)
    if not pred or not ref:
        # Either is empty: 1 when both are, as in the reference.
        return float(pred == ref)
    same = sum((Counter(pred) & Counter(ref)).values())
    if same == 0:
        return 0.0
    precision, recall = same / len(pred), same / len(ref)
    return 2 * precision * recall / (precision + recall)


def best_exact_match(prediction: str, golds: Sequence[str]) -> int:
    """EM against the best-matching of the gold answer and its aliases."""
    return max(exact_match(prediction, g) for g in golds)


def best_f1(prediction: str, golds: Sequence[str]) -> float:
    """Token F1 against the best-matching of the gold answer and its aliases."""
    return max(token_f1(prediction, g) for g in golds)


def contains_answer(response: str, golds: Sequence[str]) -> bool:
    """True when some normalized gold answer is a whole-token run in the normalized response."""
    padded = f" {normalize_answer(response)} "
    return any(f" {g} " in padded for g in (normalize_answer(x) for x in golds) if g)
