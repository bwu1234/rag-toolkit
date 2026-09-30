"""Tests for data/eval/edgar_adaptive_set.json and the script that builds it.

What they protect: the committed set is what the builder makes (a hand edit
to the JSON would drift from the spec's reviewed gold), and a bridge question
never names the company its first search is meant to find -- the property the
set exists to test.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import build_adaptive_set  # noqa: E402
import build_multihop_set  # noqa: E402

from rag.eval.dataset import EvalDataset  # noqa: E402


def _built() -> list:
    sources = {s.id: s for s in EvalDataset.load(build_multihop_set.SOURCE)}
    return build_multihop_set.build(sources, build_adaptive_set.SPEC)


def test_the_committed_set_is_what_the_builder_makes() -> None:
    committed = [s.to_dict() for s in EvalDataset.load(build_adaptive_set.OUTPUT)]
    assert committed == [s.to_dict() for s in _built()]


def test_a_bridge_question_never_names_the_company_it_asks_to_find() -> None:
    identity_parts = 0
    for sample in _built():
        first = sample.extra["parts"][0]
        if sample.extra["kind"] != "bridge" or not first["label"].startswith("The "):
            continue  # a comparison-then-follow-up names both candidates on purpose
        identity_parts += 1
        company = first["answer"].split()[0].split("(")[0]
        assert company.lower() not in sample.query.lower(), sample.id
    assert identity_parts >= 5


def test_every_question_has_a_known_kind_and_gold_evidence() -> None:
    for sample in _built():
        assert sample.extra["kind"] in {"bridge", "discovery"}
        assert all(part["spans"] for part in sample.extra["parts"]), sample.id
