"""Judge calibration against reviewed human labels.

What these protect: a false pass and a false fail are counted against the
right denominators (human FAILs and human PASSes), per category and tag;
an item whose label isn't settled -- no review, or a disagreement nobody
adjudicated -- and a fixture with no declared bounds are refused rather than
calibrated against; a single reviewer is reported as a limitation; repeated
attempts measure consistency apart from error; and drafting samples across
judge outcomes without telling reviewers what the judge said.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from rag.config.settings import LLMConfig
from rag.eval.calibrate import FixtureNotReady, calibrate, draft_items, render_calibration
from rag.eval.judge import judge_run
from tests.test_eval_judge import JUDGE, _answers, _Judge, _multihop, _run


class _ByAnswer:
    """Replies by the answer text in the prompt: the judge's verdict is part of each test case."""

    def __init__(self, replies: dict[str, list[str]]) -> None:
        self.replies = replies

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        key = next(k for k in self.replies if f"System answer: {k}" in prompt)
        return self.replies[key].pop(0) if len(self.replies[key]) > 1 else self.replies[key][0]


def _review(passed: bool, reviewer: str = "R1") -> dict[str, Any]:
    return {"reviewer": reviewer, "passed": passed, "rationale": "because", "at": "2026-10-14"}


def _item(n: int, *, human: bool | None, category: str = "single-hop", **fields: Any) -> dict[str, Any]:
    return {
        "id": f"cal-{n}", "stage": "answer", "category": category, "query": f"q{n}", "reference": f"ref {n}",
        "answer": f"ans{n}", "reviews": [] if human is None else [_review(human)], **fields,
    }


def _fixture(tmp_path: Path, items: list[dict[str, Any]], bounds: dict[str, float] | None = None) -> Path:
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps({"schema": 1, "bounds": bounds or {"false_pass_rate": 0.5, "false_fail_rate": 0.5},
                                "items": items}))
    return path


def test_false_passes_and_fails_are_counted_against_the_right_denominators(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, [
        _item(1, human=True), _item(2, human=True), _item(3, human=True),
        _item(4, human=False, tags=["wrong_period"]), _item(5, human=False, tags=["wrong_period"]),
        _item(6, human=False, category="refusal", tier="refusal"),
    ])
    judge = _ByAnswer({"ans1": ["PASS"], "ans2": ["FAIL"], "ans3": ["maybe?"], "ans4": ["PASS"],
                       "ans5": ["FAIL"], "ans6": ["FAIL"]})

    summary = calibrate(fixture, judge, JUDGE)["summary"]  # type: ignore[arg-type]
    [first] = summary["attempts"]

    assert (first["false_pass"]["count"], first["false_pass"]["n"]) == (1, 3)  # ans4, of 3 human FAILs
    assert (first["false_fail"]["count"], first["false_fail"]["n"]) == (1, 3)  # ans2, of 3 human PASSes
    assert first["unparseable"] == 1 and first["agreement"]["count"] == 3
    period = summary["groups"]["tag: wrong_period"]
    assert (period["false_pass"]["count"], period["false_pass"]["n"]) == (1, 2)
    assert summary["groups"]["category: refusal"]["false_pass"]["count"] == 0
    assert summary["labels"]["single_reviewer"] is True


def test_the_refusal_rubric_judges_refusal_items(tmp_path: Path) -> None:
    seen: list[str] = []

    class _Spy:
        def generate(self, prompt: str, *, system: str | None = None) -> str:
            seen.append(system or "")
            return "PASS"

    fixture = _fixture(tmp_path, [_item(1, human=True, category="refusal", tier="refusal")])
    calibrate(fixture, _Spy(), JUDGE)  # type: ignore[arg-type]
    assert "DECLINED" in seen[0]


@pytest.mark.parametrize("items, bounds, match", [
    ([_item(1, human=None)], None, "have no review"),
    ([{**_item(1, human=True), "reviews": [_review(True, "R1"), _review(False, "R2")]}], None,
     "disagree with no adjudication on cal-1"),
    ([_item(1, human=True)], {"false_pass_rate": 0.1}, "no declared bounds"),
])
def test_an_unsettled_fixture_is_refused(
    tmp_path: Path, items: list[dict[str, Any]], bounds: dict[str, float] | None, match: str
) -> None:
    fixture = _fixture(tmp_path, items)
    if bounds is not None:
        data = json.loads(fixture.read_text())
        data["bounds"] = bounds
        fixture.write_text(json.dumps(data))
    with pytest.raises(FixtureNotReady, match=match):
        calibrate(fixture, _Judge(), JUDGE)  # type: ignore[arg-type]


def test_an_adjudication_settles_a_disagreement_and_double_review_is_reported(tmp_path: Path) -> None:
    disputed = {**_item(1, human=None), "reviews": [_review(True, "R1"), _review(False, "R2")],
                "adjudication": {"passed": False, "reason": "wrong fiscal year", "by": "R1"}}
    agreed = {**_item(2, human=None), "reviews": [_review(True, "R1"), _review(True, "R2")]}
    record = calibrate(_fixture(tmp_path, [disputed, agreed]), _Judge(), JUDGE)  # type: ignore[arg-type]

    labels = record["summary"]["labels"]
    assert labels == {"items": 2, "reviewers": ["R1", "R2"], "single_reviewer": False,
                      "double_reviewed": 2, "review_agreements": 1, "adjudicated": 1, "double_review_rule": None}
    assert record["summary"]["attempts"][0]["false_pass"]["count"] == 1  # PASS on the adjudicated FAIL
    assert "Single reviewer" not in render_calibration(record)


def test_repeated_attempts_report_consistency_and_bounds(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, [_item(1, human=True), _item(2, human=False)],
                       bounds={"false_pass_rate": 0.1, "false_fail_rate": 0.99})
    judge = _ByAnswer({"ans1": ["PASS", "PASS", "PASS"], "ans2": ["FAIL", "PASS", "FAIL"]})

    record = calibrate(fixture, judge, JUDGE, num_attempts=3)  # type: ignore[arg-type]
    summary = record["summary"]

    assert summary["consistency"] == {"attempts": 3, "stable": 1, "flipped": ["cal-2"]}
    assert [a["false_pass"]["count"] for a in summary["attempts"]] == [0, 1, 0]
    # One human FAIL can't show a false-pass rate under 10%: the interval is too wide.
    assert summary["bounds"]["false_pass"]["within"] is False
    assert summary["bounds"]["false_fail"]["within"] is True
    table = render_calibration(record)
    assert "exploratory" in table and "Single reviewer" in table
    assert len(record["verdicts"]) == 3 and record["verdicts"][0]["cal-2"]["reply"] == "FAIL"


def test_a_multihop_item_is_judged_as_its_one_part(tmp_path: Path) -> None:
    prompts: list[str] = []

    class _Spy:
        def generate(self, prompt: str, *, system: str | None = None) -> str:
            prompts.append(prompt)
            return "PASS"

    item = {**_item(1, human=True, category="multihop part"), "stage": "multihop", "part": "Delta, FY2024"}
    calibrate(_fixture(tmp_path, [item]), _Spy(), LLMConfig(model="j"))  # type: ignore[arg-type]
    assert len(prompts) == 1 and "Part being graded: Delta, FY2024" in prompts[0]


def test_drafting_spans_judge_outcomes_without_revealing_them(tmp_path: Path) -> None:
    answers = _run(tmp_path, _answers(4))
    judge_run(answers, _Judge("PASS", "FAIL", "PASS", "FAIL"), JUDGE)  # type: ignore[arg-type]
    multihop = _run(tmp_path, _multihop(), stage="multihop")
    judge_run(multihop, _Judge("PASS", "FAIL", "PASS"), JUDGE)  # type: ignore[arg-type]

    items = draft_items([answers.run_id, multihop.run_id], per_cell=1, runs_dir=tmp_path)

    # One per (category, outcome) cell: single-hop PASS and FAIL, part PASS and FAIL, conclusion PASS.
    assert sorted(i["category"] for i in items) == [
        "multihop conclusion", "multihop part", "multihop part", "single-hop", "single-hop"]
    assert all(i["reviews"] == [] and i["adjudication"] is None for i in items)
    assert not any("PASS" in json.dumps(i) or "FAIL" in json.dumps(i) for i in items)
    part = next(i for i in items if i["category"] == "multihop part")
    assert part["reference"] in {"1", "2"} and part["source"]["run"] == multihop.run_id
