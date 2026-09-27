"""Tests for the separate answer judge and the multi-hop eval (Milestone 19, phase 0).

No LLM is involved: the chat service and judge are fakes, so these pin the
bookkeeping -- which parts get judged, what counts as complete, what evidence
recall is computed over -- which is exactly what would silently corrupt a
pipeline-vs-agent comparison if it were wrong.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pytest

from rag.config.settings import EvalConfig, LLMConfig, RagConfig
from rag.eval.answer_eval import resolve_judge_config, run_answer_eval
from rag.eval.dataset import EvalDataset, ExpectedSpan
from rag.eval.multihop_eval import CONCLUSION_LABEL, format_tokens, parts_of, run_multihop_eval
from rag.eval.relevance import unmatched_spans
from rag.generation.chat_service import ChatAnswer, ChatService, Citation

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from build_multihop_set import OUTPUT, SOURCE, build  # noqa: E402
import run_answer_matrix  # noqa: E402


class _FakeChatService(ChatService):
    def __init__(self, answer: str, citations: list[Citation] | None = None) -> None:
        self._answer = answer
        self._citations = citations or []

    def ask(self, query: str) -> ChatAnswer:  # type: ignore[override]
        return ChatAnswer(answer=self._answer, citations=self._citations)


class _UsageChatService(ChatService):
    """Returns one answer per call, each carrying the usage given for it."""

    def __init__(self, *usages: tuple[int, float, int | None, int | None]) -> None:
        self._usages = list(usages)

    def ask(self, query: str) -> ChatAnswer:  # type: ignore[override]
        calls, ms, prompt, completion = self._usages.pop(0)
        return ChatAnswer(answer="x", citations=[], llm_calls=calls, llm_ms=ms,
                          prompt_tokens=prompt, completion_tokens=completion)


class _ScriptedJudge:
    """Returns verdicts in order, recording each prompt."""

    def __init__(self, *verdicts: str) -> None:
        self._verdicts = list(verdicts)
        self.prompts: list[str] = []

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        self.prompts.append(prompt)
        return self._verdicts.pop(0)


def _citation(text: str) -> Citation:
    return Citation(chunk_id="c", document_id="d.md", text=text, score=1.0)


def _sample(conclusion: str | None = "A is larger.", kind: str = "cross_company") -> dict:
    sample: dict = {
        "id": "mh-1",
        "query": "Compare A and B.",
        "expected_answer": "A: 10; B: 5",
        "tier": "multihop",
        "kind": kind,
        "parts": [
            {"label": "A", "answer": "10", "spans": ["A reported 10"], "source_id": "a"},
            {"label": "B", "answer": "5", "spans": ["B reported 5"], "source_id": "b"},
        ],
    }
    if conclusion:
        sample["conclusion"] = conclusion
    return sample


# ---------------------------------------------------------------------------
# Judge resolution
# ---------------------------------------------------------------------------


def test_judge_defaults_to_the_generator_and_warns(caplog: pytest.LogCaptureFixture) -> None:
    config = RagConfig(llm=LLMConfig(model="gen"))

    with caplog.at_level(logging.WARNING):
        judge = resolve_judge_config(config)

    assert judge.model == "gen"
    assert "self-graded" in caplog.text


def test_configured_judge_is_used_and_does_not_warn(caplog: pytest.LogCaptureFixture) -> None:
    config = RagConfig(
        llm=LLMConfig(model="gen"),
        eval=EvalConfig(judge=LLMConfig(model="judge", temperature=0.0, timeout_s=900)),
    )

    with caplog.at_level(logging.WARNING):
        judge = resolve_judge_config(config)

    assert (judge.model, judge.temperature, judge.timeout_s) == ("judge", 0.0, 900)
    assert caplog.text == ""


def test_judge_model_override_keeps_other_judge_settings() -> None:
    config = RagConfig(eval=EvalConfig(judge=LLMConfig(model="judge", timeout_s=900)))

    judge = resolve_judge_config(config, "other")

    assert (judge.model, judge.timeout_s) == ("other", 900)
    assert config.eval.judge is not None and config.eval.judge.model == "judge"  # not mutated


def test_judge_override_of_the_generator_does_not_mutate_it() -> None:
    config = RagConfig(llm=LLMConfig(model="gen"))

    resolve_judge_config(config, "judge")

    assert config.llm.model == "gen"


def test_repo_config_leaves_the_judge_unset_for_reproducibility() -> None:
    # null reproduces every pre-Milestone 19 result; the fixed judge is opt-in.
    from rag.config.settings import load_config

    assert load_config().eval.judge is None


# ---------------------------------------------------------------------------
# answer_eval turn stats
# ---------------------------------------------------------------------------


def test_answer_eval_counts_empty_answers() -> None:
    dataset = EvalDataset.from_dicts([{"id": "q", "query": "q?", "expected_answer": "A."}])

    report = run_answer_eval(dataset, _FakeChatService("   "), _ScriptedJudge("FAIL"))

    assert report.num_empty == 1
    assert report.sample_results[0].retrieval_rounds == 1


# ---------------------------------------------------------------------------
# Evidence matching
# ---------------------------------------------------------------------------


def test_unmatched_spans_searches_the_union_of_texts() -> None:
    spans = [ExpectedSpan("alpha one"), ExpectedSpan("beta  two"), ExpectedSpan("gamma")]

    missing = unmatched_spans(spans, ["xx Alpha One xx", "beta two"])

    assert missing == ["gamma"]


# ---------------------------------------------------------------------------
# Parts and the runner
# ---------------------------------------------------------------------------


def test_parts_include_the_conclusion_last() -> None:
    sample = EvalDataset.from_dicts([_sample()]).samples[0]

    parts = parts_of(sample)

    assert [p.label for p in parts] == ["A", "B", CONCLUSION_LABEL]
    assert parts[-1].spans == []


def test_parts_of_rejects_a_single_hop_sample() -> None:
    sample = EvalDataset.from_dicts([{"id": "x", "query": "q", "expected_answer": "a"}]).samples[0]

    with pytest.raises(ValueError, match="no 'parts'"):
        parts_of(sample)


def test_complete_requires_every_part_and_the_conclusion() -> None:
    dataset = EvalDataset.from_dicts([_sample()])
    chat = _FakeChatService("A is 10, B not found.", [_citation("A reported 10")])

    report = run_multihop_eval(dataset, chat, _ScriptedJudge("PASS", "FAIL", "PASS"))

    result = report.sample_results[0]
    assert not result.complete
    assert result.completeness == pytest.approx(2 / 3)
    assert report.complete_rate == 0.0
    assert result.evidence_recall == pytest.approx(0.5)
    assert result.missing_spans == ["B reported 5"]


def test_each_part_is_judged_against_its_own_reference() -> None:
    dataset = EvalDataset.from_dicts([_sample(conclusion=None)])
    judge = _ScriptedJudge("PASS", "PASS")

    report = run_multihop_eval(dataset, _FakeChatService("A 10, B 5"), judge)

    assert report.complete_rate == 1.0
    assert "Part being graded: A\nReference answer for this part: 10" in judge.prompts[0]
    assert "Part being graded: B\nReference answer for this part: 5" in judge.prompts[1]


def test_unparseable_part_verdict_is_not_a_pass() -> None:
    dataset = EvalDataset.from_dicts([_sample(conclusion=None)])

    report = run_multihop_eval(dataset, _FakeChatService("x"), _ScriptedJudge("PASS", "MAYBE"))

    assert report.num_unparseable == 1
    assert not report.sample_results[0].complete


def test_complete_rate_is_reported_per_kind() -> None:
    a, b = _sample(conclusion=None, kind="cross_period"), _sample(conclusion=None, kind="aggregation")
    b["id"] = "mh-2"
    dataset = EvalDataset.from_dicts([a, b])

    report = run_multihop_eval(dataset, _FakeChatService("x"), _ScriptedJudge("PASS", "PASS", "PASS", "FAIL"))

    assert report.complete_rate_by_kind == {"aggregation": 0.0, "cross_period": 1.0}


def _two_samples() -> EvalDataset:
    a, b = _sample(conclusion=None), _sample(conclusion=None)
    b["id"] = "mh-2"
    return EvalDataset.from_dicts([a, b])


def test_each_sample_carries_its_turns_llm_usage() -> None:
    chat = _UsageChatService((3, 1500.0, 900, 40), (1, 500.0, 300, 20))

    report = run_multihop_eval(_two_samples(), chat, _ScriptedJudge(*["PASS"] * 4))

    first = report.sample_results[0]
    assert (first.llm_calls, first.llm_ms, first.prompt_tokens, first.completion_tokens) == (3, 1500.0, 900, 40)
    assert report.mean_llm_calls == 2.0
    assert report.mean_llm_s == pytest.approx(1.0)
    assert (report.mean_prompt_tokens, report.mean_completion_tokens) == (600.0, 30.0)
    assert report.num_with_tokens == 2


def test_token_means_skip_samples_whose_provider_reported_none() -> None:
    # Unknown is not zero: averaging a None in as 0 would halve the mean here.
    chat = _UsageChatService((2, 100.0, 800, 50), (2, 100.0, None, None))

    report = run_multihop_eval(_two_samples(), chat, _ScriptedJudge(*["PASS"] * 4))

    assert (report.mean_prompt_tokens, report.mean_completion_tokens) == (800.0, 50.0)
    assert report.num_with_tokens == 1
    assert report.mean_llm_calls == 2.0  # calls are always counted
    assert format_tokens(report) == ", 800 prompt / 50 generated tokens (from 1 of 2)"


def test_token_means_are_unknown_when_no_sample_reported_them() -> None:
    chat = _UsageChatService((1, 10.0, None, None), (1, 10.0, None, None))

    report = run_multihop_eval(_two_samples(), chat, _ScriptedJudge(*["PASS"] * 4))

    assert report.mean_prompt_tokens is None and report.mean_completion_tokens is None
    assert report.num_with_tokens == 0
    assert format_tokens(report) == ""


def test_judge_calls_are_not_counted_as_the_turns_cost() -> None:
    # The judge is called outside ask(), so however many parts it grades, the
    # sample's usage is only what the answering turn reported.
    chat = _UsageChatService((1, 10.0, 100, 10))
    dataset = EvalDataset.from_dicts([_sample()])

    report = run_multihop_eval(dataset, chat, _ScriptedJudge("PASS", "PASS", "PASS"))

    assert report.sample_results[0].llm_calls == 1


def test_matrix_saves_each_parts_judge_reply_and_the_missing_evidence() -> None:
    dataset = EvalDataset.from_dicts([_sample(conclusion=None)])
    chat = _FakeChatService("A is 10.", [_citation("A reported 10")])

    saved = run_answer_matrix.run_multihop(
        chat, _ScriptedJudge("PASS matches", "FAIL says not found"), dataset
    )["samples"][0]

    assert saved["parts"] == {"A": True, "B": False}
    assert saved["judge_outputs"] == {"A": "PASS matches", "B": "FAIL says not found"}
    assert saved["missing_spans"] == ["B reported 5"]


# ---------------------------------------------------------------------------
# The committed set
# ---------------------------------------------------------------------------


def test_committed_multihop_set_matches_its_builder() -> None:
    # Gold is inherited from edgar_eval_set.json; a hand edit to either file
    # without a rebuild would leave the two disagreeing about the same fact.
    built = [s.to_dict() for s in build({s.id: s for s in EvalDataset.load(SOURCE)})]

    assert built == json.loads(OUTPUT.read_text(encoding="utf-8"))


def test_committed_multihop_set_is_multi_part_and_sized_per_plan() -> None:
    dataset = EvalDataset.load(OUTPUT)

    assert 30 <= len(dataset) <= 40
    for sample in dataset:
        parts = sample.extra["parts"]
        assert len(parts) >= 2, sample.id
        assert all(p["spans"] for p in parts), sample.id
