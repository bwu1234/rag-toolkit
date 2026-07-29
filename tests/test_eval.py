"""Tests for the evaluation pipeline.

Covers:
- Retrieval metric functions (pure, no I/O)
- EvalDataset load/save round-trip
- retrieval_eval runner (fake Retriever, no real index)
- answer_eval runner (fake ChatService + fake LLMClient, no real LLM)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from rag.eval.dataset import EvalDataset, EvalSample
from rag.eval.metrics import (
    hit_rate,
    mean,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from rag.eval.retrieval_eval import EvalReport, SampleResult, print_report, run_retrieval_eval
from rag.eval.answer_eval import (
    AnswerEvalReport,
    AnswerSampleResult,
    _judge_prompt,
    _parse_verdict,
    run_answer_eval,
)
from rag.generation.chat_service import ChatAnswer, ChatService, Citation
from rag.retrieval.retriever import RetrievalResult, Retriever
from rag.vectorstore.base import ScoredChunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scored(chunk_id: str, document_id: str, score: float = 0.9) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk_id,
        text="some text",
        document_id=document_id,
        source=Path(document_id),
        doc_type="markdown",
        score=score,
        metadata={},
    )


class _FakeRetriever(Retriever):
    """Returns a fixed list of chunks for every query."""

    def __init__(self, chunks: list[ScoredChunk]) -> None:
        self._chunks = chunks
        self.seen_queries: list[str] = []

    def retrieve(self, query: str) -> RetrievalResult:
        self.seen_queries.append(query)
        return RetrievalResult(chunks=self._chunks, candidate_count=len(self._chunks))


class _FakeChatService(ChatService):
    """Returns a preset answer for every query."""

    def __init__(self, answer: str, citations: list[Citation] | None = None) -> None:
        self._answer = answer
        self._citations = citations or []
        self.seen_queries: list[str] = []

    def ask(self, query: str) -> ChatAnswer:
        self.seen_queries.append(query)
        return ChatAnswer(answer=self._answer, citations=self._citations)


class _FakeLLMClient:
    """Returns a preset verdict string for every generate() call."""

    def __init__(self, verdict: str) -> None:
        self._verdict = verdict
        self.seen_calls: list[tuple[str, str | None]] = []

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        self.seen_calls.append((prompt, system))
        return self._verdict


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------


def test_hit_rate_is_1_when_any_expected_doc_retrieved() -> None:
    assert hit_rate(["doc-a", "doc-b"], ["doc-b"]) == 1.0


def test_hit_rate_is_0_when_no_expected_doc_retrieved() -> None:
    assert hit_rate(["doc-x", "doc-y"], ["doc-z"]) == 0.0


def test_hit_rate_is_0_for_empty_expected() -> None:
    assert hit_rate(["doc-a"], []) == 0.0


def test_recall_at_k_partial_coverage() -> None:
    # retrieved covers 1 of 2 expected
    assert recall_at_k(["doc-a", "doc-c"], ["doc-a", "doc-b"]) == pytest.approx(0.5)


def test_recall_at_k_full_coverage() -> None:
    assert recall_at_k(["doc-a", "doc-b", "doc-c"], ["doc-a", "doc-b"]) == pytest.approx(1.0)


def test_recall_at_k_no_coverage() -> None:
    assert recall_at_k(["doc-x"], ["doc-a", "doc-b"]) == pytest.approx(0.0)


def test_recall_at_k_empty_expected() -> None:
    assert recall_at_k(["doc-a"], []) == 0.0


def test_precision_at_k_all_relevant() -> None:
    assert precision_at_k(["doc-a", "doc-b"], ["doc-a", "doc-b", "doc-c"]) == pytest.approx(1.0)


def test_precision_at_k_partial() -> None:
    assert precision_at_k(["doc-a", "doc-x"], ["doc-a"]) == pytest.approx(0.5)


def test_precision_at_k_none_relevant() -> None:
    assert precision_at_k(["doc-x"], ["doc-a"]) == pytest.approx(0.0)


def test_precision_at_k_empty_retrieved() -> None:
    assert precision_at_k([], ["doc-a"]) == 0.0


def test_reciprocal_rank_first_result_relevant() -> None:
    assert reciprocal_rank(["doc-a", "doc-b"], ["doc-a"]) == pytest.approx(1.0)


def test_reciprocal_rank_second_result_relevant() -> None:
    assert reciprocal_rank(["doc-x", "doc-a"], ["doc-a"]) == pytest.approx(0.5)


def test_reciprocal_rank_no_relevant() -> None:
    assert reciprocal_rank(["doc-x", "doc-y"], ["doc-a"]) == 0.0


def test_reciprocal_rank_empty_expected() -> None:
    assert reciprocal_rank(["doc-a"], []) == 0.0


def test_mean_of_values() -> None:
    assert mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)


def test_mean_empty_list() -> None:
    assert mean([]) == 0.0


# ---------------------------------------------------------------------------
# EvalDataset
# ---------------------------------------------------------------------------


def test_eval_sample_round_trips_through_dict() -> None:
    sample = EvalSample(
        id="s1",
        query="what is X?",
        expected_doc_ids=["doc-a"],
        expected_answer="X is Y.",
    )
    reconstructed = EvalSample.from_dict(sample.to_dict())
    assert reconstructed.id == sample.id
    assert reconstructed.query == sample.query
    assert reconstructed.expected_doc_ids == sample.expected_doc_ids
    assert reconstructed.expected_answer == sample.expected_answer


def test_eval_sample_extra_fields_preserved() -> None:
    sample = EvalSample.from_dict({"id": "s1", "query": "q", "_note": "a comment"})
    assert sample.extra == {"_note": "a comment"}
    assert sample.to_dict()["_note"] == "a comment"


def test_eval_dataset_load_save_round_trip(tmp_path: Path) -> None:
    original = EvalDataset.from_dicts([
        {"id": "q1", "query": "hello?", "expected_doc_ids": ["a.pdf"], "expected_answer": "hi"},
        {"id": "q2", "query": "bye?", "expected_doc_ids": []},
    ])
    save_path = tmp_path / "set.json"
    original.save(save_path)

    loaded = EvalDataset.load(save_path)
    assert len(loaded) == 2
    assert loaded.samples[0].id == "q1"
    assert loaded.samples[0].expected_doc_ids == ["a.pdf"]
    assert loaded.samples[1].expected_answer is None


def test_eval_dataset_load_rejects_non_array(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text('{"key": "value"}')
    with pytest.raises(ValueError, match="must be a JSON array"):
        EvalDataset.load(bad)


# ---------------------------------------------------------------------------
# run_retrieval_eval
# ---------------------------------------------------------------------------


def _make_retrieval_dataset() -> EvalDataset:
    return EvalDataset.from_dicts([
        {"id": "q1", "query": "what is A?", "expected_doc_ids": ["a.pdf"]},
        {"id": "q2", "query": "what is B?", "expected_doc_ids": ["b.pdf"]},
        {"id": "q3", "query": "what is C?", "expected_doc_ids": ["c.pdf"]},
    ])


def test_run_retrieval_eval_all_hits() -> None:
    chunks = [_scored("a::chunk0", "a.pdf"), _scored("b::chunk0", "b.pdf"), _scored("c::chunk0", "c.pdf")]
    retriever = _FakeRetriever(chunks)
    dataset = _make_retrieval_dataset()

    report = run_retrieval_eval(dataset, retriever)

    assert report.num_samples == 3
    assert report.mean_hit_rate == pytest.approx(1.0)
    assert report.mean_recall == pytest.approx(1.0)
    # All 3 chunks returned per query; expected docs land at ranks 1, 2, 3 ->
    # MRR = (1/1 + 1/2 + 1/3) / 3 = 11/18
    assert report.mrr == pytest.approx((1.0 + 0.5 + 1 / 3) / 3)
    assert len(retriever.seen_queries) == 3


def test_run_retrieval_eval_all_misses() -> None:
    retriever = _FakeRetriever([_scored("x::chunk0", "x.pdf")])
    dataset = _make_retrieval_dataset()

    report = run_retrieval_eval(dataset, retriever)

    assert report.mean_hit_rate == pytest.approx(0.0)
    assert report.mrr == pytest.approx(0.0)


def test_run_retrieval_eval_empty_results() -> None:
    retriever = _FakeRetriever([])
    dataset = _make_retrieval_dataset()

    report = run_retrieval_eval(dataset, retriever)

    assert report.mean_hit_rate == 0.0
    assert report.num_samples == 3


def test_run_retrieval_eval_partial_hits() -> None:
    # Only a.pdf is returned — q1 hits, q2 and q3 miss
    retriever = _FakeRetriever([_scored("a::chunk0", "a.pdf")])
    dataset = _make_retrieval_dataset()

    report = run_retrieval_eval(dataset, retriever)

    assert report.mean_hit_rate == pytest.approx(1 / 3)


def test_print_retrieval_report_does_not_raise(capsys) -> None:
    report = EvalReport(
        num_samples=2,
        mean_hit_rate=0.5,
        mean_recall=0.5,
        mean_precision=0.25,
        mrr=0.75,
        sample_results=[
            SampleResult("q1", "query one", ["a.pdf"], ["a.pdf"], 1.0, 1.0, 1.0, 1.0),
            SampleResult("q2", "query two", ["b.pdf"], ["x.pdf"], 0.0, 0.0, 0.0, 0.0),
        ],
    )
    print_report(report, verbose=True)
    out = capsys.readouterr().out
    assert "Hit rate" in out
    assert "MRR" in out
    assert "q1" in out


# ---------------------------------------------------------------------------
# _parse_verdict
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw,expected", [
    ("PASS", True),
    ("PASS it looks good", True),
    ("pass", True),
    ("FAIL missing detail", False),
    ("fail", False),
    ("", None),
    ("MAYBE", None),
    ("  PASS  ", True),
])
def test_parse_verdict(raw: str, expected: bool | None) -> None:
    assert _parse_verdict(raw) is expected


# ---------------------------------------------------------------------------
# run_answer_eval
# ---------------------------------------------------------------------------


def _make_answer_dataset() -> EvalDataset:
    return EvalDataset.from_dicts([
        {"id": "q1", "query": "what is X?", "expected_answer": "X is Y."},
        {"id": "q2", "query": "what is Z?", "expected_answer": "Z is W."},
        {"id": "q3", "query": "no answer", "expected_doc_ids": ["a.pdf"]},  # no expected_answer
    ])


def test_run_answer_eval_all_pass() -> None:
    dataset = _make_answer_dataset()
    chat = _FakeChatService("X is Y.")
    judge = _FakeLLMClient("PASS looks correct")

    report = run_answer_eval(dataset, chat, judge)

    assert report.num_evaluated == 2
    assert report.num_skipped == 1
    assert report.num_passed == 2
    assert report.num_failed == 0
    assert report.pass_rate == pytest.approx(1.0)


def test_run_answer_eval_all_fail() -> None:
    dataset = _make_answer_dataset()
    chat = _FakeChatService("I don't know.")
    judge = _FakeLLMClient("FAIL answer is wrong")

    report = run_answer_eval(dataset, chat, judge)

    assert report.num_passed == 0
    assert report.num_failed == 2
    assert report.pass_rate == pytest.approx(0.0)


def test_run_answer_eval_unparseable_verdict() -> None:
    dataset = EvalDataset.from_dicts([
        {"id": "q1", "query": "q?", "expected_answer": "A."},
    ])
    chat = _FakeChatService("Some answer.")
    judge = _FakeLLMClient("MAYBE I'm not sure")

    report = run_answer_eval(dataset, chat, judge)

    assert report.num_unparseable == 1
    assert report.sample_results[0].passed is None


def test_run_answer_eval_skips_samples_without_expected_answer() -> None:
    dataset = EvalDataset.from_dicts([
        {"id": "q1", "query": "q?", "expected_doc_ids": ["a.pdf"]},  # no expected_answer
    ])
    chat = _FakeChatService("answer")
    judge = _FakeLLMClient("PASS")

    report = run_answer_eval(dataset, chat, judge)

    assert report.num_evaluated == 0
    assert report.num_skipped == 1
    assert len(chat.seen_queries) == 0  # ChatService never called


def test_run_answer_eval_judge_receives_expected_and_actual() -> None:
    dataset = EvalDataset.from_dicts([
        {"id": "q1", "query": "What is X?", "expected_answer": "X is Y."},
    ])
    chat = _FakeChatService("X is probably Y.")
    judge = _FakeLLMClient("PASS")

    run_answer_eval(dataset, chat, judge)

    prompt, system = judge.seen_calls[0]
    assert "What is X?" in prompt
    assert "X is Y." in prompt
    assert "X is probably Y." in prompt
    assert system is not None and "judge" in system.lower()
