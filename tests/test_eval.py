"""Tests for the evaluation pipeline.

Covers:
- Retrieval metric functions (pure, no I/O)
- EvalDataset load/save round-trip
- retrieval_eval runner (fake Retriever, no real index)
- answer_eval runner (fake ChatService + fake LLMClient, no real LLM)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.eval.dataset import EvalDataset, EvalSample, ExpectedSpan
from rag.eval.metrics import (
    hit_rate,
    mean,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from rag.eval.relevance import judge_ranking, normalize
from rag.eval.retrieval_eval import print_report, run_retrieval_eval
from rag.eval.answer_eval import (
    _parse_verdict,
    judge_for,
    run_answer_eval,
)
from rag.generation.chat_service import ChatAnswer, ChatService, Citation
from rag.retrieval.retriever import RetrievalResult, Retriever
from rag.vectorstore.base import ScoredChunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scored(
    chunk_id: str,
    document_id: str,
    score: float = 0.9,
    text: str = "some text",
) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk_id,
        text=text,
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


def test_hit_rate_is_1_when_any_result_relevant() -> None:
    assert hit_rate([0, 1]) == 1.0


def test_hit_rate_is_0_when_nothing_relevant() -> None:
    assert hit_rate([0, 0]) == 0.0


def test_hit_rate_is_0_for_empty_ranking() -> None:
    assert hit_rate([]) == 0.0


def test_recall_at_k_partial_coverage() -> None:
    assert recall_at_k(1, 2) == pytest.approx(0.5)


def test_recall_at_k_full_coverage() -> None:
    assert recall_at_k(2, 2) == pytest.approx(1.0)


def test_recall_at_k_no_coverage() -> None:
    assert recall_at_k(0, 2) == pytest.approx(0.0)


def test_recall_at_k_nothing_expected() -> None:
    assert recall_at_k(0, 0) == 0.0


def test_precision_at_k_all_relevant() -> None:
    assert precision_at_k([1, 1]) == pytest.approx(1.0)


def test_precision_at_k_partial() -> None:
    assert precision_at_k([1, 0]) == pytest.approx(0.5)


def test_precision_at_k_none_relevant() -> None:
    assert precision_at_k([0]) == pytest.approx(0.0)


def test_precision_at_k_empty_retrieved() -> None:
    assert precision_at_k([]) == 0.0


def test_reciprocal_rank_first_result_relevant() -> None:
    assert reciprocal_rank([1, 0]) == pytest.approx(1.0)


def test_reciprocal_rank_second_result_relevant() -> None:
    assert reciprocal_rank([0, 1]) == pytest.approx(0.5)


def test_reciprocal_rank_no_relevant() -> None:
    assert reciprocal_rank([0, 0]) == 0.0


def test_reciprocal_rank_empty_ranking() -> None:
    assert reciprocal_rank([]) == 0.0


def test_ndcg_is_1_for_ideal_ranking() -> None:
    assert ndcg_at_k([3, 1, 0], [3, 1, 0]) == pytest.approx(1.0)


def test_ndcg_penalises_a_relevant_result_ranked_lower() -> None:
    # Same results, worse order -> strictly lower NDCG. This is the ordering
    # sensitivity that hit rate / recall / precision all discard.
    assert ndcg_at_k([0, 1], [1, 0]) < ndcg_at_k([1, 0], [1, 0])


def test_ndcg_rewards_higher_grade_first() -> None:
    ideal = [3, 1]
    assert ndcg_at_k([1, 3], ideal) < ndcg_at_k([3, 1], ideal)


def test_ndcg_is_0_when_nothing_relevant_was_achievable() -> None:
    assert ndcg_at_k([0, 0], [0, 0]) == 0.0


def test_mean_of_values() -> None:
    assert mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)


def test_mean_empty_list() -> None:
    assert mean([]) == 0.0


# ---------------------------------------------------------------------------
# Relevance judging
# ---------------------------------------------------------------------------


def test_normalize_folds_whitespace_case_and_punctuation() -> None:
    assert normalize("The  Company’s\nrevenue") == normalize("the company's revenue")


def test_normalize_folds_em_dash_to_the_fetchers_double_hyphen() -> None:
    assert normalize("revenue—up 8%") == normalize("revenue--up 8%")


def test_span_matching_grades_a_chunk_containing_the_quote() -> None:
    sample = EvalSample(
        id="s1",
        query="q",
        expected_spans=[ExpectedSpan(text="gross margin was 46.5%")],
    )
    chunks = [_scored("c0", "d.md", text="Noise."), _scored("c1", "d.md", text="Gross margin was 46.5% for the quarter.")]
    judgment = judge_ranking(sample, chunks)
    assert judgment.mode == "span"
    assert judgment.gains == [0, 1]
    assert judgment.covered == 1


def test_span_matching_survives_whitespace_and_quote_differences() -> None:
    sample = EvalSample(
        id="s1", query="q", expected_spans=[ExpectedSpan(text="the Company's outlook")]
    )
    chunks = [_scored("c0", "d.md", text="We reaffirm the  Company’s\noutlook today.")]
    assert judge_ranking(sample, chunks).gains == [1]


def test_span_matching_reports_unmatched_spans() -> None:
    sample = EvalSample(
        id="s1",
        query="q",
        expected_spans=[ExpectedSpan(text="present"), ExpectedSpan(text="absent")],
    )
    judgment = judge_ranking(sample, [_scored("c0", "d.md", text="this is present")])
    assert judgment.covered == 1
    assert judgment.unmatched_spans == ["absent"]


def test_chunk_matching_two_spans_takes_the_higher_grade_not_the_sum() -> None:
    sample = EvalSample(
        id="s1",
        query="q",
        expected_spans=[ExpectedSpan("alpha", grade=1), ExpectedSpan("beta", grade=3)],
    )
    judgment = judge_ranking(sample, [_scored("c0", "d.md", text="alpha and beta")])
    assert judgment.gains == [3]


def test_spans_take_precedence_over_doc_ids_when_both_present() -> None:
    sample = EvalSample(
        id="s1",
        query="q",
        expected_doc_ids=["right.md"],
        expected_spans=[ExpectedSpan(text="the actual answer")],
    )
    # Right document, wrong passage: document matching would call this a hit.
    judgment = judge_ranking(sample, [_scored("c0", "right.md", text="unrelated boilerplate")])
    assert judgment.mode == "span"
    assert judgment.gains == [0]


def test_document_matching_is_used_when_no_spans_given() -> None:
    sample = EvalSample(id="s1", query="q", expected_doc_ids=["right.md"])
    chunks = [_scored("c0", "wrong.md"), _scored("c1", "right.md")]
    judgment = judge_ranking(sample, chunks)
    assert judgment.mode == "document"
    assert judgment.gains == [0, 1]
    assert judgment.covered == 1


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
    assert report.overall.mean_recall == pytest.approx(1.0)
    # All 3 chunks returned per query; expected docs land at ranks 1, 2, 3 ->
    # MRR = (1/1 + 1/2 + 1/3) / 3 = 11/18
    assert report.overall.mrr == pytest.approx((1.0 + 0.5 + 1 / 3) / 3)
    assert len(retriever.seen_queries) == 3


def test_run_retrieval_eval_all_misses() -> None:
    retriever = _FakeRetriever([_scored("x::chunk0", "x.pdf")])
    dataset = _make_retrieval_dataset()

    report = run_retrieval_eval(dataset, retriever)

    assert report.mean_hit_rate == pytest.approx(0.0)
    assert report.overall.mrr == pytest.approx(0.0)


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


def test_run_retrieval_eval_reports_a_recall_curve() -> None:
    chunks = [_scored("a::chunk0", "a.pdf"), _scored("b::chunk0", "b.pdf"), _scored("c::chunk0", "c.pdf")]
    report = run_retrieval_eval(_make_retrieval_dataset(), _FakeRetriever(chunks))

    # 3 results returned, so only cutoffs at or below 3 are reported -- a k=10
    # entry here would be a flat line masquerading as a finding.
    assert sorted(report.overall.recall_by_k) == [1, 3]
    # Each query's document sits at a different rank, so only 1 of 3 is covered
    # at k=1, and all 3 by k=3.
    assert report.overall.recall_by_k[1] == pytest.approx(1 / 3)
    assert report.overall.recall_by_k[3] == pytest.approx(1.0)


def test_run_retrieval_eval_splits_summaries_when_modes_are_mixed() -> None:
    dataset = EvalDataset.from_dicts([
        {"id": "doc-mode", "query": "q1", "expected_doc_ids": ["a.pdf"]},
        {"id": "span-mode", "query": "q2", "expected_spans": ["never appears"]},
    ])
    report = run_retrieval_eval(dataset, _FakeRetriever([_scored("a::chunk0", "a.pdf")]))

    assert [s.label for s in report.by_mode] == ["document", "span"]
    assert report.overall.label == "all samples"


def test_run_retrieval_eval_does_not_split_when_all_samples_share_a_mode() -> None:
    report = run_retrieval_eval(_make_retrieval_dataset(), _FakeRetriever([]))
    assert report.by_mode == []
    assert report.overall.label == "document"


def test_print_retrieval_report_does_not_raise(capsys) -> None:
    dataset = EvalDataset.from_dicts([
        {"id": "q1", "query": "query one", "expected_doc_ids": ["a.pdf"]},
        {"id": "q2", "query": "query two", "expected_doc_ids": ["b.pdf"]},
    ])
    report = run_retrieval_eval(dataset, _FakeRetriever([_scored("a::chunk0", "a.pdf")]))

    print_report(report, verbose=True)
    out = capsys.readouterr().out
    assert "Hit rate" in out
    assert "MRR" in out
    assert "NDCG" in out
    assert "q1" in out


def test_print_report_surfaces_unmatched_spans_in_verbose_mode(capsys) -> None:
    dataset = EvalDataset.from_dicts([
        {"id": "q1", "query": "q", "expected_spans": ["a quote that is absent"]},
    ])
    report = run_retrieval_eval(dataset, _FakeRetriever([_scored("c0", "a.pdf", text="unrelated")]))

    print_report(report, verbose=True)
    assert "a quote that is absent" in capsys.readouterr().out


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


# ---------------------------------------------------------------------------
# Judge rubric selection
# ---------------------------------------------------------------------------


def test_answerable_samples_use_the_reference_comparison_rubric() -> None:
    system, _ = judge_for(EvalSample(id="s", query="q", expected_answer="a"))
    assert "reference answer" in system.lower()


def test_refusal_samples_use_the_did_it_decline_rubric() -> None:
    """The default rubric fails anything that 'refuses to answer' -- correct for
    answerable questions, exactly backwards for a refusal set."""
    sample = EvalSample(id="s", query="q", expected_answer="must decline", extra={"tier": "refusal"})
    system, _ = judge_for(sample)

    assert "declined" in system.lower()
    assert "wording" in system.lower(), "verbosity of the refusal must be explicitly irrelevant"


def test_refusal_rubric_does_not_penalise_refusing() -> None:
    """Regression: two correct refusals differing only in detail were graded
    FAIL and PASS by the reference-comparison rubric, which is what turned the
    refusal set into a measure of prose similarity rather than of behaviour."""
    answerable_system, _ = judge_for(EvalSample(id="s", query="q", expected_answer="a"))
    refusal_system, _ = judge_for(
        EvalSample(id="s", query="q", expected_answer="a", extra={"tier": "refusal"})
    )

    assert "refuses to answer" in answerable_system
    assert "refuses to answer" not in refusal_system
