"""Hand-computed fixtures for qrels scoring (public benchmarks plan, phase 2).

Each expected value is worked out in the test from the trec_eval definitions
(linear gain, log2(rank + 1) discount, ideal from all positive grades), so a
reader can check the arithmetic without running anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from rag.eval.dataset import MODE_QRELS, EvalDataset, EvalSample
from rag.eval.qrels import (
    DocRanking,
    RankedDoc,
    RunFileError,
    document_ranking,
    ndcg_cut,
    qrels_grades,
    read_run,
    recall_cut,
    score_rankings,
    trec_order,
    write_run,
)
from rag.eval.relevance import judge_ranking
from rag.eval.retrieval_eval import STAGE_1, STAGE_FINAL, run_qrels_eval, save_qrels_run
from rag.retrieval.retriever import RetrievalResult
from rag.vectorstore.base import ScoredChunk

GRADES = {"a": 2, "b": 1, "c": 0}


def _sample(qid: str, doc_ids: list) -> EvalSample:
    return EvalSample.from_dict({"id": qid, "query": f"query {qid}", "expected_doc_ids": doc_ids, "matching_mode": MODE_QRELS})


# ---------------------------------------------------------------------------
# nDCG@k
# ---------------------------------------------------------------------------


def test_ndcg_mixed_grades_uses_linear_gain_and_ignores_unjudged_and_zero() -> None:
    # Ranks: x (unjudged) 1, a (2) 2, c (0) 3, b (1) 4.
    # DCG  = 2/log2(3) + 1/log2(5) = 1.26186 + 0.43068 = 1.69254
    # IDCG = 2/log2(2) + 1/log2(3) = 2.00000 + 0.63093 = 2.63093
    assert ndcg_cut(["x", "a", "c", "b"], GRADES, 10) == pytest.approx(0.6433224, abs=1e-7)


def test_ndcg_is_not_the_exponential_gain_variant() -> None:
    # Ranks: b (1), a (2). Linear: (1 + 2/log2(3)) / (2 + 1/log2(3)) = 2.26186 / 2.63093.
    # The 2**g - 1 variant would give (1 + 3/log2(3)) / (3 + 1/log2(3)) = 0.8412.
    assert ndcg_cut(["b", "a"], GRADES, 10) == pytest.approx(0.8597187, abs=1e-7)


def test_ndcg_short_list_is_defined_and_the_ideal_counts_every_positive() -> None:
    # One relevant of two retrieved at rank 1: DCG 1; IDCG 1 + 1/log2(3) = 1.63093.
    assert ndcg_cut(["a"], {"a": 1, "b": 1}, 10) == pytest.approx(0.6131472, abs=1e-7)


def test_ndcg_empty_list_and_no_positive_labels_score_zero() -> None:
    assert ndcg_cut([], GRADES, 10) == 0.0
    assert ndcg_cut(["c"], {"c": 0}, 10) == 0.0


def test_ndcg_ideal_is_truncated_at_k() -> None:
    grades = {f"d{i}": 1 for i in range(12)}
    # Ten relevant documents in the top ten is perfect, although two more exist.
    assert ndcg_cut([f"d{i}" for i in range(10)], grades, 10) == pytest.approx(1.0)


def test_ndcg_only_reads_the_first_k() -> None:
    assert ndcg_cut([f"x{i}" for i in range(10)] + ["a"], GRADES, 10) == 0.0


# ---------------------------------------------------------------------------
# R@k
# ---------------------------------------------------------------------------


def test_recall_counts_distinct_positives_and_not_grade_zero() -> None:
    # Relevant = {a, b}; c is judged 0. Top 2 = [c, a] -> 1 of 2.
    assert recall_cut(["c", "a", "b"], GRADES, 2) == 0.5
    assert recall_cut(["c", "a", "b"], GRADES, 3) == 1.0


def test_recall_with_a_short_list_treats_missing_ranks_as_nonrelevant() -> None:
    assert recall_cut(["a"], GRADES, 100) == 0.5


def test_recall_without_positive_labels_is_zero() -> None:
    assert recall_cut(["c"], {"c": 0}, 100) == 0.0


# ---------------------------------------------------------------------------
# Ranking normalisation: order, ties, duplicates, self-matches
# ---------------------------------------------------------------------------


def test_ties_break_by_document_id_descending() -> None:
    order = trec_order([RankedDoc("a", 1.0), RankedDoc("c", 0.5), RankedDoc("b", 1.0)])
    assert [d.doc_id for d in order] == ["b", "a", "c"]


def test_tie_order_changes_the_score_the_way_trec_eval_would() -> None:
    # a and b tie; b sorts first, so relevant a lands at rank 2: 1/log2(3).
    ranking = document_ranking([("a", 1.0), ("b", 1.0)])
    assert ndcg_cut(ranking.doc_ids, {"a": 1}, 10) == pytest.approx(0.6309298, abs=1e-7)


def test_document_ranking_keeps_first_occurrence_and_counts_what_it_removed() -> None:
    ranking = document_ranking([("a", 0.9), ("b", 0.8), ("a", 0.7), ("q1", 0.6)], query_id="q1", remove_query=True)
    assert ranking.doc_ids == ["a", "b"]
    assert (ranking.duplicates_removed, ranking.self_matches_removed) == (1, 1)
    assert document_ranking([("q1", 0.6)], query_id="q1").doc_ids == ["q1"]


# ---------------------------------------------------------------------------
# Run files
# ---------------------------------------------------------------------------


def test_run_file_round_trips_scores_exactly(tmp_path: Path) -> None:
    score = 0.1 + 0.2  # 0.30000000000000004: rounding it would change nothing here, but can create ties
    rankings = {"q1": DocRanking((RankedDoc("a", score), RankedDoc("b", 0.3)))}
    write_run(tmp_path / "run.trec", rankings, tag="t")

    back = read_run(tmp_path / "run.trec")
    assert back["q1"].docs == (RankedDoc("a", score), RankedDoc("b", 0.3))


def test_read_run_ignores_the_rank_column_and_resorts(tmp_path: Path) -> None:
    (tmp_path / "run.trec").write_text("q1 Q0 low 1 0.1 t\nq1 Q0 high 2 0.9 t\n", encoding="utf-8")
    assert read_run(tmp_path / "run.trec")["q1"].doc_ids == ["high", "low"]


@pytest.mark.parametrize(
    "text, message",
    [
        ("q1 Q0 a 1 0.9 t\nq1 Q0 a 2 0.8 t\n", "twice"),
        ("q1 Q0 a 1 0.9\n", "6 fields"),
        ("q1 Q0 a 1 high t\n", "not a number"),
    ],
)
def test_read_run_refuses_malformed_files(tmp_path: Path, text: str, message: str) -> None:
    (tmp_path / "run.trec").write_text(text, encoding="utf-8")
    with pytest.raises(RunFileError, match=message):
        read_run(tmp_path / "run.trec")


def test_write_run_refuses_ids_with_whitespace(tmp_path: Path) -> None:
    with pytest.raises(RunFileError, match="whitespace"):
        write_run(tmp_path / "run.trec", {"q 1": DocRanking(())}, tag="t")


# ---------------------------------------------------------------------------
# Whole-set scoring
# ---------------------------------------------------------------------------


def test_grades_include_zero_labels() -> None:
    assert qrels_grades(_sample("q", ["b", {"id": "a", "grade": 2}, {"id": "c", "grade": 0}])) == GRADES


def test_a_query_judged_only_non_relevant_counts_at_zero() -> None:
    samples = [_sample("q1", ["a"]), _sample("q2", [{"id": "c", "grade": 0}])]
    rankings = {"q1": document_ranking([("a", 1.0)]), "q2": document_ranking([("c", 1.0)])}

    assert score_rankings(samples, rankings).means["nDCG@10"] == 0.5


def test_mean_is_over_every_declared_query_including_missing_and_empty() -> None:
    samples = [_sample("q1", ["a"]), _sample("q2", ["b"]), _sample("q3", ["c"])]
    rankings = {
        "q1": document_ranking([("a", 1.0)]),  # perfect
        "q2": DocRanking(()),  # returned nothing
        # q3 missing entirely; "extra" is not a declared query and is ignored.
        "extra": document_ranking([("a", 1.0)]),
    }

    scores = score_rankings(samples, rankings)

    assert [q.values["nDCG@10"] for q in scores.per_query] == [1.0, 0.0, 0.0]
    assert scores.means["nDCG@10"] == pytest.approx(1 / 3)
    assert scores.means["R@100"] == pytest.approx(1 / 3)
    assert [q.returned for q in scores.per_query] == [1, 0, 0]


# ---------------------------------------------------------------------------
# The runner's qrels path
# ---------------------------------------------------------------------------


def _chunk(doc_id: str, score: float, n: int = 0) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=f"{doc_id}::chunk{n}", text="t", document_id=doc_id, source=Path("/c"), doc_type="beir", score=score
    )


@dataclass
class _FakeRetriever:
    """Stage 1 returns a ranked list; the final stage keeps its top two."""

    top_k: int = 100
    rerank_top_k: int = 10
    candidates: dict[str, list[ScoredChunk]] = field(default_factory=dict)

    def retrieve(self, query: str, **_: object) -> RetrievalResult:
        ranked = self.candidates[query]
        return RetrievalResult(chunks=ranked[:2], candidate_count=len(ranked), candidates=ranked)


def _dataset() -> EvalDataset:
    return EvalDataset(samples=[_sample("q1", ["a", "b"]), _sample("a", ["b"])])


def test_run_qrels_eval_scores_each_stage_on_its_own_ranking(tmp_path: Path) -> None:
    retriever = _FakeRetriever(
        candidates={
            # q1: 'a' twice (two chunks of one document), then 'x', then 'b' at stage-1 rank 3.
            "query q1": [_chunk("a", 0.9), _chunk("a", 0.8, 1), _chunk("x", 0.7), _chunk("b", 0.6)],
            # Query id 'a' equals a document id: the self-match is dropped.
            "query a": [_chunk("a", 0.9), _chunk("b", 0.5)],
        }
    )

    report = run_qrels_eval(_dataset(), retriever)  # type: ignore[arg-type]

    final, stage1 = report.scores[STAGE_FINAL], report.scores[STAGE_1]
    # Final keeps chunks [a, a] -> documents [a]: R@100 = 1/2 for q1.
    assert final.per_query[0].values["R@100"] == 0.5
    # Stage 1 reaches b: R@100 = 1 for q1.
    assert stage1.per_query[0].values["R@100"] == 1.0
    # Query 'a': 'a' removed, 'b' at rank 1 in both stages.
    assert report.rankings[STAGE_1]["a"].doc_ids == ["b"]
    assert report.self_matches_removed == {STAGE_1: 1, STAGE_FINAL: 1}
    assert report.duplicates_removed == {STAGE_1: 1, STAGE_FINAL: 1}

    save_qrels_run(report, tmp_path, tag="test", settings={"note": "fixture"})
    assert read_run(tmp_path / "stage1.trec")["q1"].doc_ids == ["a", "x", "b"]
    assert (tmp_path / "final.trec").exists() and (tmp_path / "scores.json").exists()


def test_run_qrels_eval_refuses_a_stage_1_too_shallow_for_r_at_100() -> None:
    with pytest.raises(ValueError, match="--candidate-depth"):
        run_qrels_eval(_dataset(), _FakeRetriever(top_k=20))  # type: ignore[arg-type]


def test_legacy_judging_still_refuses_qrels_samples() -> None:
    with pytest.raises(NotImplementedError):
        judge_ranking(_sample("q", ["a"]), [])
