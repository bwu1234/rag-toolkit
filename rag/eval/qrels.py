"""Standard retrieval metrics over graded qrels, scored the way ``trec_eval`` scores them.

Phase 2 of ``docs/public-benchmarks-plan.md``. The legacy modes in
:mod:`rag.eval.relevance` grade *chunks* and use exponential gains, which is
the right design for span-level EDGAR questions and cannot be compared with a
published BEIR number. This module grades **distinct documents** against the
graded labels of a ``matching_mode: "qrels"`` sample, and matches the pinned
evaluator (``trec_eval`` 9.0.4, ``-c``; see ``docs/beir-reference-protocol.md``)
on every choice that changes a number:

* **nDCG@k** -- linear gain: the qrels grade itself, discounted by
  ``log2(rank + 1)``. The ideal ranking is every positive grade the query's
  qrels hold, highest first, cut at ``k``. Unjudged and grade-0 documents
  gain nothing. A list shorter than ``k`` still has a defined score.
* **R@k** -- distinct documents graded >= 1 in the first ``k``, divided by all
  documents graded >= 1. Ranks past the end of a short list count as
  non-relevant.
* **Queries with no positive grade** score 0 and still count toward the mean,
  as ``trec_eval -c`` counts them. So does a query the run returned nothing
  for.
* **Order** -- ``trec_eval`` ignores the rank column and sorts by score
  descending, then document id descending (``form_res_rels.c``). A ranking is
  put in that order before any cutoff, so tied scores are broken the same way.
* **One rank per document** -- a document appears once, at its first
  occurrence; a run file with a repeated document is refused, as
  ``trec_eval`` refuses it. Chunk-to-document collapsing keeps the first
  occurrence and counts what it removed.
* **Self-matches** -- with ``remove_query``, a document whose id equals the
  query id is dropped, the reference ``--remove-query`` rule. It is live on
  FiQA, where query and document ids collide.

Pure functions plus TREC run-file I/O; no retrieval, no config.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import log2
from pathlib import Path

from rag.eval.dataset import DEFAULT_GRADE, MODE_QRELS, EvalSample

#: The relevance threshold ``trec_eval`` applies by default (``-l 1``).
RELEVANCE_LEVEL = 1


@dataclass(frozen=True)
class RankedDoc:
    """One document in a ranking, with the score its order derives from."""

    doc_id: str
    score: float


@dataclass(frozen=True)
class DocRanking:
    """A per-query document ranking in evaluation order, and what was removed to make it."""

    docs: tuple[RankedDoc, ...]
    #: Later occurrences of an already-ranked document (several chunks of one document).
    duplicates_removed: int = 0
    #: Documents dropped because their id equals the query id (``remove_query``).
    self_matches_removed: int = 0

    @property
    def doc_ids(self) -> list[str]:
        return [d.doc_id for d in self.docs]


def trec_order(docs: Iterable[RankedDoc]) -> list[RankedDoc]:
    """``docs`` in ``trec_eval``'s order: score descending, then document id descending.

    Two stable sorts express the mixed direction. Python compares strings by
    code point, which is the same order as ``strcmp`` on their UTF-8 bytes.
    """
    by_id = sorted(docs, key=lambda d: d.doc_id, reverse=True)
    return sorted(by_id, key=lambda d: d.score, reverse=True)


def document_ranking(
    ranked: Iterable[tuple[str, float]], *, query_id: str | None = None, remove_query: bool = False
) -> DocRanking:
    """Collapse a ranked list of ``(document_id, score)`` into an evaluation-order ranking.

    Keeps each document's first occurrence (the pipeline's best-ranked chunk
    of it) before any cutoff, drops the self-match when asked, then applies
    :func:`trec_order`.
    """
    seen: set[str] = set()
    kept: list[RankedDoc] = []
    duplicates = self_matches = 0
    for doc_id, score in ranked:
        if doc_id in seen:
            duplicates += 1
            continue
        seen.add(doc_id)
        if remove_query and doc_id == query_id:
            self_matches += 1
            continue
        kept.append(RankedDoc(doc_id, float(score)))
    return DocRanking(tuple(trec_order(kept)), duplicates, self_matches)


def qrels_grades(sample: EvalSample) -> dict[str, int]:
    """Every judged document of a qrels sample and its grade, grade-0 labels included."""
    if sample.matching_mode != MODE_QRELS:
        raise ValueError(f"Sample {sample.id!r} is not a qrels sample (matching_mode={sample.matching_mode!r})")
    grades = {doc_id: sample.doc_grades.get(doc_id, DEFAULT_GRADE) for doc_id in sample.expected_doc_ids}
    grades.update({doc_id: g for doc_id, g in sample.doc_grades.items() if g == 0})
    return grades


def ndcg_cut(ranking: Sequence[str], grades: Mapping[str, int], k: int) -> float:
    """``trec_eval``'s ``ndcg_cut.k``: linear-gain nDCG over the first ``k`` of ``ranking``."""
    dcg = sum(
        grades.get(doc_id, 0) / log2(rank + 1)
        for rank, doc_id in enumerate(ranking[:k], start=1)
        if grades.get(doc_id, 0) > 0
    )
    ideal = sorted((g for g in grades.values() if g > 0), reverse=True)[:k]
    ideal_dcg = sum(g / log2(rank + 1) for rank, g in enumerate(ideal, start=1))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def recall_cut(ranking: Sequence[str], grades: Mapping[str, int], k: int) -> float:
    """``trec_eval``'s ``recall.k``: share of relevant documents in the first ``k`` of ``ranking``."""
    relevant = {doc_id for doc_id, g in grades.items() if g >= RELEVANCE_LEVEL}
    if not relevant:
        return 0.0
    return len(relevant.intersection(ranking[:k])) / len(relevant)


# ---------------------------------------------------------------------------
# TREC run files
# ---------------------------------------------------------------------------


class RunFileError(ValueError):
    """A run file is malformed or ranks a document twice for one query."""


def write_run(path: Path, rankings: Mapping[str, DocRanking], *, tag: str) -> None:
    """Write ``rankings`` as a TREC run file: ``qid Q0 docid rank score tag``.

    Scores are written with ``repr`` so they read back as the identical float:
    rounding them could create ties that change ``trec_eval``'s order.
    """
    for token, what in [(tag, "tag"), *((qid, "query id") for qid in rankings)]:
        if not token or any(ch.isspace() for ch in token):
            raise RunFileError(f"{what} {token!r} cannot be written to a whitespace-separated run file")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for qid, ranking in rankings.items():
            for rank, doc in enumerate(ranking.docs, start=1):
                if any(ch.isspace() for ch in doc.doc_id):
                    raise RunFileError(f"document id {doc.doc_id!r} contains whitespace")
                f.write(f"{qid} Q0 {doc.doc_id} {rank} {doc.score!r} {tag}\n")


def read_run(path: Path) -> dict[str, DocRanking]:
    """Read a TREC run file into evaluation-order rankings.

    The rank column is ignored, as ``trec_eval`` ignores it. A document ranked
    twice for one query is an error rather than silently deduplicated.
    """
    ranked: dict[str, list[RankedDoc]] = {}
    seen: dict[str, set[str]] = {}
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, start=1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) != 6:
                raise RunFileError(f"{path.name}:{n}: expected 6 fields, got {len(fields)}")
            qid, _q0, doc_id, _rank, raw_score, _tag = fields
            try:
                score = float(raw_score)
            except ValueError:
                raise RunFileError(f"{path.name}:{n}: score {raw_score!r} is not a number") from None
            if doc_id in seen.setdefault(qid, set()):
                raise RunFileError(f"{path.name}:{n}: query {qid!r} ranks document {doc_id!r} twice")
            seen[qid].add(doc_id)
            ranked.setdefault(qid, []).append(RankedDoc(doc_id, score))
    return {qid: DocRanking(tuple(trec_order(docs))) for qid, docs in ranked.items()}


# ---------------------------------------------------------------------------
# Scoring a whole query set
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QueryScore:
    """One query's unrounded scores, and how long each ranking actually was."""

    query_id: str
    values: dict[str, float]
    returned: int


@dataclass(frozen=True)
class QrelsScores:
    """Per-query and mean values of each metric over the full declared query set."""

    per_query: list[QueryScore]
    means: dict[str, float]


#: Metric name -> (function, cutoff). Names follow the published tables.
METRICS = {"nDCG@10": (ndcg_cut, 10), "R@100": (recall_cut, 100)}


def score_rankings(samples: Sequence[EvalSample], rankings: Mapping[str, DocRanking]) -> QrelsScores:
    """Score ``rankings`` (query id -> ranking) against every qrels sample in ``samples``.

    The mean is over *all* samples: a query the run never returned scores 0,
    as ``trec_eval -c`` scores it. Rankings for queries not in ``samples`` are
    ignored, as ``trec_eval`` ignores them.
    """
    per_query: list[QueryScore] = []
    for sample in samples:
        grades = qrels_grades(sample)
        ranking = rankings.get(sample.id, DocRanking(()))
        ids = ranking.doc_ids
        per_query.append(
            QueryScore(
                query_id=sample.id,
                values={name: fn(ids, grades, k) for name, (fn, k) in METRICS.items()},
                returned=len(ids),
            )
        )
    means = {
        name: (sum(q.values[name] for q in per_query) / len(per_query) if per_query else 0.0)
        for name in METRICS
    }
    return QrelsScores(per_query=per_query, means=means)
