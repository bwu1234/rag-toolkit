"""Retrieval evaluation runner.

Loads an eval set, runs the configured Retriever against each query, and
reports per-sample and aggregate retrieval metrics.

Usage::

    python -m rag.eval.retrieval_eval                          # uses default eval set
    python -m rag.eval.retrieval_eval --eval-set path/to/set.json
    python -m rag.eval.retrieval_eval --config path/to/config.yaml

The script exits with code 1 if the eval set is empty or the index is empty
for every query, so it can be used in CI to catch a broken retriever.

Metrics reported
----------------
* **Hit rate** — fraction of queries that retrieved anything relevant.
* **Recall@k** — mean fraction of expected items covered, reported as a curve
  over several k so the candidate-set ceiling is visible.  If recall is flat
  from k=5 to k=20, a bigger ``rerank_top_k`` buys nothing and the loss is
  upstream in retrieval; if it climbs, the reranker is discarding good results.
* **Precision@k** — mean fraction of retrieved results that were relevant.
* **MRR** — mean reciprocal rank of the first relevant result.
* **NDCG@k** — rank- and grade-weighted quality; the only reported metric that
  distinguishes "found it at rank 1" from "found it at rank 5", and the only
  one that reads span grades.

Results are grouped by matching mode (span, span_and_document, document) when a
set mixes them, because they are not comparable — see :mod:`rag.eval.relevance`.
They are also grouped by the samples' ``kind`` field when a set has one (the
``underspecified`` tier's ``implicit`` and ``paraphrase``), because the chunking
plan reports each kind on its own rather than averaged together. Every hit rate
carries a 95% Wilson interval: that tier's noise floor on its own.

Every run also reports **unmatchable spans**: expected spans that no chunk the
configured chunker makes of the corpus contains.  Those samples score as misses
whatever retrieval does, so a chunker that cuts answers in two is visibly
penalized rather than silently scoring lower.

All metrics are computed over whatever ``retrieval.rerank_top_k`` results the
Retriever returns (i.e. after any configured reranking pass).

Qrels sets (``matching_mode: "qrels"``, converted from public benchmarks by
``scripts/beir_to_eval_set.py``) take a separate path, :func:`run_qrels_eval`:
distinct-document nDCG@10 and R@100 scored the ``trec_eval`` way
(:mod:`rag.eval.qrels`), each on a named stage. R@100 reads the stage-1
ranking, which is why it needs ``--candidate-depth`` of at least 100; nDCG@10
reads the final ranking. ``--save-run DIR`` writes both rankings as TREC run
files plus the unrounded per-query scores, so an external evaluator can score
the identical rankings.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from rag.chunking.models import Chunk
from rag.config.settings import RagConfig, load_config
from rag.eval.dataset import MODE_DOCUMENT, MODE_QRELS, EvalDataset, EvalSample
from rag.eval.metrics import (
    hit_rate,
    mean,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    wilson_interval,
)
from rag.eval.qrels import METRICS, DocRanking, QrelsScores, document_ranking, score_rankings, write_run
from rag.eval.relevance import UnmatchableSpan, find_unmatchable_spans, judge_ranking
from rag.events import PipelineEvent
from rag.ingestion.corpora import chunk_selected_corpora
from rag.logging_config import configure_logging
from rag.query_filter import QueryFilter
from rag.retrieval.builder import build_retriever
from rag.retrieval.retriever import Retriever

logger = logging.getLogger(__name__)

# Default eval set path relative to repo root
_DEFAULT_EVAL_SET = Path(__file__).resolve().parents[2] / "data" / "eval" / "eval_set.json"

#: Cutoffs for the recall curve. Values above the number of results the
#: retriever actually returns are dropped rather than reported as a flat line,
#: which would read as a finding rather than an artifact of ``rerank_top_k``.
RECALL_K_VALUES = (1, 3, 5, 10, 20)


@dataclass
class SampleResult:
    """Retrieval result for a single eval sample."""

    sample_id: str
    query: str
    mode: str
    expected_doc_ids: list[str]
    retrieved_doc_ids: list[str]
    hit: float
    recall: float
    precision: float
    rr: float  # reciprocal rank
    ndcg: float
    recall_by_k: dict[int, float] = field(default_factory=dict)
    #: Spans the ranking never surfaced — the actionable detail on a miss.
    unmatched_spans: list[str] = field(default_factory=list)
    #: The sample's ``kind`` field, when the set sorts its questions into kinds.
    kind: str | None = None
    #: Documents ``retrieval.document_routing`` filtered this sample to; empty if it didn't route.
    routed_to: list[str] = field(default_factory=list)


@dataclass
class MetricSummary:
    """Aggregate metrics over a set of samples."""

    label: str
    num_samples: int
    mean_hit_rate: float
    mean_recall: float
    mean_precision: float
    mrr: float
    mean_ndcg: float
    recall_by_k: dict[int, float] = field(default_factory=dict)
    #: 95% Wilson interval on the hit rate: the noise floor of this one rate.
    hit_ci: tuple[float, float] = (0.0, 1.0)


@dataclass
class EvalReport:
    """Aggregate retrieval eval results."""

    overall: MetricSummary
    sample_results: list[SampleResult]
    #: Per-matching-mode breakdown, populated only when a set mixes modes.
    by_mode: list[MetricSummary] = field(default_factory=list)
    #: Per-``kind`` breakdown, populated only when the set's samples carry one.
    by_kind: list[MetricSummary] = field(default_factory=list)
    #: Spans no corpus chunk contains, or None when the corpus wasn't checked.
    unmatchable_spans: list[UnmatchableSpan] | None = None

    @property
    def num_samples(self) -> int:
        return self.overall.num_samples

    @property
    def mean_hit_rate(self) -> float:
        return self.overall.mean_hit_rate


def _summarize(label: str, results: list[SampleResult]) -> MetricSummary:
    k_values = sorted({k for r in results for k in r.recall_by_k})
    return MetricSummary(
        label=label,
        num_samples=len(results),
        mean_hit_rate=mean([r.hit for r in results]),
        mean_recall=mean([r.recall for r in results]),
        mean_precision=mean([r.precision for r in results]),
        mrr=mean([r.rr for r in results]),
        mean_ndcg=mean([r.ndcg for r in results]),
        recall_by_k={
            k: mean([r.recall_by_k[k] for r in results if k in r.recall_by_k])
            for k in k_values
        },
        hit_ci=wilson_interval(sum(1 for r in results if r.hit), len(results)),
    )


def run_retrieval_eval(
    dataset: EvalDataset,
    retriever: Retriever,
    *,
    corpus_chunks: list[Chunk] | None = None,
    filters_for: Callable[[EvalSample], QueryFilter | None] | None = None,
) -> EvalReport:
    """Run retrieval for every sample and return an :class:`EvalReport`.

    Pass ``corpus_chunks`` -- what the configured chunker makes of the evaluated
    corpus -- to have the report count spans no chunk contains.

    Pass ``filters_for`` to retrieve each sample with a metadata filter, e.g.
    one derived from its labels to measure what a caller who names the
    company or period would get (see ``scripts/run_matrix.py``).
    """
    results: list[SampleResult] = []

    for sample in dataset:
        query_filter = filters_for(sample) if filters_for is not None else None
        retrieved = retriever.retrieve(sample.query, query_filter=query_filter)
        chunks = retrieved.chunks
        judgment = judge_ranking(sample, chunks)

        # Recall at each cutoff is computed by re-judging a prefix of the
        # ranking rather than by slicing gains: one expected item can be matched
        # by several chunks, so coverage is not recoverable from the gain list.
        recall_by_k = {
            k: recall_at_k(*_coverage(sample, chunks[:k]))
            for k in RECALL_K_VALUES
            if k <= len(chunks)
        }

        results.append(
            SampleResult(
                sample_id=sample.id,
                query=sample.query,
                mode=judgment.mode,
                expected_doc_ids=sample.expected_doc_ids,
                retrieved_doc_ids=[c.document_id for c in chunks],
                hit=hit_rate(judgment.gains),
                recall=recall_at_k(judgment.covered, judgment.total_expected),
                precision=precision_at_k(judgment.gains),
                rr=reciprocal_rank(judgment.gains),
                ndcg=ndcg_at_k(judgment.ndcg_gains, judgment.ideal_gains),
                recall_by_k=recall_by_k,
                unmatched_spans=judgment.unmatched_spans,
                kind=sample.extra.get("kind"),
                routed_to=retrieved.routed_to,
            )
        )

    modes = sorted({r.mode for r in results})
    by_mode = (
        [_summarize(m, [r for r in results if r.mode == m]) for m in modes]
        if len(modes) > 1
        else []
    )
    kinds = sorted({r.kind for r in results if r.kind is not None})
    by_kind = [_summarize(k, [r for r in results if r.kind == k]) for k in kinds]
    if kinds and any(r.kind is None for r in results):
        by_kind.append(_summarize("(no kind)", [r for r in results if r.kind is None]))
    label = modes[0] if len(modes) == 1 else "all samples"
    return EvalReport(
        overall=_summarize(label, results),
        sample_results=results,
        by_mode=by_mode,
        by_kind=by_kind,
        unmatchable_spans=(
            find_unmatchable_spans(dataset, corpus_chunks) if corpus_chunks is not None else None
        ),
    )


#: The stages a qrels run saves, and the metric each is read for.
STAGE_1 = "stage1"
STAGE_FINAL = "final"
HEADLINE = {STAGE_FINAL: "nDCG@10", STAGE_1: "R@100"}


@dataclass
class QrelsReport:
    """A qrels run: both stages' rankings and scores, and the settings that shaped them."""

    candidate_depth: int
    final_depth: int
    remove_query: bool
    rankings: dict[str, dict[str, DocRanking]]
    scores: dict[str, QrelsScores]
    #: Chunks collapsed into an already-ranked document, per stage.
    duplicates_removed: dict[str, int] = field(default_factory=dict)
    #: Documents dropped as self-matches (id == query id), per stage.
    self_matches_removed: dict[str, int] = field(default_factory=dict)
    #: Per query id, milliseconds per retrieval stage (``embed``, ``vector_search``,
    #: ``sparse_search``, ``fusion``, ``rerank``) plus ``total``, the whole call.
    latency_ms: dict[str, dict[str, float]] = field(default_factory=dict)

    @property
    def num_samples(self) -> int:
        return len(self.scores[STAGE_FINAL].per_query)

    def latency_summary(self) -> dict[str, dict[str, float]]:
        """Median and 95th percentile per stage, over the queries that ran it.

        The first query carries any lazy model load, so the mean would mislead;
        the median doesn't see it.
        """
        stages = sorted({s for per_query in self.latency_ms.values() for s in per_query})
        summary: dict[str, dict[str, float]] = {}
        for stage in stages:
            values = sorted(q[stage] for q in self.latency_ms.values() if stage in q)
            p95 = statistics.quantiles(values, n=20)[-1] if len(values) > 1 else values[0]
            summary[stage] = {"p50": statistics.median(values), "p95": p95}
        return summary


def check_qrels_depths(candidate_depth: int, final_depth: int) -> None:
    """Refuse depths at which the headline metrics would be silently capped.

    Raises:
        ValueError: stage 1 keeps fewer than 100 results (R@100) or the final
            ranking fewer than 10 (nDCG@10).
    """
    if candidate_depth < 100:
        raise ValueError(
            f"R@100 needs a stage-1 ranking of at least 100; retrieval.top_k is {candidate_depth}. "
            "Pass --candidate-depth 100 or more."
        )
    if final_depth < 10:
        raise ValueError(
            f"nDCG@10 needs a final ranking of at least 10; retrieval.rerank_top_k is {final_depth}. "
            "Pass --final-depth 10 or more."
        )


def run_qrels_eval(dataset: EvalDataset, retriever: Retriever, *, remove_query: bool = True) -> QrelsReport:
    """Retrieve every qrels sample and score both stages against its graded labels.

    ``remove_query`` applies the reference ``--remove-query`` rule (drop a
    document whose id equals the query id) to both stages. It is on by
    default so every variant is scored under the protocol the published
    numbers used.
    """
    samples = list(dataset)
    if any(s.matching_mode != MODE_QRELS for s in samples):
        raise ValueError("A qrels run needs every sample to use matching_mode 'qrels'; this set mixes modes.")
    check_qrels_depths(retriever.top_k, retriever.rerank_top_k)

    rankings: dict[str, dict[str, DocRanking]] = {STAGE_1: {}, STAGE_FINAL: {}}
    latency_ms: dict[str, dict[str, float]] = {}
    for sample in samples:
        stages: dict[str, float] = {}

        def record(event: PipelineEvent, stages: dict[str, float] = stages) -> None:
            stages[event.stage] = stages.get(event.stage, 0.0) + (event.elapsed_ms or 0.0)

        start = time.perf_counter()
        retrieved = retriever.retrieve(sample.query, on_event=record)
        stages["total"] = (time.perf_counter() - start) * 1000
        latency_ms[sample.id] = stages
        for stage, chunks in ((STAGE_1, retrieved.candidates), (STAGE_FINAL, retrieved.chunks)):
            rankings[stage][sample.id] = document_ranking(
                ((c.document_id, c.score) for c in chunks), query_id=sample.id, remove_query=remove_query
            )

    return QrelsReport(
        candidate_depth=retriever.top_k,
        final_depth=retriever.rerank_top_k,
        remove_query=remove_query,
        rankings=rankings,
        scores={stage: score_rankings(samples, ranks) for stage, ranks in rankings.items()},
        duplicates_removed={s: sum(r.duplicates_removed for r in ranks.values()) for s, ranks in rankings.items()},
        self_matches_removed={
            s: sum(r.self_matches_removed for r in ranks.values()) for s, ranks in rankings.items()
        },
        latency_ms=latency_ms,
    )


def print_qrels_report(report: QrelsReport) -> None:
    """Print both stages' metrics, marking the one each stage is reported for."""
    print(f"\n{'=' * 62}")
    print(f"  Qrels Retrieval Eval  ({report.num_samples} queries, trec_eval -c semantics)")
    print(f"{'=' * 62}")
    print(
        f"  Depths: stage 1 = {report.candidate_depth}, final = {report.final_depth};  "
        f"remove-query {'on' if report.remove_query else 'off'}"
    )
    for stage, label in ((STAGE_FINAL, "final"), (STAGE_1, "stage 1")):
        cells = []
        for name in METRICS:
            value = report.scores[stage].means[name]
            cells.append(f"{name} {value:.4f}{'*' if HEADLINE[stage] == name else ' '}")
        print(f"  {label:<8} " + "   ".join(cells))
    print("  * the reported figure: nDCG@10 on the final ranking, R@100 on stage 1")
    for stage in (STAGE_1, STAGE_FINAL):
        short = sum(1 for q in report.scores[stage].per_query if q.returned < (100 if stage == STAGE_1 else 10))
        print(
            f"  {stage:<8} {report.duplicates_removed[stage]} duplicate chunk(s) collapsed, "
            f"{report.self_matches_removed[stage]} self-match(es) removed, {short} short list(s)"
        )
    if report.latency_ms:
        cells = [f"{stage} {v['p50']:.0f}/{v['p95']:.0f}" for stage, v in report.latency_summary().items()]
        print("  ms p50/p95  " + "  ".join(cells))
    print(f"{'=' * 62}")


def save_qrels_run(report: QrelsReport, out_dir: Path, *, tag: str, settings: dict) -> None:
    """Write ``stage1.trec``, ``final.trec`` and ``scores.json`` (unrounded) to ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for stage, ranks in report.rankings.items():
        write_run(out_dir / f"{stage}.trec", ranks, tag=tag)
    payload = {
        "settings": {
            **settings,
            "candidate_depth": report.candidate_depth,
            "final_depth": report.final_depth,
            "remove_query": report.remove_query,
        },
        "means": {stage: scores.means for stage, scores in report.scores.items()},
        "duplicates_removed": report.duplicates_removed,
        "self_matches_removed": report.self_matches_removed,
        "latency_ms": report.latency_summary(),
        "latency_ms_per_query": report.latency_ms,
        "per_query": {
            stage: {q.query_id: {**q.values, "returned": q.returned} for q in scores.per_query}
            for stage, scores in report.scores.items()
        },
    }
    (out_dir / "scores.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _coverage(sample, chunks) -> tuple[int, int]:
    judgment = judge_ranking(sample, chunks)
    return judgment.covered, judgment.total_expected


def _print_summary(summary: MetricSummary, *, indent: str = "  ") -> None:
    low, high = summary.hit_ci
    print(f"{indent}Hit rate    {summary.mean_hit_rate:.3f}  (95% CI {low:.3f}-{high:.3f})")
    print(f"{indent}Recall@k    {summary.mean_recall:.3f}")
    print(f"{indent}Precision@k {summary.mean_precision:.3f}")
    print(f"{indent}MRR         {summary.mrr:.3f}")
    print(f"{indent}NDCG@k      {summary.mean_ndcg:.3f}")
    if summary.recall_by_k:
        curve = "  ".join(f"@{k}={v:.3f}" for k, v in sorted(summary.recall_by_k.items()))
        print(f"{indent}Recall curve {curve}")


def print_report(report: EvalReport, *, verbose: bool = False) -> None:
    """Print a formatted eval report to stdout."""
    print(f"\n{'=' * 62}")
    print(f"  Retrieval Eval  ({report.num_samples} sample(s), {report.overall.label})")
    print(f"{'=' * 62}")
    _print_summary(report.overall)

    if report.by_mode:
        print(f"{'-' * 62}")
        print("  Samples with different matching modes are graded differently")
        print("  and are NOT comparable; the combined figures above are only a")
        print("  rough indicator. Read the per-mode breakdown instead:")
        for summary in report.by_mode:
            print(f"\n  [{summary.label}]  ({summary.num_samples} sample(s))")
            _print_summary(summary, indent="    ")
    if report.by_kind:
        print(f"{'-' * 62}")
        print("  By kind (reported separately; the combined figures above mix them):")
        for summary in report.by_kind:
            print(f"\n  [{summary.label}]  ({summary.num_samples} sample(s))")
            _print_summary(summary, indent="    ")
    if report.unmatchable_spans is not None:
        print(f"{'-' * 62}")
        affected = len({u.sample_id for u in report.unmatchable_spans})
        print(
            f"  Unmatchable spans  {len(report.unmatchable_spans)} "
            f"({affected} sample(s) that can't score, whatever retrieval does)"
        )
    print(f"{'=' * 62}")

    unmatchable_by_sample: dict[str, list[str]] = {}
    for u in report.unmatchable_spans or []:
        unmatchable_by_sample.setdefault(u.sample_id, []).append(u.span)

    if verbose:
        print()
        for r in report.sample_results:
            status = "HIT " if r.hit else "MISS"
            print(f"[{status}] {r.sample_id!r}  rr={r.rr:.3f} ndcg={r.ndcg:.3f} ({r.mode})")
            print(f"  query:    {r.query!r}")
            print(f"  expected: {r.expected_doc_ids}")
            retrieved_preview = r.retrieved_doc_ids[:5]
            suffix = (
                f" ... (+{len(r.retrieved_doc_ids) - 5} more)"
                if len(r.retrieved_doc_ids) > 5
                else ""
            )
            print(f"  retrieved: {retrieved_preview}{suffix}")
            unmatchable = set(unmatchable_by_sample.get(r.sample_id, []))
            for span in r.unmatched_spans:
                preview = span if len(span) <= 80 else f"{span[:77]}..."
                tag = "UNMATCHABLE" if span in unmatchable else "NOT FOUND"
                print(f"  {tag}: {preview!r}")
            print()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rag.eval.retrieval_eval",
        description="Evaluate retrieval quality against a labelled eval set.",
    )
    parser.add_argument(
        "--eval-set",
        default=None,
        metavar="PATH",
        help=f"Path to eval set JSON (default: {_DEFAULT_EVAL_SET})",
    )
    parser.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help="Path to config YAML (default: rag/config/config.yaml)",
    )
    parser.add_argument(
        "--corpus", action="append", default=None, metavar="NAME",
        help=(
            "Corpus to evaluate against, overriding corpora.active. Repeat to target "
            "a pooled index (--corpus baseline --corpus edgar); the isolated-vs-pooled "
            "difference is the cross-corpus interference cost."
        ),
    )
    parser.add_argument(
        "--candidate-depth", type=int, default=None, metavar="N",
        help="Stage-1 candidates per retriever, overriding retrieval.top_k (qrels R@100 needs >= 100).",
    )
    parser.add_argument(
        "--final-depth", type=int, default=None, metavar="N",
        help="Results kept after reranking, overriding retrieval.rerank_top_k.",
    )
    parser.add_argument(
        "--save-run", type=Path, default=None, metavar="DIR",
        help="Qrels sets: write stage1.trec, final.trec and unrounded scores.json to DIR.",
    )
    parser.add_argument(
        "--keep-self-matches", action="store_true",
        help="Qrels sets: do not drop documents whose id equals the query id (the reference drops them).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print per-sample results in addition to aggregate metrics",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = _build_parser().parse_args(argv)

    eval_path = Path(args.eval_set) if args.eval_set else _DEFAULT_EVAL_SET
    if not eval_path.exists():
        logger.error("Eval set not found: %s", eval_path)
        return 1

    logger.info("Loading eval set from %s", eval_path)
    dataset = EvalDataset.load(eval_path)
    if not dataset:
        logger.error("Eval set is empty: %s", eval_path)
        return 1

    config: RagConfig = load_config(args.config)
    if args.candidate_depth is not None:
        config.retrieval.top_k = args.candidate_depth
    if args.final_depth is not None:
        config.retrieval.rerank_top_k = args.final_depth

    modes = {s.matching_mode for s in dataset}
    if MODE_QRELS in modes:
        if len(modes) > 1:
            logger.error("Eval set mixes qrels samples with other modes; they cannot be scored together.")
            return 1
        try:
            check_qrels_depths(config.retrieval.top_k, config.retrieval.rerank_top_k)
        except ValueError as exc:
            logger.error("%s", exc)
            return 1
        retriever = build_retriever(config, corpora=args.corpus)
        qrels_report = run_qrels_eval(dataset, retriever, remove_query=not args.keep_self_matches)
        print_qrels_report(qrels_report)
        if args.save_run is not None:
            save_qrels_run(
                qrels_report,
                args.save_run,
                tag=f"rag-{config.retrieval.mode}",
                settings={
                    "eval_set": str(eval_path),
                    "config": args.config,
                    "corpus": args.corpus,
                    "retrieval_mode": config.retrieval.mode,
                    "embedding": f"{config.embedding.provider}:{config.embedding.model}",
                    "reranker": config.reranker.provider,
                },
            )
            logger.info("Saved rankings and scores to %s", args.save_run)
        return 0

    # Chunked here rather than read back from the index so the count reflects
    # the configured chunker even when the index is stale; `rag.cli
    # index-report` says whether the two agree.
    _selection, _documents, corpus_chunks = chunk_selected_corpora(config, args.corpus)

    span_samples = sum(1 for s in dataset if s.matching_mode != MODE_DOCUMENT)
    logger.info(
        "Building retriever (embedding=%s, vector_store=%s, reranker=%s)",
        config.embedding.model,
        config.vector_store.provider,
        config.reranker.provider,
    )
    retriever = build_retriever(config, corpora=args.corpus)

    logger.info(
        "Running retrieval eval on %d sample(s) (%d span-matched, %d document-matched)",
        len(dataset),
        span_samples,
        len(dataset) - span_samples,
    )
    report = run_retrieval_eval(dataset, retriever, corpus_chunks=corpus_chunks)
    print_report(report, verbose=args.verbose)

    # Exit 1 if every single query was a miss (likely an empty index)
    if report.mean_hit_rate == 0.0 and report.num_samples > 0:
        logger.warning("Hit rate is 0.0 — is the vector index built? Run `python -m rag.cli index` first.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
