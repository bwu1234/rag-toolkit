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
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

from rag.chunking.models import Chunk
from rag.config.settings import RagConfig, load_config
from rag.eval.dataset import MODE_DOCUMENT, EvalDataset
from rag.eval.metrics import (
    hit_rate,
    mean,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    wilson_interval,
)
from rag.eval.relevance import UnmatchableSpan, find_unmatchable_spans, judge_ranking
from rag.ingestion.corpora import chunk_selected_corpora
from rag.logging_config import configure_logging
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
) -> EvalReport:
    """Run retrieval for every sample and return an :class:`EvalReport`.

    Pass ``corpus_chunks`` -- what the configured chunker makes of the evaluated
    corpus -- to have the report count spans no chunk contains.
    """
    results: list[SampleResult] = []

    for sample in dataset:
        chunks = retriever.retrieve(sample.query).chunks
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
