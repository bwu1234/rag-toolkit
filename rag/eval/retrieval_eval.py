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
* **Hit rate** — fraction of queries where at least one expected document
  was retrieved (equivalent to Recall@k for single-expected-doc queries).
* **Recall@k** — mean fraction of expected documents covered per query.
* **Precision@k** — mean fraction of retrieved results that were relevant.
* **MRR** — mean reciprocal rank of the first relevant result.

All metrics are computed over whatever ``retrieval.rerank_top_k`` results the
Retriever returns (i.e. after any configured reranking pass).
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from rag.config.settings import RagConfig, load_config
from rag.eval.dataset import EvalDataset
from rag.eval.metrics import hit_rate, mean, precision_at_k, recall_at_k, reciprocal_rank
from rag.logging_config import configure_logging
from rag.retrieval.builder import build_retriever
from rag.retrieval.retriever import Retriever

logger = logging.getLogger(__name__)

# Default eval set path relative to repo root
_DEFAULT_EVAL_SET = Path(__file__).resolve().parents[2] / "data" / "eval" / "eval_set.json"


@dataclass
class SampleResult:
    """Retrieval result for a single eval sample."""

    sample_id: str
    query: str
    expected_doc_ids: list[str]
    retrieved_doc_ids: list[str]
    hit: float
    recall: float
    precision: float
    rr: float  # reciprocal rank


@dataclass
class EvalReport:
    """Aggregate retrieval eval results."""

    num_samples: int
    mean_hit_rate: float
    mean_recall: float
    mean_precision: float
    mrr: float
    sample_results: list[SampleResult]


def run_retrieval_eval(dataset: EvalDataset, retriever: Retriever) -> EvalReport:
    """Run retrieval for every sample and return an :class:`EvalReport`."""
    results: list[SampleResult] = []

    for sample in dataset:
        chunks = retriever.retrieve(sample.query)
        retrieved_doc_ids = [c.document_id for c in chunks]

        results.append(
            SampleResult(
                sample_id=sample.id,
                query=sample.query,
                expected_doc_ids=sample.expected_doc_ids,
                retrieved_doc_ids=retrieved_doc_ids,
                hit=hit_rate(retrieved_doc_ids, sample.expected_doc_ids),
                recall=recall_at_k(retrieved_doc_ids, sample.expected_doc_ids),
                precision=precision_at_k(retrieved_doc_ids, sample.expected_doc_ids),
                rr=reciprocal_rank(retrieved_doc_ids, sample.expected_doc_ids),
            )
        )

    return EvalReport(
        num_samples=len(results),
        mean_hit_rate=mean([r.hit for r in results]),
        mean_recall=mean([r.recall for r in results]),
        mean_precision=mean([r.precision for r in results]),
        mrr=mean([r.rr for r in results]),
        sample_results=results,
    )


def print_report(report: EvalReport, *, verbose: bool = False) -> None:
    """Print a formatted eval report to stdout."""
    print(f"\n{'=' * 60}")
    print(f"  Retrieval Eval  ({report.num_samples} sample(s))")
    print(f"{'=' * 60}")
    print(f"  Hit rate    {report.mean_hit_rate:.3f}")
    print(f"  Recall@k    {report.mean_recall:.3f}")
    print(f"  Precision@k {report.mean_precision:.3f}")
    print(f"  MRR         {report.mrr:.3f}")
    print(f"{'=' * 60}")

    if verbose:
        print()
        for r in report.sample_results:
            status = "HIT " if r.hit else "MISS"
            print(f"[{status}] {r.sample_id!r}  rr={r.rr:.3f}")
            print(f"  query:    {r.query!r}")
            print(f"  expected: {r.expected_doc_ids}")
            retrieved_preview = r.retrieved_doc_ids[:5]
            suffix = f" ... (+{len(r.retrieved_doc_ids) - 5} more)" if len(r.retrieved_doc_ids) > 5 else ""
            print(f"  retrieved: {retrieved_preview}{suffix}")
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
    logger.info(
        "Building retriever (embedding=%s, vector_store=%s, reranker=%s)",
        config.embedding.model,
        config.vector_store.provider,
        config.reranker.provider,
    )
    retriever = build_retriever(config)

    logger.info("Running retrieval eval on %d sample(s)", len(dataset))
    report = run_retrieval_eval(dataset, retriever)
    print_report(report, verbose=args.verbose)

    # Exit 1 if every single query was a miss (likely an empty index)
    if report.mean_hit_rate == 0.0 and report.num_samples > 0:
        logger.warning("Hit rate is 0.0 — is the vector index built? Run `python -m rag.cli index` first.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
