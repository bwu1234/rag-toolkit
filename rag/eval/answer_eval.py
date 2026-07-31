"""Answer evaluation runner (LLM-as-judge).

Loads an eval set, runs the full retrieve→rerank→generate pipeline via
:class:`~rag.generation.chat_service.ChatService` for each query, then asks
the configured LLM to judge each answer against the sample's
``expected_answer``.

Usage::

    python -m rag.eval.answer_eval                             # uses default eval set
    python -m rag.eval.answer_eval --eval-set path/to/set.json
    python -m rag.eval.answer_eval --config path/to/config.yaml

The judge prompt is deliberately minimal: it asks the LLM to output exactly
``PASS`` or ``FAIL`` (optionally followed by a brief reason on the same line)
so the result is easy to parse without fragile regex.  The raw judge output is
preserved in :attr:`AnswerSampleResult.judge_output` for debugging.

Samples without an ``expected_answer`` are skipped and counted as ``n/a`` --
they can participate in retrieval eval but not answer eval.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from rag.config.settings import load_config
from rag.eval.dataset import EvalDataset
from rag.generation.builder import build_chat_service
from rag.generation.chat_service import ChatService
from rag.generation.llm import LLMClient
from rag.logging_config import configure_logging

logger = logging.getLogger(__name__)

_DEFAULT_EVAL_SET = Path(__file__).resolve().parents[2] / "data" / "eval" / "eval_set.json"

_JUDGE_SYSTEM_PROMPT = (
    "You are a strict but fair answer-quality judge. "
    "You will be given a question, a reference answer, and a system answer. "
    "Output exactly one word on the first line: PASS if the system answer is "
    "factually consistent with and substantially covers the reference answer, "
    "or FAIL if it is wrong, incomplete, or refuses to answer. "
    "You may add a brief reason after the verdict on the same line, separated by a space."
)


def _judge_prompt(query: str, expected: str, actual: str) -> str:
    return (
        f"Question: {query}\n\n"
        f"Reference answer: {expected}\n\n"
        f"System answer: {actual}\n\n"
        "Verdict (PASS or FAIL, optionally followed by a brief reason):"
    )


def _parse_verdict(judge_output: str) -> bool | None:
    """Return True for PASS, False for FAIL, None if unparseable."""
    first_word = judge_output.strip().split()[0].upper() if judge_output.strip() else ""
    if first_word == "PASS":
        return True
    if first_word == "FAIL":
        return False
    return None


@dataclass
class AnswerSampleResult:
    """Answer eval result for a single sample."""

    sample_id: str
    query: str
    expected_answer: str
    actual_answer: str
    judge_output: str
    passed: bool | None  # None = unparseable verdict
    num_citations: int


@dataclass
class AnswerEvalReport:
    """Aggregate answer eval results."""

    num_evaluated: int       # samples with expected_answer
    num_skipped: int         # samples without expected_answer
    num_passed: int
    num_failed: int
    num_unparseable: int
    pass_rate: float         # num_passed / num_evaluated (0.0 if 0 evaluated)
    sample_results: list[AnswerSampleResult]


def run_answer_eval(
    dataset: EvalDataset,
    chat_service: ChatService,
    llm_client: LLMClient,
) -> AnswerEvalReport:
    """Run the full pipeline + LLM judge for every sample with an expected answer."""
    results: list[AnswerSampleResult] = []
    skipped = 0

    for sample in dataset:
        if not sample.expected_answer:
            skipped += 1
            continue

        chat_answer = chat_service.ask(sample.query)
        judge_out = llm_client.generate(
            _judge_prompt(sample.query, sample.expected_answer, chat_answer.answer),
            system=_JUDGE_SYSTEM_PROMPT,
        )
        verdict = _parse_verdict(judge_out)

        results.append(
            AnswerSampleResult(
                sample_id=sample.id,
                query=sample.query,
                expected_answer=sample.expected_answer,
                actual_answer=chat_answer.answer,
                judge_output=judge_out.strip(),
                passed=verdict,
                num_citations=len(chat_answer.citations),
            )
        )

    n = len(results)
    passed = sum(1 for r in results if r.passed is True)
    failed = sum(1 for r in results if r.passed is False)
    unparseable = sum(1 for r in results if r.passed is None)

    return AnswerEvalReport(
        num_evaluated=n,
        num_skipped=skipped,
        num_passed=passed,
        num_failed=failed,
        num_unparseable=unparseable,
        pass_rate=passed / n if n > 0 else 0.0,
        sample_results=results,
    )


def print_report(report: AnswerEvalReport, *, verbose: bool = False) -> None:
    """Print a formatted answer eval report to stdout."""
    print(f"\n{'=' * 60}")
    print(f"  Answer Eval  ({report.num_evaluated} evaluated, {report.num_skipped} skipped)")
    print(f"{'=' * 60}")
    print(f"  Pass rate   {report.pass_rate:.3f}  ({report.num_passed}/{report.num_evaluated})")
    if report.num_unparseable:
        print(f"  Unparseable verdicts: {report.num_unparseable}")
    print(f"{'=' * 60}")

    if verbose:
        print()
        for r in report.sample_results:
            verdict_str = "PASS" if r.passed else ("FAIL" if r.passed is False else "????")
            print(f"[{verdict_str}] {r.sample_id!r}  citations={r.num_citations}")
            print(f"  query:    {r.query!r}")
            print(f"  judge:    {r.judge_output!r}")
            answer_preview = r.actual_answer[:200].replace("\n", " ")
            print(f"  answer:   {answer_preview}{'...' if len(answer_preview) == 200 else ''}")
            print()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rag.eval.answer_eval",
        description="Evaluate answer quality using an LLM-as-judge.",
    )
    parser.add_argument("--eval-set", default=None, metavar="PATH",
                        help=f"Path to eval set JSON (default: {_DEFAULT_EVAL_SET})")
    parser.add_argument("--config", default=None, metavar="PATH",
                        help="Path to config YAML (default: rag/config/config.yaml)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print per-sample results in addition to aggregate metrics")
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

    config = load_config(args.config)
    logger.info(
        "Building chat service (embedding=%s, llm=%s:%s)",
        config.embedding.model,
        config.llm.provider,
        config.llm.model,
    )
    chat_service = build_chat_service(config)

    # Reuse the same LLM client for judging — same model judges as generates.
    from rag.generation.factory import get_llm_client
    judge_llm = get_llm_client(config.llm)

    logger.info("Running answer eval on %d sample(s)", len(dataset))
    report = run_answer_eval(dataset, chat_service, judge_llm)
    print_report(report, verbose=args.verbose)

    return 0 if report.pass_rate >= 0.5 or report.num_evaluated == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
