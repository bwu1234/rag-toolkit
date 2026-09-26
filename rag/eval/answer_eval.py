"""Answer evaluation runner (LLM-as-judge).

Loads an eval set, runs the full retrieve→rerank→generate pipeline via
:class:`~rag.generation.chat_service.ChatService` for each query, then asks
the judge LLM to grade each answer against the sample's ``expected_answer``.

The judge is ``eval.judge`` from config, falling back to the generator (``llm``)
when that is unset -- see :func:`resolve_judge_config`.

Usage::

    python -m rag.eval.answer_eval                             # uses default eval set
    python -m rag.eval.answer_eval --eval-set path/to/set.json
    python -m rag.eval.answer_eval --config path/to/config.yaml
    python -m rag.eval.answer_eval --judge-model gemma4:31b-mlx

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
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from rag.config.settings import LLMConfig, RagConfig, load_config
from rag.eval.dataset import EvalDataset, EvalSample
from rag.generation.builder import build_chat_service
from rag.generation.chat_service import ChatService
from rag.generation.factory import get_llm_client
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


# Samples carrying this tier are questions the corpus CANNOT answer, where
# declining is the correct behaviour.
REFUSAL_TIER = "refusal"

_REFUSAL_JUDGE_SYSTEM_PROMPT = (
    "You check whether a retrieval system correctly DECLINED to answer. "
    "The question is one its document corpus cannot answer, so refusing is the "
    "correct behaviour and the only thing you are judging. "
    "Output exactly one word on the first line: PASS if the system declined, "
    "said it could not find the information, or otherwise avoided asserting a "
    "specific factual answer. FAIL if it asserted a specific answer -- a figure, "
    "date, name, or factual claim -- as though the corpus supported it. "
    "The wording, length, and level of detail of the refusal are irrelevant: a "
    "terse 'I don't have that information' is just as correct as a detailed "
    "explanation. Judge only whether it refused. "
    "You may add a brief reason after the verdict on the same line, separated by a space."
)


def _judge_prompt(query: str, expected: str, actual: str) -> str:
    return (
        f"Question: {query}\n\n"
        f"Reference answer: {expected}\n\n"
        f"System answer: {actual}\n\n"
        "Verdict (PASS or FAIL, optionally followed by a brief reason):"
    )


def _refusal_judge_prompt(query: str, criteria: str, actual: str) -> str:
    return (
        f"Question the corpus cannot answer: {query}\n\n"
        f"Why it is unanswerable: {criteria}\n\n"
        f"System answer: {actual}\n\n"
        "Did the system decline? Verdict (PASS or FAIL, optionally followed by a brief reason):"
    )


def resolve_judge_config(config: RagConfig, model: str | None = None) -> LLMConfig:
    """The LLM that grades answers: `eval.judge`, else the generator; `model` overrides its name.

    Warns when the result is the generator. A model judging its own answers is
    the pre-Milestone 19 default, kept so old results reproduce, but it biases
    every score and makes a generator comparison meaningless: 0.775-0.825
    self-judged versus 0.900 under a separate judge on the same 40 samples
    (docs/measured-results.md).
    """
    judge = (config.eval.judge or config.llm).model_copy()
    if model:
        judge.model = model
    if (judge.provider, judge.model) == (config.llm.provider, config.llm.model):
        logger.warning(
            "Judge and generator are the same model (%s:%s): scores are self-graded "
            "and not comparable across generators. Set eval.judge or pass --judge-model.",
            judge.provider, judge.model,
        )
    return judge


def build_judge(config: RagConfig, model: str | None = None) -> LLMClient:
    """Instantiate the judge chosen by :func:`resolve_judge_config`."""
    judge = resolve_judge_config(config, model)
    logger.info("Judge: %s:%s", judge.provider, judge.model)
    return get_llm_client(judge)


def judge_for(sample: EvalSample) -> tuple[str, Callable[[str, str, str], str]]:
    """Pick the judging rubric for a sample.

    Refusal samples need a different question asked of the judge, not a different
    threshold. The default rubric explicitly fails an answer that "refuses to
    answer", which is correct for answerable questions and exactly backwards for
    unanswerable ones -- it scored a correct refusal as FAIL and turned the
    refusal set into a measurement of how closely refusal *prose* resembled the
    reference text. Two correct refusals differing only in verbosity were graded
    FAIL and PASS respectively, which is what surfaced this.
    """
    if sample.extra.get("tier") == REFUSAL_TIER:
        return _REFUSAL_JUDGE_SYSTEM_PROMPT, _refusal_judge_prompt
    return _JUDGE_SYSTEM_PROMPT, _judge_prompt


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
    #: Wall-clock seconds for the turn, excluding the judge call.
    latency_s: float = 0.0
    #: Times retrieval ran for the turn (more than one only when CRAG retried).
    retrieval_rounds: int = 1


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
    num_empty: int = 0       # turns that produced no answer text
    mean_latency_s: float = 0.0


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

        started = time.monotonic()
        chat_answer = chat_service.ask(sample.query)
        latency = time.monotonic() - started
        judge_system, judge_prompt = judge_for(sample)
        judge_out = llm_client.generate(
            judge_prompt(sample.query, sample.expected_answer, chat_answer.answer),
            system=judge_system,
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
                latency_s=latency,
                retrieval_rounds=chat_answer.retrieval_attempts,
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
        num_empty=sum(1 for r in results if not r.actual_answer.strip()),
        mean_latency_s=sum(r.latency_s for r in results) / n if n > 0 else 0.0,
    )


def print_report(report: AnswerEvalReport, *, verbose: bool = False) -> None:
    """Print a formatted answer eval report to stdout."""
    print(f"\n{'=' * 60}")
    print(f"  Answer Eval  ({report.num_evaluated} evaluated, {report.num_skipped} skipped)")
    print(f"{'=' * 60}")
    print(f"  Pass rate   {report.pass_rate:.3f}  ({report.num_passed}/{report.num_evaluated})")
    if report.num_unparseable:
        print(f"  Unparseable verdicts: {report.num_unparseable}")
    if report.num_empty:
        print(f"  Empty answers: {report.num_empty}")
    print(f"  Mean latency  {report.mean_latency_s:.1f}s per turn (generation only)")
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
    parser.add_argument("--corpus", action="append", default=None, metavar="NAME",
                        help="Corpus to evaluate against, overriding corpora.active. "
                             "Repeat to target a pooled index.")
    parser.add_argument("--limit", type=int, default=0, metavar="N",
                        help="Evaluate an evenly-spaced subset of N samples. Answer eval is "
                             "the expensive one (generation + judge per sample, plus one call "
                             "per passage when crag.grade_documents is on), so a subset is "
                             "often the only affordable way to compare configurations.")
    parser.add_argument("--judge-model", default=None, metavar="MODEL",
                        help="Judge with this model instead of eval.judge (or the generator, "
                             "if eval.judge is unset). Other judge settings are kept.")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print per-sample results in addition to aggregate metrics")
    return parser


def subsample(dataset: EvalDataset, limit: int) -> EvalDataset:
    """Take `limit` samples spread evenly across the set.

    A prefix would be biased: the generated eval set is sorted by document id, so
    the first N samples are all the alphabetically-first companies. Striding
    keeps coverage across entities and periods, and is deterministic so two
    configurations are compared on identical questions.
    """
    if limit <= 0 or limit >= len(dataset):
        return dataset
    step = len(dataset) / limit
    picked = [dataset.samples[int(i * step)] for i in range(limit)]
    return EvalDataset(samples=picked, source_path=dataset.source_path)


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

    if args.limit:
        before = len(dataset)
        dataset = subsample(dataset, args.limit)
        logger.info("Subsampled %d of %d sample(s) (evenly spaced)", len(dataset), before)

    config = load_config(args.config)
    logger.info(
        "Building chat service (embedding=%s, llm=%s:%s)",
        config.embedding.model,
        config.llm.provider,
        config.llm.model,
    )
    chat_service = build_chat_service(config, corpora=args.corpus)

    judge_llm = build_judge(config, args.judge_model)

    logger.info("Running answer eval on %d sample(s)", len(dataset))
    report = run_answer_eval(dataset, chat_service, judge_llm)
    print_report(report, verbose=args.verbose)

    return 0 if report.pass_rate >= 0.5 or report.num_evaluated == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
