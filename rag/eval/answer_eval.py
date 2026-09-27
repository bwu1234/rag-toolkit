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
    python -m rag.eval.answer_eval --judge-provider gemini --judge-model gemma-4-31b-it

The judge prompt is deliberately minimal: it asks the LLM to output exactly
``PASS`` or ``FAIL`` (optionally followed by a brief reason on the same line)
so the result is easy to parse without fragile regex.  The raw judge output is
preserved in :attr:`AnswerSampleResult.judge_output` for debugging.

Samples without an ``expected_answer`` are skipped and counted as ``n/a`` --
they can participate in retrieval eval but not answer eval.

Retrieval miss or generation error?
-----------------------------------
A FAIL alone does not say which stage to fix. For every sample carrying
``expected_spans``, the runner also checks whether those spans were among the
passages the generator was shown (``ChatAnswer.citations``, which is every
passage in the prompt), under the same normalization as the retrieval eval.
The pass rate is then reported separately for **evidence retrieved** -- where a
FAIL is a generation (or judge) error -- and **evidence missed**, where a FAIL
is a retrieval miss no generator can fix in one pass. A PASS without the
evidence is possible too: the fact may sit in a passage the span label did not
anticipate, or the model may have answered from memory.

Samples with no spans (refusals, the document-matched baseline set) are left
out of both buckets rather than guessed at.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, get_args
from pathlib import Path

from rag.config.settings import LLMConfig, LLMProvider, RagConfig, load_config
from rag.eval.dataset import EvalDataset, EvalSample
from rag.eval.relevance import sample_unmatched_spans
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


def resolve_judge_config(
    config: RagConfig, model: str | None = None, provider: LLMProvider | None = None
) -> LLMConfig:
    """The LLM that grades answers: `eval.judge`, else the generator; `model` / `provider` override it.

    Overriding only `model` keeps every other judge setting. Overriding
    `provider` to a different one starts a fresh config instead, keeping only
    the provider-neutral settings (temperature, token cap, timeout): the old
    provider's endpoint, rate limits and reasoning knobs mean nothing to the
    new one. Without that, `--judge-model gemma4:31b-mlx` on a Gemini
    generator config would send an Ollama tag to the Gemini API.

    Warns when the result is the generator. A model judging its own answers is
    the pre-Milestone 19 default, kept so old results reproduce, but it biases
    every score and makes a generator comparison meaningless: 0.775-0.825
    self-judged versus 0.900 under a separate judge on the same 40 samples
    (docs/measured-results.md).
    """
    base = config.eval.judge or config.llm
    if provider is not None and provider != base.provider:
        if not model:
            raise ValueError(
                f"Switching the judge to provider {provider!r} needs a model for it too (--judge-model)"
            )
        judge = LLMConfig(
            provider=provider,
            model=model,
            temperature=base.temperature,
            max_tokens=base.max_tokens,
            timeout_s=base.timeout_s,
        )
    else:
        judge = base.model_copy(update={"model": model} if model else {})
    if judge.provider == "gemini" and ":" in judge.model:
        # A colon can't be in a Gemini model id (it would break the
        # `models/{model}:generateContent` path); it is how Ollama writes tags.
        raise ValueError(
            f"Judge model {judge.model!r} looks like an Ollama tag, but the judge's provider is gemini "
            f"(inherited from {'eval.judge' if config.eval.judge else 'llm'}). "
            "Pass --judge-provider ollama, or set eval.judge in the config."
        )
    if (judge.provider, judge.model) == (config.llm.provider, config.llm.model):
        logger.warning(
            "Judge and generator are the same model (%s:%s): scores are self-graded "
            "and not comparable across generators. Set eval.judge or pass --judge-model.",
            judge.provider, judge.model,
        )
    return judge


def build_judge(config: RagConfig, model: str | None = None, provider: LLMProvider | None = None) -> LLMClient:
    """Instantiate the judge chosen by :func:`resolve_judge_config`."""
    judge = resolve_judge_config(config, model, provider)
    logger.info("Judge: %s:%s", judge.provider, judge.model)
    return get_llm_client(judge)


def add_judge_arguments(parser: argparse.ArgumentParser, *, note: str = "") -> None:
    """`--judge-model` / `--judge-provider`, shared by every runner that grades answers."""
    parser.add_argument("--judge-model", default=None, metavar="MODEL",
                        help="Judge with this model instead of eval.judge (or the generator, "
                             "if eval.judge is unset). Other judge settings are kept." + note)
    parser.add_argument("--judge-provider", default=None, choices=get_args(LLMProvider),
                        help="The judge's provider, when it differs from eval.judge's (or the "
                             "generator's) -- e.g. a local judge for a hosted generator. Needs "
                             "--judge-model; only temperature, max_tokens and timeout carry over.")


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
    #: Gold spans the sample declares; 0 when it has none (refusals, doc-matched sets).
    evidence_total: int = 0
    #: Gold spans in none of the passages the generator was shown.
    missing_spans: list[str] = field(default_factory=list)

    @property
    def evidence_retrieved(self) -> bool | None:
        """Whether every gold span was in the prompt; None when there are no spans."""
        if self.evidence_total == 0:
            return None
        return not self.missing_spans

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnswerSampleResult:
        return cls(**data)


@dataclass
class EvidenceBucket:
    """Pass counts for the samples whose evidence was (or was not) retrieved."""

    num_evaluated: int = 0
    num_passed: int = 0

    @property
    def pass_rate(self) -> float:
        return self.num_passed / self.num_evaluated if self.num_evaluated else 0.0

    @property
    def num_failed(self) -> int:
        """Includes unparseable verdicts, which are not passes either."""
        return self.num_evaluated - self.num_passed


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
    #: Samples whose gold spans were all in the prompt: a FAIL here is generation.
    evidence_retrieved: EvidenceBucket = field(default_factory=EvidenceBucket)
    #: Samples missing a gold span from the prompt: a FAIL here is retrieval.
    evidence_missed: EvidenceBucket = field(default_factory=EvidenceBucket)


def run_answer_eval(
    dataset: EvalDataset,
    chat_service: ChatService,
    llm_client: LLMClient,
    *,
    completed: Mapping[str, AnswerSampleResult] | None = None,
    on_result: Callable[[AnswerSampleResult], None] | None = None,
) -> AnswerEvalReport:
    """Run the full pipeline + LLM judge for every sample with an expected answer.

    ``completed`` holds results from an earlier, interrupted run of the same
    setup: those samples are reused, not re-asked. ``on_result`` is called with
    each newly computed result as soon as it exists, which is how the answer
    matrix checkpoints. Report order follows the dataset either way.
    """
    results: list[AnswerSampleResult] = []
    skipped = 0

    for sample in dataset:
        if not sample.expected_answer:
            skipped += 1
            continue
        if completed and sample.id in completed:
            results.append(completed[sample.id])
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
        passages = [(citation.document_id, citation.text) for citation in chat_answer.citations]

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
                evidence_total=len(sample.expected_spans),
                missing_spans=sample_unmatched_spans(sample, passages),
            )
        )
        if on_result is not None:
            on_result(results[-1])

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
        evidence_retrieved=_bucket([r for r in results if r.evidence_retrieved is True]),
        evidence_missed=_bucket([r for r in results if r.evidence_retrieved is False]),
    )


def _bucket(results: list[AnswerSampleResult]) -> EvidenceBucket:
    return EvidenceBucket(
        num_evaluated=len(results),
        num_passed=sum(1 for r in results if r.passed is True),
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
    found, missed = report.evidence_retrieved, report.evidence_missed
    if found.num_evaluated or missed.num_evaluated:
        print(f"{'-' * 60}")
        print("  By whether the gold span reached the prompt:")
        print(f"    evidence retrieved  {found.num_passed}/{found.num_evaluated} passed"
              f"  -> {found.num_failed} generation/judge failure(s)")
        print(f"    evidence missed     {missed.num_passed}/{missed.num_evaluated} passed"
              f"  -> {missed.num_failed} retrieval failure(s)")
    print(f"{'=' * 60}")

    if verbose:
        print()
        for r in report.sample_results:
            verdict_str = "PASS" if r.passed else ("FAIL" if r.passed is False else "????")
            evidence = {True: "retrieved", False: "MISSED", None: "n/a"}[r.evidence_retrieved]
            print(f"[{verdict_str}] {r.sample_id!r}  citations={r.num_citations}  "
                  f"evidence={evidence}")
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
    add_judge_arguments(parser)
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
    # Before the chat service, so a bad judge fails before anything is loaded.
    judge_llm = build_judge(config, args.judge_model, args.judge_provider)
    chat_service = build_chat_service(config, corpora=args.corpus)

    logger.info("Running answer eval on %d sample(s)", len(dataset))
    report = run_answer_eval(dataset, chat_service, judge_llm)
    print_report(report, verbose=args.verbose)

    return 0 if report.pass_rate >= 0.5 or report.num_evaluated == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
