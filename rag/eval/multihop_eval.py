"""Multi-hop answer evaluation: per-part judging plus evidence recall.

A multi-hop sample asks about several companies or periods at once ("compare
Delta's and United's ..."), so it has several *parts*, each with its own gold
answer and gold spans, plus an optional *conclusion* (which is larger, how it
changed). Built by ``scripts/build_multihop_set.py`` from pairs of verified
single-hop samples, so every part's gold is inherited rather than invented.

Why not one PASS/FAIL per answer
--------------------------------
The whole-answer rubric passes an answer that honestly says "the passages don't
cover United" -- which is the right call for grounding, and exactly the case
Milestone 19 exists to fix. In the 9b-vs-27b probe the judge scored both models
5/6 on multi-hop while a hand count found 1/5 vs 5/5 complete, correct answers
(docs/measured-results.md). So each part is judged on its own, and an answer
is **complete** only when every part, conclusion included, passes.

Evidence recall
---------------
The fraction of a sample's gold spans present in the **union** of everything
the turn retrieved. The pipeline retrieves once, so that is its citations; an
agent may search several times, and a span found by any search counts. It
separates "never found the evidence" from "found it and answered badly", which
need different fixes.

Cost
----
Each sample carries the turn's own LLM usage from ``ChatAnswer`` -- calls,
milliseconds inside them, prompt and generated tokens -- so an agent's extra
searches can be set against the pipeline's cost on the same questions. The
judge's calls are not in it: ``ask()`` meters only what it makes, and the judge
is called outside it. Token means cover only the samples whose provider
reported counts; with none, they are ``None`` (unknown), never zero.

Usage::

    python -m rag.eval.multihop_eval --corpus edgar_md --judge-model gemma4:31b-mlx
    python -m rag.eval.multihop_eval --eval-set path/to/set.json -v
    python -m rag.eval.multihop_eval --corpus edgar_md --oracle   # gold evidence, no retrieval
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from rag.config.settings import load_config
from rag.eval.answer_eval import (
    add_judge_arguments,
    generate_output,
    groundedness_counts,
    judge_call,
    not_judged,
    resolve_judge_config,
    subsample,
    trajectory,
)
from rag.eval.dataset import EvalDataset, EvalSample, ExpectedSpan
from rag.eval.oracle import add_oracle_argument, build_oracle_retriever
from rag.eval.provenance import IndexConflict
from rag.eval.relevance import unmatched_spans
from rag.eval.runs import RUNS_DIR, RunIncomplete
from rag.chat import build_chat_service
from rag.generation.chat_service import ChatResponder
from rag.llm.base import LLMClient
from rag.llm.factory import get_llm_client
from rag.logging_config import configure_logging
from rag.observability.records import AgentToolCall, is_read_passage, read_chars

logger = logging.getLogger(__name__)

_DEFAULT_EVAL_SET = (
    Path(__file__).resolve().parents[2] / "data" / "eval" / "edgar_multihop_set.json"
)

#: Label the conclusion is judged under, as though it were one more part.
CONCLUSION_LABEL = "The overall comparison or conclusion the question asks for"

_PART_JUDGE_SYSTEM_PROMPT = (
    "You grade ONE part of an answer to a question that spans several companies "
    "or periods. You are given the full question, the part being graded with its "
    "reference answer, and the system's full answer. "
    "Output exactly one word on the first line: PASS if the system answer states "
    "the reference fact for this part correctly -- the same figure or claim, "
    "attributed to the right company and period. Equivalent forms are fine: "
    "rounding, different units ($1,198 million vs $1.2 billion), or paraphrase. "
    "FAIL if the answer omits this part, says the information was not found, gives "
    "a different figure, or attributes it to the wrong company or period. "
    "Ignore what the answer says about other parts of the question. "
    "You may add a brief reason after the verdict on the same line, separated by a space."
)


def _part_judge_prompt(query: str, label: str, expected: str, actual: str) -> str:
    return (
        f"Full question: {query}\n\n"
        f"Part being graded: {label}\n"
        f"Reference answer for this part: {expected}\n\n"
        f"System answer: {actual}\n\n"
        "Verdict for this part (PASS or FAIL, optionally followed by a brief reason):"
    )


@dataclass(frozen=True)
class MultihopPart:
    """One company/period a multi-hop question asks about, or its conclusion."""

    label: str
    answer: str
    spans: list[ExpectedSpan] = field(default_factory=list)


def parts_of(sample: EvalSample) -> list[MultihopPart]:
    """The parts to judge for a sample: its ``parts``, then its ``conclusion`` if any.

    Raises ``ValueError`` on a sample with no parts, which is a single-hop sample
    in the wrong file -- judging it would silently score nothing.
    """
    raw_parts = sample.extra.get("parts") or []
    if not raw_parts:
        raise ValueError(f"Multi-hop sample {sample.id!r} has no 'parts'")
    parts = [
        MultihopPart(
            label=str(p["label"]),
            answer=str(p["answer"]),
            spans=[ExpectedSpan.from_json(s) for s in p.get("spans", [])],
        )
        for p in raw_parts
    ]
    conclusion = sample.extra.get("conclusion")
    if conclusion:
        parts.append(MultihopPart(label=CONCLUSION_LABEL, answer=str(conclusion)))
    return parts


@dataclass
class PartResult:
    label: str
    passed: bool | None  # None = unparseable verdict
    judge_output: str


@dataclass
class MultihopSampleResult:
    sample_id: str
    query: str
    kind: str
    actual_answer: str
    part_results: list[PartResult]
    evidence_total: int
    missing_spans: list[str]
    latency_s: float
    retrieval_rounds: int
    llm_calls: int = 0
    llm_ms: float = 0.0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    #: CRAG's groundedness verdict on the answer; None when unchecked or inconclusive.
    grounded: bool | None = None
    #: Agentic turns: every search call the model made, in order. Empty for a pipeline turn.
    agent_calls: list[AgentToolCall] = field(default_factory=list)
    #: Set when the turn produced no answer (`ChatAnswer.generation_failure`):
    #: scored as a failure without a judge call, never as a refusal.
    generation_failure: str | None = None
    #: Gold spans in none of the turn's *search* passages; None when the agent
    #: read no document windows, so it equals `missing_spans`.
    missing_spans_from_search: list[str] | None = None

    @property
    def completeness(self) -> float:
        """Fraction of parts (conclusion included) judged correct."""
        return sum(1 for p in self.part_results if p.passed is True) / len(self.part_results)

    @property
    def complete(self) -> bool:
        return all(p.passed is True for p in self.part_results)

    @property
    def evidence_recall(self) -> float:
        if self.evidence_total == 0:
            return 1.0
        return (self.evidence_total - len(self.missing_spans)) / self.evidence_total

    @property
    def evidence_recall_from_search(self) -> float:
        """`evidence_recall` over search passages alone, leaving out the agent's read windows."""
        missing = self.missing_spans if self.missing_spans_from_search is None else self.missing_spans_from_search
        if self.evidence_total == 0:
            return 1.0
        return (self.evidence_total - len(missing)) / self.evidence_total

    @property
    def read_chars(self) -> int:
        """Characters of document windows the agent read; 0 for a turn without reads."""
        return read_chars(self.agent_calls)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MultihopSampleResult:
        parts = [PartResult(**p) for p in data["part_results"]]
        searches = [AgentToolCall.from_dict(s) for s in data.get("agent_calls", [])]
        return cls(**{**data, "part_results": parts, "agent_calls": searches})


@dataclass
class MultihopReport:
    num_samples: int
    #: Share of samples where every part and the conclusion passed.
    complete_rate: float
    #: Mean per-sample fraction of parts passed -- partial credit.
    mean_completeness: float
    #: Mean per-sample fraction of gold spans anywhere in what was retrieved.
    evidence_recall: float
    num_empty: int
    num_unparseable: int
    mean_latency_s: float
    mean_retrieval_rounds: float
    #: The answering turn's LLM usage per sample; the judge's calls are excluded.
    mean_llm_calls: float
    mean_llm_s: float
    #: Means over the samples that reported counts only; None when none did.
    mean_prompt_tokens: float | None
    mean_completion_tokens: float | None
    #: How many samples the token means cover, so a partial mean is visible as one.
    num_with_tokens: int
    #: complete_rate per sample kind (cross_period, cross_company, aggregation).
    complete_rate_by_kind: dict[str, float]
    sample_results: list[MultihopSampleResult]
    #: `evidence_recall` over search passages alone: the gap to it is what the
    #: agent's read windows added. Equal to it for a row without reads.
    evidence_recall_from_search: float = 0.0
    mean_read_chars: float = 0.0


def judge_multihop(sample: EvalSample, output: dict[str, Any], judge: LLMClient) -> dict[str, Any]:
    """Judge each part of one stored answer, conclusion included: the judgment record for it."""
    note = not_judged(output)
    verdicts = []
    for part in parts_of(sample):
        if note is not None:
            verdicts.append({"label": part.label, "passed": False, "judge_output": note})
            continue
        verdicts.append(judge_part(judge, sample.query, part.label, part.answer, output["answer"]))
    return {"sample_id": sample.id, "verdicts": verdicts}


def judge_part(judge: LLMClient, query: str, label: str, expected: str, answer: str) -> dict[str, Any]:
    """Judge one part of an answer: the verdict as a judgment records it."""
    return judge_call(judge, _PART_JUDGE_SYSTEM_PROMPT, _part_judge_prompt(query, label, expected, answer), label)


def multihop_result(sample: EvalSample, output: dict[str, Any], judgment: dict[str, Any]) -> MultihopSampleResult:
    """One sample's result from its stored output and judgment, evidence scored against ``sample``'s labels.

    Evidence is the union of every passage the turn retrieved -- the stored
    ``passages`` -- as `run_multihop_eval` has always measured it.
    """
    gold = [span for part in parts_of(sample) for span in part.spans]
    texts = [p["text"] for p in output["passages"]]
    searched = [p["text"] for p in output["passages"] if not is_read_passage(p["chunk_id"])]
    return MultihopSampleResult(
        sample_id=sample.id,
        query=output["query"],
        kind=str(sample.extra.get("kind", "unknown")),
        actual_answer=output["answer"],
        part_results=[
            PartResult(label=v["label"], passed=v["passed"], judge_output=v["judge_output"])
            for v in judgment["verdicts"]
        ],
        evidence_total=len(gold),
        missing_spans=unmatched_spans(gold, texts),
        latency_s=output["latency_s"],
        retrieval_rounds=output["retrieval_rounds"],
        llm_calls=output["llm_calls"],
        llm_ms=output["llm_ms"],
        prompt_tokens=output["prompt_tokens"],
        completion_tokens=output["completion_tokens"],
        grounded=output["grounded"],
        agent_calls=[AgentToolCall.from_dict(c) for c in output["agent_calls"]],
        generation_failure=output["generation_failure"],
        missing_spans_from_search=unmatched_spans(gold, searched) if len(searched) < len(texts) else None,
    )


def run_multihop_eval(
    dataset: EvalDataset,
    chat_service: ChatResponder,
    judge: LLMClient,
    *,
    completed: Mapping[str, MultihopSampleResult] | None = None,
    on_result: Callable[[MultihopSampleResult], None] | None = None,
) -> MultihopReport:
    """Answer every sample, judge each of its parts, and measure evidence recall.

    ``completed`` and ``on_result`` work as in ``run_answer_eval``: reuse an
    interrupted run's samples, and hand over each new one as it finishes.
    """
    results: list[MultihopSampleResult] = []
    for sample in dataset:
        if completed and sample.id in completed:
            results.append(completed[sample.id])
            continue
        parts_of(sample)  # a sample without parts fails before it is asked
        output = generate_output(sample, chat_service)
        results.append(multihop_result(sample, output, judge_multihop(sample, output, judge)))
        if on_result is not None:
            on_result(results[-1])

    return summarize(results)


def summarize(results: list[MultihopSampleResult]) -> MultihopReport:
    n = len(results)

    def avg(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    def known_avg(values: list[int | None]) -> float | None:
        known = [v for v in values if v is not None]
        return sum(known) / len(known) if known else None

    by_kind: dict[str, list[MultihopSampleResult]] = {}
    for r in results:
        by_kind.setdefault(r.kind, []).append(r)

    return MultihopReport(
        num_samples=n,
        complete_rate=avg([float(r.complete) for r in results]),
        mean_completeness=avg([r.completeness for r in results]),
        evidence_recall=avg([r.evidence_recall for r in results]),
        num_empty=sum(1 for r in results if r.generation_failure or not r.actual_answer.strip()),
        num_unparseable=sum(1 for r in results for p in r.part_results if p.passed is None),
        mean_latency_s=avg([r.latency_s for r in results]),
        mean_retrieval_rounds=avg([float(r.retrieval_rounds) for r in results]),
        mean_llm_calls=avg([float(r.llm_calls) for r in results]),
        mean_llm_s=avg([r.llm_ms / 1000 for r in results]),
        mean_prompt_tokens=known_avg([r.prompt_tokens for r in results]),
        mean_completion_tokens=known_avg([r.completion_tokens for r in results]),
        num_with_tokens=sum(
            1 for r in results if r.prompt_tokens is not None or r.completion_tokens is not None
        ),
        complete_rate_by_kind={
            kind: avg([float(r.complete) for r in rs]) for kind, rs in sorted(by_kind.items())
        },
        sample_results=results,
        evidence_recall_from_search=avg([r.evidence_recall_from_search for r in results]),
        mean_read_chars=avg([float(r.read_chars) for r in results]),
    )


def multihop_scores(report: MultihopReport) -> dict[str, Any]:
    """A report as the answer matrix's row records it, and a multi-hop run's score revision holds it."""
    return {
        "num_evaluated": report.num_samples,
        "complete_rate": round(report.complete_rate, 4),
        "mean_completeness": round(report.mean_completeness, 4),
        "evidence_recall": round(report.evidence_recall, 4),
        "complete_rate_by_kind": {k: round(v, 4) for k, v in report.complete_rate_by_kind.items()},
        "num_empty": report.num_empty,
        "num_unparseable": report.num_unparseable,
        "mean_latency_s": round(report.mean_latency_s, 1),
        "mean_retrieval_rounds": round(report.mean_retrieval_rounds, 2),
        # The answering turn's cost; the judge's calls are not in it.
        "mean_llm_calls": round(report.mean_llm_calls, 2),
        "mean_llm_s": round(report.mean_llm_s, 1),
        "mean_prompt_tokens": _round_or_none(report.mean_prompt_tokens),
        "mean_completion_tokens": _round_or_none(report.mean_completion_tokens),
        "num_with_tokens": report.num_with_tokens,
        "groundedness": groundedness_counts([(r.grounded, r.complete) for r in report.sample_results]),
        # Per-sample detail, so a hand check of the judge needs no rerun.
        "samples": [
            {
                "id": r.sample_id,
                "complete": r.complete,
                "parts": {p.label: p.passed for p in r.part_results},
                # The judge's full reply per part, reason included: what a hand
                # check of a verdict needs, and what `parts` reduces to a bool.
                "judge_outputs": {p.label: p.judge_output for p in r.part_results},
                "evidence_recall": round(r.evidence_recall, 3),
                "missing_spans": r.missing_spans,
                "llm_calls": r.llm_calls,
                "llm_ms": round(r.llm_ms),
                "prompt_tokens": r.prompt_tokens,
                "completion_tokens": r.completion_tokens,
                "grounded": r.grounded,
                "latency_s": round(r.latency_s, 1),
                "answer": r.actual_answer,
                **trajectory(r.agent_calls),
            }
            for r in report.sample_results
        ],
    }


def _round_or_none(value: float | None) -> float | None:
    return None if value is None else round(value, 1)


def print_report(report: MultihopReport, *, verbose: bool = False) -> None:
    print(f"\n{'=' * 60}")
    print(f"  Multi-hop Eval  ({report.num_samples} samples)")
    print(f"{'=' * 60}")
    print(f"  Complete & correct   {report.complete_rate:.3f}")
    print(f"  Mean completeness    {report.mean_completeness:.3f}")
    print(f"  Evidence recall      {report.evidence_recall:.3f}")
    if report.mean_read_chars:
        print(f"    from search alone  {report.evidence_recall_from_search:.3f}"
              f"  (reads: {report.mean_read_chars:,.0f} chars per turn)")
    for kind, rate in report.complete_rate_by_kind.items():
        print(f"    complete, {kind:<16} {rate:.3f}")
    if report.num_empty:
        print(f"  No answer (empty or generation failed): {report.num_empty}")
    if report.num_unparseable:
        print(f"  Unparseable part verdicts: {report.num_unparseable}")
    print(f"  Mean latency  {report.mean_latency_s:.1f}s per turn, "
          f"{report.mean_retrieval_rounds:.1f} retrieval round(s)")
    print(f"  LLM per turn  {report.mean_llm_calls:.1f} call(s), {report.mean_llm_s:.1f}s"
          f"{format_tokens(report)}")
    print(f"{'=' * 60}")

    if verbose:
        print()
        for r in report.sample_results:
            print(f"[{'COMPLETE' if r.complete else 'PARTIAL '}] {r.sample_id}  "
                  f"parts={r.completeness:.2f}  evidence={r.evidence_recall:.2f}")
            print(f"  query:  {r.query!r}")
            for p in r.part_results:
                verdict = "PASS" if p.passed else ("FAIL" if p.passed is False else "????")
                print(f"    [{verdict}] {p.label}: {p.judge_output[:120]!r}")
            for span in r.missing_spans:
                print(f"    missing evidence: {span[:100]!r}")
            print(f"  answer: {r.actual_answer[:300].replace(chr(10), ' ')}")
            print()


def format_tokens(report: MultihopReport) -> str:
    """Mean tokens per turn as a suffix, or "" when no sample reported any."""
    if report.num_with_tokens == 0:
        return ""

    def fmt(v: float | None) -> str:
        return "?" if v is None else f"{v:,.0f}"

    coverage = (
        f" (from {report.num_with_tokens} of {report.num_samples})"
        if report.num_with_tokens < report.num_samples else ""
    )
    return (f", {fmt(report.mean_prompt_tokens)} prompt / "
            f"{fmt(report.mean_completion_tokens)} generated tokens{coverage}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rag.eval.multihop_eval",
        description="Evaluate multi-company / multi-period answers part by part.",
    )
    parser.add_argument("--eval-set", default=None, metavar="PATH",
                        help=f"Path to multi-hop set JSON (default: {_DEFAULT_EVAL_SET})")
    parser.add_argument("--config", default=None, metavar="PATH")
    parser.add_argument("--corpus", action="append", default=None, metavar="NAME",
                        help="Corpus to evaluate against, overriding corpora.active.")
    parser.add_argument("--limit", type=int, default=0, metavar="N",
                        help="Evaluate an evenly-spaced subset of N samples.")
    add_judge_arguments(parser)
    add_oracle_argument(parser)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR, metavar="DIR",
                        help="Where the run folder is written (default: data/eval/runs).")
    parser.add_argument("--no-run", action="store_true",
                        help="Don't write a run folder: generate and judge together and print the report "
                             "only (and don't check the index).")
    parser.add_argument("--fresh", action="store_true",
                        help="Start a new run and judgment attempt even if an interrupted one matches.")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = _build_parser().parse_args(argv)

    eval_path = Path(args.eval_set) if args.eval_set else _DEFAULT_EVAL_SET
    if not eval_path.exists():
        logger.error("Eval set not found: %s (build it with scripts/build_multihop_set.py)", eval_path)
        return 1
    full = EvalDataset.load(eval_path)
    dataset = subsample(full, args.limit)
    for sample in dataset:
        parts_of(sample)  # a sample without parts fails before anything is generated

    config = load_config(args.config)
    judge_config = resolve_judge_config(config, args.judge_model, args.judge_provider)
    logger.info("Judge: %s:%s", judge_config.provider, judge_config.model)
    judge = get_llm_client(judge_config)
    retriever = None
    if args.oracle:
        retriever, unfound = build_oracle_retriever(config, dataset, args.corpus)
        if unfound:
            logger.error("The oracle can't find %d gold span(s) in the chunks; see above", len(unfound))
            return 1
    chat_service = build_chat_service(config, corpora=args.corpus, retriever=retriever)

    logger.info("Running multi-hop eval on %d sample(s)", len(dataset))
    report: MultihopReport
    if args.no_run:
        report = run_multihop_eval(dataset, chat_service, judge)
        print_report(report, verbose=args.verbose)
    else:
        from rag.eval.judge import evaluate_answers_into_run, open_answer_run
        from rag.eval.score import stored_report

        try:
            run = open_answer_run(
                "multihop", config, args.corpus, runs_dir=args.runs_dir, dataset_path=eval_path, dataset=full,
                selected=dataset, oracle=args.oracle, fresh=args.fresh,
                variant=Path(args.config).stem if args.config else "config",
            )
            attempt, revision = evaluate_answers_into_run(run, dataset, chat_service, judge, judge_config,
                                                          fresh=args.fresh, started=time.monotonic())
        except (IndexConflict, RunIncomplete) as exc:
            logger.error("%s", exc)
            return 1
        stored = stored_report(run, dataset, attempt)
        assert isinstance(stored, MultihopReport)
        report = stored
        print_report(report, verbose=args.verbose)
        print(f"Run: {run.path}  (judgment {attempt.ref}, score revision {revision})")
    return 0


if __name__ == "__main__":
    # Under `python -m` this file runs as `__main__`, and `rag.eval.score`
    # imports it again as `rag.eval.multihop_eval`: two `MultihopReport`
    # classes, so `main`'s isinstance check on a stored report fails. Run the
    # imported module's `main` so there is only one.
    from rag.eval.multihop_eval import main as package_main

    sys.exit(package_main())
