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
    python -m rag.eval.answer_eval --corpus edgar_md --oracle     # gold evidence, no retrieval

The judge prompt is deliberately minimal: it asks the LLM to output exactly
``PASS`` or ``FAIL`` (optionally followed by a brief reason on the same line)
so the result is easy to parse without fragile regex.  The raw judge output is
preserved in :attr:`AnswerSampleResult.judge_output` for debugging.

Each run writes a run folder (``--no-run`` to skip): every answer is generated
and stored with the passages it was shown and its model calls verbatim
(:func:`generate_output`), then judged as a separate attempt
(``rag.eval.judge``), so a run can be re-judged or rescored without
regenerating. ``run_answer_eval`` keeps the two together for ``--no-run``.

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
from rag.eval.dataset import EvalDataset, EvalSample, subsample
from rag.eval.metrics import wilson_interval
from rag.eval.relevance import sample_unmatched_spans
from rag.eval.provenance import IndexConflict
from rag.eval.runs import RUNS_DIR, Run, RunIncomplete
from rag.eval.oracle import add_oracle_argument, build_oracle_retriever
from rag.chat import build_chat_service
from rag.generation.chat_service import ChatResponder
from rag.llm.factory import get_llm_client
from rag.llm.base import LLMClient
from rag.logging_config import configure_logging
from rag.observability.records import AgentToolCall, TurnRecord, is_read_passage, read_chars

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
    #: CRAG's groundedness verdict on the answer; None when unchecked or inconclusive.
    grounded: bool | None = None
    #: The answering turn's LLM usage, as in the multi-hop results; the judge's calls are excluded.
    llm_calls: int = 0
    llm_ms: float = 0.0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    #: Agentic turns: every agent tool call the model made, in order; search-specific details are present only for rag_search. Empty for a pipeline turn.
    agent_calls: list[AgentToolCall] = field(default_factory=list)
    #: Set when the turn produced no answer (`ChatAnswer.generation_failure`):
    #: scored as a failure without a judge call, never as a refusal.
    generation_failure: str | None = None
    #: Gold spans in none of the turn's *search* passages; None when the agent
    #: read no document windows, so it equals `missing_spans`. A 6k-character
    #: read contains a span more easily than a 1.2k passage, so recall from
    #: reads is reported apart from recall from search.
    missing_spans_from_search: list[str] | None = None

    @property
    def read_chars(self) -> int:
        """Characters of document windows the agent read; 0 for a turn without reads."""
        return read_chars(self.agent_calls)

    @property
    def evidence_only_from_reads(self) -> bool:
        """Whether reads, not searches, brought the last gold span into the prompt."""
        return (
            self.missing_spans_from_search is not None
            and bool(self.missing_spans_from_search)
            and not self.missing_spans
        )

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
        searches = [AgentToolCall.from_dict(s) for s in data.get("agent_calls", [])]
        return cls(**{**data, "agent_calls": searches})


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
    num_empty: int = 0       # turns that produced no answer: empty text, or a generation failure
    mean_latency_s: float = 0.0
    #: Samples whose gold spans were all in the prompt: a FAIL here is generation.
    evidence_retrieved: EvidenceBucket = field(default_factory=EvidenceBucket)
    #: Samples missing a gold span from the prompt: a FAIL here is retrieval.
    evidence_missed: EvidenceBucket = field(default_factory=EvidenceBucket)
    #: Of `evidence_retrieved`, the samples whose spans were complete only with
    #: the agent's read windows: what reading added over searching.
    num_evidence_only_from_reads: int = 0
    mean_read_chars: float = 0.0


#: Names how `_parse_verdict` reads a reply. Part of the judge protocol
#: (`rag.eval.judge`): change it when the parser changes what a reply means.
VERDICT_PARSER = "first-word-pass-fail-v1"


def generate_output(sample: EvalSample, chat_service: ChatResponder) -> dict[str, Any]:
    """Answer one sample: the output record a run stores, before any judging.

    ``passages`` are every passage the generator was shown
    (`ChatAnswer.citations`), with their text, so evidence is scored later
    without an index. ``turn`` is the turn's record with every LLM call it made
    verbatim (`TurnRecord.llm_exchanges`): rendered system prompts and
    messages, tools offered, tool results and raw replies -- condensing,
    retries and every agent step, not only the final answer. None from a
    responder that keeps no record (a test fake).
    """
    records: list[TurnRecord] = []
    started = time.monotonic()
    answer = chat_service.ask(sample.query, on_record=records.append)
    latency = time.monotonic() - started
    return {
        "sample_id": sample.id,
        "query": sample.query,
        "answer": answer.answer,
        "generation_failure": answer.generation_failure,
        "latency_s": latency,
        "retrieval_rounds": answer.retrieval_attempts,
        "grounded": answer.grounded,
        "llm_calls": answer.llm_calls,
        "llm_ms": answer.llm_ms,
        "prompt_tokens": answer.prompt_tokens,
        "completion_tokens": answer.completion_tokens,
        "agent_calls": [asdict(call) for call in answer.agent_calls],
        "passages": [asdict(citation) for citation in answer.citations],
        "cited_chunk_ids": answer.cited_chunk_ids,
        "invalid_citations": answer.invalid_citations,
        "turn": records[-1].to_dict() if records else None,
    }


def not_judged(output: dict[str, Any]) -> str | None:
    """The note recorded instead of a verdict when ``output`` has no answer to judge.

    A generation failure's explanatory message must not reach the refusal
    judge, which would rightly see "I couldn't answer" as a refusal and pass it.
    """
    failure = output.get("generation_failure")
    return None if failure is None else f"[not judged: generation failed ({failure})]"


def judge_call(judge: LLMClient, system: str, prompt: str, label: str | None = None) -> dict[str, Any]:
    """One judge call as a judgment records it: inputs, full reply, parsed verdict."""
    reply = judge.generate(prompt, system=system)
    return {
        "label": label, "passed": _parse_verdict(reply), "judge_output": reply.strip(),
        "system": system, "prompt": prompt, "reply": reply,
    }


def judge_answer(sample: EvalSample, output: dict[str, Any], judge: LLMClient) -> dict[str, Any]:
    """Judge one stored answer against ``sample``'s reference: the judgment record for it."""
    if (note := not_judged(output)) is not None:
        return {"sample_id": sample.id, "verdicts": [{"label": None, "passed": False, "judge_output": note}]}
    system, template = judge_for(sample)
    prompt = template(sample.query, sample.expected_answer or "", output["answer"])
    return {"sample_id": sample.id, "verdicts": [judge_call(judge, system, prompt)]}


def answer_result(sample: EvalSample, output: dict[str, Any], judgment: dict[str, Any]) -> AnswerSampleResult:
    """One sample's result from its stored output and judgment, evidence scored against ``sample``'s labels."""
    passages = output["passages"]
    shown = [(p["document_id"], p["text"]) for p in passages]
    searched = [(p["document_id"], p["text"]) for p in passages if not is_read_passage(p["chunk_id"])]
    [verdict] = judgment["verdicts"]
    return AnswerSampleResult(
        sample_id=sample.id,
        query=output["query"],
        expected_answer=sample.expected_answer or "",
        actual_answer=output["answer"],
        judge_output=verdict["judge_output"],
        passed=verdict["passed"],
        num_citations=len(passages),
        latency_s=output["latency_s"],
        retrieval_rounds=output["retrieval_rounds"],
        evidence_total=len(sample.expected_spans),
        missing_spans=sample_unmatched_spans(sample, shown),
        grounded=output["grounded"],
        llm_calls=output["llm_calls"],
        llm_ms=output["llm_ms"],
        prompt_tokens=output["prompt_tokens"],
        completion_tokens=output["completion_tokens"],
        agent_calls=[AgentToolCall.from_dict(c) for c in output["agent_calls"]],
        generation_failure=output["generation_failure"],
        missing_spans_from_search=(
            sample_unmatched_spans(sample, searched) if len(searched) < len(shown) else None
        ),
    )


def generate_into_run(run: Run, selected: EvalDataset, chat_service: ChatResponder, *,
                      started: float | None = None) -> int:
    """Answer each selected sample ``run`` has no output for yet, then mark the run complete.

    Returns how many samples were already recorded (resumed). ``started`` is
    a `time.monotonic` reading to record the run's wall time from.

    Raises:
        RunIncomplete: the outputs don't cover exactly ``selected``.
    """
    done = run.load_outputs()
    for sample in selected:
        if sample.id not in done:
            run.append_output(generate_output(sample, chat_service))
    run.complete(elapsed_s=time.monotonic() - started if started is not None else None)
    return len(done)


def run_answer_eval(
    dataset: EvalDataset,
    chat_service: ChatResponder,
    llm_client: LLMClient,
    *,
    completed: Mapping[str, AnswerSampleResult] | None = None,
    on_result: Callable[[AnswerSampleResult], None] | None = None,
) -> AnswerEvalReport:
    """Run the full pipeline + LLM judge for every sample with an expected answer.

    ``completed`` holds results from an earlier, interrupted run of the same
    setup: those samples are reused, not re-asked. ``on_result`` is called with
    each newly computed result as soon as it exists. Report order follows the
    dataset either way. A run folder generates and judges as separate steps
    instead (`rag.eval.judge`); this keeps the two together for a quick look.
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
        output = generate_output(sample, chat_service)
        results.append(answer_result(sample, output, judge_answer(sample, output, llm_client)))
        if on_result is not None:
            on_result(results[-1])
    return summarize_answers(results, skipped=skipped)


def summarize_answers(results: list[AnswerSampleResult], *, skipped: int = 0) -> AnswerEvalReport:
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
        num_empty=sum(1 for r in results if r.generation_failure or not r.actual_answer.strip()),
        mean_latency_s=sum(r.latency_s for r in results) / n if n > 0 else 0.0,
        evidence_retrieved=_bucket([r for r in results if r.evidence_retrieved is True]),
        evidence_missed=_bucket([r for r in results if r.evidence_retrieved is False]),
        num_evidence_only_from_reads=sum(1 for r in results if r.evidence_only_from_reads),
        mean_read_chars=sum(r.read_chars for r in results) / n if n > 0 else 0.0,
    )


def _bucket(results: list[AnswerSampleResult]) -> EvidenceBucket:
    return EvidenceBucket(
        num_evaluated=len(results),
        num_passed=sum(1 for r in results if r.passed is True),
    )


def groundedness_counts(verdicts: list[tuple[bool | None, bool]]) -> dict[str, int] | None:
    """How CRAG's verdicts line up with the judge's, from (grounded, passed) pairs.

    None when nothing was checked, so rows without the checker show no counts
    rather than zeros. `unchecked` also holds inconclusive checks.
    """
    checked = [(g, ok) for g, ok in verdicts if g is not None]
    if not checked:
        return None
    return {
        "checked": len(checked),
        "unchecked": len(verdicts) - len(checked),
        "ungrounded": sum(1 for g, _ in checked if g is False),
        "ungrounded_and_failed": sum(1 for g, ok in checked if g is False and not ok),
        "grounded_and_failed": sum(1 for g, ok in checked if g is True and not ok),
    }


def mean_known(values: list[int | None]) -> float | None:
    """Mean of the values a provider reported, rounded; None when none did."""
    known = [v for v in values if v is not None]
    return round(sum(known) / len(known), 1) if known else None


def trajectory(calls: list[AgentToolCall]) -> dict[str, Any]:
    """An agentic sample's trajectory, for reading a run without re-running it.

    Every tool call, its arguments as the model wrote them, and what each
    returned. Absent on pipeline samples.
    """
    return {"calls": [asdict(c) for c in calls]} if calls else {}


def answer_scores(report: AnswerEvalReport, samples: EvalDataset) -> dict[str, Any]:
    """A report as the answer matrix's row records it, and an answer run's score revision holds it.

    ``samples`` supplies each sample's ``kind``, for the per-kind rates.
    """
    kind_of = {s.id: s.extra["kind"] for s in samples if "kind" in s.extra}
    by_kind: dict[str, dict[str, int]] = {}
    for r in report.sample_results:
        if r.sample_id in kind_of:
            counts = by_kind.setdefault(kind_of[r.sample_id], {"num_evaluated": 0, "num_passed": 0})
            counts["num_evaluated"] += 1
            counts["num_passed"] += r.passed is True
    results = report.sample_results
    return {
        "num_evaluated": report.num_evaluated,
        "num_passed": report.num_passed,
        "num_failed": report.num_failed,
        "num_unparseable": report.num_unparseable,
        "num_empty": report.num_empty,
        "pass_rate": round(report.pass_rate, 4),
        "pass_ci": [round(x, 4) for x in wilson_interval(report.num_passed, report.num_evaluated)],
        "by_kind": {
            kind: {**c, "pass_ci": [round(x, 4) for x in wilson_interval(c["num_passed"], c["num_evaluated"])]}
            for kind, c in sorted(by_kind.items())
        },
        "mean_latency_s": round(report.mean_latency_s, 1),
        # The answering turn's cost, as for multi-hop; the judge's calls are not in it.
        "mean_llm_calls": round(sum(r.llm_calls for r in results) / max(len(results), 1), 2),
        "mean_prompt_tokens": mean_known([r.prompt_tokens for r in results]),
        "mean_completion_tokens": mean_known([r.completion_tokens for r in results]),
        "groundedness": groundedness_counts([(r.grounded, r.passed is True) for r in results]),
        "failed_ids": [r.sample_id for r in results if r.passed is not True],
        # Failures split by stage. Samples without gold spans (refusals, the
        # doc-matched baseline set) cannot be attributed and are counted apart.
        "failed_retrieval": report.evidence_missed.num_failed,
        "failed_generation": report.evidence_retrieved.num_failed,
        "failed_unattributed": (
            report.num_evaluated - report.num_passed
            - report.evidence_missed.num_failed - report.evidence_retrieved.num_failed
        ),
        # Per-sample detail, for paired comparison across variants.
        "samples": [
            {"id": r.sample_id, "passed": r.passed, "evidence_retrieved": r.evidence_retrieved,
             "grounded": r.grounded, "llm_calls": r.llm_calls, "latency_s": round(r.latency_s, 1),
             "answer": r.actual_answer, **trajectory(r.agent_calls)}
            for r in results
        ],
    }


def print_report(report: AnswerEvalReport, *, verbose: bool = False) -> None:
    """Print a formatted answer eval report to stdout."""
    print(f"\n{'=' * 60}")
    print(f"  Answer Eval  ({report.num_evaluated} evaluated, {report.num_skipped} skipped)")
    print(f"{'=' * 60}")
    print(f"  Pass rate   {report.pass_rate:.3f}  ({report.num_passed}/{report.num_evaluated})")
    if report.num_unparseable:
        print(f"  Unparseable verdicts: {report.num_unparseable}")
    if report.num_empty:
        print(f"  No answer (empty or generation failed): {report.num_empty}")
    print(f"  Mean latency  {report.mean_latency_s:.1f}s per turn (generation only)")
    found, missed = report.evidence_retrieved, report.evidence_missed
    if found.num_evaluated or missed.num_evaluated:
        print(f"{'-' * 60}")
        print("  By whether the gold span reached the prompt:")
        print(f"    evidence retrieved  {found.num_passed}/{found.num_evaluated} passed"
              f"  -> {found.num_failed} generation/judge failure(s)")
        print(f"    evidence missed     {missed.num_passed}/{missed.num_evaluated} passed"
              f"  -> {missed.num_failed} retrieval failure(s)")
    if report.mean_read_chars:
        print(f"  Reads  {report.mean_read_chars:,.0f} chars per turn; evidence complete only with "
              f"reads in {report.num_evidence_only_from_reads} sample(s)")
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
    add_oracle_argument(parser)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR, metavar="DIR",
                        help="Where the run folder is written (default: data/eval/runs).")
    parser.add_argument("--no-run", action="store_true",
                        help="Don't write a run folder: generate and judge together and print the report "
                             "only (and don't check the index).")
    parser.add_argument("--fresh", action="store_true",
                        help="Start a new run and judgment attempt even if an interrupted one matches.")
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
    full = EvalDataset.load(eval_path)
    if not full:
        logger.error("Eval set is empty: %s", eval_path)
        return 1

    dataset = full
    if args.limit:
        dataset = subsample(full, args.limit)
        logger.info("Subsampled %d of %d sample(s) (evenly spaced)", len(dataset), len(full))
    if not args.no_run:
        # A run holds only what can be judged; `run_answer_eval` skips the rest.
        dataset = EvalDataset(samples=[s for s in dataset if s.expected_answer], source_path=dataset.source_path)

    config = load_config(args.config)
    logger.info(
        "Building chat service (embedding=%s, llm=%s:%s)",
        config.embedding.model,
        config.llm.provider,
        config.llm.model,
    )
    # Before the chat service, so a bad judge fails before anything is loaded.
    judge_config = resolve_judge_config(config, args.judge_model, args.judge_provider)
    logger.info("Judge: %s:%s", judge_config.provider, judge_config.model)
    judge_llm = get_llm_client(judge_config)
    retriever = None
    if args.oracle:
        retriever, unfound = build_oracle_retriever(config, dataset, args.corpus)
        if unfound:
            logger.error("The oracle can't find %d gold span(s) in the chunks; see above", len(unfound))
            return 1
    chat_service = build_chat_service(config, corpora=args.corpus, retriever=retriever)

    logger.info("Running answer eval on %d sample(s)", len(dataset))
    report: AnswerEvalReport
    if args.no_run:
        report = run_answer_eval(dataset, chat_service, judge_llm)
        print_report(report, verbose=args.verbose)
    else:
        from rag.eval.judge import evaluate_answers_into_run, open_answer_run
        from rag.eval.score import stored_report

        try:
            run = open_answer_run(
                "answer", config, args.corpus, runs_dir=args.runs_dir, dataset_path=eval_path, dataset=full,
                selected=dataset, oracle=args.oracle, fresh=args.fresh,
                variant=Path(args.config).stem if args.config else "config",
            )
            attempt, revision = evaluate_answers_into_run(run, dataset, chat_service, judge_llm, judge_config,
                                                          fresh=args.fresh, started=time.monotonic())
        except (IndexConflict, RunIncomplete) as exc:
            logger.error("%s", exc)
            return 1
        stored = stored_report(run, dataset, attempt)
        assert isinstance(stored, AnswerEvalReport)
        report = stored
        print_report(report, verbose=args.verbose)
        print(f"Run: {run.path}  (judgment {attempt.ref}, score revision {revision})")

    return 0 if report.pass_rate >= 0.5 or report.num_evaluated == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
