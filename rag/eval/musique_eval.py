"""MuSiQue-Ans evaluation: short-answer EM/F1 against aliases, plus evidence recall per hop.

The outside multi-hop check for Milestone 19 (docs/public-benchmarks-plan.md,
MuSiQue follow-on, phase C). Samples come from
``scripts/musique_to_eval_set.py``: one per question, the supporting
paragraphs as gold spans and one part per decomposition step.

Scoring
-------
MuSiQue grades the final answer, with no judge, by the official EM and token
F1 (`rag.eval.answer_match`). The system answers in long form with citations,
so a fixed extraction step reduces each answer to the short answer it commits
to: one temperature-0 call to the extractor model (the judge's), which copies
the answer span or says NONE. Two checks on that step:

* ``contains`` -- whether a gold answer occurs as whole tokens anywhere in the
  long answer. It needs no extraction, so a row whose EM moves while its
  containment doesn't points at the extractor, not the answers.
* ``num_none`` -- answers the extractor found no committed answer in (the
  grounded prompt's "not in the passages"), reported apart from wrong answers.

Intermediate answers are never scored: the decomposition steps feed per-hop
evidence recall only.

Evidence
--------
Evidence recall is the share of a question's supporting paragraphs anywhere
in what the turn retrieved (its citations: the pipeline's one round, or every
search an agent ran). ``all_evidence`` is the share of questions with every
supporting paragraph found, and ``evidence_by_hop`` is the recall of step
1, 2, ... separately, which shows whether a search reached the later hops.

The closed-book responder
-------------------------
MuSiQue has been public since 2021, so the generator may know answers without
retrieval. `ClosedBookResponder` answers from the model alone; its row is the
memorization floor every retrieval row is read against.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from rag.eval.answer_match import best_exact_match, best_f1, contains_answer
from rag.eval.dataset import EvalDataset, EvalSample, ExpectedSpan
from rag.eval.relevance import unmatched_spans
from rag.events import EventSink
from rag.generation.chat_service import ChatAnswer, ChatResponder, TurnTrace
from rag.llm.base import LLMClient
from rag.generation.query_rewriter import ChatTurn
from rag.observability.records import AgentToolCall
from rag.query_filter import QueryFilter

logger = logging.getLogger(__name__)

NONE_ANSWER = "NONE"

_EXTRACT_SYSTEM_PROMPT = (
    "You read a response to a question and copy out the short answer it commits to: "
    "the entity, name, date, number or short phrase the response gives as the answer, "
    "exactly as the response writes it, with no explanation and no citation markers. "
    f"If the response gives no answer -- it says the information is not available, "
    f"or it hedges between candidates without choosing -- output {NONE_ANSWER}. "
    "Output one line only."
)

_CLOSED_BOOK_SYSTEM_PROMPT = (
    "Answer the question from your own knowledge. Think it through briefly if it "
    "needs several steps, then state the answer as a short phrase."
)


def _extract_prompt(question: str, response: str) -> str:
    return f"Question: {question}\n\nResponse:\n{response}\n\nShort answer (or {NONE_ANSWER}):"


def extract_answer(extractor: LLMClient, question: str, response: str) -> str:
    """The short answer `response` commits to, or "" when it commits to none."""
    if not response.strip():
        return ""
    out = extractor.generate(_extract_prompt(question, response), system=_EXTRACT_SYSTEM_PROMPT)
    line = next((ln.strip() for ln in out.strip().splitlines() if ln.strip()), "")
    return "" if line.strip(" .").upper() == NONE_ANSWER else line


class ClosedBookResponder(ChatResponder):
    """Answers from the model alone, with no retrieval: the memorization control.

    `llm` should be metered (`rag.observability.usage.metered_client`) so the
    row reports its calls and tokens like any other.
    """

    def __init__(self, llm: LLMClient) -> None:
        super().__init__()
        self._llm = llm

    def _answer(
        self,
        query: str,
        history: list[ChatTurn] | None,
        on_event: EventSink,
        trace: TurnTrace,
        query_filter: QueryFilter | None = None,
    ) -> ChatAnswer:
        return ChatAnswer(answer=self._llm.generate(query, system=_CLOSED_BOOK_SYSTEM_PROMPT), retrieval_attempts=0)


@dataclass
class MusiqueSampleResult:
    sample_id: str
    query: str
    #: The stratum: 2hop, 3hop or 4hop.
    kind: str
    actual_answer: str
    #: The extracted short answer; "" when the response committed to none.
    extracted: str
    em: int
    f1: float
    contains: bool
    evidence_total: int
    missing_spans: list[str]
    #: Per decomposition step, whether its supporting paragraph was retrieved.
    hop_found: list[bool]
    latency_s: float
    retrieval_rounds: int
    llm_calls: int = 0
    llm_ms: float = 0.0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    grounded: bool | None = None
    #: Agentic turns: every search call the model made, in order. Empty for a pipeline turn.
    agent_calls: list[AgentToolCall] = field(default_factory=list)

    @property
    def evidence_recall(self) -> float:
        if self.evidence_total == 0:
            return 1.0
        return (self.evidence_total - len(self.missing_spans)) / self.evidence_total

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MusiqueSampleResult:
        searches = [AgentToolCall.from_dict(s) for s in data.get("agent_calls", [])]
        return cls(**{**data, "agent_calls": searches})


@dataclass
class MusiqueReport:
    num_samples: int
    em: float
    f1: float
    contains: float
    num_none: int
    evidence_recall: float
    #: Share of questions with every supporting paragraph retrieved.
    all_evidence: float
    #: Recall of step k's supporting paragraph, k = 1, 2, ..., over questions with that many steps.
    evidence_by_hop: list[float]
    by_kind: dict[str, dict[str, float]]
    mean_latency_s: float
    mean_retrieval_rounds: float
    mean_llm_calls: float
    mean_llm_s: float
    mean_prompt_tokens: float | None
    mean_completion_tokens: float | None
    num_with_tokens: int
    sample_results: list[MusiqueSampleResult] = field(default_factory=list)


#: Seed for `stratified_subset`, so a sampled run's questions are reproducible.
SAMPLE_SEED = 19


def stratified_subset(dataset: EvalDataset, size: int, *, seed: int = SAMPLE_SEED) -> EvalDataset:
    """`size` samples with each kind's (hop count's) share kept, by largest remainder; source order.

    For the agent rows, which take too long for all of dev. Same seed and size,
    same questions, so rows sampled separately still pair question by question.
    """
    samples = list(dataset)
    if not 0 < size < len(samples):
        return dataset
    strata: dict[str, list[EvalSample]] = {}
    for s in samples:
        strata.setdefault(str(s.extra.get("kind", "unknown")), []).append(s)
    exact = {k: size * len(group) / len(samples) for k, group in strata.items()}
    quota = {k: int(v) for k, v in exact.items()}
    for k in sorted(exact, key=lambda k: (exact[k] - quota[k], k), reverse=True)[: size - sum(quota.values())]:
        quota[k] += 1
    rng = random.Random(seed)
    chosen = {s.id for k in sorted(strata) for s in rng.sample(strata[k], quota[k])}
    return EvalDataset(samples=[s for s in samples if s.id in chosen])


def golds_of(sample: EvalSample) -> list[str]:
    if not sample.expected_answer:
        raise ValueError(f"MuSiQue sample {sample.id!r} has no expected_answer")
    return [sample.expected_answer, *(str(a) for a in sample.extra.get("answer_aliases", []))]


def score_sample(
    sample: EvalSample, answer: ChatAnswer, extractor: LLMClient, latency_s: float
) -> MusiqueSampleResult:
    golds = golds_of(sample)
    extracted = extract_answer(extractor, sample.query, answer.answer)
    retrieved = [c.text for c in answer.citations]
    steps = sample.extra.get("parts") or []
    hop_found = [
        not unmatched_spans([ExpectedSpan.from_json(s) for s in step["spans"]], retrieved) for step in steps
    ]
    return MusiqueSampleResult(
        sample_id=sample.id,
        query=sample.query,
        kind=str(sample.extra.get("kind", "unknown")),
        actual_answer=answer.answer,
        extracted=extracted,
        em=best_exact_match(extracted, golds),
        f1=best_f1(extracted, golds),
        contains=contains_answer(answer.answer, golds),
        evidence_total=len(sample.expected_spans),
        missing_spans=unmatched_spans(list(sample.expected_spans), retrieved),
        hop_found=hop_found,
        latency_s=latency_s,
        retrieval_rounds=answer.retrieval_attempts,
        llm_calls=answer.llm_calls,
        llm_ms=answer.llm_ms,
        prompt_tokens=answer.prompt_tokens,
        completion_tokens=answer.completion_tokens,
        grounded=answer.grounded,
        agent_calls=answer.agent_calls,
    )


def run_musique_eval(
    dataset: EvalDataset,
    chat_service: ChatResponder,
    extractor: LLMClient,
    *,
    completed: Mapping[str, MusiqueSampleResult] | None = None,
    on_result: Callable[[MusiqueSampleResult], None] | None = None,
) -> MusiqueReport:
    """Answer every sample, extract and score its short answer, and measure evidence recall."""
    results: list[MusiqueSampleResult] = []
    for sample in dataset:
        if completed and sample.id in completed:
            results.append(completed[sample.id])
            continue
        started = time.monotonic()
        answer = chat_service.ask(sample.query)
        results.append(score_sample(sample, answer, extractor, time.monotonic() - started))
        if on_result is not None:
            on_result(results[-1])
    return summarize(results)


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _known_avg(values: list[int | None]) -> float | None:
    known = [v for v in values if v is not None]
    return sum(known) / len(known) if known else None


def summarize(results: list[MusiqueSampleResult]) -> MusiqueReport:
    by_kind: dict[str, list[MusiqueSampleResult]] = {}
    for r in results:
        by_kind.setdefault(r.kind, []).append(r)
    depth = max((len(r.hop_found) for r in results), default=0)
    return MusiqueReport(
        num_samples=len(results),
        em=_avg([float(r.em) for r in results]),
        f1=_avg([r.f1 for r in results]),
        contains=_avg([float(r.contains) for r in results]),
        num_none=sum(1 for r in results if not r.extracted),
        evidence_recall=_avg([r.evidence_recall for r in results]),
        all_evidence=_avg([float(not r.missing_spans) for r in results]),
        evidence_by_hop=[
            _avg([float(r.hop_found[k]) for r in results if len(r.hop_found) > k]) for k in range(depth)
        ],
        by_kind={
            kind: {
                "n": len(rs),
                "em": _avg([float(r.em) for r in rs]),
                "f1": _avg([r.f1 for r in rs]),
                "contains": _avg([float(r.contains) for r in rs]),
                "evidence_recall": _avg([r.evidence_recall for r in rs]),
                "all_evidence": _avg([float(not r.missing_spans) for r in rs]),
            }
            for kind, rs in sorted(by_kind.items())
        },
        mean_latency_s=_avg([r.latency_s for r in results]),
        mean_retrieval_rounds=_avg([float(r.retrieval_rounds) for r in results]),
        mean_llm_calls=_avg([float(r.llm_calls) for r in results]),
        mean_llm_s=_avg([r.llm_ms / 1000 for r in results]),
        mean_prompt_tokens=_known_avg([r.prompt_tokens for r in results]),
        mean_completion_tokens=_known_avg([r.completion_tokens for r in results]),
        num_with_tokens=sum(1 for r in results if r.prompt_tokens is not None or r.completion_tokens is not None),
        sample_results=results,
    )
