#!/usr/bin/env python
"""Run an answer-eval matrix over CRAG configurations and record the results.

CRAG (Milestone 10) shipped functionally verified and numerically unmeasured.
This measures it the only way it can be measured: end to end, with an LLM judge,
because every one of its three checks is a judgment about generated output rather
than about retrieval.

Why two eval sets
-----------------
CRAG's document grader exists to reject passages that clear the similarity floor
without answering the question. The retrieval matrix showed `retrieval.min_score`
is inert on this corpus -- 0.0, 0.1 and 0.3 gave identical results -- so if
anything is filtering irrelevant passages, it has to be the grader.

That makes the **refusal set** the sharp test, not the main set:

* `edgar_eval_set.json` -- answerable questions. CRAG can only *hurt* here, by
  grading out a passage that was actually needed. This measures the cost.
* `edgar_refusal_set.json` -- hard negatives, where declining is the correct
  answer. This measures the benefit.

A CRAG config that improves refusals while leaving answerable questions alone is
a win; one that improves refusals by refusing everything is not, which is why
both numbers are always reported together.

Cost
----
`grade_documents` is one LLM call per retrieved passage (`rerank_top_k`, so 5),
on a path that otherwise costs one generation plus one judge call. Enabling it
roughly quadruples the per-sample cost. Use `--limit` on the answerable set.

Stopping and resuming
---------------------
Each finished sample is saved to `<results-dir>/.partial/` as it completes, so
rerunning the same command after a crash or Ctrl-C picks up where it stopped.
A checkpoint written by a different config, judge, dataset or code is refused
rather than mixed in; `--fresh` discards it. The results file itself is only
written when a set finishes.

The judge
---------
Resolved **once**, from the base config (`eval.judge`, else `llm`) plus
`--judge-model`, and shared by every variant -- so a variant that changes
`llm.model` changes the generator and never the judge. Before Milestone 19 the
judge was rebuilt from each variant's own `llm`, which made any generator
comparison self-graded (see docs/measured-results.md).

Results are filed per judge: `answer_<corpus>.json` when the judge is the
generator (every result recorded before Milestone 19), and
`answer_<corpus>__judge-<model>.json` otherwise, so rows graded by different
judges never merge into one table.

Reading the table
-----------------
Answerable pass rate is shown with a paired 95% interval against the first
variant (`crag=off`) and the win/loss count of questions that flipped -- see
:mod:`rag.eval.paired`. Failures are split into **retrieval** (the gold span
never reached the prompt) and **generation** (it did, and the answer still
failed), because the two need different fixes.

The multi-hop set
-----------------
`edgar_multihop_set.json` asks about several companies or periods per
question. It is judged part by part (see `rag.eval.multihop_eval`) and reports
complete-and-correct rate, mean completeness and evidence recall -- the
baseline the Milestone 19 agent has to beat.

The chunking plan's tiers
-------------------------
`--sets period,underspecified` adds the two tiers frozen in Phase 0 of
docs/chunking-indexing-plan.md. They are opt-in, always run in full
(`--limit` applies to the answerable set only), and are reported in their own
table, never averaged into the answerable rate. `underspecified` is also
broken down by `kind` (implicit, paraphrase). Each pass rate carries a 95%
Wilson interval, the tier's noise floor on its own.

Milestone 19 phase 4
--------------------
`--family m19` swaps the CRAG variants for the phase 4 matrix in
docs/milestone-19-plan.md: the pipeline baseline, the oracle ceiling, and the
agent under each strategy and model. It writes to `data/eval/results_m19/` by
default, so its baseline can never be paired against a CRAG row recorded on the
old labels. Every row pins its generator, so a row's name stays true if the
config's default model changes. The agent rows also pin the agent model's
`max_tokens` and timeout, so they differ from each other in one factor only.

**The oracle** answers from each question's gold chunks instead of retrieval
(`rag.eval.oracle`). It skips any set without gold spans (the refusal set).

**Repeats.** The 9b samples at temperature 0.2, and the phase 0 run moved by
+/-2 multi-hop questions between identical runs. `--repeat N` runs every
variant N times as separate rows (`name #1` ... `name #N`), each with its own
checkpoint, and adds a table of the spread across them.

Usage
-----
    python scripts/run_answer_matrix.py --corpus edgar --limit 40
    python scripts/run_answer_matrix.py --corpus edgar --variant crag=off \
        --judge-model gemma4:31b-mlx
    python scripts/run_answer_matrix.py --family m19 --corpus edgar --limit 40 \
        --judge-model gemma4:31b-mlx --variant "pipeline / 9b" --repeat 3
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, get_args

from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.config.settings import LLMConfig, RagConfig, load_config  # noqa: E402
from rag.eval.answer_eval import (  # noqa: E402
    AnswerSampleResult,
    add_judge_arguments,
    resolve_judge_config,
    run_answer_eval,
    subsample,
)
from rag.eval.checkpoint import (  # noqa: E402
    CheckpointMismatch,
    SampleCheckpoint,
    code_version,
    digest,
)
from rag.eval.dataset import EvalDataset  # noqa: E402
from rag.eval.metrics import wilson_interval  # noqa: E402
from rag.eval.multihop_eval import MultihopSampleResult, run_multihop_eval  # noqa: E402
from rag.eval.oracle import build_oracle_retriever  # noqa: E402
from rag.eval.paired import compare_by_id, format_difference  # noqa: E402
from rag.generation.builder import build_chat_service  # noqa: E402
from rag.generation.chat_service import ChatResponder  # noqa: E402
from rag.generation.factory import get_llm_client  # noqa: E402
from rag.generation.llm import LLMClient  # noqa: E402
from rag.logging_config import configure_logging  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_ANSWERABLE = Path("data/eval/edgar_eval_set.json")
DEFAULT_REFUSALS = Path("data/eval/edgar_refusal_set.json")
DEFAULT_MULTIHOP = Path("data/eval/edgar_multihop_set.json")
DEFAULT_PERIOD = Path("data/eval/edgar_period_set.json")
DEFAULT_UNDERSPECIFIED = Path("data/eval/edgar_underspecified_set.json")
#: What runs without `--sets`, as before the tiers existed.
DEFAULT_SETS = ("answerable", "refusals", "multihop")
#: The chunking plan's tiers: single-answer like `answerable`, reported apart.
TIER_SETS = ("period", "underspecified")
SETS = DEFAULT_SETS + TIER_SETS


@dataclass
class Variant:
    name: str
    overrides: dict[str, Any] = field(default_factory=dict)
    #: Answer from the gold chunks instead of retrieval (see `rag.eval.oracle`).
    oracle: bool = False


# Each check is isolated so a difference is attributable to one mechanism.
# `max_retries: 0` in the isolated grader variant keeps the retry loop out of the
# picture -- otherwise "grading helped" and "searching again after grading
# emptied the results helped" are indistinguishable.
VARIANTS: list[Variant] = [
    Variant("crag=off", {"crag.enabled": False}),
    Variant("crag=on (all)", {
        "crag.enabled": True,
        "crag.grade_documents": True,
        "crag.check_groundedness": True,
        "crag.max_retries": 1,
    }),
    Variant("crag=grade only", {
        "crag.enabled": True,
        "crag.grade_documents": True,
        "crag.check_groundedness": False,
        "crag.max_retries": 0,
    }),
    Variant("crag=grade+retry", {
        "crag.enabled": True,
        "crag.grade_documents": True,
        "crag.check_groundedness": False,
        "crag.max_retries": 1,
    }),
    Variant("crag=groundedness only", {
        "crag.enabled": True,
        "crag.grade_documents": False,
        "crag.check_groundedness": True,
        "crag.max_retries": 0,
    }),
    # Chunking plan Phase 2: the deterministic chunk header, from the index
    # data/eval/config_header.yaml builds. It changes retrieval and the prompt
    # together (the header replaces the file name as the passage's source
    # label), so this is the pipeline-level effect. CRAG off, pairing with
    # `crag=off`.
    Variant("header=on", {
        "crag.enabled": False,
        "paths.index_dir": "data/index_header",
        "chunking.header.template": "{company} ({ticker}) {form}, period ended {period_end}",
        # Pinned: the shipped default turned this on after this row was measured.
        "reranker.include_header": False,
    }),
    # The same, with the cross-encoder scoring the header too.
    Variant("header=on rerank_header", {
        "crag.enabled": False,
        "paths.index_dir": "data/index_header",
        "chunking.header.template": "{company} ({ticker}) {form}, period ended {period_end}",
        "reranker.include_header": True,
    }),
]


# Milestone 19 phase 4 (docs/milestone-19-plan.md). One factor per row against
# the row above it. The generator is pinned on every row. The agent rows pin
# their own model with a larger `max_tokens` (the 27b's reasoning trace counts
# against it) and the agent's timeout, the same on every agent row, so 9b vs 27b
# is the model alone. Pipeline answers average ~184 generated tokens, far under
# either cap.
_GENERATOR_9B = "qwen3.5:9b-mlx"
_AGENT_9B = {"model": _GENERATOR_9B, "max_tokens": 4096, "timeout_s": 600}
_AGENT_27B = {"model": "qwen3.8:27b-mlx", "max_tokens": 4096, "timeout_s": 600}
_PIPELINE = {"chat.mode": "pipeline", "crag.enabled": False, "llm.model": _GENERATOR_9B}
_AGENTIC = {"chat.mode": "agentic", "crag.enabled": False, "llm.model": _GENERATOR_9B}

M19_VARIANTS: list[Variant] = [
    Variant("pipeline / 9b", _PIPELINE),
    Variant("oracle / 9b", _PIPELINE, oracle=True),
    Variant("agentic react / 9b", {**_AGENTIC, "agent.strategy": "react", "agent.llm": _AGENT_9B}),
    Variant("agentic planned / 9b", {**_AGENTIC, "agent.strategy": "planned", "agent.llm": _AGENT_9B}),
    Variant("agentic react / 27b", {**_AGENTIC, "agent.strategy": "react", "agent.llm": _AGENT_27B}),
    Variant("agentic react / 27b, think=low", {
        **_AGENTIC, "agent.strategy": "react", "agent.llm": {**_AGENT_27B, "think": "low"},
    }),
    # The agent's checker reports a verdict and changes nothing, so this row's
    # answers match the plain 27b row's up to sampling; what it measures is
    # whether the verdicts flag the answers the judge failed.
    Variant("agentic react / 27b + groundedness", {
        **_AGENTIC, "agent.strategy": "react", "agent.llm": _AGENT_27B,
        "crag.enabled": True, "crag.check_groundedness": True,
        "crag.grade_documents": False, "crag.max_retries": 0,
    }),
]

FAMILIES: dict[str, list[Variant]] = {"crag": VARIANTS, "m19": M19_VARIANTS}
#: Each family writes apart, so a row is only ever paired within its own family.
DEFAULT_RESULTS_DIRS = {"crag": Path("data/eval/results"), "m19": Path("data/eval/results_m19")}


def apply_overrides(config: RagConfig, overrides: dict[str, Any]) -> RagConfig:
    """Set each dotted path. A dict value for a nested model field is validated into it.

    The dict is merged over the field's current value, or builds a fresh model
    when the field is unset (`agent.llm: null`), so `{"model": ...}` means
    "that model, with every other setting at its default".
    """
    updated = config.model_copy(deep=True)
    for path, value in overrides.items():
        target: Any = updated
        parts = path.split(".")
        for part in parts[:-1]:
            target = getattr(target, part)
        name = parts[-1]
        if isinstance(value, dict):
            current = getattr(target, name)
            if isinstance(current, BaseModel):
                value = type(current).model_validate({**current.model_dump(exclude_unset=True), **value})
            else:
                annotation = type(target).model_fields[name].annotation
                model = next(t for t in (annotation, *get_args(annotation))
                             if isinstance(t, type) and issubclass(t, BaseModel))
                value = model.model_validate(value)
        setattr(target, name, value)
    return updated


def repeated(variants: list[Variant], times: int) -> list[Variant]:
    """Each variant `times` times, as `name #1` ... `name #N`; unchanged when `times` is 1."""
    if times <= 1:
        return variants
    return [Variant(f"{v.name} #{k}", v.overrides, v.oracle) for v in variants for k in range(1, times + 1)]


def base_name(name: str) -> str:
    """A repeat's variant name without its `#k`."""
    return re.sub(r" #\d+$", "", name)


def generator_of(config: RagConfig) -> str:
    """The model that writes the answer: the agent's under `chat.mode: agentic`."""
    llm = (config.agent.llm or config.llm) if config.chat.mode == "agentic" else config.llm
    return f"{llm.provider}:{llm.model}"


def mode_of(config: RagConfig) -> str:
    return f"agentic:{config.agent.strategy}" if config.chat.mode == "agentic" else "pipeline"


def checkpoint_fingerprint(
    config: RagConfig, judge: LLMConfig, dataset: EvalDataset, corpus: str, code: str,
    *, oracle: bool = False,
) -> dict[str, str]:
    """What a set's result depends on; a checkpoint is resumed only if all of it matches."""
    fingerprint = {
        "config": digest(config.model_dump(mode="json")),
        "judge": digest(judge.model_dump(mode="json")),
        "dataset": digest([s.to_dict() for s in dataset]),
        "corpus": corpus,
        "code": code,
    }
    if oracle:
        # The oracle row's config equals the pipeline row's; this keeps them apart.
        fingerprint["retrieval"] = "oracle"
    return fingerprint


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


def _mean_known(values: list[int | None]) -> float | None:
    known = [v for v in values if v is not None]
    return round(sum(known) / len(known), 1) if known else None


def run_one(
    label: str,
    chat_service: ChatResponder,
    judge: LLMClient,
    dataset: EvalDataset,
    checkpoint: SampleCheckpoint | None = None,
) -> dict[str, Any]:
    completed = {
        sid: AnswerSampleResult.from_dict(r) for sid, r in (checkpoint.load() if checkpoint else {}).items()
    }
    if completed:
        logger.info("  %s: resuming, %d sample(s) from the checkpoint", label, len(completed))
    started = time.monotonic()
    report = run_answer_eval(
        dataset, chat_service, judge, completed=completed,
        on_result=(lambda r: checkpoint.append(r.to_dict())) if checkpoint else None,
    )
    elapsed = time.monotonic() - started
    logger.info(
        "  %s: %d/%d passed (%.3f) in %.0fs",
        label, report.num_passed, report.num_evaluated, report.pass_rate, elapsed,
    )
    kind_of = {s.id: s.extra["kind"] for s in dataset if "kind" in s.extra}
    by_kind: dict[str, dict[str, int]] = {}
    for r in report.sample_results:
        if r.sample_id in kind_of:
            counts = by_kind.setdefault(kind_of[r.sample_id], {"num_evaluated": 0, "num_passed": 0})
            counts["num_evaluated"] += 1
            counts["num_passed"] += r.passed is True
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
        "mean_llm_calls": round(sum(r.llm_calls for r in report.sample_results)
                                / max(len(report.sample_results), 1), 2),
        "mean_prompt_tokens": _mean_known([r.prompt_tokens for r in report.sample_results]),
        "mean_completion_tokens": _mean_known([r.completion_tokens for r in report.sample_results]),
        "groundedness": groundedness_counts(
            [(r.grounded, r.passed is True) for r in report.sample_results]
        ),
        # elapsed_s covers this session only; resumed samples ran in an earlier one.
        "elapsed_s": round(elapsed, 1),
        "resumed_samples": len(completed),
        "failed_ids": [r.sample_id for r in report.sample_results if r.passed is not True],
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
             "answer": r.actual_answer}
            for r in report.sample_results
        ],
    }


def paired_pass_delta(result: dict[str, Any], baseline: dict[str, Any]) -> str:
    """Pass-rate difference vs `baseline`, paired by sample id; unparseable counts as a fail."""
    delta = result["pass_rate"] - baseline["pass_rate"]
    if "samples" not in result or "samples" not in baseline:
        return f"{delta:+.3f} (no CI)"

    def scores(run: dict[str, Any]) -> dict[str, float]:
        return {s["id"]: 1.0 if s["passed"] is True else 0.0 for s in run["samples"]}

    try:
        diff = compare_by_id(scores(baseline), scores(result))
    except ValueError:
        return f"{delta:+.3f} (unpaired)"
    return format_difference(diff, binary=True)


def run_multihop(
    chat_service: ChatResponder,
    judge: LLMClient,
    dataset: EvalDataset,
    checkpoint: SampleCheckpoint | None = None,
) -> dict[str, Any]:
    completed = {
        sid: MultihopSampleResult.from_dict(r) for sid, r in (checkpoint.load() if checkpoint else {}).items()
    }
    if completed:
        logger.info("  multihop: resuming, %d sample(s) from the checkpoint", len(completed))
    started = time.monotonic()
    report = run_multihop_eval(
        dataset, chat_service, judge, completed=completed,
        on_result=(lambda r: checkpoint.append(r.to_dict())) if checkpoint else None,
    )
    elapsed = time.monotonic() - started
    logger.info(
        "  multihop: %.3f complete, %.3f completeness, %.3f evidence recall in %.0fs",
        report.complete_rate, report.mean_completeness, report.evidence_recall, elapsed,
    )
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
        "elapsed_s": round(elapsed, 1),
        "resumed_samples": len(completed),
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
            }
            for r in report.sample_results
        ],
    }


def _round_or_none(value: float | None) -> float | None:
    return None if value is None else round(value, 1)


def _tokens_cell(m: dict[str, Any]) -> str:
    """Prompt / generated tokens per turn; "—" when no sample reported them."""
    if m["num_with_tokens"] == 0:
        return "—"
    prompt, gen = m["mean_prompt_tokens"], m["mean_completion_tokens"]
    cell = f"{'?' if prompt is None else f'{prompt:,.0f}'} / {'?' if gen is None else f'{gen:,.0f}'}"
    if m["num_with_tokens"] < m["num_evaluated"]:
        cell += f" ({m['num_with_tokens']} of {m['num_evaluated']})"
    return cell


def judge_suffix(judge: LLMConfig, generator: LLMConfig) -> str:
    """Filename suffix naming the judge; empty for the legacy self-judged results."""
    if (judge.provider, judge.model) == (generator.provider, generator.model):
        return ""
    return "__judge-" + re.sub(r"[^A-Za-z0-9.]+", "-", judge.model).strip("-")


def render_table(results: list[dict[str, Any]]) -> str:
    reference = next((r for r in results if r.get("answerable")), None)
    ref_name = f"`{reference['variant']}`" if reference else "first"
    lines = [
        f"| variant | answerable pass | Δ vs {ref_name} [95% CI] | fails: retrieval / generation "
        "| n | s | refusal pass | n | s |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        a, f = r.get("answerable"), r.get("refusals")
        if a:
            delta = (
                paired_pass_delta(a, reference["answerable"])
                if reference and r is not reference else "—"
            )
            split = "—"
            if "failed_retrieval" in a:
                split = f"{a['failed_retrieval']} / {a['failed_generation']}"
                if a["failed_unattributed"]:
                    split += f" (+{a['failed_unattributed']} n/a)"
            a_cell = (f"{a['pass_rate']:.3f} | {delta} | {split} | {a['num_evaluated']} "
                      f"| {a['elapsed_s']:.0f}")
        else:
            a_cell = "— | — | — | — | —"
        f_cell = f"{f['pass_rate']:.3f} | {f['num_evaluated']} | {f['elapsed_s']:.0f}" if f else "— | — | —"
        lines.append(f"| `{r['variant']}` | {a_cell} | {f_cell} |")

    tiers = [(r, name) for r in results for name in TIER_SETS if r.get(name)]
    if tiers:
        lines += [
            "",
            "Chunking-plan tiers, each reported on its own (CI: 95% Wilson interval "
            "on that rate alone):",
            "",
            "| variant | set | pass | 95% CI | fails: retrieval / generation | n | s |",
            "|---|---|---|---|---|---|---|",
        ]
        for r, name in tiers:
            t = r[name]
            split = f"{t['failed_retrieval']} / {t['failed_generation']}"
            if t["failed_unattributed"]:
                split += f" (+{t['failed_unattributed']} n/a)"
            rows = [(name, t)] + [(f"{name}: {k}", c) for k, c in t.get("by_kind", {}).items()]
            for label, c in rows:
                low, high = c["pass_ci"]
                # Stage split and time belong to the whole set, not one kind.
                whole = c is t
                elapsed = f"{t['elapsed_s']:.0f}" if whole else "—"
                lines.append(
                    f"| `{r['variant']}` | {label} | {c['num_passed'] / c['num_evaluated']:.3f} "
                    f"| [{low:.3f}, {high:.3f}] | {split if whole else '—'} | {c['num_evaluated']} "
                    f"| {elapsed} |"
                )

    multihop = [r for r in results if r.get("multihop")]
    if multihop:
        lines += [
            "",
            "| variant | multi-hop complete | completeness | evidence recall | n | s/turn "
            "| LLM calls/turn | LLM s/turn | prompt / gen tokens/turn |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for r in multihop:
            m = r["multihop"]
            # Rows recorded before the cost columns existed show "—" for them.
            cost = (
                f"{m['mean_llm_calls']:.1f} | {m['mean_llm_s']:.1f} | {_tokens_cell(m)}"
                if "mean_llm_calls" in m else "— | — | —"
            )
            lines.append(
                f"| `{r['variant']}` | {m['complete_rate']:.3f} | {m['mean_completeness']:.3f} "
                f"| {m['evidence_recall']:.3f} | {m['num_evaluated']} | {m['mean_latency_s']:.1f} "
                f"| {cost} |"
            )

    checked = [(r, name, r[name]["groundedness"]) for r in results for name in SETS
               if r.get(name) and r[name].get("groundedness")]
    if checked:
        lines += [
            "",
            "Groundedness verdicts against the judge (a fail is a judged FAIL, or an "
            "incomplete multi-hop answer):",
            "",
            "| variant | set | checked | flagged ungrounded | flagged & failed | passed check & failed "
            "| unchecked |",
            "|---|---|---|---|---|---|---|",
        ]
        for r, name, g in checked:
            lines.append(
                f"| `{r['variant']}` | {name} | {g['checked']} | {g['ungrounded']} "
                f"| {g['ungrounded_and_failed']} | {g['grounded_and_failed']} | {g['unchecked']} |"
            )

    spread = render_repeats(results)
    if spread:
        lines += ["", spread]
    return "\n".join(lines)


def render_repeats(results: list[dict[str, Any]]) -> str:
    """Per variant run more than once: each run's headline count, then their mean and range."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        if base_name(r["variant"]) != r["variant"]:
            groups.setdefault(base_name(r["variant"]), []).append(r)
    rows = []
    for name, runs in groups.items():
        for set_name in SETS:
            counts = [
                (run[set_name]["num_passed"] if "num_passed" in run[set_name]
                 else round(run[set_name]["complete_rate"] * run[set_name]["num_evaluated"]),
                 run[set_name]["num_evaluated"])
                for run in runs if run.get(set_name)
            ]
            if len(counts) < 2:
                continue
            passed = [p for p, _ in counts]
            label = "complete" if set_name == "multihop" else "pass"
            rows.append(
                f"| `{name}` | {set_name} {label} | {', '.join(f'{p}/{n}' for p, n in counts)} "
                f"| {sum(passed) / len(passed):.1f} | {max(passed) - min(passed)} |"
            )
    if not rows:
        return ""
    return "\n".join([
        "Spread across repeats (counts per run; a difference between variants smaller than "
        "the range is within run-to-run noise):",
        "",
        "| variant | set | per run | mean | range |",
        "|---|---|---|---|---|",
        *rows,
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description="Answer-eval matrix: CRAG or Milestone 19 variants over the answerable, refusal and multi-hop sets.")
    parser.add_argument("--family", choices=sorted(FAMILIES), default="crag",
                        help="Which variant list to run: the CRAG matrix, or Milestone 19 phase 4's.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--corpus", action="append", default=None, metavar="NAME")
    parser.add_argument("--answerable", type=Path, default=DEFAULT_ANSWERABLE)
    parser.add_argument("--refusals", type=Path, default=DEFAULT_REFUSALS)
    parser.add_argument("--multihop", type=Path, default=DEFAULT_MULTIHOP)
    parser.add_argument("--period", type=Path, default=DEFAULT_PERIOD)
    parser.add_argument("--underspecified", type=Path, default=DEFAULT_UNDERSPECIFIED)
    parser.add_argument("--sets", default=",".join(DEFAULT_SETS),
                        help=f"Comma-separated subset of {', '.join(SETS)} to run "
                             f"(default: {','.join(DEFAULT_SETS)}).")
    add_judge_arguments(parser, note=" Fixed across all variants.")
    parser.add_argument("--limit", type=int, default=40,
                        help="Answerable samples to evaluate (evenly spaced). 0 = all.")
    parser.add_argument("--results-dir", type=Path, default=None,
                        help="Default: data/eval/results (crag) or data/eval/results_m19 (m19).")
    parser.add_argument("--repeat", type=int, default=1, metavar="N",
                        help="Run each variant N times, as separate rows, to measure run-to-run noise.")
    parser.add_argument("--fresh", action="store_true",
                        help="Discard saved per-sample checkpoints instead of resuming from them.")
    parser.add_argument("--variant", action="append", default=None, metavar="NAME")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    configure_logging()

    family = FAMILIES[args.family]
    if args.results_dir is None:
        args.results_dir = DEFAULT_RESULTS_DIRS[args.family]
    variants = family
    if args.variant:
        wanted = set(args.variant)
        if unknown_variants := wanted - {v.name for v in family}:
            parser.error(f"unknown variant(s) in family {args.family}: {', '.join(sorted(unknown_variants))}")
        variants = [v for v in variants if v.name in wanted]
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")
    variants = repeated(variants, args.repeat)
    if args.list:
        for v in variants:
            print(f"  {v.name:<36} {'[oracle] ' if v.oracle else ''}{v.overrides}")
        return 0

    sets = [name.strip() for name in args.sets.split(",") if name.strip()]
    if unknown := set(sets) - set(SETS):
        parser.error(f"unknown set(s): {', '.join(sorted(unknown))}")

    base = load_config(args.config)
    judge_config = resolve_judge_config(base, args.judge_model, args.judge_provider)
    judge = get_llm_client(judge_config)
    paths = {
        "answerable": args.answerable, "refusals": args.refusals, "multihop": args.multihop,
        "period": args.period, "underspecified": args.underspecified,
    }
    datasets = {name: EvalDataset.load(paths[name]) for name in sets}
    if "answerable" in datasets:
        datasets["answerable"] = subsample(datasets["answerable"], args.limit)
    selection = base.corpus_selection(args.corpus)
    logger.info(
        "Answer matrix: %d variant(s), sets %s, corpus %s, judge %s:%s",
        len(variants),
        ", ".join(f"{name}={len(datasets[name])}" for name in sets),
        selection.describe(), judge_config.provider, judge_config.model,
    )

    args.results_dir.mkdir(parents=True, exist_ok=True)
    stem = f"answer_{selection.slug}{judge_suffix(judge_config, base.llm)}"
    destination = args.results_dir / f"{stem}.json"
    accumulated: dict[str, dict[str, Any]] = {}
    if destination.exists():
        accumulated = {r["variant"]: r for r in json.loads(destination.read_text())["results"]}

    def flush() -> list[dict[str, Any]]:
        order = {v.name: i for i, v in enumerate(family)}
        ordered = sorted(accumulated.values(), key=lambda r: (order.get(base_name(r["variant"]), 999), r["variant"]))
        destination.write_text(json.dumps({
            "corpus": selection.describe(),
            "collection": selection.collection_name,
            "judge": f"{judge_config.provider}:{judge_config.model}",
            "answerable_set": str(args.answerable),
            "refusal_set": str(args.refusals),
            "multihop_set": str(args.multihop),
            "period_set": str(args.period),
            "underspecified_set": str(args.underspecified),
            "results": ordered,
        }, indent=2) + "\n")
        (args.results_dir / f"{stem}.md").write_text(
            f"# Answer eval — {selection.describe()}, judge "
            f"`{judge_config.model}`\n\n{render_table(ordered)}\n"
        )
        return ordered

    # Each finished sample is appended to a checkpoint under .partial/, so a
    # stopped run resumes where it left off. Every checkpoint is checked
    # against its set's fingerprint before anything runs: a mismatch found
    # after an hour of earlier variants would waste that hour.
    code = code_version()
    configs = {v.name: apply_overrides(base, v.overrides) for v in variants}
    # The oracle needs gold spans, so it skips any set without them (refusals).
    sets_for = {
        v.name: [n for n in sets if not v.oracle or all(s.expected_spans for s in datasets[n])]
        for v in variants
    }
    for v in variants:
        if skipped := [n for n in sets if n not in sets_for[v.name]]:
            logger.info("[%s] skipping %s: the oracle needs gold spans", v.name, ", ".join(skipped))
    checkpoints: dict[tuple[str, str], SampleCheckpoint] = {}
    for variant in variants:
        for name in sets_for[variant.name]:
            variant_slug = re.sub(r"[^A-Za-z0-9.]+", "-", variant.name).strip("-")
            checkpoint = SampleCheckpoint(
                args.results_dir / ".partial" / f"{stem}__{variant_slug}__{name}.jsonl",
                checkpoint_fingerprint(configs[variant.name], judge_config, datasets[name],
                                       selection.slug, code, oracle=variant.oracle),
            )
            if args.fresh:
                checkpoint.discard()
            try:
                checkpoint.load()
            except CheckpointMismatch as exc:
                logger.error("%s", exc)
                return 1
            checkpoints[variant.name, name] = checkpoint

    ordered: list[dict[str, Any]] = []
    for variant in variants:
        logger.info("[%s] %s", variant.name, variant.overrides)
        config = configs[variant.name]
        if variant.oracle:
            retriever, unfound = build_oracle_retriever(
                config, [s for name in sets_for[variant.name] for s in datasets[name]], args.corpus,
            )
            if unfound:
                logger.error("[%s] %d gold span(s) are in no chunk; see above", variant.name, len(unfound))
                return 1
            chat_service = build_chat_service(config, corpora=args.corpus, retriever=retriever)
        else:
            chat_service = build_chat_service(config, corpora=args.corpus)
        # Merge into the stored row, so `--sets multihop` adds a column to a
        # variant rather than discarding its answerable/refusal numbers.
        record: dict[str, Any] = {
            **accumulated.get(variant.name, {}),
            "variant": variant.name,
            "overrides": variant.overrides,
            "generator": generator_of(config),
            "mode": mode_of(config),
            "retrieval": "oracle" if variant.oracle else "configured",
        }
        for name in sets_for[variant.name]:
            checkpoint = checkpoints[variant.name, name]
            if name == "multihop":
                record[name] = run_multihop(chat_service, judge, datasets[name], checkpoint)
            else:
                record[name] = run_one(name, chat_service, judge, datasets[name], checkpoint)
            accumulated[variant.name] = record
            # Write the finished set, then drop its checkpoint: the results file
            # only ever holds whole sets, and a crash between the two just
            # resumes into an immediate re-summary.
            ordered = flush()
            checkpoint.discard()

    print(f"\n{render_table(ordered)}\n")
    print(f"Wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
