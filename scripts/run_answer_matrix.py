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

Usage
-----
    python scripts/run_answer_matrix.py --corpus edgar --limit 40
    python scripts/run_answer_matrix.py --corpus edgar --variant crag=off \
        --judge-model gemma4:31b-mlx
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
from typing import Any

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
from rag.eval.multihop_eval import MultihopSampleResult, run_multihop_eval  # noqa: E402
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
SETS = ("answerable", "refusals", "multihop")


@dataclass
class Variant:
    name: str
    overrides: dict[str, Any] = field(default_factory=dict)


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
]


def apply_overrides(config: RagConfig, overrides: dict[str, Any]) -> RagConfig:
    updated = config.model_copy(deep=True)
    for path, value in overrides.items():
        target: Any = updated
        parts = path.split(".")
        for part in parts[:-1]:
            target = getattr(target, part)
        setattr(target, parts[-1], value)
    return updated


def checkpoint_fingerprint(
    config: RagConfig, judge: LLMConfig, dataset: EvalDataset, corpus: str, code: str
) -> dict[str, str]:
    """What a set's result depends on; a checkpoint is resumed only if all of it matches."""
    return {
        "config": digest(config.model_dump(mode="json")),
        "judge": digest(judge.model_dump(mode="json")),
        "dataset": digest([s.to_dict() for s in dataset]),
        "corpus": corpus,
        "code": code,
    }


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
    return {
        "num_evaluated": report.num_evaluated,
        "num_passed": report.num_passed,
        "num_failed": report.num_failed,
        "num_unparseable": report.num_unparseable,
        "num_empty": report.num_empty,
        "pass_rate": round(report.pass_rate, 4),
        "mean_latency_s": round(report.mean_latency_s, 1),
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
            {"id": r.sample_id, "passed": r.passed, "evidence_retrieved": r.evidence_retrieved}
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
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Answer-eval matrix: CRAG variants over the answerable, refusal and multi-hop sets.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--corpus", action="append", default=None, metavar="NAME")
    parser.add_argument("--answerable", type=Path, default=DEFAULT_ANSWERABLE)
    parser.add_argument("--refusals", type=Path, default=DEFAULT_REFUSALS)
    parser.add_argument("--multihop", type=Path, default=DEFAULT_MULTIHOP)
    parser.add_argument("--sets", default=",".join(SETS),
                        help=f"Comma-separated subset of {', '.join(SETS)} to run.")
    add_judge_arguments(parser, note=" Fixed across all variants.")
    parser.add_argument("--limit", type=int, default=40,
                        help="Answerable samples to evaluate (evenly spaced). 0 = all.")
    parser.add_argument("--results-dir", type=Path, default=Path("data/eval/results"))
    parser.add_argument("--fresh", action="store_true",
                        help="Discard saved per-sample checkpoints instead of resuming from them.")
    parser.add_argument("--variant", action="append", default=None, metavar="NAME")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    configure_logging()

    variants = VARIANTS
    if args.variant:
        wanted = set(args.variant)
        variants = [v for v in variants if v.name in wanted]
    if args.list:
        for v in variants:
            print(f"  {v.name:<24} {v.overrides}")
        return 0

    sets = [name.strip() for name in args.sets.split(",") if name.strip()]
    if unknown := set(sets) - set(SETS):
        parser.error(f"unknown set(s): {', '.join(sorted(unknown))}")

    base = load_config(args.config)
    judge_config = resolve_judge_config(base, args.judge_model, args.judge_provider)
    judge = get_llm_client(judge_config)
    datasets = {
        "answerable": subsample(EvalDataset.load(args.answerable), args.limit),
        "refusals": EvalDataset.load(args.refusals),
        "multihop": EvalDataset.load(args.multihop),
    }
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
        order = {v.name: i for i, v in enumerate(VARIANTS)}
        ordered = sorted(accumulated.values(), key=lambda r: order.get(r["variant"], 999))
        destination.write_text(json.dumps({
            "corpus": selection.describe(),
            "collection": selection.collection_name,
            "judge": f"{judge_config.provider}:{judge_config.model}",
            "answerable_set": str(args.answerable),
            "refusal_set": str(args.refusals),
            "multihop_set": str(args.multihop),
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
    checkpoints: dict[tuple[str, str], SampleCheckpoint] = {}
    for variant in variants:
        for name in sets:
            variant_slug = re.sub(r"[^A-Za-z0-9.]+", "-", variant.name).strip("-")
            checkpoint = SampleCheckpoint(
                args.results_dir / ".partial" / f"{stem}__{variant_slug}__{name}.jsonl",
                checkpoint_fingerprint(configs[variant.name], judge_config, datasets[name],
                                       selection.slug, code),
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
        chat_service = build_chat_service(config, corpora=args.corpus)
        # Merge into the stored row, so `--sets multihop` adds a column to a
        # variant rather than discarding its answerable/refusal numbers.
        record: dict[str, Any] = {
            **accumulated.get(variant.name, {}),
            "variant": variant.name,
            "overrides": variant.overrides,
            "generator": f"{config.llm.provider}:{config.llm.model}",
        }
        for name in sets:
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
