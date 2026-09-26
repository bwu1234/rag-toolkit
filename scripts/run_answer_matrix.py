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
    resolve_judge_config,
    run_answer_eval,
    subsample,
)
from rag.eval.dataset import EvalDataset  # noqa: E402
from rag.eval.multihop_eval import run_multihop_eval  # noqa: E402
from rag.generation.builder import build_chat_service  # noqa: E402
from rag.generation.chat_service import ChatService  # noqa: E402
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


def run_one(
    label: str, chat_service: ChatService, judge: LLMClient, dataset: EvalDataset
) -> dict[str, Any]:
    started = time.monotonic()
    report = run_answer_eval(dataset, chat_service, judge)
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
        "elapsed_s": round(elapsed, 1),
        "failed_ids": [r.sample_id for r in report.sample_results if r.passed is not True],
    }


def run_multihop(chat_service: ChatService, judge: LLMClient, dataset: EvalDataset) -> dict[str, Any]:
    started = time.monotonic()
    report = run_multihop_eval(dataset, chat_service, judge)
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
        "elapsed_s": round(elapsed, 1),
        # Per-sample detail, so a hand check of the judge needs no rerun.
        "samples": [
            {
                "id": r.sample_id,
                "complete": r.complete,
                "parts": {p.label: p.passed for p in r.part_results},
                "evidence_recall": round(r.evidence_recall, 3),
                "answer": r.actual_answer,
            }
            for r in report.sample_results
        ],
    }


def judge_suffix(judge: LLMConfig, generator: LLMConfig) -> str:
    """Filename suffix naming the judge; empty for the legacy self-judged results."""
    if (judge.provider, judge.model) == (generator.provider, generator.model):
        return ""
    return "__judge-" + re.sub(r"[^A-Za-z0-9.]+", "-", judge.model).strip("-")


def render_table(results: list[dict[str, Any]]) -> str:
    lines = [
        "| variant | answerable pass | n | s | refusal pass | n | s |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        a, f = r.get("answerable"), r.get("refusals")
        a_cell = f"{a['pass_rate']:.3f} | {a['num_evaluated']} | {a['elapsed_s']:.0f}" if a else "— | — | —"
        f_cell = f"{f['pass_rate']:.3f} | {f['num_evaluated']} | {f['elapsed_s']:.0f}" if f else "— | — | —"
        lines.append(f"| `{r['variant']}` | {a_cell} | {f_cell} |")

    multihop = [r for r in results if r.get("multihop")]
    if multihop:
        lines += [
            "",
            "| variant | multi-hop complete | completeness | evidence recall | n | s/turn |",
            "|---|---|---|---|---|---|",
        ]
        for r in multihop:
            m = r["multihop"]
            lines.append(
                f"| `{r['variant']}` | {m['complete_rate']:.3f} | {m['mean_completeness']:.3f} "
                f"| {m['evidence_recall']:.3f} | {m['num_evaluated']} | {m['mean_latency_s']:.1f} |"
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
    parser.add_argument("--judge-model", default=None, metavar="MODEL",
                        help="Judge with this model instead of eval.judge / the generator. "
                             "Fixed across all variants.")
    parser.add_argument("--limit", type=int, default=40,
                        help="Answerable samples to evaluate (evenly spaced). 0 = all.")
    parser.add_argument("--results-dir", type=Path, default=Path("data/eval/results"))
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
    judge_config = resolve_judge_config(base, args.judge_model)
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

    ordered: list[dict[str, Any]] = []
    for variant in variants:
        logger.info("[%s] %s", variant.name, variant.overrides)
        config = apply_overrides(base, variant.overrides)
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
            if name == "multihop":
                record[name] = run_multihop(chat_service, judge, datasets[name])
            else:
                record[name] = run_one(name, chat_service, judge, datasets[name])
            accumulated[variant.name] = record
            # Checkpoint after every set: these runs are tens of minutes each and
            # an interruption must not discard the ones already finished.
            ordered = flush()

    print(f"\n{render_table(ordered)}\n")
    print(f"Wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
