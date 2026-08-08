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

Usage
-----
    python scripts/run_answer_matrix.py --corpus edgar --limit 40
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.config.settings import RagConfig, load_config  # noqa: E402
from rag.eval.answer_eval import run_answer_eval, subsample  # noqa: E402
from rag.eval.dataset import EvalDataset  # noqa: E402
from rag.generation.builder import build_chat_service  # noqa: E402
from rag.generation.factory import get_llm_client  # noqa: E402
from rag.logging_config import configure_logging  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_ANSWERABLE = Path("data/eval/edgar_eval_set.json")
DEFAULT_REFUSALS = Path("data/eval/edgar_refusal_set.json")


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
    label: str, config: RagConfig, dataset: EvalDataset, corpora: list[str] | None
) -> dict[str, Any]:
    chat_service = build_chat_service(config, corpora=corpora)
    judge = get_llm_client(config.llm)
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
        "pass_rate": round(report.pass_rate, 4),
        "elapsed_s": round(elapsed, 1),
    }


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
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure CRAG via answer eval.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--corpus", action="append", default=None, metavar="NAME")
    parser.add_argument("--answerable", type=Path, default=DEFAULT_ANSWERABLE)
    parser.add_argument("--refusals", type=Path, default=DEFAULT_REFUSALS)
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

    base = load_config(args.config)
    answerable = subsample(EvalDataset.load(args.answerable), args.limit)
    refusals = EvalDataset.load(args.refusals)
    selection = base.corpus_selection(args.corpus)
    logger.info(
        "Answer matrix: %d variant(s), %d answerable + %d refusal sample(s), corpus %s",
        len(variants), len(answerable), len(refusals), selection.describe(),
    )

    args.results_dir.mkdir(parents=True, exist_ok=True)
    destination = args.results_dir / f"answer_{selection.slug}.json"
    accumulated: dict[str, dict[str, Any]] = {}
    if destination.exists():
        accumulated = {r["variant"]: r for r in json.loads(destination.read_text())["results"]}

    def flush() -> list[dict[str, Any]]:
        order = {v.name: i for i, v in enumerate(VARIANTS)}
        ordered = sorted(accumulated.values(), key=lambda r: order.get(r["variant"], 999))
        destination.write_text(json.dumps({
            "corpus": selection.describe(),
            "collection": selection.collection_name,
            "answerable_set": str(args.answerable),
            "refusal_set": str(args.refusals),
            "results": ordered,
        }, indent=2) + "\n")
        (args.results_dir / f"answer_{selection.slug}.md").write_text(
            f"# Answer eval / CRAG — {selection.describe()}\n\n{render_table(ordered)}\n"
        )
        return ordered

    ordered: list[dict[str, Any]] = []
    for variant in variants:
        logger.info("[%s] %s", variant.name, variant.overrides)
        config = apply_overrides(base, variant.overrides)
        record: dict[str, Any] = {"variant": variant.name, "overrides": variant.overrides}
        record["answerable"] = run_one("answerable", config, answerable, args.corpus)
        record["refusals"] = run_one("refusals", config, refusals, args.corpus)
        accumulated[variant.name] = record
        # Checkpoint after every variant: these runs are tens of minutes each and
        # an interruption must not discard the ones already finished.
        ordered = flush()

    print(f"\n{render_table(ordered)}\n")
    print(f"Wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
