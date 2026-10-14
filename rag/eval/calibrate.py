"""Judge calibration: how often the judge's verdict matches a reviewed human label, by category.

    python -m rag.eval.calibrate draft --runs RUN [RUN ...] --per-cell 8
    python -m rag.eval.calibrate run data/eval/judge_calibration.json --judge-model gemma4:31b-mlx --attempts 3

A pass rate that matches an earlier judge's says nothing about whether
either is right. Calibration compares verdicts with human labels on a fixed
set of stored answers (eval harness plan, "Calibrate the judge in Phase 4";
procedure in docs/judge-calibration-guide.md).

The fixture
-----------
One JSON file. Each item is one verdict slot -- a single answer, or one part
of a multi-hop answer -- carrying everything the judge reads (question,
reference, tier or part, answer), so it stays self-contained when the run it
was drawn from is gone. Its human label is a list of ``reviews`` (reviewer,
verdict, rationale, when) plus an ``adjudication`` when reviewers disagree.
An item's label is the adjudication, else the reviews' shared verdict; an item
with no review, or with disagreeing reviews and no adjudication, is refused.
The fixture also declares its ``bounds`` -- the highest false-pass and
false-fail rates a judge may show -- and calibration refuses a fixture
without them, so the bar is in the committed file before any judge is
measured against it.

``draft`` builds candidate items from judged runs, sampled evenly across
category and judge outcome, so the fixture isn't only the current system's
misses. Drafted items carry no verdict: reviewers label blind.

The report
----------
Per attempt: agreement, false passes (judge PASS, human FAIL) over human
FAILs and false fails over human PASSes, each with a 95% Wilson interval,
and unparseable verdicts, counted apart. By category and by tag (e.g.
``wrong_period``) for the first attempt. Across attempts: how many items got
the same verdict every time -- judge noise, separate from judge error. The
bounds are checked against the first attempt's upper interval: a judge whose
interval reaches past a bound is marked exploratory for claims that need it.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rag.config.settings import LLMConfig, load_config
from rag.eval.answer_eval import VERDICT_PARSER, add_judge_arguments, judge_answer, resolve_judge_config
from rag.eval.dataset import EvalSample
from rag.eval.judge import RUBRICS, attempts
from rag.eval.metrics import wilson_interval
from rag.eval.multihop_eval import CONCLUSION_LABEL, judge_part, parts_of
from rag.eval.runs import COMPLETE, RUNS_DIR, load_run, utc_now, write_json
from rag.hashing import stable_hash
from rag.llm.base import LLMClient
from rag.llm.factory import get_llm_client
from rag.logging_config import configure_logging

logger = logging.getLogger(__name__)

_REPO = Path(__file__).resolve().parents[2]
DEFAULT_FIXTURE = _REPO / "data" / "eval" / "judge_calibration.json"
DEFAULT_DRAFT = _REPO / "data" / "eval" / "drafts" / "judge_calibration_draft.json"
RESULTS_DIR = _REPO / "data" / "eval" / "results_calibration"
GUIDE = "docs/judge-calibration-guide.md"
SCHEMA = 1


class FixtureNotReady(ValueError):
    """The fixture can't calibrate a judge yet: unlabeled items, unresolved reviews, or no bounds."""


@dataclass(frozen=True)
class Item:
    """One verdict slot with what the judge reads and what the reviewers decided."""

    id: str
    stage: str
    category: str
    query: str
    reference: str
    answer: str
    tier: str | None = None
    part: str | None = None
    tags: list[str] = field(default_factory=list)
    reviews: list[dict[str, Any]] = field(default_factory=list)
    adjudication: dict[str, Any] | None = None
    source: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Item:
        item = cls(**data)
        if item.stage not in RUBRICS:
            raise ValueError(f"Item {item.id!r}: unknown stage {item.stage!r}")
        if item.stage == "multihop" and not item.part:
            raise ValueError(f"Item {item.id!r}: a multihop item names the part it labels")
        return item

    @property
    def human(self) -> bool | None:
        """The label: the adjudication, else the reviews' shared verdict; None when there is none."""
        if self.adjudication is not None:
            return bool(self.adjudication["passed"])
        verdicts = {bool(r["passed"]) for r in self.reviews}
        return verdicts.pop() if len(verdicts) == 1 else None


def load_fixture(path: Path) -> tuple[dict[str, Any], list[Item]]:
    """The fixture's settings and its items; refuses duplicate ids."""
    data = json.loads(path.read_text(encoding="utf-8"))
    items = [Item.from_dict(i) for i in data["items"]]
    if duplicates := sorted(i for i, n in Counter(item.id for item in items).items() if n > 1):
        raise ValueError(f"{path} has duplicate item ids: {', '.join(duplicates)}")
    return {k: v for k, v in data.items() if k != "items"}, items


def check_ready(meta: dict[str, Any], items: list[Item]) -> None:
    """Raise `FixtureNotReady` unless every item has a label and the bounds are declared."""
    problems = []
    if unlabeled := [i.id for i in items if not i.reviews]:
        problems.append(f"{len(unlabeled)} item(s) have no review ({', '.join(unlabeled[:5])}...)")
    if unresolved := [i.id for i in items if i.reviews and i.human is None]:
        problems.append(f"reviews disagree with no adjudication on {', '.join(unresolved)}")
    bounds = meta.get("bounds") or {}
    if not {"false_pass_rate", "false_fail_rate"} <= bounds.keys():
        problems.append("no declared bounds (false_pass_rate, false_fail_rate); declare them before measuring a judge")
    if problems:
        raise FixtureNotReady("The fixture isn't ready to calibrate against: " + "; ".join(problems))


def judge_item(item: Item, client: LLMClient) -> dict[str, Any]:
    """Judge one item exactly as an answer run's attempt would judge that slot."""
    if item.stage == "multihop":
        return judge_part(client, item.query, item.part or "", item.reference, item.answer)
    sample = EvalSample(id=item.id, query=item.query, expected_answer=item.reference,
                        extra={"tier": item.tier} if item.tier else {})
    [verdict] = judge_answer(sample, {"answer": item.answer, "generation_failure": None}, client)["verdicts"]
    return dict(verdict)


# ---------------------------------------------------------------------------
# The summary
# ---------------------------------------------------------------------------


def _rate(count: int, n: int) -> dict[str, Any]:
    low, high = wilson_interval(count, n)
    return {"count": count, "n": n, "rate": round(count / n, 4) if n else None,
            "ci": [round(low, 4), round(high, 4)]}


def _confusion(items: list[Item], verdicts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    human_pass = [i for i in items if i.human]
    human_fail = [i for i in items if i.human is False]
    judged = {i.id: verdicts[i.id]["passed"] for i in items}
    return {
        "n": len(items),
        "human_pass": len(human_pass),
        "human_fail": len(human_fail),
        "agreement": _rate(sum(judged[i.id] is i.human for i in items), len(items)),
        "false_pass": _rate(sum(judged[i.id] is True for i in human_fail), len(human_fail)),
        "false_fail": _rate(sum(judged[i.id] is False for i in human_pass), len(human_pass)),
        "unparseable": sum(judged[i.id] is None for i in items),
    }


def summarize_calibration(
    meta: dict[str, Any], items: list[Item], attempt_verdicts: list[dict[str, dict[str, Any]]]
) -> dict[str, Any]:
    """Labels, per-attempt error rates, the first attempt by category and tag, and consistency."""
    reviewers = sorted({r["reviewer"] for i in items for r in i.reviews})
    double = [i for i in items if len({r["reviewer"] for r in i.reviews}) >= 2]
    first = attempt_verdicts[0]
    groups: dict[str, list[Item]] = {}
    for item in items:
        groups.setdefault(f"category: {item.category}", []).append(item)
        for tag in item.tags:
            groups.setdefault(f"tag: {tag}", []).append(item)
    stable = [i.id for i in items if len({str(v[i.id]["passed"]) for v in attempt_verdicts}) == 1]
    bounds = meta["bounds"]
    primary = _confusion(items, first)
    return {
        "labels": {
            "items": len(items),
            "reviewers": reviewers,
            # One reviewer can't show independent agreement; the report says so.
            "single_reviewer": len(reviewers) < 2,
            "double_reviewed": len(double),
            "review_agreements": sum(len({bool(r["passed"]) for r in i.reviews}) == 1 for i in double),
            "adjudicated": sum(i.adjudication is not None for i in items),
            # The predeclared rule for which items get a second review.
            "double_review_rule": meta.get("double_review"),
        },
        "attempts": [_confusion(items, v) for v in attempt_verdicts],
        "groups": {name: _confusion(group, first) for name, group in sorted(groups.items())},
        "consistency": {"attempts": len(attempt_verdicts), "stable": len(stable),
                        "flipped": sorted(set(i.id for i in items) - set(stable))},
        "bounds": {
            name: {"bound": bounds[f"{name}_rate"], "upper_ci": primary[name]["ci"][1],
                   "within": primary[name]["ci"][1] <= bounds[f"{name}_rate"]}
            for name in ("false_pass", "false_fail")
        },
    }


def _cell(rate: dict[str, Any]) -> str:
    if not rate["n"]:
        return "— (0)"
    return f"{rate['count']}/{rate['n']} = {rate['rate']:.3f} [{rate['ci'][0]:.3f}, {rate['ci'][1]:.3f}]"


def render_calibration(record: dict[str, Any]) -> str:
    s = record["summary"]
    labels = s["labels"]
    lines = [
        f"# Judge calibration — `{record['judge']}` on `{record['fixture']}`",
        "",
        f"- Items: {labels['items']}; reviewers: {', '.join(labels['reviewers'])}; "
        f"double-reviewed: {labels['double_reviewed']} ({labels['review_agreements']} agreed); "
        f"adjudicated: {labels['adjudicated']}"
        + (f" (double-review rule: {labels['double_review_rule']})" if labels["double_review_rule"] else ""),
    ]
    if labels["single_reviewer"]:
        lines.append("- **Single reviewer:** the labels have no independent check; agreement with the "
                     "judge is agreement with one person's reading.")
    lines += [
        f"- Judge protocol: rubric `{record['rubric_hash']}`, parser `{record['parser']}`, "
        f"fixture `{record['fixture_hash']}`",
        "",
        "| attempt | agreement | false pass (of human FAIL) | false fail (of human PASS) | unparseable |",
        "|---|---|---|---|---|",
    ]
    for k, a in enumerate(s["attempts"], start=1):
        lines.append(f"| {k} | {_cell(a['agreement'])} | {_cell(a['false_pass'])} | {_cell(a['false_fail'])} "
                     f"| {a['unparseable']} |")
    lines += [
        "",
        "First attempt by category and tag:",
        "",
        "| group | n | human PASS / FAIL | false pass | false fail | unparseable |",
        "|---|---|---|---|---|---|",
    ]
    for name, g in s["groups"].items():
        lines.append(f"| {name} | {g['n']} | {g['human_pass']} / {g['human_fail']} | {_cell(g['false_pass'])} "
                     f"| {_cell(g['false_fail'])} | {g['unparseable']} |")
    c = s["consistency"]
    if c["attempts"] > 1:
        lines += ["", f"Consistency: {c['stable']} of {labels['items']} items got the same verdict in all "
                      f"{c['attempts']} attempts" + (f"; changed: {', '.join(c['flipped'])}" if c["flipped"] else "")]
    lines += ["", "Declared bounds, against the first attempt's upper 95% bound:", ""]
    for name, b in s["bounds"].items():
        verdict = "within" if b["within"] else "**outside: exploratory for claims that need it**"
        lines.append(f"- {name.replace('_', ' ')}: upper {b['upper_ci']:.3f} vs bound {b['bound']:.3f} — {verdict}")
    return "\n".join(lines) + "\n"


def calibrate(
    fixture_path: Path, client: LLMClient, judge: LLMConfig, *, num_attempts: int = 1
) -> dict[str, Any]:
    """Judge every fixture item ``num_attempts`` times; return the full record, verdicts and summary.

    Raises:
        FixtureNotReady: see `check_ready`.
    """
    meta, items = load_fixture(fixture_path)
    check_ready(meta, items)
    attempt_verdicts = []
    for k in range(num_attempts):
        logger.info("Calibration attempt %d of %d over %d item(s)", k + 1, num_attempts, len(items))
        attempt_verdicts.append({item.id: judge_item(item, client) for item in items})
    stages = sorted({i.stage for i in items})
    return {
        "schema": SCHEMA,
        "fixture": str(fixture_path),
        "fixture_hash": stable_hash(json.loads(fixture_path.read_text(encoding="utf-8"))),
        "judge": f"{judge.provider}:{judge.model}",
        "judge_config": judge.model_dump(mode="json"),
        "rubric_hash": stable_hash({stage: RUBRICS[stage].prompts for stage in stages}),
        "parser": VERDICT_PARSER,
        "recorded_at": utc_now().isoformat(timespec="seconds"),
        "verdicts": attempt_verdicts,
        "summary": summarize_calibration(meta, items, attempt_verdicts),
    }


# ---------------------------------------------------------------------------
# Drafting candidates
# ---------------------------------------------------------------------------


def draft_items(run_refs: list[str], *, per_cell: int, seed: int = 0, runs_dir: Path = RUNS_DIR) -> list[dict[str, Any]]:
    """Candidate items from judged runs, up to ``per_cell`` per (category, judge outcome), unlabeled.

    The judge's verdict picks the cell, so the draft spans passes and fails
    rather than only current misses, but isn't written into the item.
    Generation failures are left out: they fail without a judge call.
    """
    cells: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for ref in run_refs:
        run = load_run(ref, runs_dir)
        stage = run.manifest["stage"]
        if stage not in RUBRICS or run.manifest["status"] != COMPLETE:
            raise ValueError(f"{run.run_id} is not a complete answer run")
        judged = attempts(run, complete_only=True)
        if not judged:
            raise ValueError(f"{run.run_id} has no complete judgment to sample outcomes from")
        verdicts = judged[-1].records()
        samples = {s.id: s for s in run.labels()}
        for output in run.read_outputs():
            if output["generation_failure"] is not None:
                continue
            sample = samples[output["sample_id"]]
            source = {"run": run.run_id, "sample_id": sample.id}
            for verdict in verdicts[sample.id]["verdicts"]:
                if stage == "answer":
                    tier = sample.extra.get("tier")
                    item = {"stage": stage, "category": "refusal" if tier == "refusal" else "single-hop",
                            "query": sample.query, "reference": sample.expected_answer or "",
                            "tier": tier, "part": None}
                else:
                    reference = next(p.answer for p in parts_of(sample) if p.label == verdict["label"])
                    item = {"stage": stage,
                            "category": "multihop conclusion" if verdict["label"] == CONCLUSION_LABEL
                            else "multihop part",
                            "query": sample.query, "reference": reference, "tier": None, "part": verdict["label"]}
                item.update(answer=output["answer"], tags=[], reviews=[], adjudication=None,
                            source={**source, "label": verdict["label"]})
                outcome = {True: "judged PASS", False: "judged FAIL", None: "unparseable"}[verdict["passed"]]
                cells.setdefault((item["category"], outcome), []).append(item)
    rng = random.Random(seed)
    drafted = []
    for key in sorted(cells):
        pool = cells[key]
        drafted += rng.sample(pool, min(per_cell, len(pool)))
    rng.shuffle(drafted)  # so reviewers can't read the cell from the order
    return [{"id": f"cal-{n:04d}", **item} for n, item in enumerate(drafted, start=1)]


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    parser = argparse.ArgumentParser(prog="python -m rag.eval.calibrate", description="Judge calibration against reviewed human labels.")
    commands = parser.add_subparsers(dest="command", required=True)
    draft = commands.add_parser("draft", help="Sample unlabeled candidate items from judged answer runs.")
    draft.add_argument("--runs", nargs="+", required=True, metavar="RUN")
    draft.add_argument("--per-cell", type=int, default=8, metavar="N",
                       help="Items per (category, judge outcome) cell (default: 8)")
    draft.add_argument("--seed", type=int, default=0)
    draft.add_argument("--out", type=Path, default=DEFAULT_DRAFT)
    draft.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    run = commands.add_parser("run", help="Judge a reviewed fixture and report against its labels.")
    run.add_argument("fixture", type=Path, nargs="?", default=DEFAULT_FIXTURE)
    run.add_argument("--config", default=None, metavar="PATH")
    add_judge_arguments(run)
    run.add_argument("--attempts", type=int, default=1, metavar="N",
                     help="Judge every item N times, to measure judge noise as well as error (default: 1)")
    run.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)

    if args.command == "draft":
        try:
            items = draft_items(args.runs, per_cell=args.per_cell, seed=args.seed, runs_dir=args.runs_dir)
        except (FileNotFoundError, ValueError) as exc:
            logger.error("%s", exc)
            return 1
        args.out.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.out, {"schema": SCHEMA, "guide": GUIDE, "bounds": None, "double_review": None,
                              "items": items})
        print(f"Wrote {len(items)} unlabeled item(s) to {args.out}; label them per {GUIDE}")
        return 0

    try:
        judge = resolve_judge_config(load_config(args.config), args.judge_model, args.judge_provider)
        record = calibrate(args.fixture, get_llm_client(judge), judge, num_attempts=args.attempts)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1
    slug = re.sub(r"[^A-Za-z0-9.]+", "-", judge.model).strip("-")
    stem = f"{args.fixture.stem}__{slug}__{record['recorded_at'][:19].replace(':', '')}"
    args.results_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.results_dir / f"{stem}.json", record)
    markdown = render_calibration(record)
    (args.results_dir / f"{stem}.md").write_text(markdown, encoding="utf-8")
    print(markdown, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
