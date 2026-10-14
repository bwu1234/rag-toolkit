"""Reports over pinned score revisions: the matrix table, rebuilt from run folders.

    python -m rag.eval.report --baseline <run>:<revision> --runs <run>:<revision> ... [--out FILE.md]

Each reference names a run (folder name or path) and one of its score
revisions. A bare run is accepted only while it has exactly one revision, so
a reference can't silently move to a later rescoring; the report prints the
revisions it resolved, which is what a published comparison cites. Score
revisions are immutable, so a report built from the same references is the
same report whatever is rescored afterwards.

Pairing needs the runs to be comparable, which is not config-hash equality --
variants differ in config on purpose. A comparison is **refused** when the
runs evaluated different samples, were scored against different labels or by
a different scorer or settings, or (unless ``--declare corpus``) read a
different corpus. It is **flagged** when the config differs in paths the
variant doesn't declare (a matrix variant declares its overrides; add more
with ``--declare PATH``), when the code differs (``--declare code``), or
when the same index recipe was rebuilt (``--declare index``). See
docs/eval-harness-plan.md, "Comparison eligibility".

The table is `render_table`, the one the matrix script writes, so a report
over a matrix's runs reproduces its ``results/*.md`` table exactly.
"""

from __future__ import annotations

import argparse
import fnmatch
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag.eval.metrics import wilson_interval
from rag.eval.paired import compare_by_id, format_difference
from rag.eval.provenance import flatten, without_denylisted
from rag.eval.runs import COMPLETE, RUNS_DIR, Run, load_run
from rag.logging_config import configure_logging

logger = logging.getLogger(__name__)

#: Score settings that may differ between paired revisions: where the
#: unmatchable-span count was read from is the run's own corpus, not a metric definition.
_PER_RUN_SETTINGS = {"unmatchable_spans"}


class Ineligible(ValueError):
    """The runs can't be paired: they measured different things or were scored differently."""


def paired_delta(
    result: dict[str, Any], baseline: dict[str, Any], metric: str, *, binary: bool = False
) -> str:
    """Table cell for `metric`, candidate minus baseline, paired by sample id.

    Rows recorded before per-sample scores were stored get the bare difference
    of means, marked so it is not read as tested.
    """
    delta = result["metrics"][metric] - baseline["metrics"][metric]
    if "samples" not in result or "samples" not in baseline:
        return f"{delta:+.3f} (no CI)"
    try:
        diff = compare_by_id(
            {sid: s[metric] for sid, s in baseline["samples"].items()},
            {sid: s[metric] for sid, s in result["samples"].items()},
        )
    except ValueError:
        return f"{delta:+.3f} (unpaired)"
    return format_difference(diff, binary=binary)


def hit_interval(result: dict[str, Any], kind: str | None = None) -> str:
    """95% Wilson interval on a row's hit rate (or one kind's), from its per-sample scores."""
    samples = [
        s for s in result.get("samples", {}).values() if kind is None or s.get("kind") == kind
    ]
    if not samples:
        return "—"
    low, high = wilson_interval(sum(1 for s in samples if s["hit_rate"]), len(samples))
    return f"[{low:.3f}, {high:.3f}]"


def render_table(results: list[dict[str, Any]], *, baseline_variant: str = "baseline") -> str:
    """Markdown table, grouped by axis, with paired deltas against the baseline row."""
    baseline = next((r for r in results if r["variant"] == baseline_variant), None)
    lines = [
        "| variant | hit | hit 95% CI | recall | prec | MRR | NDCG | Δ hit [95% CI] | Δ NDCG [95% CI] "
        "| unmatch. | s |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    last_axis = None
    for result in results:
        if last_axis is not None and result["axis"] != last_axis:
            lines.append("| | | | | | | | | | | |")
        last_axis = result["axis"]
        m = result["metrics"]
        if baseline and result["variant"] != baseline_variant:
            hit_str = paired_delta(result, baseline, "hit_rate", binary=True)
            ndcg_str = paired_delta(result, baseline, "ndcg")
        else:
            hit_str = ndcg_str = "—"
        unmatchable = result.get("unmatchable_spans")
        unmatchable_str = "—" if unmatchable is None else str(len(unmatchable))
        lines.append(
            f"| `{result['variant']}` | {m['hit_rate']:.3f} | {hit_interval(result)} | {m['recall']:.3f} | "
            f"{m['precision']:.3f} | {m['mrr']:.3f} | {m['ndcg']:.3f} | {hit_str} | "
            f"{ndcg_str} | {unmatchable_str} | {result['elapsed_s']:.0f} |"
        )
    with_kinds = [r for r in results if r.get("by_kind")]
    if with_kinds:
        lines += [
            "",
            "By kind, each reported on its own (the rows above mix them):",
            "",
            "| variant | kind | n | hit | hit 95% CI | NDCG | Δ hit [95% CI] |",
            "|---|---|---|---|---|---|---|",
        ]
        for result in with_kinds:
            for kind, k in result["by_kind"].items():
                delta = "—"
                if baseline and result["variant"] != baseline_variant and kind in baseline.get("by_kind", {}):
                    delta = paired_delta(
                        _only_kind(result, kind), _only_kind(baseline, kind), "hit_rate", binary=True
                    )
                lines.append(
                    f"| `{result['variant']}` | {kind} | {k['num_samples']} | "
                    f"{k['metrics']['hit_rate']:.3f} | {hit_interval(result, kind)} | "
                    f"{k['metrics']['ndcg']:.3f} | {delta} |"
                )
    lines += [
        "",
        "`hit 95% CI` is a Wilson interval on that rate alone, the noise floor of one "
        "run on this many questions. Δ is variant minus `baseline`, paired by sample. `*` marks a 95% interval "
        "that excludes zero; `W/L` counts the questions the variant gained / lost and "
        "`p` is McNemar's exact test on them -- trust it over the CI when W+L is small. "
        "`(no CI)` rows predate per-sample scores and need a re-run to be tested. "
        "`unmatch.` counts expected spans no chunk contains under that variant's "
        "chunking (`—` predates the count).",
    ]
    return "\n".join(lines)


def _only_kind(result: dict[str, Any], kind: str) -> dict[str, Any]:
    """A row narrowed to one kind's samples and metrics, for a per-kind paired delta."""
    return {
        "metrics": result["by_kind"][kind]["metrics"],
        "samples": {sid: s for sid, s in result["samples"].items() if s.get("kind") == kind},
    }


# ---------------------------------------------------------------------------
# Pinned references
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Pinned:
    """One run and the score revision a report reads from it."""

    run: Run
    revision: str
    scores: dict[str, Any]

    @property
    def ref(self) -> str:
        return f"{self.run.run_id}:{self.revision}"

    def row(self) -> dict[str, Any]:
        """The run as a matrix row, the shape `render_table` reads."""
        manifest, extra = self.run.manifest, self.run.manifest.get("extra") or {}
        return {
            "variant": manifest["variant"],
            "axis": extra.get("axis", manifest["variant"]),
            "elapsed_s": manifest.get("elapsed_s", 0.0),
            **{key: self.scores[key] for key in
               ("num_samples", "metrics", "by_kind", "recall_by_k", "unmatchable_spans", "samples")},
        }


def resolve(ref: str, runs_dir: Path = RUNS_DIR) -> Pinned:
    """``<run>:<revision>``, or a bare ``<run>`` that has exactly one score revision."""
    run_ref, _, revision = ref.rpartition(":") if ":" in Path(ref).name else (ref, "", "")
    run = load_run(run_ref, runs_dir)
    if run.manifest["status"] != COMPLETE:
        raise Ineligible(f"{run.run_id} is {run.manifest['status']}; only complete runs can be reported")
    if not revision:
        revisions = run.score_revisions()
        if len(revisions) != 1:
            raise ValueError(
                f"{run.run_id} has {len(revisions)} score revisions ({', '.join(revisions)}); "
                "name one as <run>:<revision>"
            )
        revision = revisions[0]
    return Pinned(run, revision, run.read_scores(revision))


# ---------------------------------------------------------------------------
# Eligibility
# ---------------------------------------------------------------------------


def config_differences(baseline: Run, candidate: Run) -> dict[str, tuple[Any, Any]]:
    """Every config path whose value differs, ignoring `CONFIG_HASH_DENYLIST` fields."""
    old = flatten(without_denylisted(baseline.manifest["config"]))
    new = flatten(without_denylisted(candidate.manifest["config"]))
    return {path: (old.get(path), new.get(path))
            for path in sorted(old.keys() | new.keys()) if old.get(path) != new.get(path)}


def _declared(pinned: Pinned, extra_declared: list[str]) -> list[str]:
    extra = pinned.run.manifest.get("extra") or {}
    return [*(extra.get("overrides") or {}), *extra_declared]


def _matches(path: str, patterns: list[str]) -> bool:
    return any(path == p or path.startswith(f"{p}.") or fnmatch.fnmatchcase(path, p) for p in patterns)


def check_pair(baseline: Pinned, candidate: Pinned, declared: list[str]) -> list[str]:
    """Refuse a pair that can't be compared; return the confounders of one that can.

    Raises:
        Ineligible: naming each reason.
    """
    b, c = baseline.run.manifest, candidate.run.manifest
    problems = []
    if b["stage"] != c["stage"]:
        problems.append(f"different stages ({b['stage']} vs {c['stage']})")
    if b["dataset"]["selection_hash"] != c["dataset"]["selection_hash"]:
        problems.append(
            f"different samples evaluated ({b['dataset']['selection']}, {b['dataset']['path']} vs "
            f"{c['dataset']['selection']}, {c['dataset']['path']})"
        )
    if baseline.scores["dataset_hash"] != candidate.scores["dataset_hash"]:
        problems.append(
            "scored against different labels; rescore one onto the other's with "
            "`python -m rag.eval.score <run> --eval-set PATH`"
        )
    if baseline.scores["scorer"] != candidate.scores["scorer"]:
        problems.append(f"different scorers ({baseline.scores['scorer']} vs {candidate.scores['scorer']})")
    settings = [{k: v for k, v in p.scores["settings"].items() if k not in _PER_RUN_SETTINGS}
                for p in (baseline, candidate)]
    if settings[0] != settings[1]:
        problems.append(f"different scoring settings ({settings[0]} vs {settings[1]})")
    declared = _declared(candidate, declared)
    b_index, c_index = b.get("index") or {}, c.get("index") or {}
    if b_index.get("corpus_hash") != c_index.get("corpus_hash") and "corpus" not in declared:
        problems.append("different corpus snapshots; pass --declare corpus if that is the treatment")
    if problems:
        raise Ineligible(f"{candidate.ref} can't be paired with {baseline.ref}: " + "; ".join(problems))

    confounders = [
        f"`{path}`: {old!r} -> {new!r}"
        for path, (old, new) in config_differences(baseline.run, candidate.run).items()
        if not _matches(path, declared)
    ]
    b_filters, c_filters = ((m.get("extra") or {}).get("filters") for m in (b, c))
    if b_filters != c_filters and "eval.derived_filters" not in declared and "filters" not in (c.get("extra") or {}):
        confounders.append(f"derived filters: {b_filters!r} -> {c_filters!r}")
    if b["code_version"] != c["code_version"] and "code" not in declared:
        confounders.append(f"code: {b['code_version']} -> {c['code_version']}")
    if (b_index.get("index_hash") == c_index.get("index_hash")
            and b_index.get("index_build_id") != c_index.get("index_build_id") and "index" not in declared):
        confounders.append(
            f"same index recipe, different build: {b_index.get('index_build_id')} -> {c_index.get('index_build_id')}"
        )
    return confounders


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


def build_report(baseline: Pinned, candidates: list[Pinned], *, declared: list[str] | None = None) -> str:
    """The matrix-style markdown report, followed by its pins and any confounders.

    Raises:
        Ineligible: some candidate can't be paired with the baseline.
    """
    declared = declared or []
    confounders = {c.ref: check_pair(baseline, c, declared) for c in candidates}
    variants = [p.run.manifest["variant"] for p in (baseline, *candidates)]
    if len(set(variants)) != len(variants):
        raise ValueError(f"Two references share a variant name ({', '.join(variants)}); a table row needs one each")

    m = baseline.run.manifest
    lines = [
        f"# Retrieval matrix — {m['corpus']['selection']}",
        "",
        f"- Eval set: `{m['dataset']['path']}` ({m['dataset']['num_samples']} samples)",
        f"- Collection: `{m['corpus']['collection']}`",
        "",
        render_table([p.row() for p in (baseline, *candidates)], baseline_variant=m["variant"]),
        "",
        "## Pinned score revisions",
        "",
        *(f"- `{p.run.manifest['variant']}`: `{p.ref}`" + (" (baseline)" if p is baseline else "")
          for p in (baseline, *candidates)),
    ]
    flagged = {ref: found for ref, found in confounders.items() if found}
    if flagged:
        lines += ["", "## Undeclared differences (possible confounders)", ""]
        for ref, found in flagged.items():
            lines.append(f"- `{ref}`:")
            lines += [f"  - {item}" for item in found]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    parser = argparse.ArgumentParser(
        prog="python -m rag.eval.report",
        description="Report pinned run score revisions against a baseline, as the matrix table.",
    )
    parser.add_argument("--baseline", required=True, metavar="RUN[:REVISION]")
    parser.add_argument("--runs", nargs="+", default=[], metavar="RUN[:REVISION]")
    parser.add_argument("--declare", action="append", default=[], metavar="PATH",
                        help="A difference the comparison intends: a config path or glob, or "
                             "'code', 'index', 'corpus'. Repeatable.")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--out", type=Path, default=None, help="Write the report here as well as printing it.")
    args = parser.parse_args(argv)

    try:
        baseline = resolve(args.baseline, args.runs_dir)
        candidates = [resolve(ref, args.runs_dir) for ref in args.runs]
        report = build_report(baseline, candidates, declared=args.declare)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1
    print(report, end="")
    if args.out is not None:
        args.out.write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
