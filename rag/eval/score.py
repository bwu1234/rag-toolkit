"""Rescore a completed run from its stored outputs, without retrieving again.

    python -m rag.eval.score <run>                      # the run's own labels
    python -m rag.eval.score <run> --eval-set PATH      # labels as they are now

Each rescoring adds a score revision beside the run's earlier ones and never
replaces one: the revision id hashes the outputs, labels, scorer and
settings, so rescoring unchanged labels names the revision that already
exists. Labels other than the run's own are snapshotted under the run's
``datasets/``, so every revision stays self-contained.

What a label fix may and may not change:

* **Span and document labels, grades, alternatives, matching mode, kind** --
  scoring only. The stored rankings are rescored.
* **The query, or anything else a sample was retrieved with** -- an input.
  Its stored ranking answers a different question, so rescoring is refused,
  naming the samples; they need a fresh run. A run that filtered each sample
  by its labels (the matrix's derived filters) also treats
  ``expected_doc_ids`` as an input.
* **The selected samples** must all still exist; samples added to the set
  since aren't in the run and aren't scored.

Unmatchable spans read the corpus as well as the labels. A revision on the
run's own labels carries the count recorded at run time; on changed labels
it is unavailable (``null``, shown as ``—``) rather than an old count
attached to new labels. Recomputing it needs the build's chunk inventory,
which runs don't store yet.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from rag.eval.dataset import EvalDataset, EvalSample
from rag.eval.retrieval_eval import RECALL_K_VALUES, SCORER, report_scores, run_retrieval_eval
from rag.eval.runs import COMPLETE, RUNS_DIR, Run, RunIncomplete, load_run
from rag.logging_config import configure_logging

logger = logging.getLogger(__name__)

#: Sample fields a retrieval run was executed with. Everything else is a label.
RETRIEVAL_INPUTS = ("query",)


class InputChanged(ValueError):
    """The new eval set changes what some sample was run with; its outputs can't be rescored."""


def _inputs(sample: EvalSample, *, label_filters: bool) -> dict[str, Any]:
    fields = {name: getattr(sample, name) for name in RETRIEVAL_INPUTS}
    if label_filters:
        fields["expected_doc_ids"] = sample.expected_doc_ids
    return fields


def label_changes(old: EvalDataset, new: EvalDataset, sample_ids: list[str]) -> dict[str, list[str]]:
    """For each selected sample whose record differs, the fields that changed."""
    before = {s.id: s.to_dict() for s in old}
    after = {s.id: s.to_dict() for s in new}
    changes = {}
    for sid in sample_ids:
        a, b = before[sid], after[sid]
        if fields := sorted(k for k in a.keys() | b.keys() if a.get(k) != b.get(k)):
            changes[sid] = fields
    return changes


def selected_from(run: Run, labels: EvalDataset) -> EvalDataset:
    """``labels`` narrowed to the run's selected samples, in the run's order.

    Raises:
        InputChanged: a selected sample is missing, or was run with different inputs.
    """
    by_id = {s.id: s for s in labels}
    ids = run.manifest["dataset"]["selected_ids"]
    if missing := [sid for sid in ids if sid not in by_id]:
        raise InputChanged(f"The eval set no longer has samples this run evaluated: {', '.join(missing)}")
    label_filters = bool((run.manifest.get("extra") or {}).get("filters"))
    original = {s.id: s for s in run.labels()}
    if changed := [sid for sid in ids
                   if _inputs(original[sid], label_filters=label_filters)
                   != _inputs(by_id[sid], label_filters=label_filters)]:
        raise InputChanged(
            f"{run.run_id} was run with different inputs for {', '.join(changed)}; their stored "
            "outputs answer another question, so they need a fresh run, not a rescore"
        )
    return EvalDataset(samples=[by_id[sid] for sid in ids], source_path=labels.source_path)


def rescore(run: Run, labels: EvalDataset | None = None) -> tuple[str, dict[str, list[str]]]:
    """Score ``run``'s outputs against ``labels`` (default: its own) as a new or existing revision.

    Returns the revision id and the selected samples whose labels changed.

    Raises:
        RunIncomplete: the run isn't complete.
        InputChanged: see `selected_from`.
    """
    if run.manifest["status"] != COMPLETE:
        raise RunIncomplete(f"{run.run_id} is {run.manifest['status']}; only a complete run can be scored")
    if run.manifest["stage"] != "retrieval":
        raise NotImplementedError(f"Offline scoring of {run.manifest['stage']!r} runs is not built yet")
    labels = labels if labels is not None else run.labels()
    selected = selected_from(run, labels)
    changes = label_changes(run.labels(), labels, run.manifest["dataset"]["selected_ids"])
    digest = run.save_labels(labels)

    outputs = {record["sample_id"]: record for record in run.read_outputs()}
    scores = report_scores(run_retrieval_eval(selected, None, outputs=outputs))

    # The unmatchable count recorded at run time holds only for the labels it
    # was counted against.
    recorded = next(
        (r for r in map(run.read_scores, run.score_revisions())
         if r["dataset_hash"] == digest and r["settings"].get("unmatchable_spans") is not None),
        None,
    )
    scores["unmatchable_spans"] = recorded["unmatchable_spans"] if recorded else None
    revision = run.write_scores(
        scorer=SCORER,
        settings={
            "recall_k_values": list(RECALL_K_VALUES),
            "unmatchable_spans": recorded["settings"]["unmatchable_spans"] if recorded else None,
        },
        scores=scores,
        dataset_hash=digest,
    )
    return revision, changes


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    parser = argparse.ArgumentParser(
        prog="python -m rag.eval.score",
        description="Rescore a completed run's stored outputs as a new score revision.",
    )
    parser.add_argument("run", help="Run folder or run id")
    parser.add_argument("--eval-set", type=Path, default=None, metavar="PATH",
                        help="Labels to score against (default: the run's own snapshot)")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    args = parser.parse_args(argv)

    try:
        run = load_run(args.run, args.runs_dir)
        before = set(run.score_revisions())
        revision, changes = rescore(run, EvalDataset.load(args.eval_set) if args.eval_set else None)
    except (FileNotFoundError, ValueError, RunIncomplete, NotImplementedError) as exc:
        logger.error("%s", exc)
        return 1
    for sid, fields in changes.items():
        print(f"  label change  {sid}: {', '.join(fields)}")
    state = "existing" if revision in before else "new"
    print(f"{run.run_id}:{revision}  ({state} score revision, {len(changes)} sample(s) with changed labels)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
