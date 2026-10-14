"""Rescore a completed run from its stored outputs, without retrieving again.

    python -m rag.eval.score <run>                      # the run's own labels
    python -m rag.eval.score <run> --eval-set PATH      # labels as they are now
    python -m rag.eval.score <run> --judgment ATTEMPT   # answer runs: which judging to score

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

Answer and multi-hop runs are scored from their stored answers and passages
plus one judgment attempt (`rag.eval.judge`), pinned in the revision. Evidence
is rescored from the stored passages, so a span-label fix needs neither the
index nor the judge. A changed reference answer, tier, part or conclusion
needs a new judgment: an attempt made against other references is refused.
The evidence each stage reads is named in the revision's settings --
``prompt passages`` for single answers, ``retrieved union`` for multi-hop --
and for an agent both are every passage the turn showed the model.

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

from rag.eval.answer_eval import AnswerEvalReport, answer_result, answer_scores, summarize_answers
from rag.eval.dataset import EvalDataset, EvalSample
from rag.eval.judge import Attempt, ReferencesChanged, attempts, find_attempt, reference_hash
from rag.eval.multihop_eval import MultihopReport, multihop_result, multihop_scores, summarize
from rag.eval.retrieval_eval import RECALL_K_VALUES, SCORER, report_scores, run_retrieval_eval
from rag.eval.runs import COMPLETE, RUNS_DIR, Run, RunIncomplete, load_run
from rag.logging_config import configure_logging

logger = logging.getLogger(__name__)

#: Sample fields a run was executed with. Everything else is a label.
RETRIEVAL_INPUTS = ("query",)

#: Scorer versions for answer runs, and the evidence each one scores.
ANSWER_SCORERS = {"answer": "answer-judged-v1", "multihop": "multihop-judged-v1"}
EVIDENCE = {"answer": "prompt passages", "multihop": "retrieved union"}


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


def rescore(
    run: Run, labels: EvalDataset | None = None, *, judgment: str | None = None
) -> tuple[str, dict[str, list[str]]]:
    """Score ``run``'s outputs against ``labels`` (default: its own) as a new or existing revision.

    ``judgment`` picks an answer run's judgment attempt; without it the one
    complete attempt against these references is used.

    Returns the revision id and the selected samples whose labels changed.

    Raises:
        RunIncomplete: the run isn't complete.
        InputChanged: see `selected_from`.
        ReferencesChanged: the judgment was made against other references.
    """
    if run.manifest["status"] != COMPLETE:
        raise RunIncomplete(f"{run.run_id} is {run.manifest['status']}; only a complete run can be scored")
    stage = run.manifest["stage"]
    if stage != "retrieval" and stage not in ANSWER_SCORERS:
        raise NotImplementedError(f"Offline scoring of {stage!r} runs is not built yet")
    labels = labels if labels is not None else run.labels()
    selected = selected_from(run, labels)
    changes = label_changes(run.labels(), labels, run.manifest["dataset"]["selected_ids"])
    outputs = {record["sample_id"]: record for record in run.read_outputs()}
    if stage in ANSWER_SCORERS:
        attempt = pick_attempt(run, selected, judgment)
        digest = run.save_labels(labels)
        return _score_answers(run, stage, selected, outputs, attempt, digest), changes

    digest = run.save_labels(labels)
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


def pick_attempt(run: Run, selected: EvalDataset, ref: str | None) -> Attempt:
    """The complete judgment attempt to score ``selected`` with.

    Raises:
        ReferencesChanged: ``ref`` judged other references, or only such attempts exist.
        ValueError: no attempt, or several with none named.
    """
    wanted = reference_hash(run.manifest["stage"], selected)
    outputs_digest = run.outputs_digest()
    if ref is not None:
        attempt = find_attempt(run, ref)
        if attempt.manifest["status"] != COMPLETE:
            raise RunIncomplete(f"Judgment attempt {attempt.ref} is not complete; rerun rag.eval.judge to finish it")
        if attempt.manifest["reference_hash"] != wanted:
            raise ReferencesChanged(
                f"Judgment attempt {attempt.ref} judged other reference answers than these labels have; "
                f"judge again against them: python -m rag.eval.judge {run.run_id} --eval-set PATH"
            )
        if attempt.manifest["outputs_digest"] != outputs_digest:
            raise ValueError(f"Judgment attempt {attempt.ref} judged different outputs than {run.run_id} holds")
        return attempt
    complete = attempts(run, complete_only=True)
    matching = [a for a in complete if a.manifest["reference_hash"] == wanted
                and a.manifest["outputs_digest"] == outputs_digest]
    if len(matching) == 1:
        return matching[0]
    if matching:
        raise ValueError(
            f"{run.run_id} has {len(matching)} judgment attempts against these references; "
            f"pick one with --judgment: {', '.join(a.ref for a in matching)}"
        )
    if complete:
        raise ReferencesChanged(
            f"{run.run_id}'s judgments were made against other reference answers; judge again: "
            f"python -m rag.eval.judge {run.run_id} --eval-set PATH"
        )
    raise ValueError(f"{run.run_id} has no complete judgment yet; run python -m rag.eval.judge {run.run_id}")


def stored_report(
    run: Run, selected: EvalDataset, attempt: Attempt, outputs: dict[str, dict[str, Any]] | None = None
) -> AnswerEvalReport | MultihopReport:
    """An answer run's report, rebuilt from its stored outputs and one judgment attempt."""
    if outputs is None:
        outputs = {record["sample_id"]: record for record in run.read_outputs()}
    judged = attempt.records()
    if run.manifest["stage"] == "answer":
        return summarize_answers([answer_result(s, outputs[s.id], judged[s.id]) for s in selected])
    return summarize([multihop_result(s, outputs[s.id], judged[s.id]) for s in selected])


def _score_answers(
    run: Run, stage: str, selected: EvalDataset, outputs: dict[str, dict[str, Any]], attempt: Attempt, digest: str
) -> str:
    report = stored_report(run, selected, attempt, outputs)
    scores = (answer_scores(report, selected) if isinstance(report, AnswerEvalReport)
              else multihop_scores(report))
    return run.write_scores(
        scorer=ANSWER_SCORERS[stage],
        settings={
            "evidence": EVIDENCE[stage],
            # The protocol is a metric definition: revisions under different
            # ones never pair. The attempt is this run's draw under it.
            "judge_hash": attempt.judge_hash,
            "judgment": {"attempt_id": attempt.attempt_id, "digest": attempt.digest(),
                         "judge": attempt.manifest["judge"]},
        },
        scores=scores,
        dataset_hash=digest,
    )


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    parser = argparse.ArgumentParser(
        prog="python -m rag.eval.score",
        description="Rescore a completed run's stored outputs as a new score revision.",
    )
    parser.add_argument("run", help="Run folder or run id")
    parser.add_argument("--eval-set", type=Path, default=None, metavar="PATH",
                        help="Labels to score against (default: the run's own snapshot)")
    parser.add_argument("--judgment", default=None, metavar="ATTEMPT",
                        help="Answer runs: the judgment attempt to score (<judge_hash>/<attempt_id> or the id)")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    args = parser.parse_args(argv)

    try:
        run = load_run(args.run, args.runs_dir)
        before = set(run.score_revisions())
        revision, changes = rescore(run, EvalDataset.load(args.eval_set) if args.eval_set else None,
                                    judgment=args.judgment)
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
