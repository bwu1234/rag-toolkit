"""Judge a completed answer run's stored answers, as one more attempt beside the earlier ones.

    python -m rag.eval.judge <run> --judge-model gemma4:31b-mlx
    python -m rag.eval.judge <run> --judge-model M --eval-set PATH   # references as they are now

An answer run generates once (`outputs.jsonl`); judging reads those answers
and never regenerates them. Each judging is an **attempt**, kept under the
run and never replaced::

    judgments/<judge_hash>/
      <attempt_id>.json    # judge, references, source outputs, status
      <attempt_id>.jsonl   # identity header, then per sample: rendered inputs, full replies, verdicts

``judge_hash`` names the judge **protocol**: the judge's provider, model and
settings, the rubric's prompts, the verdict parser, and a hash of the
reference labels it compares against. A new judge, a changed reference or a
changed rubric gets a new directory; judging again under the same protocol
gets a new attempt in the same one, so judge variance can be measured from
attempts that differ only in the draw. Attempts record the digest of the
outputs they judged, and an attempt still running when its process stopped is
resumed by the next judging under the same protocol, as runs are.

What the judge reads as a reference, per stage (anything else is an input or
an evidence label, which re-judging does not depend on):

* ``answer`` -- ``expected_answer``, and ``tier`` (which picks the refusal rubric).
* ``multihop`` -- each part's label and answer, and the ``conclusion``.

The model's revision is not resolved: an Ollama tag can be re-pulled under
the same name, and the attempt records the name only.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import secrets
import socket
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag.config.settings import LLMConfig, RagConfig, load_config
from rag.eval import answer_eval, multihop_eval
from rag.eval.answer_eval import generate_into_run
from rag.eval.checkpoint import SampleCheckpoint
from rag.eval.dataset import EvalDataset, EvalSample
from rag.eval.provenance import checked_index, code_version
from rag.eval.runs import (
    COMPLETE,
    RUNNING,
    RUNS_DIR,
    Run,
    RunIncomplete,
    held_elsewhere,
    load_run,
    open_run,
    utc_now,
    write_json,
)
from rag.generation.chat_service import ChatResponder
from rag.ingestion.corpora import chunk_selected_corpora
from rag.hashing import stable_hash
from rag.llm.base import LLMClient
from rag.llm.daily_budget import DailyRequestBudgetSpent
from rag.llm.factory import get_llm_client
from rag.logging_config import configure_logging

logger = logging.getLogger(__name__)

JUDGMENTS = "judgments"


@dataclass(frozen=True)
class Rubric:
    """How one stage's answers are judged."""

    #: The rubric as the judge sees it: every system prompt, and every prompt
    #: template rendered with placeholders. Hashed into the protocol.
    prompts: dict[str, str]
    references: Callable[[EvalSample], dict[str, Any]]
    judge: Callable[[EvalSample, dict[str, Any], LLMClient], dict[str, Any]]


def _answer_references(sample: EvalSample) -> dict[str, Any]:
    return {"expected_answer": sample.expected_answer, "tier": sample.extra.get("tier")}


def _multihop_references(sample: EvalSample) -> dict[str, Any]:
    return {"parts": [{"label": p.label, "answer": p.answer} for p in multihop_eval.parts_of(sample)]}


_PLACEHOLDERS = ("{query}", "{expected}", "{actual}")

RUBRICS: dict[str, Rubric] = {
    "answer": Rubric(
        prompts={
            "system": answer_eval._JUDGE_SYSTEM_PROMPT,
            "prompt": answer_eval._judge_prompt(*_PLACEHOLDERS),
            "refusal_system": answer_eval._REFUSAL_JUDGE_SYSTEM_PROMPT,
            "refusal_prompt": answer_eval._refusal_judge_prompt(*_PLACEHOLDERS),
        },
        references=_answer_references,
        judge=answer_eval.judge_answer,
    ),
    "multihop": Rubric(
        prompts={
            "system": multihop_eval._PART_JUDGE_SYSTEM_PROMPT,
            "prompt": multihop_eval._part_judge_prompt("{query}", "{label}", "{expected}", "{actual}"),
            "conclusion_label": multihop_eval.CONCLUSION_LABEL,
        },
        references=_multihop_references,
        judge=multihop_eval.judge_multihop,
    ),
}


class ReferencesChanged(ValueError):
    """A judgment was made against references other than the ones being scored."""


def reference_hash(stage: str, samples: EvalDataset) -> str:
    """Hash of what the judge compares each selected sample's answer against."""
    rubric = RUBRICS[stage]
    return stable_hash([{"id": s.id, **rubric.references(s)} for s in samples])


def judge_protocol(judge: LLMConfig, stage: str, samples: EvalDataset) -> dict[str, Any]:
    """Everything a verdict depends on besides the answer itself."""
    return {
        "judge": judge.model_dump(mode="json"),
        "stage": stage,
        "rubric": RUBRICS[stage].prompts,
        "parser": answer_eval.VERDICT_PARSER,
        "references": reference_hash(stage, samples),
    }


def protocol_hash(judge: LLMConfig, stage: str, samples: EvalDataset) -> str:
    """The ``judge_hash`` a judging of ``samples`` by ``judge`` is filed under."""
    return stable_hash(judge_protocol(judge, stage, samples))


@dataclass
class Attempt:
    """One judging of one run's outputs."""

    path: Path
    manifest: dict[str, Any]

    @property
    def judge_hash(self) -> str:
        return str(self.manifest["judge_hash"])

    @property
    def attempt_id(self) -> str:
        return str(self.manifest["attempt_id"])

    @property
    def ref(self) -> str:
        return f"{self.judge_hash}/{self.attempt_id}"

    @property
    def records_path(self) -> Path:
        return self.path.with_suffix(".jsonl")

    def checkpoint(self) -> SampleCheckpoint:
        return SampleCheckpoint(self.records_path, {
            "judge_hash": self.judge_hash, "attempt_id": self.attempt_id,
            "outputs_digest": self.manifest["outputs_digest"],
        })

    def records(self) -> dict[str, dict[str, Any]]:
        """Judgments so far, by sample id."""
        return self.checkpoint().load()

    def digest(self) -> str:
        return stable_hash(self.records_path.read_text(encoding="utf-8"))

    def complete(self, sample_ids: list[str]) -> None:
        """Mark the attempt complete once it judged exactly ``sample_ids``, once each.

        Raises:
            RunIncomplete: naming duplicate, missing or unexpected samples.
        """
        lines = self.records_path.read_text(encoding="utf-8").splitlines()[1:] if self.records_path.exists() else []
        counts = Counter(json.loads(line)["sample_id"] for line in lines if line)
        problems = []
        if duplicates := sorted(sid for sid, n in counts.items() if n > 1):
            problems.append(f"duplicate judgments for {', '.join(duplicates)}")
        if missing := [sid for sid in sample_ids if sid not in counts]:
            problems.append(f"no judgment for {', '.join(missing)}")
        if extra := sorted(set(counts) - set(sample_ids)):
            problems.append(f"judgments for unselected samples {', '.join(extra)}")
        if problems:
            raise RunIncomplete(f"judgment attempt {self.ref}: " + "; ".join(problems))
        self.manifest.update(status=COMPLETE, finished_at=utc_now().isoformat(timespec="seconds"))
        self.manifest.pop("pid", None)
        write_json(self.path, self.manifest)


def attempts(run: Run, *, complete_only: bool = False) -> list[Attempt]:
    """Every judgment attempt of ``run``, oldest first."""
    found = []
    for path in sorted((run.path / JUDGMENTS).glob("*/*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if not complete_only or manifest["status"] == COMPLETE:
            found.append(Attempt(path, manifest))
    return sorted(found, key=lambda a: a.manifest["started_at"])


def judged(run: Run, judge_hash: str) -> bool:
    """Whether ``run`` has a complete judgment attempt under the protocol ``judge_hash`` names."""
    return any(a.judge_hash == judge_hash for a in attempts(run, complete_only=True))


def find_attempt(run: Run, ref: str) -> Attempt:
    """An attempt of ``run`` by ``<judge_hash>/<attempt_id>`` or by attempt id alone."""
    matches = [a for a in attempts(run) if ref in (a.ref, a.attempt_id)]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"{run.run_id} has no judgment attempt {ref!r}; it has: "
            + (", ".join(a.ref for a in attempts(run)) or "none")
        )
    return matches[0]


def open_attempt(
    run: Run, judge: LLMConfig, selected: EvalDataset, *, labels_hash: str, resume: bool = True
) -> Attempt:
    """Resume this protocol's unfinished attempt on these outputs, or start a new one."""
    stage = run.manifest["stage"]
    protocol = judge_protocol(judge, stage, selected)
    judge_hash = protocol_hash(judge, stage, selected)
    outputs_digest = run.outputs_digest()
    directory = run.path / JUDGMENTS / judge_hash
    if resume:
        for attempt in reversed(attempts(run)):
            m = attempt.manifest
            if (m["judge_hash"], m["status"], m["outputs_digest"]) != (judge_hash, RUNNING, outputs_digest):
                continue
            if held_elsewhere(m):
                logger.warning("Judgment attempt %s is still being written (pid %s); not resuming it",
                               attempt.ref, m["pid"])
                continue
            logger.info("Resuming judgment attempt %s (%d judged)", attempt.ref, len(attempt.records()))
            m.update(pid=os.getpid(), host=socket.gethostname())
            m.setdefault("resumed_at", []).append(utc_now().isoformat(timespec="seconds"))
            write_json(attempt.path, m)
            return attempt

    started = utc_now()
    attempt_id = f"{started:%Y%m%dT%H%M%SZ}-{secrets.token_hex(3)}"
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": 1,
        "attempt_id": attempt_id,
        "judge_hash": judge_hash,
        "run_id": run.run_id,
        "status": RUNNING,
        "judge": f"{judge.provider}:{judge.model}",
        "protocol": protocol,
        "dataset_hash": labels_hash,
        "reference_hash": protocol["references"],
        "outputs_digest": outputs_digest,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": None,
    }
    attempt = Attempt(directory / f"{attempt_id}.json", manifest)
    write_json(attempt.path, manifest)
    logger.info("New judgment attempt %s (%s)", attempt.ref, manifest["judge"])
    return attempt


def judge_run(
    run: Run,
    client: LLMClient,
    judge: LLMConfig,
    labels: EvalDataset | None = None,
    *,
    resume: bool = True,
) -> Attempt:
    """Judge every stored answer of a complete run into a judgment attempt, and complete it.

    ``labels`` default to the run's own; others are snapshotted beside it.

    Raises:
        RunIncomplete: the run, or (after an interruption) the attempt, isn't complete.
        InputChanged: ``labels`` change a sample's query.
        DailyRequestBudgetSpent: a hosted judge's budget ran out; rerunning resumes the attempt.
    """
    from rag.eval.score import selected_from  # score imports this module

    stage = run.manifest["stage"]
    if run.manifest["status"] != COMPLETE:
        raise RunIncomplete(f"{run.run_id} is {run.manifest['status']}; only a complete run can be judged")
    if stage not in RUBRICS:
        raise ValueError(f"{run.run_id} is a {stage} run; only {', '.join(RUBRICS)} runs are judged")
    labels = labels if labels is not None else run.labels()
    selected = selected_from(run, labels)
    attempt = open_attempt(run, judge, selected, labels_hash=run.save_labels(labels), resume=resume)
    outputs = {record["sample_id"]: record for record in run.read_outputs()}
    done = attempt.records()
    checkpoint = attempt.checkpoint()
    for sample in selected:
        if sample.id not in done:
            checkpoint.append(RUBRICS[stage].judge(sample, outputs[sample.id], client))
    attempt.complete([s.id for s in selected])
    return attempt


def open_answer_run(
    stage: str,
    config: RagConfig,
    corpora: list[str] | None,
    *,
    runs_dir: Path,
    dataset_path: Path,
    dataset: EvalDataset,
    selected: EvalDataset,
    oracle: bool,
    fresh: bool,
    variant: str,
) -> Run:
    """Check the index and open (or resume) an answer run for ``answer_eval``/``multihop_eval``.

    The oracle reads no index, so it isn't checked.

    Raises:
        IndexConflict: the index doesn't hold what the config and corpus describe.
    """
    selection, documents, chunks = chunk_selected_corpora(config, corpora)
    index = None if oracle else checked_index(config, selection, documents, chunks, label=f"{stage}_eval")
    return open_run(
        runs_dir, stage=stage, experiment=f"{stage}_eval", variant=variant, config=config, selection=selection,
        dataset_path=dataset_path, dataset=dataset, selected=selected,
        selection_method=f"evenly spaced, {len(selected)} of {len(dataset)}" if len(selected) < len(dataset) else "all",
        index=index, code=code_version(), extra={"retrieval": "oracle" if oracle else "configured"},
        resume=not fresh,
    )


def evaluate_answers_into_run(
    run: Run,
    selected: EvalDataset,
    chat_service: ChatResponder,
    client: LLMClient,
    judge: LLMConfig,
    *,
    fresh: bool = False,
    started: float | None = None,
) -> tuple[Attempt, str]:
    """Generate what ``run`` lacks, judge it as an attempt, and score that: the attempt and revision.

    Every answer is generated before any is judged, so a local generator and
    judge each load once rather than taking turns per sample.
    """
    from rag.eval.score import rescore

    if run.manifest["status"] != COMPLETE:
        generate_into_run(run, selected, chat_service, started=started)
    attempt = judge_run(run, client, judge, resume=not fresh)
    revision, _ = rescore(run, judgment=attempt.ref)
    return attempt, revision


def main(argv: list[str] | None = None) -> int:
    from rag.eval.score import InputChanged, rescore

    configure_logging()
    parser = argparse.ArgumentParser(
        prog="python -m rag.eval.judge",
        description="Judge a completed answer run's stored answers as a new judgment attempt, then score it.",
    )
    parser.add_argument("run", help="Run folder or run id")
    parser.add_argument("--config", default=None, metavar="PATH",
                        help="Config whose eval.judge (else llm) is the judge, before --judge-model")
    answer_eval.add_judge_arguments(parser)
    parser.add_argument("--eval-set", type=Path, default=None, metavar="PATH",
                        help="References to judge against (default: the run's own snapshot)")
    parser.add_argument("--fresh", action="store_true",
                        help="Start a new attempt even if this protocol has an unfinished one")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    args = parser.parse_args(argv)

    try:
        run = load_run(args.run, args.runs_dir)
        judge = answer_eval.resolve_judge_config(load_config(args.config), args.judge_model, args.judge_provider)
        labels = EvalDataset.load(args.eval_set) if args.eval_set else None
        logger.info("Judge: %s:%s", judge.provider, judge.model)
        attempt = judge_run(run, get_llm_client(judge), judge, labels, resume=not args.fresh)
        revision, _ = rescore(run, labels, judgment=attempt.ref)
    except DailyRequestBudgetSpent as exc:
        logger.error("%s Rerun the same command after it resets; the attempt resumes.", exc)
        return 2
    except (FileNotFoundError, ValueError, RunIncomplete, InputChanged) as exc:
        logger.error("%s", exc)
        return 1
    print(f"judgment {attempt.ref}\n{run.run_id}:{revision}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
