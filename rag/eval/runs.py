"""Run folders: one eval run's manifest, dataset snapshot, raw outputs and score revisions.

A results row holds the numbers a run came to, and a rerun replaces it. A run
folder holds what the run *saw*, and is only ever added to::

    data/eval/runs/20261014T101500Z-retrieval-edgar_eval_set-3f9c2a1b/
      manifest.json      # what produced it; status: running | complete
      dataset.json       # the whole eval set as loaded, extra fields included
      outputs.jsonl      # identity header, then one line per sample
      scores/<revision>.json

The folder name is for people; the manifest is what code selects runs by.
``outputs.jsonl`` doubles as the checkpoint: a stopped run is resumed by
appending to it, but only by a run whose identity -- config, dataset,
selected samples, index build, code -- matches the header exactly, and only
when the index build is verified. An index with no build record can't show
it is the same index between two sessions, so such a run always starts over.

A run is marked complete only when its outputs cover exactly the selected
samples, once each. Scores are derived from the stored outputs, never from
live retrieval, so rescoring later (Phase 3) runs the same code over the
same records. See docs/eval-harness-plan.md, "A run folder that is only
ever added to".
"""

from __future__ import annotations

import json
import logging
import os
import socket
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rag.config.settings import CorpusSelection, RagConfig
from rag.eval.checkpoint import SampleCheckpoint
from rag.eval.dataset import EvalDataset
from rag.eval.provenance import (
    IndexProvenance,
    config_hash,
    dataset_hash,
    sample_dicts,
    selection_hash,
)
from rag.hashing import stable_hash

logger = logging.getLogger(__name__)

RUNS_DIR = Path(__file__).resolve().parents[2] / "data" / "eval" / "runs"

MANIFEST = "manifest.json"
DATASET = "dataset.json"
OUTPUTS = "outputs.jsonl"
SCORES = "scores"

RUNNING = "running"
COMPLETE = "complete"


class RunIncomplete(RuntimeError):
    """A run's outputs don't cover exactly its selected samples, once each."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _write_json(path: Path, value: Any) -> None:
    """Write via a temp file and rename, so a crash never leaves half a manifest."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _alive_here(manifest: dict[str, Any]) -> bool:
    """Whether another process on this host that opened ``manifest``'s run is still running.

    Two processes appending to one ``outputs.jsonl`` would interleave their
    samples, so a run some other live process holds is never resumed. Runs on
    another host can't be checked; run folders aren't shared between hosts.
    """
    pid = manifest.get("pid")
    if manifest.get("host") != socket.gethostname() or not isinstance(pid, int) or pid == os.getpid():
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by someone else
    return True


@dataclass
class Run:
    """An open run folder."""

    path: Path
    manifest: dict[str, Any]

    @property
    def run_id(self) -> str:
        return str(self.manifest["run_id"])

    @property
    def outputs_path(self) -> Path:
        return self.path / OUTPUTS

    def checkpoint(self) -> SampleCheckpoint:
        return SampleCheckpoint(self.outputs_path, self.manifest["identity"])

    def load_outputs(self) -> dict[str, dict[str, Any]]:
        """Outputs already recorded, by sample id (empty for a new run)."""
        return self.checkpoint().load()

    def append_output(self, record: dict[str, Any]) -> None:
        self.checkpoint().append(record)

    def read_outputs(self) -> list[dict[str, Any]]:
        """Every recorded output line, in order, duplicates included."""
        if not self.outputs_path.exists():
            return []
        lines = self.outputs_path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines[1:] if line]

    def complete(self) -> None:
        """Mark the run complete once its outputs cover exactly its selected samples.

        Raises:
            RunIncomplete: naming duplicate, missing or unexpected sample ids.
                The run stays ``running`` and can be resumed.
        """
        expected = self.manifest["dataset"]["selected_ids"]
        counts = Counter(r["sample_id"] for r in self.read_outputs())
        problems = []
        if duplicates := sorted(sid for sid, n in counts.items() if n > 1):
            problems.append(f"duplicate outputs for {', '.join(duplicates)}")
        if missing := [sid for sid in expected if sid not in counts]:
            problems.append(f"no output for {', '.join(missing)}")
        if extra := sorted(set(counts) - set(expected)):
            problems.append(f"outputs for unselected samples {', '.join(extra)}")
        if problems:
            raise RunIncomplete(f"{self.path.name}: " + "; ".join(problems))
        self.manifest.update(status=COMPLETE, finished_at=_now().isoformat(timespec="seconds"))
        self.manifest.pop("pid", None)
        _write_json(self.path / MANIFEST, self.manifest)

    def outputs_digest(self) -> str:
        return stable_hash(self.outputs_path.read_text(encoding="utf-8"))

    def write_scores(self, *, scorer: str, settings: dict[str, Any], scores: dict[str, Any]) -> str:
        """Save a score revision derived from this run's outputs; return its id.

        The id hashes what the scores depend on -- the outputs, the labels,
        the scorer and its settings -- so scoring the same run the same way
        again names the same revision, and a label fix or new scorer adds one
        beside it rather than overwriting it.
        """
        if self.manifest["status"] != COMPLETE:
            raise RunIncomplete(f"{self.path.name} is not complete; partial outputs can't be scored")
        source = {
            "outputs_digest": self.outputs_digest(),
            "dataset_hash": self.manifest["dataset"]["dataset_hash"],
            "selection_hash": self.manifest["dataset"]["selection_hash"],
            "scorer": scorer,
            "settings": settings,
        }
        revision = stable_hash(source)
        path = self.path / SCORES / f"{revision}.json"
        if not path.exists():
            path.parent.mkdir(exist_ok=True)
            _write_json(path, {
                "score_revision": revision,
                "run_id": self.run_id,
                **source,
                "scored_at": _now().isoformat(timespec="seconds"),
                **scores,
            })
        return revision


def run_identity(
    *,
    stage: str,
    variant: str,
    config: RagConfig,
    dataset: EvalDataset,
    selected: EvalDataset,
    index: IndexProvenance | None,
    code: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Everything a resumed run must share with the one it continues.

    ``extra`` holds settings outside the config that change what a run
    retrieves, such as the matrix's label-derived filters.
    """
    return {
        "stage": stage,
        "variant": variant,
        "config_hash": config_hash(config),
        "dataset_hash": dataset_hash(dataset),
        "selection_hash": selection_hash([s.id for s in selected]),
        "index_hash": index.index_hash if index else None,
        "corpus_hash": index.corpus_hash if index else None,
        "index_build_id": index.build_id if index else None,
        "code_version": code,
        **({"extra": extra} if extra else {}),
    }


def _resumable(runs_dir: Path, identity: dict[str, Any]) -> Run | None:
    """The newest unfinished run with exactly ``identity`` that nothing is still writing to."""
    if not runs_dir.exists():
        return None
    for manifest_path in sorted(runs_dir.glob(f"*/{MANIFEST}"), reverse=True):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if manifest.get("status") != RUNNING or manifest.get("identity") != identity:
            continue
        if _alive_here(manifest):
            logger.warning("%s matches but is still being written (pid %s); not resuming it",
                           manifest_path.parent.name, manifest["pid"])
            continue
        return Run(manifest_path.parent, manifest)
    return None


def open_run(
    runs_dir: Path,
    *,
    stage: str,
    experiment: str,
    variant: str,
    config: RagConfig,
    selection: CorpusSelection,
    dataset_path: Path,
    dataset: EvalDataset,
    selected: EvalDataset,
    selection_method: str,
    index: IndexProvenance | None,
    code: str,
    extra: dict[str, Any] | None = None,
    resume: bool = True,
) -> Run:
    """Resume the matching unfinished run, or start a new one.

    Resuming needs a verified index build: without a build id, two sessions
    can't be shown to have read the same index, so the run starts over.

    Args:
        selected: The samples this run evaluates, in order.
        selection_method: How ``selected`` was chosen, e.g. ``"all"``.
        index: None when the stage reads no index.
        resume: False to always start a new run.
    """
    identity = run_identity(
        stage=stage, variant=variant, config=config, dataset=dataset, selected=selected,
        index=index, code=code, extra=extra,
    )
    can_resume = index is None or index.verified
    if resume and can_resume and (run := _resumable(runs_dir, identity)) is not None:
        done = len(run.load_outputs())
        logger.info("[%s] resuming %s (%d/%d samples done)", variant, run.path.name, done, len(selected))
        run.manifest.update(pid=os.getpid(), host=socket.gethostname())
        run.manifest.setdefault("resumed_at", []).append(_now().isoformat(timespec="seconds"))
        _write_json(run.path / MANIFEST, run.manifest)
        return run
    if resume and not can_resume:
        logger.info("[%s] index build unverified, so this run can't resume an earlier one", variant)

    started = _now()
    name = f"{started:%Y%m%dT%H%M%SZ}-{stage}-{dataset_path.stem}-{identity['config_hash'][:8]}"
    path = runs_dir / name
    suffix = 1
    while path.exists():
        suffix += 1
        path = runs_dir / f"{name}-{suffix}"
    path.mkdir(parents=True)

    manifest: dict[str, Any] = {
        "schema": 1,
        "run_id": path.name,
        "stage": stage,
        "status": RUNNING,
        "experiment": experiment,
        "variant": variant,
        "identity": identity,
        "config_hash": identity["config_hash"],
        "config": config.model_dump(mode="json"),
        **({"extra": extra} if extra else {}),
        "index": index.as_record() if index is not None else None,
        "corpus": {
            "selection": selection.describe(),
            "slug": selection.slug,
            "collection": selection.collection_name,
        },
        "dataset": {
            "path": str(dataset_path),
            "dataset_hash": identity["dataset_hash"],
            "num_samples": len(dataset),
            "selection": selection_method,
            "selected_ids": [s.id for s in selected],
            "selection_hash": identity["selection_hash"],
        },
        "code_version": code,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": None,
    }
    _write_json(path / DATASET, sample_dicts(dataset))
    _write_json(path / MANIFEST, manifest)
    logger.info("[%s] new run %s", variant, path)
    return Run(path, manifest)
