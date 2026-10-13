"""Per-sample checkpoints for long eval runs: save as each sample finishes, resume after a stop.

An answer-eval set is tens of minutes of generation and judging, and an agent
variant can be hours. Writing results only when a set finishes means a crash,
a timeout or a Ctrl-C at the last sample loses all of it. So each finished
sample is appended to a JSON Lines file, and a restarted run skips the samples
already in it.

The checkpoint is not the result. The runner still writes its results file
only when a set is complete, then discards the checkpoint, so a results file
never holds half a set dressed up as a whole one.

What would make resuming wrong is mixing samples from two different setups
into one number. So the first line of every checkpoint is a **fingerprint** of
what produced it -- effective config, judge, dataset, code -- and a checkpoint
whose fingerprint differs from the current run's is refused, naming what
changed, rather than resumed. The code part is the git commit plus a hash of
uncommitted changes under ``rag/`` and ``scripts/``, so an edit between two
sessions counts as a change even before it's committed. The index's
*contents* are not covered: re-indexing the same corpus with the same config
between sessions goes undetected.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

# Re-exported: both lived here before `rag.eval.provenance` existed.
from rag.eval.provenance import code_version, digest

__all__ = ["CheckpointMismatch", "SampleCheckpoint", "code_version", "digest"]

logger = logging.getLogger(__name__)


class CheckpointMismatch(RuntimeError):
    """A checkpoint exists, but was written by a different setup than this run's."""


class SampleCheckpoint:
    """One set's finished samples, one JSON object per line, behind a fingerprint header."""

    def __init__(self, path: Path, fingerprint: dict[str, str]) -> None:
        self.path = path
        self.fingerprint = fingerprint

    def load(self) -> dict[str, dict[str, Any]]:
        """Samples already saved, by ``sample_id``; empty when there is no checkpoint.

        Raises ``CheckpointMismatch`` when the checkpoint came from a different
        setup. A truncated last line -- the process killed mid-write -- is
        dropped and the file rewritten without it, so the next append starts
        on a clean line instead of fusing onto the fragment.
        """
        if not self.path.exists():
            return {}
        lines = self.path.read_text(encoding="utf-8").splitlines()
        if not lines:
            return {}

        try:
            saved = json.loads(lines[0])["fingerprint"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise CheckpointMismatch(f"{self.path} has no readable fingerprint header") from exc
        if saved != self.fingerprint:
            changed = sorted(k for k in saved.keys() | self.fingerprint.keys()
                             if saved.get(k) != self.fingerprint.get(k))
            raise CheckpointMismatch(
                f"{self.path} was written by a different setup (changed: {', '.join(changed)}). "
                "Resuming would mix samples from both into one result; rerun with --fresh "
                "to discard it."
            )

        records: dict[str, dict[str, Any]] = {}
        for number, line in enumerate(lines[1:], start=2):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                if number != len(lines):
                    raise CheckpointMismatch(f"{self.path}:{number} is corrupt") from None
                logger.warning("Dropping a truncated last line from %s", self.path)
                self.path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
                break
            records[record["sample_id"]] = record
        return records

    def append(self, record: dict[str, Any]) -> None:
        """Save one finished sample, writing the header first if the file is new."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.path.exists() or self.path.stat().st_size == 0
        with self.path.open("a", encoding="utf-8") as f:
            if new:
                f.write(json.dumps({"fingerprint": self.fingerprint}) + "\n")
            f.write(json.dumps(record, default=str) + "\n")

    def discard(self) -> None:
        self.path.unlink(missing_ok=True)
