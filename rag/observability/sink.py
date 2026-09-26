"""Where turn and feedback records go: the `TurnSink` interface and its JSONL default.

An interface, like every other swappable stage, so the default can stay a local
file while an OpenTelemetry (or any other) exporter becomes one more adapter
rather than a rewrite -- a tracing SaaS is deliberately not the default for a
local-first system.
"""

from __future__ import annotations

import json
import logging
import threading
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from rag.observability.records import FeedbackRecord, TurnRecord

logger = logging.getLogger(__name__)


class TurnSink(ABC):
    """Accepts turn and feedback records. Write-only by design.

    Reading back is a property of a particular store (a file can be scanned, an
    OTel collector can't), so it isn't part of the interface -- see
    `read_turn_log` for the JSONL case.
    """

    @abstractmethod
    def record_turn(self, record: TurnRecord) -> None:
        raise NotImplementedError

    @abstractmethod
    def record_feedback(self, record: FeedbackRecord) -> None:
        raise NotImplementedError


class JsonlTurnSink(TurnSink):
    """Appends one JSON object per line to a local file.

    The file is opened per write, in append mode, rather than held open: the
    API and the Streamlit UI may both be running against the same log, and
    short appends to an `O_APPEND` file don't interleave mid-line the way a
    buffered long-lived handle can. The lock covers the threads within one
    process (the API's thread pool).
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def record_turn(self, record: TurnRecord) -> None:
        self._append(record.to_dict())

    def record_feedback(self, record: FeedbackRecord) -> None:
        self._append(record.to_dict())

    def _append(self, payload: dict[str, Any]) -> None:
        line = json.dumps(payload, ensure_ascii=False) + "\n"
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line)


def read_turn_log(path: Path) -> Iterator[dict[str, Any]]:
    """Yield every record in a JSONL turn log, oldest first.

    A line that doesn't parse (a write cut off by a crash) is skipped with a
    warning rather than aborting the read -- one torn line shouldn't hide the
    rest of the log.
    """

    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping unreadable line %d in %s", line_number, path)


def turns_with_feedback(records: Iterable[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any] | None]]:
    """Pair each turn record with the latest feedback given on it, oldest turn first.

    Feedback is appended after the turn it rates (see `FeedbackRecord`), so the
    join happens at read time. Feedback for a turn id not in the log -- a turn
    recorded elsewhere, or a mistyped id sent to `/feedback` -- is dropped.
    """

    turns: list[dict[str, Any]] = []
    latest: dict[str, dict[str, Any]] = {}
    for record in records:
        if record.get("kind") == "turn":
            turns.append(record)
        elif record.get("kind") == "feedback" and isinstance(record.get("turn_id"), str):
            latest[record["turn_id"]] = record
    return [(turn, latest.get(turn.get("turn_id", ""))) for turn in turns]
