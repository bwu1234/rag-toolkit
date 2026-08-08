"""Persistent checkpoint for generated chunk contexts.

Contextualization is the most expensive operation in this repo -- one LLM call
per chunk, measured at ~3.45s/chunk against `qwen3.5:9b-mlx`.  On the EDGAR
corpus (4,236 chunks) that is roughly four hours sequentially.  A run that long
*will* be interrupted: a laptop sleeps, a daemon restarts, a terminal closes.
Without a checkpoint every interruption throws away all the work.

This is a plain append-only JSONL file mapping a prompt fingerprint to the
blurb it produced.  Append-only rather than write-at-the-end is the entire
point: a run killed at 80% has 80% of its work durably on disk.

What the key covers
-------------------
The fingerprint hashes **every input to the LLM call** -- the system prompt, the
rendered user prompt (which already contains the truncated parent document and
the chunk text), the reply cap, and the model name.  Anything that would change
the generated blurb therefore changes the key and misses the cache.

That is stronger than the index's own change detection, which hashes
``chunk.text`` alone.  It closes a hazard documented in CLAUDE.md: editing
``CONTEXT_SYSTEM_PROMPT`` or ``max_document_chars`` previously left stale
contexts in the index with nothing detecting it.  Here, a changed prompt simply
does not match.

It also means the cache is **safe to keep across ``index --reset``**, which is
what makes reset affordable again -- re-embedding is cheap, regenerating
contexts is not.  ``--clear-context-cache`` forces regeneration when you
actually want it (e.g. comparing two models on identical inputs).

Only successful generations are stored.  A transient failure must not be
memoized as "this chunk has no context" forever.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import TextIO

logger = logging.getLogger(__name__)


def context_cache_path(index_dir: Path) -> Path:
    """Location of the checkpoint file for an index directory."""
    return index_dir / "contextual_cache.jsonl"


def fingerprint(*, system_prompt: str, prompt: str, max_context_chars: int, model: str) -> str:
    """Stable key over every input that can change the generated blurb."""
    digest = hashlib.sha256()
    for part in (system_prompt, prompt, str(max_context_chars), model):
        digest.update(part.encode("utf-8"))
        # Length-prefixed separator so ("ab", "c") and ("a", "bc") differ.
        digest.update(b"\x00")
    return digest.hexdigest()


class ContextCache:
    """Append-only JSONL map of prompt fingerprint -> generated context."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._entries: dict[str, str] = {}
        self._handle: TextIO | None = None
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        corrupt = 0
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    self._entries[record["key"]] = record["context"]
                except (json.JSONDecodeError, KeyError, TypeError):
                    # A run killed mid-write leaves a torn final line. That is
                    # an expected cost of append-on-completion, not corruption
                    # worth failing the run over -- drop it and continue.
                    corrupt += 1
        if corrupt:
            logger.warning(
                "Ignored %d unreadable line(s) in %s (likely a partial write from "
                "an interrupted run)",
                corrupt,
                self.path,
            )
        logger.info("Loaded %d cached chunk context(s) from %s", len(self._entries), self.path)

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, key: str) -> str | None:
        return self._entries.get(key)

    def put(self, key: str, context: str) -> None:
        """Record a generated context and flush it to disk immediately.

        Flushing per entry rather than per batch is deliberate: the whole value
        of the checkpoint is that an interruption keeps everything finished so
        far, and buffering would trade that away for negligible I/O savings
        against a multi-second LLM call.
        """
        self._entries[key] = context
        if self._handle is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.path.open("a", encoding="utf-8")
        self._handle.write(json.dumps({"key": key, "context": context}) + "\n")
        self._handle.flush()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def clear(self) -> None:
        """Drop every cached context, on disk and in memory."""
        self.close()
        self._entries.clear()
        self.path.unlink(missing_ok=True)
        logger.info("Cleared chunk context cache at %s", self.path)

    def __enter__(self) -> "ContextCache":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
