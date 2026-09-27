"""Config-driven factory for `TurnSink`, plus the config fingerprint stamped on every record."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from rag.config.settings import REPO_ROOT, RagConfig, TurnLogConfig
from rag.observability.sink import JsonlTurnSink, TurnSink


def get_turn_sink(config: TurnLogConfig) -> TurnSink | None:
    """Instantiate the sink selected by `config.provider`, or `None` for `"none"`."""

    if config.provider == "none":
        return None
    if config.provider == "jsonl":
        return JsonlTurnSink(turn_log_path(config))
    raise ValueError(
        f"Unknown turn log provider: {config.provider!r}. "
        "Add a TurnSink adapter and register it here to support a new one."
    )


def turn_log_path(config: TurnLogConfig) -> Path:
    return (REPO_ROOT / config.path).resolve()


def config_fingerprint(config: RagConfig) -> str:
    """A short, stable hash of every setting that can change what a turn does.

    `observability` and `eval` are excluded: where the log goes and which judge
    the eval runners use don't change a single answer, and including them would
    split identical-behaviour turns into different buckets.

    `llm.thinking_level` is excluded only while unset, so adding the field
    didn't re-key every turn already logged; setting it changes the hash.

    `agent` is excluded outright, for now: nothing on the query path reads it
    until the agent loop lands, so adding the section must not re-key every
    logged turn either. When `chat.mode: agentic` exists, the agent settings
    change what a turn does and belong in the hash for that mode.
    """

    exclude: dict[str, Any] = {"observability": True, "eval": True, "agent": True}
    if config.llm.thinking_level is None:
        exclude["llm"] = {"thinking_level"}
    payload = config.model_dump_json(exclude=exclude)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
