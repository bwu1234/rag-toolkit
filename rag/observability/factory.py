"""Config-driven factory for `TurnSink`, plus the config fingerprint stamped on every record."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from rag.config.settings import DEFAULT_CARRY_METADATA, DEFAULT_HNSW_EF_SEARCH, REPO_ROOT, RagConfig, TurnLogConfig
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
    `raw` (on `llm` and `agent.llm`) likewise, while it is off.
    `embedding.query_instruction` and `embedding.revision` are handled the same way, as are
    `chunking.carry_metadata`, `chunking.header`, `reranker.include_header`
    and `vector_store.hnsw_ef_search` while each is at its default.

    The daily request budget (`requests_per_day` and its reserve and log) is
    always excluded: it decides whether a turn is sent, not what it says.

    `agent` counts only under `chat.mode: agentic`, the one mode that reads
    it; a pipeline turn's hash doesn't move when agent settings change. For
    the same reason `chat.mode` itself is excluded while it's `pipeline`, so
    adding the field didn't re-key every pipeline turn logged before it.
    """

    exclude: dict[str, Any] = {"observability": True, "eval": True}
    # The daily request budget decides whether a turn is sent, never what it says.
    budget = {"requests_per_day", "requests_per_day_reserve", "daily_request_log"}
    if config.chat.mode == "pipeline":
        exclude["agent"] = True
        exclude["chat"] = {"mode"}
    elif config.agent.llm is not None:
        exclude["agent"] = {"llm": budget | ({"raw"} if not config.agent.llm.raw else set())}
    exclude["llm"] = (
        budget
        | ({"thinking_level"} if config.llm.thinking_level is None else set())
        | ({"raw"} if not config.llm.raw else set())
    )
    embedding_new = {
        name for name in ("query_instruction", "revision") if getattr(config.embedding, name) is None
    }
    if embedding_new:
        exclude["embedding"] = embedding_new
    chunking_new = set()
    if tuple(config.chunking.carry_metadata) == DEFAULT_CARRY_METADATA:
        chunking_new.add("carry_metadata")
    if config.chunking.header.template is None:
        chunking_new.add("header")
    if chunking_new:
        exclude["chunking"] = chunking_new
    if not config.reranker.include_header:
        exclude["reranker"] = {"include_header"}
    if config.vector_store.hnsw_ef_search == DEFAULT_HNSW_EF_SEARCH:
        exclude["vector_store"] = {"hnsw_ef_search"}
    payload = config.model_dump_json(exclude=exclude)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
