"""Optional MLflow export of pinned score revisions (eval harness Phase 4b pilot).

    python -m rag.eval.tracking export RUN[:REVISION] ... | --all [--replace]
    python -m rag.eval.tracking compare --baseline RUN[:REV] --candidate RUN[:REV] [--declare PATH]
    python -m rag.eval.tracking serve [--port 5050]

MLflow is a searchable *presentation* of the run folders, never their source:
everything here reads `data/eval/runs/` and writes only to a local tracking
store (SQLite plus a file artifact store under ``data/eval/mlflow/``), which
can be deleted and rebuilt from the run folders at any time. The runner,
scorer and reports never import this module, so they work without MLflow
installed (``pip install -e ".[tracking]"``).

- **One MLflow run per score revision**, in an experiment named
  ``<stage>/<eval set>``. Tags carry our identities (run id, score revision,
  dataset/selection hashes, judge attempt, config hash, index build, code
  version), so the tracker maps back to the run folder. Re-exporting a
  revision already exported is a no-op; a new revision is a new MLflow run.
- **One trace per sample**: the query, the answer or ranked chunks, the
  agent's tool calls, expectations from the labels, and each score as
  feedback (the judge's reply as the rationale), so MLflow's evaluation view
  can show two runs' samples side by side. An answering turn's trace has
  every LLM call verbatim (system prompt, tool results, reasoning) at its
  recorded time, and the run holds each sample's whole turn as Markdown
  (``transcripts/<sample>.md``, the `scripts/trace_question.py` layout). For
  a qwen3.8 agent both show each call's rendered prompt, reconstructed
  from the recorded messages (`transcript.with_rendered_prompts`).
- **Comparisons are ours.** `compare` exports `rag.eval.compare`'s paired
  differences, refusing whatever `report.check_pair` refuses, into a
  ``comparisons/<stage>/<eval set>`` experiment with a link to MLflow's
  side-by-side view of the two runs. A tracker's own aggregate deltas
  decide nothing.

Telemetry: MLflow reports usage by default; this module and ``serve`` turn it
off unless ``MLFLOW_DISABLE_TELEMETRY`` is already set.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sys
import warnings
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any

from rag.events import RETRIEVAL_STAGES
from rag.eval.compare import PAIRED_METRICS, Comparison, compare, flip_rows, render_comparison, samples_by_id
from rag.eval.provenance import flatten, without_denylisted
from rag.eval.report import Ineligible, Pinned, resolve
from rag.eval.runs import COMPLETE, RUNS_DIR, load_run
from rag.logging_config import configure_logging
from rag.observability.records import AgentToolCall, LLMExchange, StageEvent, TurnRecord
from rag.observability.transcript import render_turn_markdown, with_rendered_prompts

logger = logging.getLogger(__name__)

TRACKING_DIR = RUNS_DIR.parent / "mlflow"
#: Bumped when the exported shape changes; `--replace` re-exports older ones.
EXPORTER = "rag.eval.tracking/2"
HOST = "127.0.0.1"

#: Tag keys. ``rag.key`` is the idempotency key a re-export looks up.
KEY, STATUS, EXPORTER_TAG = "rag.key", "rag.export_status", "rag.exporter"
#: MLflow's description tag, on runs and experiments.
NOTE = "mlflow.note.content"
#: Experiment-name prefix for exported comparisons.
COMPARISONS = "comparisons/"
#: Experiment descriptions: where in the UI each kind of experiment's detail lives.
RUN_GUIDE = ("One MLflow run per score revision, one trace per sample (query, retrieved chunks or answer, "
             "agent tool calls, expectations and scores). To read one run: Evaluation runs -> the run's "
             "name -> Traces, then click a row.")
COMPARISON_GUIDE = ("Paired comparisons computed by `rag.eval.compare`: metrics only, no traces. Open a run "
                    "and follow the link in its description to the side-by-side trace view.")
DONE = "complete"

_MAX_PARAM = 6000  # MLflow's MAX_PARAM_VAL_LENGTH
_BATCH = 100  # MLflow's MAX_PARAMS_TAGS_PER_BATCH
#: Characters MLflow refuses in metric and param names (e.g. the parentheses in "(no kind)").
_UNSAFE_NAME = re.compile(r"[^\w\-. :/]")
_LATENCY_SPANS = {"embed": "EMBEDDING", "vector_search": "RETRIEVER", "rerank": "RERANKER"}
#: Turn events that are an LLM call (its exchange has the timing).
_LLM_EVENTS = {"agent_step", "agent_answer"}
_SEARCH_STAGES = {
    stage: {**_LATENCY_SPANS, "sparse_search": "RETRIEVER", "web_search": "RETRIEVER"}.get(stage, "UNKNOWN")
    for stage in RETRIEVAL_STAGES
}


def _mlflow() -> ModuleType:
    os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
    # Synchronous trace writes: an assessment logged right after a trace ends
    # must find the trace, and an interrupted export must not leave half-written ones.
    os.environ["MLFLOW_ENABLE_ASYNC_TRACE_LOGGING"] = "false"
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    try:
        import mlflow
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise SystemExit('MLflow is not installed; `pip install -e ".[tracking]"`') from exc
    return mlflow


def tracking_uri(tracking_dir: Path) -> str:
    return f"sqlite:///{tracking_dir.resolve() / 'mlflow.db'}"


def _ns(iso: str) -> int:
    return int(datetime.fromisoformat(iso).timestamp() * 1e9)


@dataclass(frozen=True)
class Exported:
    key: str
    experiment: str
    run_id: str
    created: bool


class Store:
    """The local tracking store: experiments by name, runs by `rag.key`."""

    def __init__(self, tracking_dir: Path = TRACKING_DIR) -> None:
        self.mlflow = _mlflow()
        self.dir = tracking_dir
        tracking_dir.mkdir(parents=True, exist_ok=True)
        self.mlflow.set_tracking_uri(tracking_uri(tracking_dir))
        self.client = self.mlflow.MlflowClient()

    def experiment(self, name: str) -> str:
        guide = COMPARISON_GUIDE if name.startswith(COMPARISONS) else RUN_GUIDE
        found = self.client.get_experiment_by_name(name)
        if found is None:
            location = (self.dir / "artifacts" / name.replace("/", "__")).resolve().as_uri()
            return str(self.client.create_experiment(name, artifact_location=location, tags={NOTE: guide}))
        if NOTE not in found.tags:  # exported before experiments carried a description
            self.client.set_experiment_tag(found.experiment_id, NOTE, guide)
        return str(found.experiment_id)

    def find(self, experiment_id: str, key: str) -> Any | None:
        escaped = key.replace("'", "\\'")
        runs = self.client.search_runs([experiment_id], filter_string=f"tags.`{KEY}` = '{escaped}'")
        return runs[0] if runs else None

    def delete(self, run: Any) -> None:
        """Remove a partial or replaced export: its traces, then the run itself."""
        traces = self.mlflow.search_traces(
            locations=[run.info.experiment_id], run_id=run.info.run_id, return_type="list"
        )
        if traces:
            self.client.delete_traces(run.info.experiment_id, trace_ids=[t.info.trace_id for t in traces])
        self.client.delete_run(run.info.run_id)

    def claim(self, experiment: str, key: str, *, replace: bool) -> tuple[str, Any | None]:
        """The experiment id and an export already complete under ``key`` (None: export it)."""
        experiment_id = self.experiment(experiment)
        existing = self.find(experiment_id, key)
        if existing is not None:
            if existing.data.tags.get(STATUS) == DONE and not replace:
                return experiment_id, existing
            logger.info("Removing %s export of %s", existing.data.tags.get(STATUS, "a partial"), key)
            self.delete(existing)
        return experiment_id, None

    def log_params(self, run_id: str, params: dict[str, Any]) -> None:
        from mlflow.entities import Param

        items = [Param(_UNSAFE_NAME.sub("_", k)[:250], str(v)[:_MAX_PARAM]) for k, v in params.items()]
        for i in range(0, len(items), _BATCH):
            self.client.log_batch(run_id, params=items[i:i + _BATCH])

    def log_metrics(self, run_id: str, metrics: dict[str, float], timestamp_ms: int) -> None:
        from mlflow.entities import Metric

        items = [Metric(_UNSAFE_NAME.sub("_", k)[:250], float(v), timestamp_ms, 0) for k, v in metrics.items()]
        for i in range(0, len(items), 1000):
            self.client.log_batch(run_id, metrics=items[i:i + 1000])


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


def experiment_name(pinned: Pinned, prefix: str = "") -> str:
    m = pinned.run.manifest
    return f"{prefix}{m['stage']}/{Path(m['dataset']['path']).stem}"


def _numbers(value: Any, prefix: str = "") -> Iterator[tuple[str, float]]:
    """Every number in a nested score mapping, as dotted metric keys (booleans and lists skipped)."""
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, int | float):
        yield prefix, float(value)
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from _numbers(v, f"{prefix}.{k}" if prefix else str(k))


def run_metrics(scores: dict[str, Any]) -> dict[str, float]:
    skip = {"samples", "settings", "unmatchable_spans", "failed_ids", "pass_ci"}
    metrics = dict(_numbers({k: v for k, v in scores.items() if k not in skip}))
    if scores.get("pass_ci"):
        metrics["pass_ci.low"], metrics["pass_ci.high"] = scores["pass_ci"]
    if isinstance(scores.get("unmatchable_spans"), list):
        metrics["unmatchable_spans"] = float(len(scores["unmatchable_spans"]))
    return metrics


def run_tags(pinned: Pinned) -> dict[str, str]:
    m, s = pinned.run.manifest, pinned.scores
    index, extra = m.get("index") or {}, m.get("extra") or {}
    judgment = s["settings"].get("judgment") or {}
    tags = {
        KEY: pinned.ref,
        EXPORTER_TAG: EXPORTER,
        "rag.run_id": pinned.run.run_id,
        "rag.score_revision": pinned.revision,
        "rag.stage": m["stage"],
        "rag.variant": m["variant"],
        "rag.experiment": m.get("experiment") or "",
        "rag.eval_set": m["dataset"]["path"],
        "rag.dataset_hash": s["dataset_hash"],
        "rag.run_dataset_hash": m["dataset"]["dataset_hash"],
        "rag.selection_hash": m["dataset"]["selection_hash"],
        "rag.scorer": s["scorer"],
        "rag.judge_hash": s["settings"].get("judge_hash") or "",
        "rag.judge": judgment.get("judge") or "",
        "rag.judge_attempt": judgment.get("attempt_id") or "",
        "rag.config_hash": m["config_hash"],
        "rag.corpus": m["corpus"]["selection"] if isinstance(m.get("corpus"), dict) else "",
        "rag.corpus_hash": index.get("corpus_hash") or "",
        "rag.index_hash": index.get("index_hash") or "",
        "rag.index_build_id": index.get("index_build_id") or "",
        "rag.code_version": m["code_version"],
        "rag.repeat_group": ((extra.get("repeat") or {}).get("group")) or "",
    }
    return {k: str(v) for k, v in tags.items() if v != ""}


def export_run(store: Store, pinned: Pinned, *, replace: bool = False) -> Exported:
    """Export one score revision and its samples; a no-op if it is already exported."""
    mlflow, m = store.mlflow, pinned.run.manifest
    experiment = experiment_name(pinned)
    experiment_id, existing = store.claim(experiment, pinned.ref, replace=replace)
    if existing is not None:
        return Exported(pinned.ref, experiment, existing.info.run_id, created=False)

    started = _ns(m["started_at"]) // 1_000_000
    tags = {**run_tags(pinned), STATUS: "exporting", NOTE: _run_note(pinned)}
    run = store.client.create_run(
        experiment_id, start_time=started, tags=tags,
        run_name=f"{m['variant']} · {pinned.revision[:8]}",
    )
    run_id = run.info.run_id
    overrides = (m.get("extra") or {}).get("overrides") or {}
    store.log_params(run_id, {
        **flatten(without_denylisted(m["config"])),
        **{f"override.{k}": json.dumps(v, sort_keys=True) for k, v in flatten(overrides).items()},
    })
    store.log_metrics(run_id, run_metrics(pinned.scores), started)
    with mlflow.start_run(run_id=run_id):
        _log_dataset(mlflow, pinned)
        mlflow.log_dict(m, "manifest.json")
        mlflow.log_dict(pinned.scores, f"scores/{pinned.revision}.json")
        if (attempt := _judgment_path(pinned, ".json")) is not None:
            mlflow.log_artifact(str(attempt), "judgment")  # the judge protocol and rubric behind the verdicts
        _log_traces(mlflow, pinned)
    store.client.set_tag(run_id, STATUS, DONE)
    finished = m.get("finished_at")
    store.client.set_terminated(run_id, end_time=_ns(finished) // 1_000_000 if finished else None)
    return Exported(pinned.ref, experiment, run_id, created=True)


def _run_note(pinned: Pinned) -> str:
    m = pinned.run.manifest
    return (
        f"Exported from run folder `{pinned.run.run_id}`, score revision `{pinned.revision}` "
        f"(`{pinned.scores['scorer']}`). The run folder is canonical; this run can be rebuilt "
        f"from it with `python -m rag.eval.tracking export {pinned.ref} --replace`.\n\n"
        f"Eval set `{m['dataset']['path']}`, {m['dataset']['num_samples']} samples in the set, "
        f"selection `{m['dataset'].get('selection', '')}`."
    )


def _log_dataset(mlflow: ModuleType, pinned: Pinned) -> None:
    """The eval set as a metadata-only dataset input: name and label hash, no rows uploaded."""
    from mlflow.data.dataset_source_registry import resolve_dataset_source
    from mlflow.data.meta_dataset import MetaDataset

    path = Path(pinned.run.manifest["dataset"]["path"])
    with warnings.catch_warnings():  # two registered local sources match a file:// URI; either will do
        warnings.filterwarnings("ignore", message="The specified dataset source can be interpreted")
        source = resolve_dataset_source((pinned.run.path / "dataset.json").resolve().as_uri())
    # MLflow marks `profile` abstract in its stubs but MetaDataset defines it; its own docs build one this way.
    dataset = MetaDataset(source, name=path.stem, digest=pinned.scores["dataset_hash"][:36])  # type: ignore[abstract]
    mlflow.log_input(dataset, context="eval")


# ---------------------------------------------------------------------------
# Traces: one per sample
# ---------------------------------------------------------------------------


def _judgment_path(pinned: Pinned, suffix: str) -> Path | None:
    """The judgment attempt file a revision's verdicts came from, if it names one."""
    judgment = pinned.scores["settings"].get("judgment") or {}
    judge_hash = pinned.scores["settings"].get("judge_hash")
    if not judgment or not judge_hash:
        return None
    path = pinned.run.path / "judgments" / judge_hash / f"{judgment['attempt_id']}{suffix}"
    return path if path.exists() else None


def judge_replies(pinned: Pinned) -> dict[str, list[dict[str, Any]]]:
    """Sample id -> the judgment attempt's verdicts, for answer revisions that name one."""
    path = _judgment_path(pinned, ".jsonl")
    if path is None:
        return {}
    lines = path.read_text(encoding="utf-8").splitlines()[1:]
    return {r["sample_id"]: r.get("verdicts") or [] for r in map(json.loads, filter(None, lines))}


#: An assessment to log: name, value, rationale, source.
Feedback = tuple[str, Any, str | None, Any]


def _log_traces(mlflow: ModuleType, pinned: Pinned) -> None:
    from mlflow.entities import AssessmentSource, AssessmentSourceType

    stage = pinned.run.manifest["stage"]
    outputs = {r["sample_id"]: r for r in pinned.run.read_outputs()}
    labels = {s.id: s.to_dict() for s in pinned.run.labels(pinned.scores["dataset_hash"])}
    replies = judge_replies(pinned)
    scorer = AssessmentSource(source_type=AssessmentSourceType.CODE, source_id=pinned.scores["scorer"])
    judge = AssessmentSource(
        source_type=AssessmentSourceType.LLM_JUDGE,
        source_id=(pinned.scores["settings"].get("judgment") or {}).get("judge") or "judge",
    )
    clock = _ns(pinned.run.manifest["started_at"])
    models = _models(pinned.run.manifest["config"])
    for sid, score in samples_by_id(pinned.scores).items():
        out, label = outputs[sid], labels.get(sid, {})
        tags = {k: str(v) for k, v in {"rag.sample_id": sid, "rag.kind": label.get("kind") or score.get("kind"),
                                       "rag.tier": label.get("tier")}.items() if v}
        if stage == "retrieval":
            trace_id, took = _retrieval_trace(mlflow, out, tags, clock)
            feedback: list[Feedback] = [(k, score[k], None, scorer)
                                        for k in ("hit_rate", "recall", "mrr", "ndcg") if k in score]
        else:
            record = _turn_record(out, models)
            if record is not None:
                tags["rag.transcript"] = f"transcripts/{_UNSAFE_NAME.sub('_', sid)}.md"
                mlflow.log_text(_transcript(pinned, sid, record, label), tags["rag.transcript"])
            trace_id, took = _answer_trace(mlflow, out, record, models, tags, clock)
            feedback = _answer_feedback(stage, score, replies.get(sid, []), scorer, judge)
        clock += took
        for name in ("expected_answer", "expected_doc_ids", "expected_spans"):
            if label.get(name):
                mlflow.log_expectation(trace_id=trace_id, name=name, value=label[name])
        for name, value, rationale, source in feedback:
            mlflow.log_feedback(trace_id=trace_id, name=name, value=value, rationale=rationale, source=source)


def _documents(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Chunks in MLflow's retriever-document shape, which its UI renders as a ranked list."""
    return [
        {"id": c["chunk_id"], "page_content": c.get("text", ""),
         "metadata": {"doc_uri": c.get("document_id"), "chunk_id": c["chunk_id"],
                      **{k: c[k] for k in ("rank", "score") if c.get(k) is not None}}}
        for c in chunks
    ]


def _retrieval_trace(mlflow: ModuleType, out: dict[str, Any], tags: dict[str, str], start: int) -> tuple[str, int]:
    latency = out.get("latency_ms") or {}
    inputs = {"query": out["query"], **({"filter": out["filter"]} if out.get("filter") else {})}
    attributes = {k: out[k] for k in ("routed_to", "search_queries", "candidate_count", "dropped_below_min_score")
                  if out.get(k) not in (None, [])}
    root = mlflow.start_span_no_context("retrieve", "RETRIEVER", inputs=inputs, attributes=attributes,
                                        tags=tags, start_time_ns=start)
    # Stage timings are recorded, so the timeline is real: stages run in order.
    t = start
    for stage, span_type in _LATENCY_SPANS.items():
        if stage in latency:
            span = mlflow.start_span_no_context(stage, span_type, parent_span=root, start_time_ns=t)
            t += int(latency[stage] * 1e6)
            span.end(end_time_ns=t)
    total = int(latency.get("total", (t - start) / 1e6) * 1e6)
    root.set_outputs(_documents(out.get("results") or []))
    root.end(end_time_ns=start + total)
    return root.trace_id, total


@dataclass(frozen=True)
class _Models:
    """Who answered: the agent's model (its `chat` calls) and the pipeline's (`generate` calls)."""

    chat: dict[str, Any]
    generate: dict[str, Any]

    @property
    def renders_qwen38(self) -> bool:
        """Whether the agent's prompts can be reconstructed with `ollama_raw`'s qwen3.8 renderer."""
        return self.chat.get("provider") == "ollama" and str(self.chat.get("model", "")).startswith("qwen3.8")


def _models(config: dict[str, Any]) -> _Models:
    llm = config.get("llm") or {}
    return _Models(chat=(config.get("agent") or {}).get("llm") or llm, generate=llm)


def _turn_record(out: dict[str, Any], models: _Models) -> TurnRecord | None:
    """The sample's stored turn, rendered prompts reconstructed where the agent's model allows; None if absent."""
    if not out.get("turn"):
        return None
    record = TurnRecord.from_dict(out["turn"])
    return with_rendered_prompts(record, models.chat.get("think")) if models.renders_qwen38 else record


def _transcript(pinned: Pinned, sid: str, record: TurnRecord, label: dict[str, Any]) -> str:
    m = pinned.run.manifest
    context = [("Sample", f"{sid} ({m['dataset']['path']})"), ("Variant", m["variant"]),
               ("Run", f"`{pinned.ref}`")]
    expected = label.get("expected_answer")
    return render_turn_markdown(record, title=f"Trace: {sid}", context=context,
                                expected_answer=expected if isinstance(expected, str) else None)


def _ms(value: float | None) -> int:
    return int((value or 0.0) * 1e6)


def _answer_trace(
    mlflow: ModuleType, out: dict[str, Any], record: TurnRecord | None, models: _Models,
    tags: dict[str, str], start: int,
) -> tuple[str, int]:
    calls = out.get("agent_calls") or []
    took = int((out.get("latency_s") or 0.0) * 1e9)
    attributes = {k: out[k] for k in ("llm_calls", "llm_ms", "prompt_tokens", "completion_tokens",
                                      "retrieval_rounds", "cited_chunk_ids", "invalid_citations",
                                      "generation_failure", "grounded")
                  if out.get(k) not in (None, [])}
    root = mlflow.start_span_no_context("answer", "AGENT" if calls else "CHAIN", inputs={"query": out["query"]},
                                        attributes=attributes, tags=tags, start_time_ns=start)
    passages = {p["chunk_id"]: p for p in out.get("passages") or []}
    if record is not None and record.llm_exchanges is not None:
        took = max(took, _turn_spans(mlflow, root, record, passages, models, start) - start)
    else:
        # Only the sample's total latency is recorded, so tool spans are
        # zero-length markers in call order rather than an invented timeline.
        for call in calls:
            _tool_span(mlflow, root, AgentToolCall.from_dict(call), None, [], passages, start)
    if passages:
        span = mlflow.start_span_no_context("passages shown", "RETRIEVER", parent_span=root,
                                            start_time_ns=start + took)
        span.set_outputs(_documents(list(passages.values())))
        span.end(end_time_ns=start + took)
    root.set_outputs({"answer": out.get("answer") or ""})
    root.end(end_time_ns=start + took)
    return root.trace_id, took


def _turn_spans(
    mlflow: ModuleType, root: Any, record: TurnRecord, passages: dict[str, dict[str, Any]], models: _Models,
    start: int,
) -> int:
    """Each LLM call at its recorded time, and each tool call after the call that asked for it; returns the end."""
    exchanges = record.llm_exchanges or []
    end = start
    for number, exchange in enumerate(exchanges, 1):
        end = max(end, _llm_span(mlflow, root, exchange, number, models, start))
    calls: dict[int, list[AgentToolCall]] = defaultdict(list)
    for call in record.agent_calls:
        calls[call.step].append(call)
    # A tool's event follows the LLM call that asked for it and comes after the
    # search stages it ran; its own duration covers theirs. Tools run in order.
    step, cursor = 0, start
    stages: list[StageEvent] = []
    for event in record.events:
        if event.stage in _LLM_EVENTS:
            step += 1
            if step <= len(exchanges):
                cursor = start + _ms(exchanges[step - 1].started_ms + exchanges[step - 1].elapsed_ms)
            stages = []
        elif event.stage in _SEARCH_STAGES:
            stages.append(event)
        else:
            asked = calls[step].pop(0) if calls[step] else None
            cursor = _tool_span(mlflow, root, asked, event, stages, passages, cursor)
            stages = []
        end = max(end, cursor)
    for leftover in (c for step_calls in calls.values() for c in step_calls):  # no event recorded for it
        _tool_span(mlflow, root, leftover, None, [], passages, start)
    return max(end, start + _ms(record.total_ms))


def _tool_span(
    mlflow: ModuleType, root: Any, call: AgentToolCall | None, event: StageEvent | None, stages: list[StageEvent],
    passages: dict[str, dict[str, Any]], start: int,
) -> int:
    """One tool call with its search stages laid out in order inside it; returns its end."""
    name = call.tool if call is not None else event.stage if event is not None else "tool"
    inputs = {} if call is None else {
        k: v for k, v in {"query": call.query, "filters": call.filters, "filters_raw": call.filters_raw,
                          "documents": call.documents, "expression": call.expression, "start": call.start,
                          "end": call.end}.items() if v not in (None, [], "")}
    attributes: dict[str, Any] = {"timing": "recorded" if event is not None else "not recorded"}
    if call is not None:
        attributes["step"] = call.step
    if event is not None:
        attributes["event"] = event.message
    span = mlflow.start_span_no_context(name, "TOOL", parent_span=root, inputs=inputs, attributes=attributes,
                                        start_time_ns=start)
    t = start
    for stage in stages:
        stage_attributes = {"event": stage.message}
        if stage.hits is not None:
            stage_attributes["score"] = stage.score_scale or "not recorded"
        child = mlflow.start_span_no_context(stage.stage, _SEARCH_STAGES[stage.stage], parent_span=span,
                                             attributes=stage_attributes, start_time_ns=t)
        if stage.hits is not None:
            child.set_outputs(_documents([
                {"chunk_id": h.chunk_id, "document_id": h.document_id, "score": h.score, "rank": rank,
                 "text": passages.get(h.chunk_id, {}).get("text", "")}
                for rank, h in enumerate(stage.hits, 1)
            ]))
        t += _ms(stage.elapsed_ms)
        child.end(end_time_ns=t)
    end = start + _ms(event.elapsed_ms) if event is not None else start
    if call is not None:
        found = [passages[c] for c in call.chunk_ids if c in passages]
        span.set_outputs({k: v for k, v in {"status": call.status, "note": call.note, "result": call.result,
                                             "passages": call.passages, "new_passages": call.new_passages,
                                             "documents": _documents(found)}.items() if v not in (None, [])})
    span.end(end_time_ns=max(end, t))
    return max(end, t)


def _llm_span(mlflow: ModuleType, root: Any, exchange: LLMExchange, number: int, models: _Models, start: int) -> int:
    """One LLM call in OpenAI chat format, which MLflow's trace view renders as a conversation; returns its end."""
    from mlflow.tracing.constant import SpanAttributeKey

    model = (models.chat if exchange.kind == "chat" else models.generate).get("model")
    begin = start + _ms(exchange.started_ms)
    if exchange.kind == "chat":
        messages, ids = _openai_messages(exchange.messages)
    else:
        messages, ids = ([{"role": "system", "content": exchange.system}] if exchange.system else []) + [
            {"role": "user", "content": exchange.prompt or ""}], 0
    attributes: dict[str, Any] = {
        SpanAttributeKey.MESSAGE_FORMAT: "openai",
        **{k: v for k, v in {"stop_reason": exchange.stop_reason, "thinking": exchange.thinking,
                             "error": exchange.error}.items() if v},
    }
    if exchange.tools:
        attributes[SpanAttributeKey.CHAT_TOOLS] = [{"type": "function", "function": t} for t in exchange.tools]
    if exchange.prompt_tokens is not None and exchange.completion_tokens is not None:
        attributes[SpanAttributeKey.CHAT_USAGE] = {
            "input_tokens": exchange.prompt_tokens, "output_tokens": exchange.completion_tokens,
            "total_tokens": exchange.prompt_tokens + exchange.completion_tokens}
    if exchange.raw_prompt is not None:
        # Fenced: MLflow's default view renders attribute strings as Markdown,
        # which would turn the template's "# Tools" into a heading.
        reconstructed = exchange.raw_output is None
        attributes["rendered_prompt" + (" (reconstructed)" if reconstructed else "")] = _verbatim(exchange.raw_prompt)
        if not reconstructed:
            attributes["raw_output"] = _verbatim(exchange.raw_output or "")
    span = mlflow.start_span_no_context(f"{exchange.kind} {number}", "CHAT_MODEL", parent_span=root,
                                        inputs={"model": model, "messages": messages}, attributes=attributes,
                                        start_time_ns=begin)
    if exchange.error:
        span.set_status("ERROR")
    else:
        reply = _openai_assistant({"content": exchange.response, "thinking": exchange.thinking,
                                   "tool_calls": exchange.tool_calls}, ids)[0]
        span.set_outputs({"object": "chat.completion", "model": model, "choices": [
            {"index": 0, "message": reply, "finish_reason": exchange.stop_reason}]})
    end = begin + _ms(exchange.elapsed_ms)
    span.end(end_time_ns=end)
    return end


def _verbatim(text: str) -> str:
    """``text`` in a code fence longer than any backtick run inside it."""
    fence = "`" * max(3, max((len(run) for run in re.findall(r"`+", text)), default=0) + 1)
    return f"{fence}text\n{text}\n{fence}"


def _openai_assistant(message: dict[str, Any], ids: int) -> tuple[dict[str, Any], int]:
    """A recorded assistant turn as an OpenAI message; tool calls get ids numbered on from ``ids``."""
    out: dict[str, Any] = {"role": "assistant", "content": message.get("content") or ""}
    if message.get("thinking"):
        out["reasoning_content"] = message["thinking"]
    calls = []
    for call in message.get("tool_calls") or []:
        ids += 1
        calls.append({"id": f"call_{ids}", "type": "function", "function": {
            "name": call["name"], "arguments": json.dumps(call["arguments"], ensure_ascii=False)}})
    if calls:
        out["tool_calls"] = calls
    return out, ids


def _openai_messages(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Recorded messages (`usage.message_dict`) as OpenAI chat messages, results paired to calls by id."""
    out: list[dict[str, Any]] = []
    ids, waiting = 0, []
    for message in messages:
        if message["role"] == "assistant":
            turn, first = _openai_assistant(message, ids), ids
            out.append(turn[0])
            ids = turn[1]
            waiting += [f"call_{i}" for i in range(first + 1, ids + 1)]
        elif message["role"] == "tool":
            out.append({"role": "tool", "tool_call_id": waiting.pop(0) if waiting else "",
                        "name": message.get("name"), "content": message.get("content") or ""})
        else:
            out.append({"role": message["role"], "content": message.get("content") or ""})
    return out, ids


def _answer_feedback(
    stage: str, score: dict[str, Any], verdicts: list[dict[str, Any]], scorer: Any, judge: Any,
) -> list[Feedback]:
    feedback: list[Feedback]
    if stage == "multihop":
        parts = list((score.get("parts") or {}).items())
        replies = score.get("judge_outputs") or {}
        feedback = [("complete", score.get("complete"), None, scorer)]
        # Numbered, not named: part names differ per question, and one column per
        # name would scatter the evaluation table across dozens of columns.
        feedback += [(f"part {i}", passed, f"{name}: {replies.get(name, '')}".strip(), judge)
                     for i, (name, passed) in enumerate(parts, 1)]
        if score.get("evidence_recall") is not None:
            missing = score.get("missing_spans") or []
            feedback.append(("evidence_recall", score["evidence_recall"],
                             ("missing: " + " | ".join(missing)) if missing else None, scorer))
        return feedback
    reply = " | ".join(v.get("judge_output") or "" for v in verdicts) or None
    feedback = [("passed", score.get("passed"), reply, judge)]
    feedback += [(k, score[k], None, scorer) for k in ("evidence_retrieved", "grounded") if score.get(k) is not None]
    return feedback


# ---------------------------------------------------------------------------
# Comparisons
# ---------------------------------------------------------------------------


def comparison_key(comparison: Comparison) -> str:
    declared = ",".join(sorted(comparison.declared))
    return f"{comparison.baseline.ref}..{comparison.candidate.ref}" + (f"|{declared}" if declared else "")


def export_comparison(store: Store, comparison: Comparison, *, replace: bool = False) -> Exported:
    """Export both revisions (if needed) and the harness's paired comparison of them."""
    base = export_run(store, comparison.baseline)
    cand = export_run(store, comparison.candidate)
    key = comparison_key(comparison)
    experiment = experiment_name(comparison.baseline, prefix=COMPARISONS)
    experiment_id, existing = store.claim(experiment, key, replace=replace)
    if existing is not None:
        return Exported(key, experiment, existing.info.run_id, created=False)

    side_by_side = (f"#/experiments/{store.experiment(base.experiment)}/evaluation-runs"
                    f"?selectedRunUuid={cand.run_id}&compareToRunUuid={base.run_id}")
    note = (f"[Open the two runs side by side in MLflow]({side_by_side})\n\n"
            f"Computed by `rag.eval.compare` and eligible under `report.check_pair`; MLflow's own "
            f"aggregate deltas are a view, not the comparison.\n\n{render_comparison(comparison)}")
    b, c = comparison.baseline, comparison.candidate
    run = store.client.create_run(experiment_id, run_name=f"{c.run.manifest['variant']} vs {b.run.manifest['variant']}",
                                  tags={KEY: key, EXPORTER_TAG: EXPORTER, STATUS: "exporting",
                                        NOTE: note[:8000],
                                        "rag.baseline": b.ref, "rag.candidate": c.ref,
                                        "rag.baseline_mlflow_run": base.run_id, "rag.candidate_mlflow_run": cand.run_id,
                                        "rag.confounders": str(len(comparison.confounders))})
    run_id = run.info.run_id
    store.log_params(run_id, {"baseline": b.ref, "candidate": c.ref, "declared": ",".join(comparison.declared)})
    metrics: dict[str, float] = {}
    for mc in comparison.metrics:
        d, name = mc.difference, mc.metric.name
        metrics.update({f"{name}.baseline": mc.baseline_mean, f"{name}.candidate": mc.candidate_mean,
                        f"{name}.diff": d.mean_diff, f"{name}.ci_low": d.ci_low, f"{name}.ci_high": d.ci_high,
                        f"{name}.n": d.n, f"{name}.wins": d.wins, f"{name}.losses": d.losses})
        if mc.metric.binary:
            metrics[f"{name}.mcnemar_p"] = d.p_value
    store.log_metrics(run_id, {k: v for k, v in metrics.items() if v not in (float("inf"), float("-inf"))},
                      int(datetime.now().timestamp() * 1000))
    with store.mlflow.start_run(run_id=run_id):
        store.mlflow.log_text(render_comparison(comparison), "comparison.md")
        rows = flip_rows(comparison)
        store.mlflow.log_table({k: [r[k] for r in rows] for k in rows[0]} if rows else {}, "flips.json")
    store.client.set_tag(run_id, STATUS, DONE)
    store.client.set_terminated(run_id)
    return Exported(key, experiment, run_id, created=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _all_refs(runs_dir: Path) -> list[str]:
    refs = []
    for path in sorted(p for p in runs_dir.iterdir() if (p / "manifest.json").exists()):
        run = load_run(path)
        if run.manifest["status"] == COMPLETE and run.manifest["stage"] in PAIRED_METRICS:
            refs += [f"{run.run_id}:{rev}" for rev in run.score_revisions()]
    return refs


def _print(exported: Exported) -> None:
    print(f"{exported.key} -> {exported.experiment} {exported.run_id} "
          f"({'exported' if exported.created else 'already exported'})")


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    parser = argparse.ArgumentParser(prog="python -m rag.eval.tracking", description="Optional MLflow export of pinned score revisions.")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--tracking-dir", type=Path, default=TRACKING_DIR)
    sub = parser.add_subparsers(dest="command", required=True)
    ex = sub.add_parser("export", help="export score revisions as MLflow runs with per-sample traces")
    ex.add_argument("refs", nargs="*", metavar="RUN[:REVISION]")
    ex.add_argument("--all", action="store_true", help="every revision of every complete run")
    ex.add_argument("--replace", action="store_true", help="delete and re-export revisions already exported")
    cp = sub.add_parser("compare", help="export the harness's paired comparison of two revisions")
    cp.add_argument("--baseline", required=True, metavar="RUN[:REVISION]")
    cp.add_argument("--candidate", required=True, metavar="RUN[:REVISION]")
    cp.add_argument("--declare", action="append", default=[], metavar="PATH",
                    help="an intended difference, as for rag.eval.report. Repeatable.")
    cp.add_argument("--replace", action="store_true")
    sv = sub.add_parser("serve", help="the MLflow UI over the tracking store, on loopback")
    sv.add_argument("--port", type=int, default=5050)  # macOS AirPlay Receiver holds 5000
    args = parser.parse_args(argv)

    if args.command == "serve":
        mlflow_bin = shutil.which("mlflow") or sys.exit('MLflow is not installed; `pip install -e ".[tracking]"`')
        os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
        args.tracking_dir.mkdir(parents=True, exist_ok=True)
        print(f"MLflow UI on http://{HOST}:{args.port}/ (telemetry "
              f"{'off' if os.environ['MLFLOW_DISABLE_TELEMETRY'].lower() == 'true' else 'ON'}); Ctrl-C stops it",
              flush=True)
        os.execv(mlflow_bin, [mlflow_bin, "server", "--backend-store-uri", tracking_uri(args.tracking_dir),
                              "--host", HOST, "--port", str(args.port)])

    try:
        if args.command == "export":
            if not args.refs and not args.all:
                parser.error("name revisions to export, or pass --all")
            refs = _all_refs(args.runs_dir) if args.all else args.refs
            pinned = [resolve(ref, args.runs_dir) for ref in refs]
            store = Store(args.tracking_dir)
            for p in pinned:
                _print(export_run(store, p, replace=args.replace))
        else:
            comparison = compare(resolve(args.baseline, args.runs_dir), resolve(args.candidate, args.runs_dir),
                                 tuple(args.declare))
            print(render_comparison(comparison), end="")
            _print(export_comparison(Store(args.tracking_dir), comparison, replace=args.replace))
    except (FileNotFoundError, ValueError) as exc:  # Ineligible is a ValueError
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["Exported", "Ineligible", "Store", "export_comparison", "export_run", "main"]
