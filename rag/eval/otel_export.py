"""Export pinned eval score revisions as OpenTelemetry traces over OTLP (Milestone 26).

    python -m rag.eval.otel_export RUN[:REVISION] ... | --all [--endpoint URL] [--capture-content]

The eval runners never write the turn log, so their turns never reach the
live `OtelTurnSink`; this reads the run folders instead, the way
`rag.eval.tracking` does for MLflow, and sends each sample to any OTLP
backend (Phoenix, Langfuse, a collector) under its own service name, so eval
traffic stays apart from live turns there too.

- **One trace per sample.** An answering sample is its stored turn, drawn by
  `TraceWriter` -- model calls at their recorded times, the agent's searches
  with their stages -- with the sample's scores and label ids on the root
  span. A retrieval sample is its stages, laid end to end from the run's
  start since only their lengths are stored.
- **Ids are keyed** on `<run>:<revision>/<sample>`, the exporter version,
  the service/resource and whether text is captured. Exporting a revision
  again with the same settings sends the same trace and span ids, which a
  backend that keys spans by id stores once; Phoenix keeps the first write
  and ignores the repeat. Other settings, or a new exporter version, are new
  traces rather than a write the backend would drop.
- **Text stays home by default**: query, answer, prompts, judge replies,
  expected answers and part names go only with `--capture-content`. Scores,
  ids, timings and token counts always go.

The endpoint defaults to `observability.turn_log.otel` in the config;
headers (a Langfuse key) come from `OTEL_EXPORTER_OTLP_HEADERS`. Exits 1 if
any span failed to export.
"""

from __future__ import annotations

import argparse
import logging
import threading
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter, SpanExportResult

from rag.config.settings import load_config
from rag.eval.compare import PAIRED_METRICS, samples_by_id
from rag.eval.report import Pinned, resolve
from rag.eval.runs import COMPLETE, RUNS_DIR, load_run
from rag.eval.tracking import EXPORTER_TAG, KEY, judge_replies, run_tags
from rag.logging_config import configure_logging
from rag.observability.otel import TraceWriter, otlp_exporter, stage_attributes
from rag.observability.records import TurnRecord

logger = logging.getLogger(__name__)

SERVICE_NAME = "rag-toolkit-eval"
#: Part of every trace's key: bump it when the spans drawn change, so a
#: re-export reaches a backend that keeps the first write of an id (Phoenix).
EXPORTER = "rag.eval.otel_export/1"
#: Spans the batch processor may hold. Its default (2048) is under one large
#: revision, and a full queue drops spans with only a log line.
_QUEUE = 1 << 16
#: Per-sample score fields that are not scores: the turn's own output, carried by its spans.
_NOT_SCORES = frozenset({"id", "answer", "calls", "parts", "judge_outputs", "missing_spans"})


class _Ended(SpanProcessor):
    """Counts the spans drawn, to check against the spans the exporter saw."""

    def __init__(self) -> None:
        self.count = 0
        self._lock = threading.Lock()

    def on_end(self, span: ReadableSpan) -> None:
        with self._lock:
            self.count += 1


class _CountingExporter(SpanExporter):
    """Passes spans on, counting the ones that failed: the batch processor only logs a failure."""

    def __init__(self, inner: SpanExporter) -> None:
        self.inner = inner
        self.sent = 0
        self.failed = 0
        self._lock = threading.Lock()

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        result = self.inner.export(spans)
        with self._lock:
            if result is SpanExportResult.SUCCESS:
                self.sent += len(spans)
            else:
                self.failed += len(spans)
        return result

    def shutdown(self) -> None:
        self.inner.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self.inner.force_flush(timeout_millis)


def export_revision(writer: TraceWriter, pinned: Pinned) -> int:
    """Draw every sample of one score revision; returns how many."""

    manifest = pinned.run.manifest
    stage = manifest["stage"]
    outputs = {row["sample_id"]: row for row in pinned.run.read_outputs()}
    labels = {s.id: s.to_dict() for s in pinned.run.labels(pinned.scores["dataset_hash"])}
    replies = judge_replies(pinned) if writer.capture_content else {}
    run = {key: value for key, value in run_tags(pinned).items() if key not in (KEY, EXPORTER_TAG)}
    clock = _epoch_ns(manifest["started_at"])
    samples = samples_by_id(pinned.scores)
    for sid, score in samples.items():
        out = outputs[sid]
        attributes = {
            **run,
            "rag.sample_id": sid,
            **_score_attributes(score, writer.capture_content),
            **_label_attributes(labels.get(sid, {}), writer.capture_content),
        }
        if writer.capture_content and replies.get(sid):
            attributes["rag.eval.judge_reply"] = " | ".join(v.get("judge_output") or "" for v in replies[sid])
        with writer.trace(key=f"{EXPORTER}|{writer.identity}|{pinned.ref}/{sid}"):
            if stage == "retrieval":
                clock += _retrieval_trace(writer, out, attributes, clock)
            elif out.get("turn"):
                writer.turn(TurnRecord.from_dict(out["turn"]), attributes=attributes)
            else:
                # Stored before runs kept the turn: the answer and its scores, no spans inside.
                at = clock
                if writer.capture_content:
                    attributes |= {"rag.query": out.get("query"), "rag.answer": out.get("answer")}
                writer.span("answer", start_ns=at, attributes=attributes).end(
                    end_time=at + int((out.get("latency_s") or 0.0) * 1e9)
                )
    return len(samples)


def _retrieval_trace(writer: TraceWriter, out: dict[str, Any], attributes: dict[str, Any], start: int) -> int:
    """A retrieval sample's stages, in the order they ran; returns its length in ns."""

    latency = dict(out.get("latency_ms") or {})
    total = latency.pop("total", None)
    results = out.get("results") or []
    root_attributes = {
        **attributes,
        "gen_ai.operation.name": "retrieval",
        "openinference.span.kind": "RETRIEVER",
        "rag.timing": "laid end to end from the run's start",
        "rag.candidate_count": out.get("candidate_count"),
        "rag.routed_to": out.get("routed_to"),
        "rag.results.chunk_ids": [r["chunk_id"] for r in results],
        "rag.results.document_ids": [r["document_id"] for r in results],
        "rag.results.scores": [r["score"] for r in results],
    }
    if writer.capture_content:
        root_attributes |= {"rag.query": out.get("query"), "rag.search_queries": out.get("search_queries")}
    root = writer.span("retrieval", start_ns=start, attributes=root_attributes)
    t = start
    for stage, ms in latency.items():
        span = writer.span(stage, start_ns=t, parent=root, attributes=stage_attributes(stage))
        t += int(ms * 1e6)
        span.end(end_time=t)
    end = max(t, start + int((total or 0.0) * 1e6))
    root.end(end_time=end)
    return end - start


def _score_attributes(score: dict[str, Any], capture_content: bool) -> dict[str, Any]:
    """Every scalar score as ``rag.eval.<name>``; a multihop answer's parts as a pass list."""

    attributes: dict[str, Any] = {
        f"rag.eval.{name}": value for name, value in score.items()
        if name not in _NOT_SCORES and isinstance(value, bool | int | float | str)
    }
    parts = score.get("parts")
    if isinstance(parts, dict) and parts:
        attributes["rag.eval.parts_passed"] = [bool(passed) for passed in parts.values()]
        if capture_content:
            attributes["rag.eval.parts"] = list(parts)
            replies = score.get("judge_outputs") or {}
            attributes["rag.eval.judge_reply"] = " | ".join(f"{n}: {replies.get(n, '')}" for n in parts)
    if capture_content and score.get("missing_spans"):
        attributes["rag.eval.missing_spans"] = list(score["missing_spans"])
    return attributes


def _label_attributes(label: dict[str, Any], capture_content: bool) -> dict[str, Any]:
    attributes: dict[str, Any] = {
        "rag.eval.kind": label.get("kind"),
        "rag.eval.tier": label.get("tier"),
        "rag.eval.expected_doc_ids": label.get("expected_doc_ids"),
    }
    if capture_content:
        expected = label.get("expected_answer")
        attributes["rag.eval.expected_answer"] = expected if isinstance(expected, str) else None
        attributes["rag.eval.expected_spans"] = label.get("expected_spans")
    return attributes


def _all_refs(runs_dir: Path) -> list[str]:
    refs = []
    for path in sorted(p for p in runs_dir.iterdir() if (p / "manifest.json").exists()):
        run = load_run(path)
        if run.manifest["status"] == COMPLETE and run.manifest["stage"] in PAIRED_METRICS:
            refs += [f"{run.run_id}:{rev}" for rev in run.score_revisions()]
    return refs


def _epoch_ns(iso: str) -> int:
    return int(datetime.fromisoformat(iso).timestamp() * 1e9)


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    parser = argparse.ArgumentParser(
        prog="python -m rag.eval.otel_export", description="Export eval score revisions as OTLP traces."
    )
    parser.add_argument("refs", nargs="*", metavar="RUN[:REVISION]")
    parser.add_argument("--all", action="store_true", help="every revision of every complete run")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--config", type=Path, default=None,
                        help="config whose observability.turn_log.otel supplies the defaults")
    parser.add_argument("--endpoint", help="OTLP traces endpoint (default: the config's)")
    parser.add_argument("--protocol", choices=("http/protobuf", "grpc"), help="(default: the config's)")
    parser.add_argument("--service-name", default=SERVICE_NAME)
    parser.add_argument("--capture-content", action="store_true",
                        help="also send text: query, answer, prompts, judge replies, expected answers")
    args = parser.parse_args(argv)
    if not args.refs and not args.all:
        parser.error("name revisions to export, or pass --all")

    otel = load_config(args.config).observability.turn_log.otel
    endpoint = args.endpoint or (otel.endpoint if otel is not None else None)
    if endpoint is None:
        parser.error("no --endpoint, and the config sets no observability.turn_log.otel")
    protocol = args.protocol or (otel.protocol if otel is not None else "http/protobuf")
    try:
        pinned = [resolve(ref, args.runs_dir) for ref in (_all_refs(args.runs_dir) if args.all else args.refs)]
    except (FileNotFoundError, ValueError) as exc:  # Ineligible is a ValueError
        print(f"error: {exc}")
        return 1

    exporter = _CountingExporter(otlp_exporter(endpoint, protocol))
    writer = TraceWriter(BatchSpanProcessor(exporter, max_queue_size=_QUEUE), service_name=args.service_name,
                         capture_content=args.capture_content)
    ended = _Ended()
    writer.add_processor(ended)
    for p in pinned:
        samples = export_revision(writer, p)
        writer.force_flush()  # one revision at a time, so the queue never fills
        print(f"{p.ref}: {samples} sample trace(s)")
    writer.shutdown()
    lost = ended.count - exporter.sent - exporter.failed  # dropped before export, or still queued
    print(f"{exporter.sent} of {ended.count} span(s) sent to {endpoint}"
          + (f"; {exporter.failed} failed to export" if exporter.failed else "")
          + (f"; {lost} never exported" if lost else ""))
    return 1 if exporter.failed or lost else 0


if __name__ == "__main__":
    raise SystemExit(main())
