"""Turns as OpenTelemetry traces: `TraceWriter` draws them, `OtelTurnSink` exports live ones (Milestone 26).

A turn is drawn after it ends, from its `TurnRecord`, not instrumented live,
so pipeline code stays free of tracing calls. Its spans sit at the times they
ran, each inside the innermost stage that contains it: `rag.observability.timeline`,
which the MLflow export (`rag.eval.tracking`) draws from too.

Attribute names follow the OTel GenAI semantic conventions (`gen_ai.*`) where
one fits, so a GenAI-aware backend reads token counts and models without
configuration, and `rag.*` otherwise. Text -- query, answer, prompts, tool
arguments, stage messages -- is sent only with `capture_content`.

A live turn's trace id is its turn id (both are 128 random bits), so a turn in
the JSONL log or on a `ChatResponse` is found in the backend by the same id,
and feedback given later lands in the same trace. `rag.eval.otel_export` keys
each trace instead, so a re-export sends the same ids rather than a copy.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter
from opentelemetry.sdk.trace.id_generator import RandomIdGenerator
from opentelemetry.trace import Span, Status, StatusCode

from rag.config.settings import OtelExportConfig
from rag.observability.records import AgentToolCall, FeedbackRecord, LLMExchange, StageEvent, TurnRecord
from rag.observability.sink import TurnSink
from rag.observability.timeline import TOOL_STAGES, Node, nodes, parents

logger = logging.getLogger(__name__)

#: Stages that are a GenAI operation of their own.
_STAGE_OPERATIONS = {
    "embed": "embeddings",
    "vector_search": "retrieval",
    "sparse_search": "retrieval",
    "web_search": "retrieval",
}
#: OpenInference's span kind (`openinference.span.kind`) for the stages it can
#: name and the GenAI conventions can't -- a reranker above all. Phoenix reads
#: it; other backends ignore it. Every other stage is a CHAIN.
_STAGE_KINDS = {
    "embed": "EMBEDDING",
    "vector_search": "RETRIEVER",
    "sparse_search": "RETRIEVER",
    "web_search": "RETRIEVER",
    "rerank": "RERANKER",
}
_KIND = "openinference.span.kind"


@dataclass
class _IdPlan:
    """The ids the spans of the trace being drawn take."""

    trace_id: int
    seed: bytes | None = None
    """Derive span ids from this, in creation order, instead of drawing them at random."""
    spans: int = 0

    def next_span_id(self) -> int:
        assert self.seed is not None
        self.spans += 1
        digest = hashlib.sha256(self.seed + self.spans.to_bytes(4, "big")).digest()
        return int.from_bytes(digest[:8], "big") or 1


_PLAN: ContextVar[_IdPlan | None] = ContextVar("rag_otel_id_plan", default=None)


class _PlannedIdGenerator(RandomIdGenerator):
    """Random ids, except inside `TraceWriter.trace`: its trace id, and span ids derived from its key."""

    def generate_trace_id(self) -> int:
        plan = _PLAN.get()
        return plan.trace_id if plan is not None else super().generate_trace_id()

    def generate_span_id(self) -> int:
        plan = _PLAN.get()
        return plan.next_span_id() if plan is not None and plan.seed is not None else super().generate_span_id()


class TraceWriter:
    """Draws turns as traces into its own `TracerProvider`.

    Its own provider rather than the global one: the MLflow pilot and Chroma
    also load the OTel SDK, and these traces shouldn't pick up their spans or
    hand them ours. Pass a `BatchSpanProcessor` to export off the caller's
    thread, a `SimpleSpanProcessor` in tests.
    """

    def __init__(self, processor: SpanProcessor, *, service_name: str, capture_content: bool = False) -> None:
        self.capture_content = capture_content
        self._provider = TracerProvider(
            resource=Resource.create({"service.name": service_name}), id_generator=_PlannedIdGenerator()
        )
        self._provider.add_span_processor(processor)
        self._tracer = self._provider.get_tracer(__name__)

    @property
    def identity(self) -> str:
        """What, besides the record, decides the spans drawn and where they land.

        The resource (service name, and anything `OTEL_RESOURCE_ATTRIBUTES`
        adds, such as a Phoenix project) minus what describes the process
        rather than the spans -- the SDK, and the random `service.instance.id`
        it gives each process -- and whether text is captured. A keyed trace should fold it in: a backend
        that keeps the first write of an id would otherwise ignore a re-export
        made with other settings.
        """

        resource = {k: str(v) for k, v in self._provider.resource.attributes.items()
                    if not k.startswith("telemetry.sdk.") and k != "service.instance.id"}
        return json.dumps({"resource": resource, "capture_content": self.capture_content}, sort_keys=True)

    def add_processor(self, processor: SpanProcessor) -> None:
        """Another processor every span goes to, after the first."""

        self._provider.add_span_processor(processor)

    def force_flush(self) -> bool:
        """Export what's queued now; False if it didn't finish within the default timeout."""

        return self._provider.force_flush()

    def shutdown(self) -> None:
        """Flush what's queued and stop exporting. Also runs at interpreter exit."""

        self._provider.shutdown()

    @contextmanager
    def trace(self, *, turn_id: str | None = None, key: str | None = None) -> Iterator[None]:
        """Root spans started inside take a chosen trace id.

        `turn_id` uses the turn id itself (random span ids); `key` hashes a
        stable name into the trace id and every span id, so drawing the same
        thing twice yields the same ids. With neither, ids stay random.
        """

        plan: _IdPlan | None = None
        if key is not None:
            digest = hashlib.sha256(key.encode("utf-8")).digest()
            plan = _IdPlan(int.from_bytes(digest[:16], "big") or 1, seed=digest)
        elif turn_id is not None and (value := _hex_id(turn_id)) is not None:
            plan = _IdPlan(value)
        token = _PLAN.set(plan)
        try:
            yield
        finally:
            _PLAN.reset(token)

    def span(
        self, name: str, *, start_ns: int, parent: Span | None = None, attributes: Mapping[str, Any] | None = None,
    ) -> Span:
        """Start a span (a root without `parent`), its unset attributes dropped. The caller ends it."""

        context = trace.set_span_in_context(parent) if parent is not None else None
        return self._tracer.start_span(name, context, attributes=_clean(dict(attributes or {})), start_time=start_ns)

    def turn(self, record: TurnRecord, *, attributes: Mapping[str, Any] | None = None) -> None:
        """Draw `record` as a trace; `attributes` are added to its root span."""

        t0 = _epoch_ns(record.timestamp)
        agentic = record.metadata.get("mode", "").startswith("agentic")
        operation = "invoke_agent" if agentic else "invoke_workflow"
        root = self.span(
            f"{operation} rag", start_ns=t0, attributes={**self._turn_attributes(record, operation), **(attributes or {})}
        )
        if record.outcome in ("error", "generation_failed"):
            root.set_status(Status(StatusCode.ERROR, record.error or record.generation_failure or record.outcome))
        ordered = nodes(record)
        spans: list[Span] = []
        open_stages: list[int] = []
        for index, (node, parent) in enumerate(zip(ordered, parents(ordered), strict=True)):
            # A stage ends once a node starts outside it: each span ends as soon
            # as nothing more goes inside it.
            while open_stages and open_stages[-1] != parent:
                done = open_stages.pop()
                spans[done].end(end_time=t0 + _ns(ordered[done].end_ms))
            spans.append(self._node_span(node, record, spans[parent] if parent is not None else root, t0))
            if node.event is not None and node.end_ms > node.start_ms:
                open_stages.append(index)
            else:
                spans[index].end(end_time=t0 + _ns(node.end_ms))
        for done in reversed(open_stages):
            spans[done].end(end_time=t0 + _ns(ordered[done].end_ms))
        root.end(end_time=t0 + _ns(max([record.total_ms, *(node.end_ms for node in ordered)])))

    def feedback(self, record: FeedbackRecord) -> None:
        """A thumbs up/down as a zero-length root span, at the time it was given."""

        at = _epoch_ns(record.timestamp)
        attributes: dict[str, Any] = {"rag.turn_id": record.turn_id, "rag.feedback.rating": record.rating}
        if self.capture_content:
            attributes["rag.feedback.comment"] = record.comment
        self.span("feedback", start_ns=at, attributes=attributes).end(end_time=at)

    def _turn_attributes(self, record: TurnRecord, operation: str) -> dict[str, Any]:
        attributes: dict[str, Any] = {
            "gen_ai.operation.name": operation,
            _KIND: "AGENT" if operation == "invoke_agent" else "CHAIN",
            "gen_ai.agent.name": "rag",
            "rag.turn_id": record.turn_id,
            "rag.outcome": record.outcome,
            "rag.llm_calls": record.llm_calls,
            "rag.llm_ms": record.llm_ms,
            "rag.tool_calls": record.tool_calls,
            "rag.history_turns": record.history_turns,
            "rag.shown_chunk_ids": record.shown_chunk_ids,
            "rag.cited_chunk_ids": record.cited_chunk_ids,
            "rag.invalid_citations": record.invalid_citations,
            "gen_ai.usage.input_tokens": record.prompt_tokens,
            "gen_ai.usage.output_tokens": record.completion_tokens,
            "rag.grounded": record.grounded,
            "rag.stopped_reason": record.stopped_reason,
            "rag.generation_failure": record.generation_failure,
            "rag.estimated_usage_calls": record.estimated_usage_calls or None,
            **{f"rag.{key}": value for key, value in record.metadata.items()},
        }
        if self.capture_content:
            attributes |= {
                "gen_ai.input.messages": _json(_semconv_messages([{"role": "user", "content": record.query}])[0]),
                "gen_ai.output.messages": (
                    _json([{**_semconv_messages([{"role": "assistant", "content": record.answer}])[0][0],
                            "finish_reason": record.final_stop_reason or "stop"}])
                    if record.answer is not None else None
                ),
                "rag.query": record.query,
                "rag.answer": record.answer,
                "rag.rewritten_query": record.rewritten_query,
                "rag.search_queries": record.search_queries,
                "rag.query_filter": _json(record.query_filter) if record.query_filter else None,
                "rag.error": record.error,
            }
        return attributes

    def _node_span(self, node: Node, record: TurnRecord, parent: Span, t0: int) -> Span:
        if node.exchange is not None:
            name, attributes = self._exchange_attributes(node.exchange, record)
        else:
            assert node.event is not None
            name, attributes = self._event_attributes(node.event, node.call)
        span = self.span(name, start_ns=t0 + _ns(node.start_ms), parent=parent, attributes=attributes)
        error = node.exchange.error if node.exchange is not None else None
        if error or (node.call is not None and node.call.status == "error"):
            span.set_status(Status(StatusCode.ERROR, error or (node.call.note if node.call else None)))
        return span

    def _exchange_attributes(self, exchange: LLMExchange, record: TurnRecord) -> tuple[str, dict[str, Any]]:
        operation = "chat" if exchange.kind == "chat" else "text_completion"
        # The turn's metadata names one model: the agent's under agentic mode
        # (its `chat` calls), the pipeline's otherwise (its `generate` calls).
        provider, _, model = record.metadata.get("llm", "").partition(":")
        agentic = record.metadata.get("mode", "").startswith("agentic")
        named = model and (exchange.kind == "chat") == agentic
        attributes: dict[str, Any] = {
            "gen_ai.operation.name": operation,
            _KIND: "LLM",
            "gen_ai.provider.name": provider if named else None,
            "gen_ai.request.model": model if named else None,
            "gen_ai.usage.input_tokens": exchange.prompt_tokens,
            "gen_ai.usage.output_tokens": exchange.completion_tokens,
            "gen_ai.response.finish_reasons": [exchange.stop_reason] if exchange.stop_reason else None,
            "rag.tool_calls_requested": len(exchange.tool_calls) or None,
            "error.type": exchange.error.split(":", 1)[0] if exchange.error else None,
        }
        if self.capture_content:
            sent = exchange.messages or [{"role": "user", "content": exchange.prompt or ""}]
            messages, ids = _semconv_messages(sent)
            reply, _ = _semconv_messages([{"role": "assistant", "content": exchange.response,
                                           "thinking": exchange.thinking, "tool_calls": exchange.tool_calls}], ids)
            reply[0]["finish_reason"] = exchange.stop_reason or "stop"
            attributes |= {
                "gen_ai.system_instructions": (
                    _json([{"type": "text", "content": exchange.system}]) if exchange.system else None
                ),
                "gen_ai.input.messages": _json(messages),
                "gen_ai.output.messages": _json(reply),
                "gen_ai.tool.definitions": _json(exchange.tools) if exchange.tools else None,
            }
        return f"{operation} {model}" if named else operation, attributes

    def _event_attributes(self, event: StageEvent, call: AgentToolCall | None) -> tuple[str, dict[str, Any]]:
        attributes: dict[str, Any] = {"rag.stage": event.stage}
        if event.hits is not None:
            attributes |= {
                "rag.hits.chunk_ids": [hit.chunk_id for hit in event.hits],
                "rag.hits.document_ids": [hit.document_id for hit in event.hits],
                "rag.hits.scores": [hit.score for hit in event.hits],
                "rag.score_scale": event.score_scale,
            }
        if event.ended_ms is None:
            attributes["rag.timing"] = "reconstructed"
        if self.capture_content:
            attributes["rag.message"] = event.message
        tool = TOOL_STAGES.get(event.stage)
        if tool is None:
            return event.stage, attributes | stage_attributes(event.stage)
        tool = call.tool if call is not None else tool
        attributes |= {"gen_ai.operation.name": "execute_tool", _KIND: "TOOL", "gen_ai.tool.name": tool}
        if call is not None:
            attributes |= {
                "rag.tool.step": call.step,
                "rag.tool.status": call.status,
                "rag.tool.passages": call.passages,
                "rag.tool.new_passages": call.new_passages,
                "rag.tool.chunk_ids": call.chunk_ids,
                "rag.tool.documents": call.documents,
                "rag.tool.start": call.start,
                "rag.tool.end": call.end,
            }
            if self.capture_content:
                attributes |= {
                    "rag.tool.query": call.query,
                    "rag.tool.filters": _json(call.filters_raw) if call.filters_raw is not None else None,
                    "rag.tool.expression": call.expression,
                    "rag.tool.result": call.result,
                    "rag.tool.note": call.note,
                }
        return f"execute_tool {tool}", attributes


class OtelTurnSink(TurnSink):
    """Exports each live turn as a trace, and each feedback click as a span in its turn's trace."""

    wants_exchanges = True

    def __init__(self, processor: SpanProcessor, *, service_name: str, capture_content: bool = False) -> None:
        self.writer = TraceWriter(processor, service_name=service_name, capture_content=capture_content)

    def shutdown(self) -> None:
        self.writer.shutdown()

    def record_turn(self, record: TurnRecord) -> None:
        with self.writer.trace(turn_id=record.turn_id):
            self.writer.turn(record)

    def record_feedback(self, record: FeedbackRecord) -> None:
        with self.writer.trace(turn_id=record.turn_id):
            self.writer.feedback(record)


def stage_attributes(stage: str) -> dict[str, Any]:
    """What a pipeline stage's span says about itself, wherever it is drawn."""

    return {"rag.stage": stage, "gen_ai.operation.name": _STAGE_OPERATIONS.get(stage),
            _KIND: _STAGE_KINDS.get(stage, "CHAIN")}


def otlp_exporter(endpoint: str, protocol: str) -> SpanExporter:
    """An OTLP span exporter; headers come from `OTEL_EXPORTER_OTLP_HEADERS`."""

    if protocol == "grpc":
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter as GrpcExporter

        return GrpcExporter(endpoint=endpoint)
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter as HttpExporter
    except ImportError as exc:
        raise ImportError("OTLP over http/protobuf needs the exporter: pip install -e '.[otel]'") from exc
    return HttpExporter(endpoint=endpoint)


def otlp_turn_sink(config: OtelExportConfig) -> OtelTurnSink:
    """An `OtelTurnSink` batching spans to `config.endpoint` over OTLP."""

    return OtelTurnSink(
        BatchSpanProcessor(otlp_exporter(config.endpoint, config.protocol)),
        service_name=config.service_name, capture_content=config.capture_content,
    )


def _semconv_messages(messages: list[dict[str, Any]], ids: int = 0) -> tuple[list[dict[str, Any]], int]:
    """Recorded messages (`usage.message_dict`) in the GenAI conventions' shape: a role and typed parts.

    The record keeps no tool-call ids, so calls are numbered on from `ids` in
    order, and each tool result answers the oldest call still unanswered --
    the order the agent sends them in. Returns the messages and the next id.
    An output message also needs a `finish_reason`, which the caller adds:
    the conventions require it, and Phoenix ignores an output without one.
    """

    pending: list[str] = []
    out: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role", "user")
        parts: list[dict[str, Any]] = []
        if role == "tool":
            parts.append({"type": "tool_call_response", "id": pending.pop(0) if pending else None,
                          "response": message.get("content", "")})
        else:
            if message.get("thinking"):
                parts.append({"type": "reasoning", "content": message["thinking"]})
            if message.get("content"):
                parts.append({"type": "text", "content": message["content"]})
            for call in message.get("tool_calls") or []:
                ids += 1
                pending.append(f"call_{ids}")
                parts.append({"type": "tool_call", "id": f"call_{ids}", "name": call.get("name"),
                              "arguments": call.get("arguments")})
        out.append({"role": role, "parts": parts})
    return out, ids


def _hex_id(turn_id: str) -> int | None:
    try:
        value = int(turn_id, 16)
    except ValueError:
        return None
    return value if 0 < value < 1 << 128 else None


def _epoch_ns(timestamp: str) -> int:
    return int(datetime.fromisoformat(timestamp).timestamp() * 1e9)


def _ns(ms: float) -> int:
    return int(ms * 1e6)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _clean(attributes: dict[str, Any]) -> dict[str, Any]:
    """Drop what's unset: OTel attributes can't be None, and an empty list says nothing."""

    return {key: value for key, value in attributes.items() if value is not None and value != []}
