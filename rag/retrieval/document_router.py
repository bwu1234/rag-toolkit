"""Document-level routing: pick the filing first, then rank chunks inside it.

On EDGAR, 16.5% of text is identical across a company's filings, so a
question about one quarter competes with the same paragraph from every other
quarter. Chunk text can't tell those copies apart. A document-level record
that names the filing can, which is what `DocumentRouter` ranks.

Each indexed document gets one record, rendered from
`retrieval.document_routing.record_template` (by default its chunk header plus
its period end spelled out). A query is ranked against the records two ways,
BM25 and dense, and the two rankings are fused with RRF, the same pair and
fusion chunk retrieval uses.

**The gate.** A wrong route doesn't rank the answer lower; it filters it out.
The router therefore routes only when BM25 and dense independently put the
same document first, with BM25's top score strictly ahead of its second.
Otherwise it falls back and retrieval runs unfiltered. That rule has no
threshold to hand-set, which is what the plan asks of the fallback. The probe
that chose it is in docs/measured-results.md ("Document routing").

Records are built lazily from the sparse index on first use (one batched
embedding call), not persisted: at 61 filings that costs well under a second
and can never fall out of sync with the chunk index. A corpus of many
thousands of documents would want them persisted at index time instead.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from string import Formatter
from typing import Any

from rank_bm25 import BM25Okapi

from rag.embedding.base import EmbeddingModel
from rag.query_filter import DOCUMENT_ID, QueryFilter
from rag.retrieval.rrf import DEFAULT_RRF_K
from rag.retrieval.similarity import cosine_similarity
from rag.retrieval.sparse import IndexedDocument, SparseIndex, build_bm25, tokenize

logger = logging.getLogger(__name__)

#: The template field holding the chunk header rather than a metadata key.
HEADER_FIELD = "header"
#: Format spec that spells a stored date out ("February 15, 2026").
DATE_SPEC = "date"


def spell_date(value: Any) -> str:
    """A date as questions write it. Accepts a `date`, `YYYYMMDD` int or ISO string.

    Raises:
        ValueError: the value isn't a date in any of those forms.
    """

    if isinstance(value, date):
        day = value
    elif isinstance(value, int) and not isinstance(value, bool):
        day = date(value // 10000, value // 100 % 100, value % 100)
    elif isinstance(value, str):
        day = date.fromisoformat(value)
    else:
        raise ValueError(f"not a date: {value!r}")
    return f"{day:%B} {day.day}, {day.year}"


class _RecordFormatter(Formatter):
    """`str.format` plus the `:date` spec."""

    def format_field(self, value: Any, format_spec: str) -> str:
        if format_spec == DATE_SPEC:
            return spell_date(value)
        return str(super().format_field(value, format_spec))


def record_fields(template: str) -> list[str]:
    """The fields a record template names, in order."""
    return [name for _literal, name, _spec, _conv in Formatter().parse(template) if name is not None]


def check_record_template(template: str, carried: Sequence[str]) -> None:
    """Reject a template naming a field no indexed chunk stores.

    Such a field would leave every document without a record, and routing
    would silently never fire.

    Raises:
        ValueError: a field is neither `header` nor carried metadata.
    """

    allowed = {HEADER_FIELD, *carried}
    unknown = sorted(set(record_fields(template)) - allowed)
    if unknown:
        raise ValueError(
            f"retrieval.document_routing.record_template names {', '.join(unknown)}, which chunks "
            f"don't store; it can use {', '.join(sorted(allowed))} (see chunking.carry_metadata)."
        )


def render_record(template: str, document: IndexedDocument) -> str | None:
    """`template` filled from the document's header and metadata, or `None` if any field is missing."""

    values: dict[str, Any] = {**document.metadata, HEADER_FIELD: document.header}
    for name in record_fields(template):
        if values.get(name) is None or values.get(name) == "":
            return None
    try:
        return _RecordFormatter().vformat(template, (), values)
    except ValueError as exc:
        logger.warning("No routing record for %s: %s", document.document_id, exc)
        return None


@dataclass(frozen=True)
class RoutingDecision:
    """Where routing sent a query: `document_ids` to filter to, or none (fell back)."""

    document_ids: list[str]
    reason: str

    @property
    def routed(self) -> bool:
        return bool(self.document_ids)

    def to_filter(self) -> QueryFilter:
        return QueryFilter(any_of={DOCUMENT_ID: list(self.document_ids)})


class DocumentRouter:
    """Ranks one record per indexed document and routes a query when BM25 and dense agree."""

    def __init__(
        self,
        embedder: EmbeddingModel,
        sparse_index: SparseIndex,
        *,
        top_m: int,
        record_template: str,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> None:
        if top_m <= 0:
            raise ValueError(f"top_m must be positive, got {top_m}")
        self._embedder = embedder
        self._sparse_index = sparse_index
        self.top_m = top_m
        self.record_template = record_template
        self.rrf_k = rrf_k
        self._ids: list[str] | None = None
        self._bm25: BM25Okapi | None = None
        self._vectors: list[list[float]] = []

    def route(self, query: str) -> RoutingDecision:
        """The documents to restrict `query`'s retrieval to, or a fallback with its reason."""

        ids = self._ensure_built()
        if not ids:
            return RoutingDecision([], "no document has a routing record")
        assert self._bm25 is not None

        tokens = tokenize(query)
        bm25_scores = [float(s) for s in self._bm25.get_scores(tokens)] if tokens else [0.0] * len(ids)
        query_vector = self._embedder.embed_query(query)
        dense_scores = [cosine_similarity(query_vector, vector) for vector in self._vectors]

        bm25_order = sorted(range(len(ids)), key=lambda i: bm25_scores[i], reverse=True)
        dense_order = sorted(range(len(ids)), key=lambda i: dense_scores[i], reverse=True)

        best = bm25_order[0]
        if bm25_scores[best] <= 0.0:
            return RoutingDecision([], "no record shares a term with the query")
        if len(ids) > 1 and bm25_scores[best] == bm25_scores[bm25_order[1]]:
            return RoutingDecision([], f"BM25 ties for first ({ids[best]}, {ids[bm25_order[1]]})")
        if dense_order[0] != best:
            return RoutingDecision([], f"BM25 picks {ids[best]}, dense picks {ids[dense_order[0]]}")

        fused = [0.0] * len(ids)
        for order in (bm25_order, dense_order):
            for rank, index in enumerate(order, start=1):
                fused[index] += 1.0 / (self.rrf_k + rank)
        chosen = sorted(range(len(ids)), key=lambda i: fused[i], reverse=True)[: self.top_m]
        return RoutingDecision([ids[i] for i in chosen], "BM25 and dense agree")

    def _ensure_built(self) -> list[str]:
        if self._ids is not None:
            return self._ids

        documents = self._sparse_index.documents()
        records: list[tuple[str, str]] = []
        for document in documents:
            record = render_record(self.record_template, document)
            if record is not None:
                records.append((document.document_id, record))
        missing = len(documents) - len(records)
        if missing:
            # A document without a record can never be routed to, so any query
            # that routes excludes it. Loud, because it looks like a recall bug.
            logger.warning(
                "%d of %d document(s) have no routing record (missing a field in %r); "
                "a routed query can never return them",
                missing,
                len(documents),
                self.record_template,
            )

        self._ids = [document_id for document_id, _ in records]
        if records:
            texts = [text for _, text in records]
            self._bm25 = build_bm25([tokenize(text) for text in texts])
            self._vectors = self._embedder.embed_documents(texts)
        logger.info("Built document routing over %d record(s)", len(records))
        return self._ids
