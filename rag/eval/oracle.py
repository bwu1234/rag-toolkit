"""The oracle row: answer from each question's gold evidence instead of from retrieval.

Milestone 19 phase 4's diagnostic ceiling. Evidence recall says whether the
gold spans reached the prompt; it can't say what the generator would have done
with them if they all had. The oracle makes that measurable: it hands the
generator the indexed chunks that contain a question's gold spans, and nothing
else, through the pipeline's own prompt, generation and citations. If
`oracle / 9b` is not much above `pipeline / 9b` on multi-hop completeness, the
headroom is in synthesis, which an agent that searches more can't fix.

Gold chunks are found with :func:`~rag.eval.relevance.unmatched_spans`, the
matcher evidence recall uses, over the chunks the configured chunker makes of
the corpus -- the same chunks `index` embeds and `index-report` checks against
the built index. For each span, the first chunk containing it (in corpus order,
among the sample's expected documents when it names any) is used, so a
span-and-document sample never gets another filing's copy of the same sentence.
Chunks are deduplicated and kept in the sample's span order.

It is a diagnostic, never a candidate default, and it needs gold spans: a
refusal sample has none, so the oracle refuses it rather than scoring an empty
prompt. CRAG is refused too, because its retries search again with rewritten
queries that the oracle has no answer for.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from rag.chunking.models import Chunk
from rag.config.settings import RagConfig
from rag.eval.dataset import EvalDataset, EvalSample
from rag.eval.relevance import unmatched_spans
from rag.events import EventSink
from rag.ingestion.corpora import chunk_selected_corpora
from rag.query_filter import QueryFilter
from rag.retrieval.retriever import RetrievalResult
from rag.vectorstore.base import ScoredChunk

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UnfoundSpan:
    """A gold span that no chunk of the sample's documents contains."""

    sample_id: str
    span: str


def gold_chunks(sample: EvalSample, chunks: Sequence[Chunk]) -> tuple[list[Chunk], list[str]]:
    """The chunks holding `sample`'s gold spans, and the spans no chunk holds.

    Raises ``ValueError`` for a sample without gold spans (a refusal sample).
    """

    if not sample.expected_spans:
        raise ValueError(f"Sample {sample.id!r} has no gold spans; the oracle needs them")
    documents = set(sample.expected_doc_ids)
    candidates = [c for c in chunks if not documents or c.document_id in documents]
    picked: dict[str, Chunk] = {}
    unfound: list[str] = []
    for span in sample.expected_spans:
        match = next((c for c in candidates if not unmatched_spans([span], [c.text])), None)
        if match is None:
            unfound.append(span.text)
        else:
            picked.setdefault(match.id, match)
    return list(picked.values()), unfound


def _scored(chunk: Chunk) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk.id,
        text=chunk.text,
        document_id=chunk.document_id,
        source=chunk.source,
        doc_type=chunk.doc_type,
        score=1.0,
        metadata=dict(chunk.metadata),
        context=chunk.context,
        header=chunk.header,
    )


class OracleRetriever:
    """A `PassageRetriever` that answers each known question with its gold chunks.

    Keyed by the exact question text, because that is what `ChatService` passes
    on a single-turn eval. An unknown query raises rather than returning nothing,
    which would read as a retrieval miss and quietly lower the ceiling.
    """

    def __init__(self, gold: dict[str, list[Chunk]]) -> None:
        self._gold = {query: [_scored(c) for c in chunks] for query, chunks in gold.items()}

    def retrieve(
        self,
        query: str,
        *,
        query_filter: QueryFilter | None = None,
        on_event: EventSink | None = None,
    ) -> RetrievalResult:
        if query not in self._gold:
            raise KeyError(f"The oracle has no gold evidence for query {query!r}")
        chunks = list(self._gold[query])
        return RetrievalResult(chunks=chunks, candidate_count=len(chunks), candidates=chunks)


def build_oracle_retriever(
    config: RagConfig, dataset: EvalDataset | Iterable[EvalSample], corpora: list[str] | None
) -> tuple[OracleRetriever, list[UnfoundSpan]]:
    """Chunk the selected corpora as `index` would, and map each sample to its gold chunks.

    Returns the retriever and every gold span no chunk contains. A sample with
    unfound spans still runs on the chunks that were found; the caller decides
    whether a nonempty list is fatal.
    """

    if config.crag.enabled:
        raise ValueError(
            "The oracle replaces retrieval; CRAG's retries would search again with queries "
            "it can't answer. Run it with crag.enabled: false."
        )
    if config.chunking.contextual.enabled:
        # Generated contexts live in the index's cache, not in the chunker's output,
        # so the oracle's passages would lack the context the pipeline's carry.
        raise ValueError("The oracle doesn't carry contextual chunking's contexts; turn it off")
    _selection, _documents, chunks = chunk_selected_corpora(config, corpora)
    gold: dict[str, list[Chunk]] = {}
    unfound: list[UnfoundSpan] = []
    for sample in dataset:
        found, missing = gold_chunks(sample, chunks)
        if sample.query in gold and [c.id for c in gold[sample.query]] != [c.id for c in found]:
            raise ValueError(f"Two samples ask {sample.query!r} with different gold evidence")
        gold[sample.query] = found
        unfound.extend(UnfoundSpan(sample.id, span) for span in missing)
    for miss in unfound:
        logger.warning("Oracle: no chunk holds a gold span of %s: %r", miss.sample_id, miss.span[:80])
    return OracleRetriever(gold), unfound


def add_oracle_argument(parser: argparse.ArgumentParser) -> None:
    """``--oracle``, shared by the answer and multi-hop runners."""
    parser.add_argument(
        "--oracle", action="store_true",
        help="Answer from each question's gold chunks instead of retrieval (a diagnostic "
             "ceiling for generation; needs gold spans, so not the refusal set, and CRAG off).",
    )
