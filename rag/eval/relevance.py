"""Turn an eval sample plus a ranked result list into relevance judgments.

This is the seam between the eval *dataset* (what counts as a correct answer)
and the eval *metrics* (pure functions over relevance grades).  Keeping it
separate means :mod:`rag.eval.metrics` never needs to know that spans or
document ids exist -- it operates on grades in rank order, which is the general
form every ranking metric wants.

Three matching modes, chosen per sample by :attr:`~rag.eval.dataset.EvalSample.matching_mode`:

* **span** -- a chunk is relevant if it contains one of the sample's verbatim
  quotes.  Its grade is the highest grade among the spans it contains, so a
  chunk holding two spans counts once rather than being double-rewarded.
* **span_and_document** -- as ``span``, but only chunks from an expected
  document count.  For answers whose text also appears in another document
  (a policy paragraph restated in next year's filing), where ``span`` would
  credit the wrong filing's copy as a hit.
* **document** -- a chunk is relevant if it came from an expected document.
  The legacy behaviour, kept because it is a reasonable proxy on short
  documents and because the existing eval set is written this way.

The modes are **not comparable** and must not be averaged together; the runner
reports them as separate groups.  Document-level precision is inflated on long
documents by construction (every one of a 70-chunk filing's chunks is "relevant"),
which is exactly the distortion span matching exists to remove.

Matching is done on a normalized copy of both sides -- see :func:`normalize` --
so a quote survives the whitespace collapsing that cleaning and chunking apply,
and the typographic punctuation that varies between a filing and a copy-paste.
Neither ``Chunk.text`` nor the stored span is modified.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rag.chunking.models import Chunk
from rag.eval.dataset import (
    MODE_DOCUMENT,
    MODE_SPAN_AND_DOCUMENT,
    EvalDataset,
    EvalSample,
    ExpectedSpan,
)
from rag.vectorstore.base import ScoredChunk

_WHITESPACE = re.compile(r"\s+")

# Typographic variants that differ between a filing, a PDF copy-paste, and a
# hand-typed quote without differing in meaning. Collapsed on both sides before
# matching so an eval set does not fail on an apostrophe.
_PUNCTUATION_FOLD = {
    " ": " ",   # non-breaking space
    "‘": "'",   # left single quote
    "’": "'",   # right single quote
    "“": '"',   # left double quote
    "”": '"',   # right double quote
    "–": "-",   # en dash
    "—": "-",   # em dash
    "…": "...",  # ellipsis
}


def normalize(text: str) -> str:
    """Fold text into the form used for span matching.

    Case-insensitive, whitespace-collapsed, and typographically normalized.
    ``--`` folds to ``-`` as well, so a span quoting an em dash matches a corpus
    where the fetcher already rewrote it as a double hyphen.
    """
    for fancy, plain in _PUNCTUATION_FOLD.items():
        text = text.replace(fancy, plain)
    text = text.replace("--", "-")
    return _WHITESPACE.sub(" ", text).strip().casefold()


@dataclass
class Judgment:
    """Relevance judgments for one sample's ranked results."""

    mode: str
    #: Relevance grade of each retrieved chunk, in rank order.
    gains: list[int]
    #: Best achievable grades at these ranks, for NDCG's denominator.
    ideal_gains: list[int]
    #: How many distinct expected items the ranking covered.
    covered: int
    #: How many distinct expected items the sample declares.
    total_expected: int
    #: Spans the ranking never surfaced -- the useful thing to print on a miss.
    unmatched_spans: list[str] = field(default_factory=list)


def judge_ranking(sample: EvalSample, chunks: list[ScoredChunk]) -> Judgment:
    """Grade a ranked result list against a sample's ground truth."""
    if sample.matching_mode == MODE_DOCUMENT:
        return _judge_by_document(sample, chunks)
    return _judge_by_span(sample, chunks)


def _eligible(sample: EvalSample, document_id: str) -> bool:
    """Whether a chunk from ``document_id`` may count for ``sample`` at all."""
    if sample.matching_mode == MODE_SPAN_AND_DOCUMENT:
        return document_id in sample.expected_doc_ids
    return True


def _judge_by_span(sample: EvalSample, chunks: list[ScoredChunk]) -> Judgment:
    # An ineligible chunk is judged as empty text, so it keeps its rank (and
    # its zero gain) without being able to match a span.
    normalized_chunks = [
        normalize(chunk.text) if _eligible(sample, chunk.document_id) else ""
        for chunk in chunks
    ]
    spans = [(span, _needles(span)) for span in sample.expected_spans]

    gains: list[int] = []
    matched: set[str] = set()
    for chunk_text in normalized_chunks:
        grades = [span.grade for span, needles in spans if _contains(chunk_text, needles)]
        gains.append(max(grades) if grades else 0)

    for span, needles in spans:
        if any(_contains(chunk_text, needles) for chunk_text in normalized_chunks):
            matched.add(span.text)

    # The best possible ranking puts the highest-graded spans first; it cannot
    # produce more relevant results than there are spans, or than there are
    # ranks to fill.
    ideal = sorted((span.grade for span in sample.expected_spans), reverse=True)

    return Judgment(
        mode=sample.matching_mode,
        gains=gains,
        ideal_gains=_fit(ideal, len(gains)),
        covered=len(matched),
        total_expected=len(sample.expected_spans),
        unmatched_spans=[s.text for s in sample.expected_spans if s.text not in matched],
    )


def _judge_by_document(sample: EvalSample, chunks: list[ScoredChunk]) -> Judgment:
    expected = set(sample.expected_doc_ids)
    gains = [1 if chunk.document_id in expected else 0 for chunk in chunks]
    covered = len({c.document_id for c in chunks if c.document_id in expected})

    # Any number of chunks from an expected document are relevant, so the best
    # achievable ranking is "every returned result was relevant" -- unlike span
    # mode, where the ideal is bounded by the number of distinct spans.
    return Judgment(
        mode=MODE_DOCUMENT,
        gains=gains,
        ideal_gains=[1] * len(gains),
        covered=covered,
        total_expected=len(expected),
    )


def _needles(span: ExpectedSpan) -> list[str]:
    """The normalized forms of every quote that satisfies ``span``."""
    return [needle for needle in (normalize(quote) for quote in span.quotes) if needle]


def _contains(haystack: str, needles: list[str]) -> bool:
    """Whether normalized ``haystack`` holds any of one span's ``needles``."""
    return any(needle in haystack for needle in needles)


def _fit(values: list[int], length: int) -> list[int]:
    """Pad with zeros or truncate so ``values`` has exactly ``length`` entries."""
    return (values + [0] * length)[:length]


def unmatched_spans(spans: list[ExpectedSpan], texts: list[str]) -> list[str]:
    """Spans that appear in none of `texts`, under the same normalization as ranking.

    Order-free on purpose: this is for evidence recall over the *union* of
    everything a turn retrieved, where there is no single ranking to grade --
    an agent may search four times, and a span found by any search counts.
    """
    haystacks = [normalize(text) for text in texts]
    return [
        span.text for span in spans
        if not any(_contains(haystack, _needles(span)) for haystack in haystacks)
    ]


def sample_unmatched_spans(sample: EvalSample, passages: list[tuple[str, str]]) -> list[str]:
    """:func:`unmatched_spans` for one sample, honouring its matching mode.

    ``passages`` are ``(document_id, text)`` pairs. Under ``span_and_document``
    a span found only in another document's passage still counts as missing.
    """
    return unmatched_spans(
        sample.expected_spans,
        [text for document_id, text in passages if _eligible(sample, document_id)],
    )


@dataclass(frozen=True)
class UnmatchableSpan:
    """An expected span that no chunk of the corpus contains."""

    sample_id: str
    span: str


def find_unmatchable_spans(dataset: EvalDataset, chunks: list[Chunk]) -> list[UnmatchableSpan]:
    """Spans no chunk in ``chunks`` contains, under each sample's matching rule.

    Such a span can never score as found, whatever retrieval does, so a miss on
    it measures the chunker (or the label), not the retriever. ``chunks`` must
    be what the configured chunker makes of the evaluated corpus: the count is
    what exposes a chunker that cuts answers in two, including the fixed
    chunker the generated set was drafted from, which fits every span by
    construction.
    """
    by_document: dict[str, list[str]] = {}
    for chunk in chunks:
        by_document.setdefault(chunk.document_id, []).append(normalize(chunk.text))
    everything = [text for texts in by_document.values() for text in texts]

    missing: list[UnmatchableSpan] = []
    for sample in dataset:
        if sample.matching_mode == MODE_DOCUMENT:
            continue
        haystacks = (
            [text for doc_id in sample.expected_doc_ids for text in by_document.get(doc_id, [])]
            if sample.matching_mode == MODE_SPAN_AND_DOCUMENT
            else everything
        )
        for span in sample.expected_spans:
            needles = _needles(span)
            if not any(_contains(haystack, needles) for haystack in haystacks):
                missing.append(UnmatchableSpan(sample_id=sample.id, span=span.text))
    return missing
