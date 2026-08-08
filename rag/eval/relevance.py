"""Turn an eval sample plus a ranked result list into relevance judgments.

This is the seam between the eval *dataset* (what counts as a correct answer)
and the eval *metrics* (pure functions over relevance grades).  Keeping it
separate means :mod:`rag.eval.metrics` never needs to know that spans or
document ids exist -- it operates on grades in rank order, which is the general
form every ranking metric wants.

Two matching modes, chosen per sample by :attr:`~rag.eval.dataset.EvalSample.matching_mode`:

* **span** -- a chunk is relevant if it contains one of the sample's verbatim
  quotes.  Its grade is the highest grade among the spans it contains, so a
  chunk holding two spans counts once rather than being double-rewarded.
* **document** -- a chunk is relevant if it came from an expected document.
  The legacy behaviour, kept because it is a reasonable proxy on short
  documents and because the existing eval set is written this way.

The two are **not comparable** and must not be averaged together; the runner
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

from rag.eval.dataset import MODE_DOCUMENT, MODE_SPAN, EvalSample
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
    if sample.matching_mode == MODE_SPAN:
        return _judge_by_span(sample, chunks)
    return _judge_by_document(sample, chunks)


def _judge_by_span(sample: EvalSample, chunks: list[ScoredChunk]) -> Judgment:
    normalized_chunks = [normalize(chunk.text) for chunk in chunks]
    spans = [(span, normalize(span.text)) for span in sample.expected_spans]

    gains: list[int] = []
    matched: set[str] = set()
    for chunk_text in normalized_chunks:
        grades = [span.grade for span, needle in spans if needle and needle in chunk_text]
        gains.append(max(grades) if grades else 0)

    for span, needle in spans:
        if needle and any(needle in chunk_text for chunk_text in normalized_chunks):
            matched.add(span.text)

    # The best possible ranking puts the highest-graded spans first; it cannot
    # produce more relevant results than there are spans, or than there are
    # ranks to fill.
    ideal = sorted((span.grade for span in sample.expected_spans), reverse=True)

    return Judgment(
        mode=MODE_SPAN,
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


def _fit(values: list[int], length: int) -> list[int]:
    """Pad with zeros or truncate so ``values`` has exactly ``length`` entries."""
    return (values + [0] * length)[:length]
