"""Tests for the eval-set generator's validation logic (`scripts/generate_eval_set.py`).

Only the pure parts are covered -- reply parsing and the validator -- with no
LLM involved.  That is deliberate: the generator's LLM calls are throwaway (a
bad generation is simply dropped), but a **validator bug is silent and
permanent**.  It would let a mislabelled sample into the eval set, and every
configuration comparison run against that set afterwards would be measuring the
label rather than the retriever.

The validator is the only thing standing between "a local 9b paraphrased when
asked to quote" and a corrupted benchmark, so it gets tested like production
code even though it lives in a script.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from generate_eval_set import (  # noqa: E402
    MIN_SPAN_CHARS,
    Candidate,
    Validator,
    parse_reply,
    sample_chunks,
)

from rag.chunking.models import Chunk  # noqa: E402

_LONG_SPAN = "Operating income increased $10.1 billion or 17% during the period."
assert len(_LONG_SPAN) >= MIN_SPAN_CHARS, "fixture must clear the minimum span length"


def _chunk(chunk_id: str, document_id: str, text: str) -> Chunk:
    return Chunk(
        id=chunk_id,
        text=text,
        document_id=document_id,
        source=Path(document_id),
        doc_type="markdown",
        metadata={"chunk_index": 0},
    )


def _candidate(chunk: Chunk, span: str, question: str = "What was operating income?") -> Candidate:
    return Candidate(chunk=chunk, question=question, answer_span=span, answer="$10.1 billion")


def _validator(chunks: list[Chunk], max_span_chars: int = 150) -> Validator:
    return Validator(chunks, max_span_chars=max_span_chars)


# ---------------------------------------------------------------------------
# Reply parsing
# ---------------------------------------------------------------------------


def test_parse_reply_reads_a_bare_json_object() -> None:
    assert parse_reply('{"question": "q", "answer_span": "s"}') == {
        "question": "q",
        "answer_span": "s",
    }


def test_parse_reply_strips_markdown_fences() -> None:
    assert parse_reply('```json\n{"question": "q"}\n```') == {"question": "q"}


def test_parse_reply_ignores_preamble_and_trailing_chatter() -> None:
    reply = 'Sure! Here is the JSON:\n{"question": "q"}\nHope that helps.'
    assert parse_reply(reply) == {"question": "q"}


def test_parse_reply_returns_none_on_garbage() -> None:
    assert parse_reply("no json here") is None
    assert parse_reply('{"unclosed": ') is None


def test_parse_reply_rejects_a_json_array() -> None:
    # A list would blow up later on .get(); reject it at the boundary.
    assert parse_reply('["question"]') is None


# ---------------------------------------------------------------------------
# Validation: the checks that keep bad labels out
# ---------------------------------------------------------------------------


def test_verbatim_span_is_accepted() -> None:
    chunk = _chunk("d.md::chunk0", "d.md", f"Preamble. {_LONG_SPAN} Trailing text.")
    sample = _validator([chunk]).validate(_candidate(chunk, _LONG_SPAN))

    assert sample is not None
    assert sample.expected_spans[0].text == _LONG_SPAN
    assert sample.expected_doc_ids == ["d.md"]


def test_paraphrased_span_is_rejected() -> None:
    """The most common and most dangerous failure: the model rewrites the quote."""
    chunk = _chunk("d.md::chunk0", "d.md", f"Preamble. {_LONG_SPAN}")
    validator = _validator([chunk])

    paraphrase = "Operating income rose by $10.1 billion, an increase of 17%."
    assert validator.validate(_candidate(chunk, paraphrase)) is None
    assert validator.reasons["span_not_verbatim"] == 1


def test_span_matching_tolerates_whitespace_and_smart_quotes() -> None:
    """Normalization must be the same as eval time, or valid labels get dropped."""
    chunk = _chunk("d.md::chunk0", "d.md", "The  Company's\noperating income increased sharply.")
    span = "The Company’s operating income increased sharply."
    assert _validator([chunk]).validate(_candidate(chunk, span)) is not None


def test_span_longer_than_the_overlap_is_rejected() -> None:
    text = "x " * 200
    chunk = _chunk("d.md::chunk0", "d.md", text)
    validator = _validator([chunk], max_span_chars=50)

    assert validator.validate(_candidate(chunk, text[:120])) is None
    assert validator.reasons["span_too_long"] == 1


def test_bare_figure_span_is_rejected_as_too_short() -> None:
    chunk = _chunk("d.md::chunk0", "d.md", "Revenue was $359 million this quarter.")
    validator = _validator([chunk])

    assert validator.validate(_candidate(chunk, "$359 million")) is None
    assert validator.reasons["span_too_short"] == 1


def test_span_appearing_in_two_filings_is_rejected() -> None:
    """Boilerplate cannot identify one passage, so it must not become ground truth."""
    boilerplate = "Our results of operations may fluctuate materially in future periods."
    chunks = [
        _chunk("a.md::chunk0", "a.md", f"Apple filing. {boilerplate}"),
        _chunk("b.md::chunk0", "b.md", f"Microsoft filing. {boilerplate}"),
    ]
    validator = _validator(chunks)

    assert validator.validate(_candidate(chunks[0], boilerplate)) is None
    assert validator.reasons["span_not_unique"] == 1


def test_question_that_copies_the_excerpt_is_rejected() -> None:
    """A question quoting its own source is trivially retrievable and measures nothing."""
    text = f"{_LONG_SPAN} The increase was driven by higher cloud services revenue overall."
    chunk = _chunk("d.md::chunk0", "d.md", text)
    validator = _validator([chunk])

    copied = "What about: the increase was driven by higher cloud services revenue overall?"
    assert validator.validate(_candidate(chunk, _LONG_SPAN, question=copied)) is None
    assert validator.reasons["question_copies_excerpt"] == 1


def test_rejection_reasons_are_tallied_for_reporting() -> None:
    chunk = _chunk("d.md::chunk0", "d.md", f"Preamble. {_LONG_SPAN}")
    validator = _validator([chunk])

    validator.validate(_candidate(chunk, "$1"))
    validator.validate(_candidate(chunk, "$2"))
    validator.validate(_candidate(chunk, "a paraphrase that is long enough to pass length"))

    assert validator.reasons["span_too_short"] == 2
    assert validator.reasons["span_not_verbatim"] == 1


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def test_sampling_spreads_across_documents_rather_than_clustering() -> None:
    """Uniform sampling would over-weight the longest filings and miss short ones."""
    chunks = [
        _chunk(f"big.md::chunk{i}", "big.md", "b" * 500) for i in range(50)
    ] + [_chunk("small.md::chunk0", "small.md", "s" * 500)]

    selected = sample_chunks(chunks, count=4, seed=0)

    assert len({c.document_id for c in selected}) == 2, "the one-chunk document must be reached"


def test_sampling_skips_chunks_too_short_to_question() -> None:
    chunks = [_chunk("d.md::chunk0", "d.md", "tiny"), _chunk("d.md::chunk1", "d.md", "x" * 500)]
    selected = sample_chunks(chunks, count=5, seed=0)
    assert [c.id for c in selected] == ["d.md::chunk1"]


def test_sampling_is_deterministic_for_a_given_seed() -> None:
    chunks = [_chunk(f"d{i}.md::chunk0", f"d{i}.md", "x" * 500) for i in range(20)]
    assert [c.id for c in sample_chunks(chunks, 8, seed=7)] == [
        c.id for c in sample_chunks(chunks, 8, seed=7)
    ]
