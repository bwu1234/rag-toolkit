"""Eval dataset schema and I/O for the RAG evaluation pipeline.

An eval set is a JSON array of :class:`EvalSample` objects.  Each sample has:

* **query** — the natural-language question to ask the system.
* **expected_spans** — verbatim quotes from the source that answer the query.
  A retrieved chunk is relevant if it *contains* one.  This is the preferred
  ground truth; see below.
* **expected_doc_ids** — document ids (the ``document_id`` field on a
  :class:`~rag.vectorstore.base.ScoredChunk`) that a good retriever should
  surface.  The fallback when no spans are given.
* **expected_answer** — a reference answer or answer criteria string.  Used by
  the answer eval.  Can be empty/``None`` if you only want retrieval eval.

Why spans, and why verbatim quotes
----------------------------------
Document-level matching is only a proxy for "found the right passage", and the
proxy holds only while documents are short.  On a 3,300-character API reference
(~4 chunks) a chunk from the right document is almost certainly the right chunk.
On a 60,000-character SEC filing (~70 chunks) ``precision@k`` can read 1.0 while
every retrieved chunk is boilerplate from the wrong end of the document.  The
metric silently stops measuring what it claims to.

Worse, granularity would depend on file format rather than intent: the PDF
loader emits one :class:`~rag.ingestion.models.Document` per *page*, while the
Markdown loader emits one per *file*, so the same content graded at
document-level means different things depending on how it was stored.

Spans are recorded as **verbatim quotes rather than character offsets** on
purpose.  Offsets are invalidated by any change to cleaning or chunking, and
comparing across chunking configurations is the whole reason the eval exists —
a ground truth that moves when you change ``chunk_size`` cannot measure
``chunk_size``.  A quote is chunking-invariant and is located at eval time.

Keep spans **short** -- a clause, not a paragraph.  A span matches only a chunk
that contains all of it, so a span the chunker cuts in two matches nothing.
The retrieval eval chunks the corpus with the configured chunker and reports
how many spans no chunk contains as ``unmatchable_spans``, so a chunker that
splits answers shows up as a count rather than as a silently lower score.

A sample can set ``"matching_mode": "span_and_document"`` to require both: a
chunk counts only if it contains a span *and* comes from one of
``expected_doc_ids``.  Use it when the answer text appears verbatim in more
than one document (the same paragraph restated in next year's filing), where
plain span matching would credit the wrong document's copy.

Grading is optional.  A plain string is worth ``DEFAULT_GRADE``; an object
``{"text": ..., "grade": 3}`` allows graded relevance, which is what makes NDCG
meaningful (a passage stating the exact figure outranks one merely discussing
the topic).  The object form also takes ``"alternatives": [...]``: other
verbatim quotes of the same fact, for a fact the corpus states more than once
in different words.

After a set is committed it changes only to fix a label error, and each fix is
recorded on the sample as a ``label_fixes`` entry (date, reason, old values),
which round-trips through ``extra`` like any other annotation.

The format is intentionally minimal — add extra fields as comments in the JSON
file; they are round-tripped transparently through the ``extra`` dict.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

#: Relevance grade assigned to a span written as a bare string.
DEFAULT_GRADE = 1

#: Matching granularity for a sample -- see :attr:`EvalSample.matching_mode`.
MODE_SPAN = "span"
MODE_DOCUMENT = "document"
MODE_SPAN_AND_DOCUMENT = "span_and_document"
MATCHING_MODES = (MODE_SPAN, MODE_DOCUMENT, MODE_SPAN_AND_DOCUMENT)


@dataclass(frozen=True)
class ExpectedSpan:
    """A verbatim quote that a relevant chunk must contain, and its grade.

    ``alternatives`` are other verbatim quotes of the *same fact*, any one of
    which satisfies the span -- e.g. a FY2024 figure restated as the prior-year
    comparative in the FY2025 filing. Several spans on a sample mean "all of
    these facts"; alternatives on one span mean "this fact, quoted any of these
    ways", so a fact stated twice still counts once toward recall.
    """

    text: str
    grade: int = DEFAULT_GRADE
    alternatives: tuple[str, ...] = ()

    @property
    def quotes(self) -> tuple[str, ...]:
        """Every quote that satisfies this span, the primary one first."""
        return (self.text, *self.alternatives)

    def to_json(self) -> str | dict:
        """Round-trip back to the shortest accepted JSON form."""
        if self.grade == DEFAULT_GRADE and not self.alternatives:
            return self.text
        d: dict = {"text": self.text}
        if self.grade != DEFAULT_GRADE:
            d["grade"] = self.grade
        if self.alternatives:
            d["alternatives"] = list(self.alternatives)
        return d

    @classmethod
    def from_json(cls, raw: str | dict) -> "ExpectedSpan":
        if isinstance(raw, str):
            return cls(text=raw)
        if not isinstance(raw, dict) or "text" not in raw:
            raise ValueError(
                "An expected_spans entry must be a string or an object with a "
                f"'text' key, got {raw!r}"
            )
        return cls(
            text=str(raw["text"]),
            grade=int(raw.get("grade", DEFAULT_GRADE)),
            alternatives=tuple(str(a) for a in raw.get("alternatives", ())),
        )


@dataclass
class EvalSample:
    """A single evaluation example."""

    id: str
    query: str
    # Verbatim quotes that a relevant chunk must contain. Preferred ground
    # truth -- see the module docstring for why this beats document ids.
    expected_spans: list[ExpectedSpan] = field(default_factory=list)
    # Document ids (chunk.document_id) that should appear in retrieval results.
    # Used only when no spans are given: a retrieved chunk from the right
    # document counts regardless of which chunk it is, which is a usable proxy
    # for short documents and a misleading one for long ones.
    expected_doc_ids: list[str] = field(default_factory=list)
    # Reference answer or answer-quality criteria for the answer eval.
    expected_answer: str | None = None
    # Any extra fields from the JSON file are preserved here.
    extra: dict = field(default_factory=dict)
    # Explicit matching mode from the JSON file, or None to infer it. Only
    # `span_and_document` needs stating: the other two follow from which
    # fields are filled in.
    explicit_mode: str | None = None

    def __post_init__(self) -> None:
        if self.explicit_mode is None:
            return
        if self.explicit_mode not in MATCHING_MODES:
            raise ValueError(
                f"Sample {self.id!r}: unknown matching_mode {self.explicit_mode!r}; "
                f"expected one of {', '.join(MATCHING_MODES)}"
            )
        needs_spans = self.explicit_mode in (MODE_SPAN, MODE_SPAN_AND_DOCUMENT)
        needs_docs = self.explicit_mode in (MODE_DOCUMENT, MODE_SPAN_AND_DOCUMENT)
        # Refused rather than degraded: a span_and_document sample with no doc
        # ids would silently become plain span matching, which is the exact
        # wrong-period credit the mode exists to prevent.
        if (needs_spans and not self.expected_spans) or (needs_docs and not self.expected_doc_ids):
            raise ValueError(
                f"Sample {self.id!r}: matching_mode {self.explicit_mode!r} needs "
                + " and ".join(
                    name for name, needed in (("expected_spans", needs_spans), ("expected_doc_ids", needs_docs))
                    if needed
                )
            )

    @property
    def matching_mode(self) -> str:
        """Which granularity this sample is graded at.

        An explicit ``matching_mode`` wins. Otherwise spans win when present: a
        sample may legitimately carry both, and the doc ids stay useful for a
        quick eyeball of *where* a result came from even once spans decide the
        score.
        """
        if self.explicit_mode is not None:
            return self.explicit_mode
        return MODE_SPAN if self.expected_spans else MODE_DOCUMENT

    def to_dict(self) -> dict:
        d: dict = {
            "id": self.id,
            "query": self.query,
            "expected_doc_ids": self.expected_doc_ids,
        }
        if self.expected_spans:
            d["expected_spans"] = [s.to_json() for s in self.expected_spans]
        if self.expected_answer is not None:
            d["expected_answer"] = self.expected_answer
        if self.explicit_mode is not None:
            d["matching_mode"] = self.explicit_mode
        d.update(self.extra)
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "EvalSample":
        known = {"id", "query", "expected_doc_ids", "expected_spans", "expected_answer", "matching_mode"}
        extra = {k: v for k, v in data.items() if k not in known}
        return cls(
            id=str(data["id"]),
            query=str(data["query"]),
            expected_spans=[
                ExpectedSpan.from_json(s) for s in (data.get("expected_spans") or [])
            ],
            expected_doc_ids=list(data.get("expected_doc_ids") or []),
            expected_answer=data.get("expected_answer"),
            extra=extra,
            explicit_mode=data.get("matching_mode"),
        )


@dataclass
class EvalDataset:
    """A collection of :class:`EvalSample` objects, backed by a JSON file."""

    samples: list[EvalSample]
    source_path: Path | None = None

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump([s.to_dict() for s in self.samples], f, indent=2, ensure_ascii=False)
            f.write("\n")

    @classmethod
    def load(cls, path: Path | str) -> "EvalDataset":
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, list):
            raise ValueError(f"Eval set at {path} must be a JSON array, got {type(raw).__name__}")
        samples = [EvalSample.from_dict(item) for item in raw]
        return cls(samples=samples, source_path=path)

    @classmethod
    def from_dicts(cls, items: list[dict]) -> "EvalDataset":
        """Convenience constructor for tests — build from plain dicts."""
        return cls(samples=[EvalSample.from_dict(d) for d in items])
