"""Eval dataset schema and I/O for the RAG evaluation pipeline.

An eval set is a JSON array of :class:`EvalSample` objects.  Each sample has:

* **query** — the natural-language question to ask the system.
* **expected_doc_ids** — document ids (the ``document_id`` field on a
  :class:`~rag.vectorstore.base.ScoredChunk`) that a good retriever should
  surface for this query.  Used by the retrieval eval.  Can be empty if you
  only want answer eval.
* **expected_answer** — a reference answer or answer criteria string.  Used by
  the answer eval.  Can be empty/``None`` if you only want retrieval eval.

The format is intentionally minimal — add extra fields as comments in the JSON
file; they are round-tripped transparently through the ``extra`` dict.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EvalSample:
    """A single evaluation example."""

    id: str
    query: str
    # Document ids (chunk.document_id) that should appear in retrieval results.
    # Matching is document-level, not chunk-level, so a retrieved chunk from
    # the right document counts as a hit regardless of which chunk it is.
    expected_doc_ids: list[str] = field(default_factory=list)
    # Reference answer or answer-quality criteria for the answer eval.
    expected_answer: str | None = None
    # Any extra fields from the JSON file are preserved here.
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d: dict = {
            "id": self.id,
            "query": self.query,
            "expected_doc_ids": self.expected_doc_ids,
        }
        if self.expected_answer is not None:
            d["expected_answer"] = self.expected_answer
        d.update(self.extra)
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "EvalSample":
        known = {"id", "query", "expected_doc_ids", "expected_answer"}
        extra = {k: v for k, v in data.items() if k not in known}
        return cls(
            id=str(data["id"]),
            query=str(data["query"]),
            expected_doc_ids=list(data.get("expected_doc_ids") or []),
            expected_answer=data.get("expected_answer"),
            extra=extra,
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
