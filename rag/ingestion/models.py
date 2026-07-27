"""Typed data model for loaded documents, plus the Loader interface.

Granularity is intentionally loader-defined rather than uniform: PDF loaders
emit one `Document` per page (so a citation can point at a page number),
while Markdown/text loaders emit one `Document` per file (page numbers don't
exist for them). Everything downstream — cleaning, chunking, embedding —
operates on `Document.text` + `Document.metadata` uniformly, regardless of
which loader produced it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Document:
    """A single loaded unit of text with metadata for downstream chunking and citation.

    Attributes:
        id: Stable, human-readable identifier — typically
            `"<relative-path>"` or `"<relative-path>#page=<n>"`. Used to trace
            chunks back to their source for citations and eval.
        text: Raw extracted text (cleaning happens in a separate pass — see
            `rag.ingestion.cleaners` — so loaders stay focused on extraction).
        source: Absolute path to the file this document was extracted from.
        doc_type: Short tag identifying the loader/format, e.g. "pdf", "markdown", "text".
        metadata: Loader-specific extras (e.g. `page`, `page_count`, `title`).
            Kept as a plain dict (rather than a fixed schema) because different
            formats surface different metadata; consumers should use `.get()`.
    """

    id: str
    text: str
    source: Path
    doc_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


def make_document_id(relative_path: Path, *, page: int | None = None) -> str:
    """Build a stable, human-readable document id from a corpus-relative path.

    Using a readable id (rather than a hash) makes debugging and eval-set
    authoring much easier — you can look at an id and know exactly which file
    and page it refers to.
    """

    base = relative_path.as_posix()
    return f"{base}#page={page}" if page is not None else base


class Loader(ABC):
    """Interface for format-specific document loaders.

    To support a new format: subclass `Loader`, implement `load`, and register
    an extension -> loader mapping in `rag.ingestion.loaders.get_loader_for`.
    Pipeline code (the directory walker, CLI, etc.) only ever talks to this
    interface.
    """

    #: File extensions (including the leading dot, lowercase) this loader handles.
    extensions: tuple[str, ...]

    @abstractmethod
    def load(self, path: Path, *, corpus_root: Path) -> list[Document]:
        """Load `path` (known to be under `corpus_root`) into one or more `Document`s.

        `corpus_root` is passed through so loaders can compute corpus-relative
        ids via `make_document_id(path.relative_to(corpus_root), ...)`.
        """
        raise NotImplementedError
