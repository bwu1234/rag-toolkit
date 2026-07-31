"""Format-specific loaders plus a directory walker that dispatches by extension.

Adding a new format: write a `Loader` subclass, add it to `_LOADERS` below.
Nothing else in the pipeline needs to change — `load_corpus` and the CLI only
depend on the `Loader` interface.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pypdf import PdfReader

from rag.ingestion.models import Document, Loader, make_document_id

logger = logging.getLogger(__name__)


class PdfLoader(Loader):
    """Extracts one `Document` per page from a PDF using pypdf.

    Per-page granularity is a deliberate choice: it lets citations point at a
    specific page (`source.pdf, p. 4`) rather than just a filename, which
    matters a lot for trustworthy answers. The cost is more, smaller documents
    going into chunking — acceptable since chunking will likely re-merge short
    pages depending on `chunk_size`.
    """

    extensions = (".pdf",)

    def load(self, path: Path, *, corpus_root: Path) -> list[Document]:
        relative = path.relative_to(corpus_root)
        reader = PdfReader(str(path))
        title = _pdf_title(reader) or path.stem
        page_count = len(reader.pages)

        documents: list[Document] = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            documents.append(
                Document(
                    id=make_document_id(relative, page=page_number),
                    text=text,
                    source=path,
                    doc_type="pdf",
                    metadata={
                        "title": title,
                        "page": page_number,
                        "page_count": page_count,
                    },
                )
            )
        return documents


class MarkdownLoader(Loader):
    """Loads an entire Markdown file as a single `Document`.

    Markdown has no native pagination, so file-level granularity is the
    natural unit; structure-aware splitting (by heading) is left to chunking,
    which can be made markdown-aware later without touching this loader.
    """

    extensions = (".md", ".markdown")

    def load(self, path: Path, *, corpus_root: Path) -> list[Document]:
        relative = path.relative_to(corpus_root)
        text = path.read_text(encoding="utf-8", errors="replace")
        return [
            Document(
                id=make_document_id(relative),
                text=text,
                source=path,
                doc_type="markdown",
                metadata={"title": _first_heading(text) or path.stem},
            )
        ]


class TextLoader(Loader):
    """Loads a plain-text file as a single `Document`."""

    extensions = (".txt",)

    def load(self, path: Path, *, corpus_root: Path) -> list[Document]:
        relative = path.relative_to(corpus_root)
        text = path.read_text(encoding="utf-8", errors="replace")
        return [
            Document(
                id=make_document_id(relative),
                text=text,
                source=path,
                doc_type="text",
                metadata={"title": path.stem},
            )
        ]


# Extension -> loader instance. Loaders are stateless, so one shared instance
# per format is enough; `get_loader_for` looks them up case-insensitively.
_LOADERS: dict[str, Loader] = {}
for _loader in (PdfLoader(), MarkdownLoader(), TextLoader()):
    for _ext in _loader.extensions:
        _LOADERS[_ext] = _loader


def get_loader_for(path: Path) -> Loader | None:
    """Return the registered loader for `path`'s extension, or None if unsupported."""

    return _LOADERS.get(path.suffix.lower())


def load_corpus(corpus_dir: Path) -> list[Document]:
    """Walk `corpus_dir` recursively and load every file with a registered loader.

    Files with unsupported extensions (and dotfiles) are skipped with a debug
    log line rather than raising — a corpus directory routinely contains things
    like `.gitkeep`, `README` files, or formats not yet supported, and ingestion
    should degrade gracefully rather than fail the whole run over one file.
    """

    corpus_dir = corpus_dir.resolve()
    if not corpus_dir.is_dir():
        raise FileNotFoundError(f"Corpus directory does not exist: {corpus_dir}")

    documents: list[Document] = []
    skipped = 0

    for path in sorted(corpus_dir.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue

        loader = get_loader_for(path)
        if loader is None:
            logger.debug("Skipping unsupported file: %s", path.relative_to(corpus_dir))
            skipped += 1
            continue

        try:
            loaded = loader.load(path, corpus_root=corpus_dir)
        except Exception:
            logger.exception("Failed to load %s — skipping", path.relative_to(corpus_dir))
            skipped += 1
            continue

        documents.extend(loaded)
        logger.info(
            "Loaded %s -> %d document(s) [%s]",
            path.relative_to(corpus_dir),
            len(loaded),
            loaded[0].doc_type if loaded else "?",
        )

    logger.info("Ingestion complete: %d document(s), %d file(s) skipped", len(documents), skipped)
    return documents


def _pdf_title(reader: PdfReader) -> str | None:
    """Best-effort title extraction from PDF metadata (often absent or junk)."""

    try:
        title = reader.metadata.title if reader.metadata else None
    except Exception:
        return None
    return title.strip() if title and title.strip() else None


def _first_heading(markdown_text: str) -> str | None:
    """Use the first ATX heading (`# Title`) as a document title, if present."""

    for line in markdown_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip() or None
    return None
