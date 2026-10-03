"""Format-specific loaders plus a directory walker that dispatches by extension.

Adding a new format: write a `Loader` subclass, add an instance to the tuple
that builds `_LOADERS` below.
Nothing else in the pipeline needs to change — `load_corpus` and the CLI only
depend on the `Loader` interface.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import yaml
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
        raw = path.read_text(encoding="utf-8", errors="replace")
        front_matter, text = split_front_matter(raw)
        return [
            Document(
                id=make_document_id(relative),
                text=text,
                source=path,
                doc_type="markdown",
                # Front matter wins over the heading, so a file can state its
                # title outright rather than have it guessed.
                metadata={"title": _first_heading(text) or path.stem, **front_matter},
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


def beir_passage_text(title: str, text: str) -> str:
    """The one string a BEIR passage is indexed, embedded and reranked as.

    ``title + " " + text``, or the body alone when the title is empty (every
    FiQA passage). This is the document recipe of the pinned BGE reference
    vectors; the pinned BM25 index joins with ``"\n"``, which its analyzer
    treats the same as a space (docs/beir-reference-protocol.md). Neither
    part is otherwise altered. The title must be *in* the text: carried
    metadata alone never reaches the embedder, BM25 or the reranker.
    """

    return f"{title} {text}" if title else text


class BeirCorpusLoader(Loader):
    """Loads a BEIR ``corpus.jsonl``: one `Document` per line, id taken verbatim.

    ``Document.id`` is the passage's ``_id`` unchanged, so BEIR qrels match
    retrieved documents with no mapping table. The text is
    `beir_passage_text`; the title is also kept as metadata. Queries and qrels
    live outside the corpus directory (see ``scripts/fetch_beir.py``), so only
    passages can reach an index.

    Raises on a line without ``_id`` or ``text``, or on a repeated ``_id``,
    rather than skipping it: a benchmark corpus that silently loses passages
    scores against labels it no longer contains.
    """

    extensions = (".jsonl",)

    def load(self, path: Path, *, corpus_root: Path) -> list[Document]:
        documents: list[Document] = []
        seen: set[str] = set()
        with path.open(encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict) or "_id" not in row or "text" not in row:
                    raise ValueError(f"{path.name}:{line_number}: a BEIR passage needs '_id' and 'text'")
                doc_id = str(row["_id"])
                if doc_id in seen:
                    raise ValueError(f"{path.name}:{line_number}: duplicate _id {doc_id!r}")
                seen.add(doc_id)
                title = str(row.get("title") or "")
                documents.append(
                    Document(
                        id=doc_id,
                        text=beir_passage_text(title, str(row["text"])),
                        source=path,
                        doc_type="beir",
                        metadata={"title": title},
                    )
                )
        return documents


# Extension -> loader instance. Loaders are stateless, so one shared instance
# per format is enough; `get_loader_for` looks them up case-insensitively.
_LOADERS: dict[str, Loader] = {}
for _loader in (PdfLoader(), MarkdownLoader(), TextLoader(), BeirCorpusLoader()):
    for _ext in _loader.extensions:
        _LOADERS[_ext] = _loader


def get_loader_for(path: Path) -> Loader | None:
    """Return the registered loader for `path`'s extension, or None if unsupported."""

    return _LOADERS.get(path.suffix.lower())


class CorpusLoadError(Exception):
    """Some corpus files could not be loaded, so the corpus on disk was only partly read."""


def load_corpus(corpus_dir: Path, *, failed: list[Path] | None = None) -> list[Document]:
    """Walk `corpus_dir` recursively and load every file with a registered loader.

    Files with unsupported extensions (and dotfiles) are skipped with a debug
    log line rather than raising — a corpus directory routinely contains things
    like `.gitkeep`, `README` files, or formats not yet supported, and ingestion
    should degrade gracefully rather than fail the whole run over one file.

    A supported file whose loader raises is skipped too, but it is not the same
    as an absent file: its documents are missing from the result while the file
    is still in the corpus. Pass ``failed`` to collect those paths -- the indexer
    needs them so it doesn't purge the documents a bad upload replaced.
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
            if failed is not None:
                failed.append(path)
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


def split_front_matter(markdown_text: str) -> tuple[dict[str, Any], str]:
    """Split a leading YAML front matter block off `markdown_text`.

    Returns the parsed mapping and the text after the closing `---`. The block
    is removed rather than left in, so `Document.text` -- and every character
    offset and eval span measured against it -- is the same whether or not a
    file carries front matter. Values keep their YAML types: an unquoted
    `2024-09-28` is a `datetime.date`. Text without a block comes back as-is.

    Raises:
        ValueError: the block opens but never closes, or isn't a mapping.
            Loud on purpose: a file whose metadata silently vanished would
            index without the fields filters and headers depend on.
    """

    if not markdown_text.startswith("---\n"):
        return {}, markdown_text
    end = markdown_text.find("\n---\n", 3)
    if end == -1:
        raise ValueError("front matter opens with `---` but has no closing `---` line")
    parsed = yaml.safe_load(markdown_text[4:end])
    if parsed is None:
        parsed = {}
    if not isinstance(parsed, dict):
        raise ValueError(f"front matter must be a mapping, got {type(parsed).__name__}")
    return parsed, markdown_text[end + len("\n---\n") :]


def _first_heading(markdown_text: str) -> str | None:
    """Use the first ATX heading (`# Title`) as a document title, if present."""

    for line in markdown_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip() or None
    return None
