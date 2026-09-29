"""Load the selected corpora, and turn them into the chunks an index would hold.

Shared by every command that needs "what the configured pipeline makes of this
corpus": `index` embeds the result, `chunk` and `index-report` describe it, and
the retrieval eval checks its expected spans against it. Keeping one path means
the eval's `unmatchable_spans` count is measured on exactly the chunks the
indexer would write, not on a lookalike.
"""

from __future__ import annotations

import logging

from rag.chunking.chunkers import get_chunker
from rag.chunking.models import Chunk
from rag.config.settings import CorpusSelection, RagConfig
from rag.ingestion.cleaners import clean_documents
from rag.ingestion.loaders import load_corpus
from rag.ingestion.models import Document

logger = logging.getLogger(__name__)


def load_selected_corpora(
    config: RagConfig, corpora: list[str] | None, *, clean: bool = False
) -> tuple[CorpusSelection, list[Document]]:
    """Load every document in the selected corpora, refusing id collisions.

    With ``clean=True`` each corpus's documents are cleaned as they load,
    unless its registry entry sets ``clean: false`` (benchmark text that must
    be indexed exactly as published).

    `Document.id` is a corpus-relative path, so pooling two corpora that each
    contain `faq.txt` produces two documents with the same id -- and therefore
    chunks with the same id, which the vector store would silently upsert over
    one another. The index would come out short by however many documents
    collided, with nothing reporting it.

    Raising is the right response rather than namespacing ids by corpus:
    namespacing would change every `document_id` in the corpus, invalidating the
    `expected_doc_ids` already recorded in the eval sets, to fix a problem the
    current corpora do not have.
    """

    selection = config.corpus_selection(corpora)
    documents: list[Document] = []
    origin: dict[str, str] = {}

    for name, directory in zip(selection.names, selection.document_dirs):
        logger.info("Loading corpus %r from %s", name, directory)
        loaded = load_corpus(directory)
        if clean:
            entry = config.corpora.registry.get(name)
            if entry is None or entry.clean:
                loaded = clean_documents(loaded)
            else:
                logger.info("Corpus %r: cleaning skipped (clean: false), text kept as loaded", name)
        for document in loaded:
            if document.id in origin:
                raise ValueError(
                    f"Document id {document.id!r} appears in both corpus "
                    f"{origin[document.id]!r} and {name!r}. Pooled corpora must have "
                    "distinct document ids -- rename the file in one of them."
                )
            origin[document.id] = name
            documents.append(document)

    return selection, documents


def chunk_selected_corpora(
    config: RagConfig, corpora: list[str] | None
) -> tuple[CorpusSelection, list[Document], list[Chunk]]:
    """Load, clean and chunk the selected corpora with the configured chunker.

    Cleaning follows each corpus's ``clean`` setting.

    Returns the *cleaned* documents, since chunk offsets index into their text.
    """

    selection, documents = load_selected_corpora(config, corpora, clean=True)
    chunks = get_chunker(config.chunking).chunk(documents) if documents else []
    return selection, documents, chunks
