"""Contextual chunking: give each chunk a sentence saying where it came from.

Fixed-size chunking has a failure mode that no amount of retrieval tuning
fixes. Split a document into windows and most windows lose the thing that made
them findable: a chunk reading "The limit is 1,000 requests per minute and
resets on a rolling window" never names the product, the endpoint, or the
plan tier, because the surrounding document already established all three.
Embed that chunk and it lands nowhere near "what is the ACS rate limit?"; put
it in a BM25 index and the term "ACS" isn't in it to match.

The fix (Anthropic's "contextual retrieval") is to ask an LLM, once per chunk,
to write a short blurb situating that chunk in its parent document, and to
index the blurb *prepended to the chunk* rather than the chunk alone. The
retrieval unit gets back the identifying terms the split threw away.

Two properties make this safe to bolt onto an existing index:

- **`Chunk.text` is never modified.** The context lives in its own field and is
  joined on only where indexing happens (`Chunk.contextual_text`). Citations,
  previews, and `char_start`/`char_end` keep pointing at the verbatim span the
  chunker produced, so nothing a user sees is model-generated.
- **It fails open, per chunk.** A chunk whose context generation raises or comes
  back empty is indexed exactly as it would have been with this module absent.
  A degraded chunk is a much better outcome than a failed index run, and this
  runs a full corpus's worth of LLM calls where a single transient error
  otherwise throws the whole run away.

Cost is the reason it's off by default: this is one LLM call *per chunk*, at
index time, with a slice of the parent document in every prompt. That's the
most expensive thing in this repo per run -- and it's paid once per index
rather than once per query, which is exactly the trade the technique is
built on.
"""

from __future__ import annotations

import logging

from rag.chunking.models import Chunk
from rag.generation.llm import LLMClient
from rag.ingestion.models import Document

logger = logging.getLogger(__name__)

CONTEXT_SYSTEM_PROMPT = (
    "You situate an excerpt within the document it came from, so the excerpt "
    "can be found by search on its own. "
    "Given a document and one excerpt from it, write a short standalone note "
    "saying what the excerpt is about and where it sits in the document. "
    "Name the specific things the excerpt refers to only implicitly -- the "
    "product, feature, endpoint, section, or procedure the surrounding "
    "document establishes but the excerpt itself never names. "
    "Use the document's own terminology, including acronyms and identifiers. "
    "Do not summarize the excerpt's content, do not add information that is "
    "not in the document, and do not refer to 'the excerpt' or 'this chunk'. "
    "Reply with one or two plain sentences and nothing else: no preamble, no "
    "heading, no quotation marks."
)


def build_context_prompt(document_text: str, chunk_text: str) -> str:
    """Render the parent document and one excerpt into a single user-turn prompt."""

    return (
        f"<document>\n{document_text}\n</document>\n\n"
        f"<excerpt>\n{chunk_text}\n</excerpt>\n\n"
        "Note situating the excerpt in the document:"
    )


class ChunkContextualizer:
    """Attaches a generated document-context blurb to each chunk.

    Takes an `LLMClient` rather than building one, for the same reason
    `QueryCondenser` and the query expanders do: one provider, one model, one
    connection, chosen by config.

    `max_document_chars` truncates the parent document in the prompt. A long
    document would otherwise blow past the model's context window on every one
    of its chunks, and the head of a document is where the identifying material
    (title, overview, defined terms) almost always lives -- which is precisely
    what the blurb needs. `max_context_chars` bounds the reply, since a blurb
    that runs longer than the chunk starts to dominate what gets embedded.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        *,
        max_document_chars: int = 8000,
        max_context_chars: int = 400,
    ) -> None:
        self._llm_client = llm_client
        self.max_document_chars = max_document_chars
        self.max_context_chars = max_context_chars

    def contextualize(self, chunks: list[Chunk], documents: list[Document]) -> list[Chunk]:
        """Return `chunks` with `context` filled in, in the same order.

        Chunks whose parent document isn't in `documents`, and chunks whose
        generation fails, are returned unchanged -- so the result is always the
        same length and always safe to index.
        """

        by_id = {document.id: document for document in documents}
        contextualized: list[Chunk] = []
        generated = 0

        for chunk in chunks:
            document = by_id.get(chunk.document_id)
            if document is None:
                # Shouldn't happen via the CLI (chunks come from these very
                # documents), but a caller assembling the two lists separately
                # shouldn't get an exception for it.
                logger.warning(
                    "No parent document %r for chunk %r; indexing it without context",
                    chunk.document_id,
                    chunk.id,
                )
                contextualized.append(chunk)
                continue

            context = self._generate(document, chunk)
            if context is None:
                contextualized.append(chunk)
                continue

            generated += 1
            contextualized.append(
                Chunk(
                    id=chunk.id,
                    text=chunk.text,
                    document_id=chunk.document_id,
                    source=chunk.source,
                    doc_type=chunk.doc_type,
                    metadata=dict(chunk.metadata),
                    context=context,
                )
            )

        logger.info("Generated context for %d/%d chunk(s)", generated, len(chunks))
        return contextualized

    def _generate(self, document: Document, chunk: Chunk) -> str | None:
        """Return a context blurb for `chunk`, or `None` to index it unchanged."""

        prompt = build_context_prompt(document.text[: self.max_document_chars], chunk.text)
        try:
            reply = self._llm_client.generate(prompt, system=CONTEXT_SYSTEM_PROMPT)
        except Exception:  # noqa: BLE001 -- fail open; see module docstring
            logger.exception("Context generation failed for chunk %r", chunk.id)
            return None

        # Collapse to a single line: the blurb is prepended to chunk text for
        # embedding, and stray newlines only make the joined string harder to
        # read in a trace or a stored record.
        context = " ".join(reply.split()).strip().strip('"').strip()
        if not context:
            logger.warning("Context generation returned nothing for chunk %r", chunk.id)
            return None

        return context[: self.max_context_chars].strip()
