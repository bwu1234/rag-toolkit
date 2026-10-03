"""Prompt templates for retrieval-augmented generation.

Kept as plain functions returning strings (not a templating-library
dependency) -- the format is simple enough that string formatting is more
readable than it would be wrapped in a template DSL, and it keeps this module
trivially testable.
"""

from __future__ import annotations

import re

from rag.config.settings import PromptStyle
from rag.vectorstore.base import ScoredChunk

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using only the "
    "numbered context passages provided below. "
    "Cite the passages you rely on inline using their bracketed numbers, "
    "e.g. [1] or [2][3]. "
    "If the passages don't contain enough information to answer, say so "
    "plainly instead of guessing or using outside knowledge."
)

# Used only to regenerate an answer the groundedness check rejected (see
# `rag.generation.crag.GroundednessChecker`). Same instruction, stated as a
# correction: the model has already produced an unsupported answer once, so
# repeating the original prompt verbatim mostly reproduces it.
REGROUND_SYSTEM_PROMPT = (
    SYSTEM_PROMPT + " "
    "Your previous attempt at this question stated things the passages do not "
    "support. Write a new answer that stays strictly inside them. Every "
    "specific -- every number, name, condition, and procedure -- must appear in "
    "a passage you cite for it. Where the passages are silent or partial, say "
    "so explicitly rather than filling the gap; a short answer that stops at "
    "the evidence is correct, and a complete-sounding one that goes past it is "
    "not."
)


def _citation_label(chunk: ScoredChunk) -> str:
    """Build a human-readable source label, including a page number if known.

    The chunk header, when there is one, replaces the document id: "Apple Inc.
    (AAPL) 10-K, period ended 2024-09-28" tells the answering model which
    company and period a passage is about, where a file name only hints at it.
    """

    label = chunk.header or chunk.document_id
    page = chunk.metadata.get("page")
    return f"{label} (p.{page})" if page is not None else label


def format_passage(number: int, chunk: ScoredChunk) -> str:
    """One numbered passage block, as both the pipeline prompt and the agent's search results show it.

    Shared so that pipeline and agent differ only in how passages are found,
    not in how they're presented -- otherwise a phase 4 comparison would
    measure the formatting along with the loop.
    """

    return (
        f"Passage [{number}] (source: {_citation_label(chunk)}):\n"
        + (f"Context: {chunk.context}\n" if chunk.context else "")
        + chunk.text
    )


def build_rag_prompt(query: str, chunks: list[ScoredChunk]) -> str:
    """Render the retrieved chunks and question into a single user-turn prompt.

    Each chunk becomes a numbered "Passage [n]" block labeled with its source
    (and page, if known) so the model can ground its answer in -- and cite --
    specific passages. Numbering here is the single source of truth for the
    `[n]` markers the system prompt asks the model to use, and for mapping a
    cited number back to a `Citation` in `ChatService`.

    A chunk carrying generated context (see `rag.chunking.contextualizer`) gets
    it on its own labeled line rather than run together with the passage text.
    The context is a model's description of where the excerpt sits, not corpus
    content, and the answering model shouldn't be able to quote or cite it as
    though it were -- so the boundary is made explicit rather than implied.
    """

    passages = "\n\n".join(format_passage(index, chunk) for index, chunk in enumerate(chunks, start=1))

    return (
        f"Context passages:\n\n{passages}\n\n"
        f"Question: {query}\n\n"
        "Answer the question using only the passages above, citing them by "
        "number as you go."
    )


# The `plain` style (`chat.prompt: plain`): the textbook RAG template, kept as
# a baseline for measuring what the grounded prompt's instructions are worth.
# Deliberately missing everything `build_rag_prompt` adds -- no passage
# numbers, no source labels, no citation request, and no instruction to admit
# when the context falls short -- so it is not a variant of the grounded
# prompt, it is the thing the grounded prompt was built to improve on. It has
# no system prompt for the same reason: "use only the context" is itself one of
# the grounding instructions.
PLAIN_SYSTEM_PROMPT: str | None = None


def build_plain_prompt(query: str, chunks: list[ScoredChunk]) -> str:
    """Render chunks and question as undecorated context followed by the question.

    Generated context (`chunk.context`) is left out: labelling it is how the
    grounded prompt keeps the model from quoting it, and without a label it
    would reach the model as corpus text.
    """

    context = "\n\n".join(chunk.text for chunk in chunks)
    return (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        "Answer:"
    )


def system_prompt_for(style: PromptStyle) -> str | None:
    """The generation system prompt for `style`."""

    return SYSTEM_PROMPT if style == "grounded" else PLAIN_SYSTEM_PROMPT


def build_prompt(style: PromptStyle, query: str, chunks: list[ScoredChunk]) -> str:
    """Render the generation user turn in `style`."""

    return build_rag_prompt(query, chunks) if style == "grounded" else build_plain_prompt(query, chunks)


# `[1]`, `[2][3]`, and the comma-list form small models drift into despite the
# instruction (`[1, 3]`). Ranges (`[1-3]`) are deliberately not expanded: a
# model that writes one hasn't said which passages in between it relied on.
_CITATION_MARKER = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def parse_cited_passages(answer: str, passage_count: int) -> list[int]:
    """Return the 1-based passage numbers `answer` cites, in first-cited order.

    The inverse of `build_rag_prompt`'s numbering. Numbers outside
    `1..passage_count` are dropped -- a model citing `[7]` over five passages
    cited nothing real, and mapping it anywhere would invent a judgment.
    """

    cited: list[int] = []
    for match in _CITATION_MARKER.finditer(answer):
        for part in match.group(1).split(","):
            number = int(part)
            if 1 <= number <= passage_count and number not in cited:
                cited.append(number)
    return cited


def strip_citation_markers(text: str) -> str:
    """Remove `[n]` markers, e.g. from an earlier answer replayed as history.

    An earlier turn's `[3]` meant that turn's third passage. Replayed to the
    agent, it would read as a citation of *this* turn's third passage, and a
    model that copied it would cite something it never saw.
    """

    return re.sub(r"[ \t]*" + _CITATION_MARKER.pattern, "", text)
