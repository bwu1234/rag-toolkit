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


# ---------------------------------------------------------------------------
# Agentic retrieval (Milestone 19)
# ---------------------------------------------------------------------------

# Every clause answers a failure the prototype showed or a guard the plan
# names: standalone queries (search has no memory), one search per entity
# (multi-hop coverage), no outside knowledge *even as background* (the 27b's
# "for reference" FY2015 leak), and saying which parts went unanswered (the
# per-entity completeness rubric).
AGENT_SYSTEM_PROMPT = (
    "You answer questions using a search tool, rag_search, over the document "
    "corpus described below. Search before answering any question the corpus "
    "might cover; answer without searching only when the question needs no "
    "documents, such as a greeting.\n\n"
    "Search results are numbered passages, and the numbers stay the same for "
    "the whole conversation. Make every query standalone: name the specific "
    "entity, period, product or topic it is about, because the search sees "
    "only the query, not this conversation. A question about several entities "
    "or periods needs a separate search for each.{filters}{listing} {iterate}\n\n"
    "Answer only from the passages, citing them inline as [n], e.g. [2] or "
    "[3][5]. Do not add facts from your own knowledge, not even as background "
    "or 'for reference'. If the passages don't answer the question, or answer "
    "only part of it, say which parts they don't cover instead of guessing."
    "{corpus}"
)

_REACT_ITERATE = (
    "After each result, decide whether you have enough; if a passage points "
    "to something you haven't found yet, search for it."
)

_PLANNED_ITERATE = (
    "You get one round of searches: request every search the question needs "
    "now, all in this turn. You will answer from their results without "
    "searching again."
)


# Only with `agent.model_filters` on. The tool's own `filters` description
# says what the fields are; this says when to reach for one, and what to do
# when it finds nothing -- the failure a self-chosen filter adds.
_FILTERS_HINT = (
    " When the question names a company or period, pass rag_search a filter "
    "for it, so passages about other companies or periods can't crowd out the "
    "one you need. If a filtered search finds nothing, check the filter's "
    "values or search without it."
)


# Only when `agent.tools` offers rag_list_documents. The failure it answers:
# asked for a set ("which airline..."), the 27b spent its searches probing for
# carriers the corpus doesn't hold, since no ranked search says what's absent.
_LISTING_HINT = (
    " To learn what the corpus contains -- which companies, filings and "
    "periods -- call rag_list_documents rather than searching for each "
    "candidate; a question about a set (\"which airlines...\") starts there."
)


def agent_system_prompt(
    strategy: str,
    corpus_descriptions: list[tuple[str, str | None]],
    *,
    model_filters: bool = False,
    list_documents: bool = False,
) -> str:
    """The agent's system prompt for `strategy`, naming the corpora it searches.

    The corpus description is what lets the model decide *whether* to search
    -- the routing decision the agent replaces a classifier with -- so the
    registry's descriptions are included when the config has them.
    """

    described = [f"- {name}: {' '.join(text.split())}" for name, text in corpus_descriptions if text]
    corpus = "\n\nThe corpus:\n" + "\n".join(described) if described else ""
    iterate = _PLANNED_ITERATE if strategy == "planned" else _REACT_ITERATE
    filters = _FILTERS_HINT if model_filters else ""
    listing = _LISTING_HINT if list_documents else ""
    return AGENT_SYSTEM_PROMPT.format(iterate=iterate, corpus=corpus, filters=filters, listing=listing)


# Sent as its own user message before the forced tool-free turn. Where it goes
# was measured: replaying a captured 27b conversation that had hit the cap,
# this text appended to the last tool result got an empty answer 3/3 times,
# and as a separate user message a real answer 3/3 (docs/milestone-19-plan.md,
# phase 3). Gemini documents the opposite placement -- instructions inside
# the function response -- so its adapter folds this message into the last
# response itself; the agent stays provider-neutral.
AGENT_SYNTHESIS_INSTRUCTION = (
    "No more searches are available for this question. Answer it now from "
    "the passages above, citing them as [n]. Say plainly which parts of the "
    "question the passages don't cover."
)
