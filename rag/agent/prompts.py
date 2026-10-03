"""The agent's prompt text: its system prompt per strategy and the synthesis nudge.

Split from `rag.generation.prompts`, which keeps the pipeline's prompts and the
citation helpers both modes share. Plain strings and one function, as there.
"""

from __future__ import annotations

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
    "[3][5].{calculate} Do not add facts from your own knowledge, not even as background "
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


# Only when `agent.tools` offers the calculator. The failure it answers: a
# figure the filings don't print (a percentage change, a margin) computed in
# the model's reasoning, where multi-digit division goes wrong unchecked.
_CALCULATOR_HINT = (
    " For any figure you derive -- a difference, a ratio, a percentage change "
    "-- call calculator with the numbers from the passages rather than "
    "computing it yourself, and cite the passages the numbers came from."
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
    calculator: bool = False,
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
    calculate = _CALCULATOR_HINT if calculator else ""
    return AGENT_SYSTEM_PROMPT.format(
        iterate=iterate, corpus=corpus, filters=filters, listing=listing, calculate=calculate
    )


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
