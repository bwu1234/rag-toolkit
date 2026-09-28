"""Corrective RAG: the three checks that let the pipeline disagree with itself.

Everything upstream of this module is a single forward pass -- retrieve, rank,
generate -- with no step that can look at a result and reject it. The relevance
floor comes closest, but it thresholds a *similarity score*, and a score is a
statement about how alike two pieces of text are, not about whether one answers
a question asked of the other. A passage from the right document, on the right
topic, that happens not to contain the answer scores well and is passed to the
model as though it did.

CRAG (Corrective Retrieval-Augmented Generation) closes the loop with three
small LLM-backed judgments, each its own class here so they can be tested,
swapped, and switched off independently:

- `DocumentGrader` — per passage: does this help answer the question? Passages
  that don't are dropped before they reach the prompt.
- `RetryQueryRewriter` — when grading leaves nothing, the query is the suspect.
  Rewrite it (differently from `QueryCondenser`, which resolves *references*;
  this one changes *vocabulary and framing*) and search again.
- `GroundednessChecker` — after generating: is this answer actually supported by
  those passages? An unsupported answer triggers a regeneration.

`ChatService` owns the loop that wires these together; they hold no state and
know nothing about each other, so the control flow stays readable in one place
instead of being distributed across the components.

**All three fail open**, exactly like `QueryCondenser` and the query expanders:
an LLM error or an unparseable reply resolves to "keep going as though this
check hadn't run". The pipeline's job is to answer the question; a broken
checker should degrade the pipeline to what it was before CRAG existed, not
turn a working turn into a failed one. Note this makes the failure direction
asymmetric on purpose -- a grader that can't be reached keeps passages rather
than dropping them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from rag.generation.llm import LLMClient
from rag.vectorstore.base import ScoredChunk

logger = logging.getLogger(__name__)


GRADE_SYSTEM_PROMPT = (
    "You judge whether a retrieved passage is useful for answering a question. "
    "Answer YES if the passage contains information that helps answer the "
    "question, even partially. Answer NO if it is about a different subject, or "
    "is on the right subject but contains nothing that bears on what was asked. "
    "Being topically related is not enough on its own. "
    "Reply with exactly one word, YES or NO, and nothing else."
)

REWRITE_SYSTEM_PROMPT = (
    "You rewrite a search query that failed to find anything useful. "
    "Produce one new query for the same underlying information need, changing "
    "the vocabulary and the framing rather than restating the question: prefer "
    "the terms a reference document would use over the terms a person would "
    "speak, replace narrow phrasing with the general concept (or the reverse), "
    "and drop conversational framing. "
    "Reply with the rewritten query and nothing else: no preamble, no "
    "explanation, no quotation marks."
)

GROUNDEDNESS_SYSTEM_PROMPT = (
    "You check whether an answer is supported by the passages it was written "
    "from. Answer GROUNDED if every factual claim in the answer is stated in or "
    "directly follows from the passages. Answer UNGROUNDED if the answer adds "
    "specifics that are not in the passages -- numbers, names, procedures, "
    "conditions -- or contradicts them. "
    "An answer that declines to answer, or says the passages do not cover the "
    "question, is GROUNDED. Judge only whether the passages support it: do not "
    "use outside knowledge and do not judge whether it is a good answer. "
    "Reply with exactly one word, GROUNDED or UNGROUNDED, and nothing else."
)


def _first_word(reply: str) -> str:
    """Return the reply's first word, uppercased -- '' for an empty reply.

    Deliberately the same minimal parse `rag.eval.answer_eval` uses for its
    judge: a one-word protocol needs no regex, and anything that doesn't match
    is treated as unparseable and fails open rather than being coaxed into a
    verdict it may not have meant.
    """

    stripped = reply.strip()
    return stripped.split()[0].strip(".,:!*").upper() if stripped else ""


def format_passages(chunks: list[ScoredChunk]) -> str:
    """Render passages for a checker prompt, numbered as `build_rag_prompt` does."""

    return "\n\n".join(
        f"Passage [{index}]:\n{chunk.index_text}" for index, chunk in enumerate(chunks, start=1)
    )


@dataclass(frozen=True)
class GradedChunks:
    """What grading kept, and how much it threw away.

    The count is carried separately rather than recomputed from list lengths
    because it has to survive up to `ChatAnswer`: "the corpus had nothing" and
    "the corpus had five passages and the grader rejected all five" send a user
    to completely different places, and only this step knows which happened.
    """

    kept: list[ScoredChunk]
    graded_out: int = 0


class DocumentGrader:
    """Drops retrieved passages that don't bear on the question.

    One LLM call per passage rather than one call grading all of them. A single
    call would be cheaper, but it would need the model to emit a parseable
    verdict list whose length matches the input -- and a small local model that
    miscounts, skips an entry, or reorders them silently corrupts the mapping
    from verdict to passage. Per-passage grading has a one-word reply and no
    mapping to get wrong. The cost is real and it's the reason
    `crag.grade_documents` is off by default.
    """

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client

    def grade(self, query: str, chunks: list[ScoredChunk]) -> GradedChunks:
        """Return the passages worth answering from, plus how many were rejected."""

        if not chunks:
            return GradedChunks(kept=[])

        kept = [chunk for chunk in chunks if self._is_relevant(query, chunk)]
        graded_out = len(chunks) - len(kept)
        logger.info(
            "Graded %d passage(s) for %r: kept %d, dropped %d",
            len(chunks),
            query,
            len(kept),
            graded_out,
        )
        return GradedChunks(kept=kept, graded_out=graded_out)

    def _is_relevant(self, query: str, chunk: ScoredChunk) -> bool:
        prompt = (
            f"Question: {query}\n\n"
            f"Passage:\n{chunk.index_text}\n\n"
            "Does this passage help answer the question? Reply YES or NO:"
        )
        try:
            reply = self._llm_client.generate(prompt, system=GRADE_SYSTEM_PROMPT)
        except Exception:  # noqa: BLE001 -- fail open; see module docstring
            logger.exception("Grading failed for chunk %r; keeping it", chunk.chunk_id)
            return True

        verdict = _first_word(reply)
        if verdict == "YES":
            return True
        if verdict == "NO":
            return False

        logger.warning(
            "Unparseable grade %r for chunk %r; keeping it", reply.strip()[:80], chunk.chunk_id
        )
        return True


class RetryQueryRewriter:
    """Rewrites a query that retrieved nothing usable, for one more attempt.

    Distinct from `QueryCondenser` despite the family resemblance: the condenser
    resolves *references* ("what about part-time staff?" -> the full question)
    using conversation history, and runs before the first search. This runs
    after a search has already failed, has no history, and changes *wording* --
    the assumption being that the information need was fine and the words used
    to look for it weren't.
    """

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client

    def rewrite(self, query: str, *, attempt: int = 1) -> str:
        """Return a re-phrased `query`, or `query` itself if rewriting fails."""

        prompt = (
            f"This query returned no useful documents: {query}\n\n"
            "Rewritten query:"
        )
        try:
            reply = self._llm_client.generate(prompt, system=REWRITE_SYSTEM_PROMPT)
        except Exception:  # noqa: BLE001 -- fail open; see module docstring
            logger.exception("Retry rewrite failed for query %r", query)
            return query

        rewritten = " ".join(reply.split()).strip().strip('"').strip()
        if not rewritten:
            logger.warning("Retry rewrite returned nothing for query %r", query)
            return query

        logger.info("Retry %d rewrote %r into %r", attempt, query, rewritten)
        return rewritten


class GroundednessChecker:
    """Judges whether an answer is actually supported by the passages it cites.

    This is the check that makes the citation contract mean something. The
    system prompt asks the model to answer only from the numbered passages and
    cite them, but nothing verifies that it did -- an inline `[2]` is a token
    the model chose to emit, not evidence that passage 2 says what the sentence
    next to it claims. Reading the answer back against the passages is the
    cheapest thing that actually tests it.

    Returns `None`, not `False`, when the check itself fails: "the answer is
    unsupported" and "we couldn't tell" are different, and only the first is
    grounds for regenerating. Callers surface the distinction rather than
    collapsing it into a boolean.
    """

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client

    def check(self, query: str, chunks: list[ScoredChunk], answer: str) -> bool | None:
        """Return True if grounded, False if not, None if the check was inconclusive."""

        if not chunks or not answer.strip():
            return None

        prompt = (
            f"Passages:\n\n{format_passages(chunks)}\n\n"
            f"Question: {query}\n\n"
            f"Answer:\n{answer}\n\n"
            "Is the answer supported by the passages? Reply GROUNDED or UNGROUNDED:"
        )
        try:
            reply = self._llm_client.generate(prompt, system=GROUNDEDNESS_SYSTEM_PROMPT)
        except Exception:  # noqa: BLE001 -- fail open; see module docstring
            logger.exception("Groundedness check failed for query %r", query)
            return None

        verdict = _first_word(reply)
        if verdict == "GROUNDED":
            return True
        if verdict == "UNGROUNDED":
            logger.warning("Answer for %r judged ungrounded in its passages", query)
            return False

        logger.warning("Unparseable groundedness verdict %r for %r", reply.strip()[:80], query)
        return None
