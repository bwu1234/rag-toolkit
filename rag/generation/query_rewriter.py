"""Rewrites a follow-up question into a standalone one using the chat history.

Retrieval is stateless: `Retriever.retrieve` embeds exactly the string it's
handed. In a multi-turn UI that's a real failure mode -- "what about
part-time staff?" carries almost none of its meaning in its own words, so
embedding it searches the corpus for a generic phrase and returns whatever is
nearest to it rather than to what the user actually asked.

The fix is the standard condense step: before retrieving, ask the LLM to fold
the recent turns into a single self-contained question. That rewritten query
then drives *both* retrieval and the generation prompt -- `build_rag_prompt`
renders passages and one question with no conversational context of its own,
so handing it the raw follow-up would leave the model answering "what about
part-time staff?" against passages it has no way to connect to the thread.

Deliberately fail-open: any problem condensing (LLM error, empty reply) falls
back to the original query and logs it. A degraded retrieval is a much better
outcome than a failed turn, and the original query is exactly what the system
would have used before this step existed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from rag.generation.llm import LLMClient

logger = logging.getLogger(__name__)

Role = Literal["user", "assistant"]

DEFAULT_MAX_HISTORY_TURNS = 6

CONDENSE_SYSTEM_PROMPT = (
    "You rewrite follow-up questions into standalone ones. "
    "Given a conversation and the user's latest message, restate that message "
    "as a single self-contained question that preserves every detail needed to "
    "understand it without the conversation -- resolving pronouns and implicit "
    "references ('it', 'that one', 'what about X?') into explicit terms from "
    "the earlier turns. "
    "If the latest message is already self-contained, return it unchanged. "
    "Reply with the rewritten question and nothing else: no preamble, no "
    "explanation, no quotation marks."
)


@dataclass(frozen=True)
class ChatTurn:
    """One prior message in the conversation, as supplied by the caller.

    Deliberately its own tiny type rather than reusing `ChatAnswer` or an API
    schema: the UI holds Streamlit dicts, the API holds pydantic models, and
    the eval pipeline holds nothing at all. A plain role/content pair is the
    common denominator all three can cheaply produce, and it keeps
    `ChatService` from depending on any one caller's representation.
    """

    role: Role
    content: str


def format_history(history: list[ChatTurn]) -> str:
    """Render turns as a simple `User:`/`Assistant:` transcript for the prompt."""

    return "\n".join(f"{'User' if turn.role == 'user' else 'Assistant'}: {turn.content}" for turn in history)


def build_condense_prompt(query: str, history: list[ChatTurn]) -> str:
    """Render the transcript and the latest message into a single user-turn prompt."""

    return (
        f"Conversation so far:\n\n{format_history(history)}\n\n"
        f"Latest user message: {query}\n\n"
        "Rewritten standalone question:"
    )


class QueryCondenser:
    """Folds recent turns plus the latest message into one standalone query.

    Takes an `LLMClient` rather than building its own so it reuses whatever
    model the rest of the pipeline is configured with -- no second provider, no
    second set of credentials, and a rewrite that's consistent with the model
    doing the answering.
    """

    def __init__(self, llm_client: LLMClient, *, max_history_turns: int = DEFAULT_MAX_HISTORY_TURNS) -> None:
        self._llm_client = llm_client
        self.max_history_turns = max_history_turns

    def condense(self, query: str, history: list[ChatTurn]) -> str:
        """Return a standalone version of `query`, or `query` itself if that's not possible.

        Returns `query` untouched -- without calling the LLM -- when there's no
        history to fold in, which is the common case for the CLI, the eval
        pipeline, and the first turn of any conversation. Only a genuine
        follow-up pays for the extra round trip.
        """

        recent = history[-self.max_history_turns :] if self.max_history_turns else []
        if not recent:
            return query

        try:
            rewritten = self._llm_client.generate(
                build_condense_prompt(query, recent), system=CONDENSE_SYSTEM_PROMPT
            )
        except Exception:  # noqa: BLE001 -- fail open; see module docstring
            logger.exception("Query condensing failed; retrieving with the original query %r", query)
            return query

        # Small models sometimes wrap the rewrite in quotes despite being told
        # not to; stripping them costs nothing and avoids embedding stray
        # punctuation. An empty reply means the model gave us nothing usable.
        cleaned = rewritten.strip().strip('"').strip()
        if not cleaned:
            logger.warning("Query condensing returned an empty rewrite; using the original query %r", query)
            return query

        logger.info("Condensed %r into %r", query, cleaned)
        return cleaned
