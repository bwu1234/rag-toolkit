"""Pydantic request/response models for the chat API.

Kept separate from `rag.generation.chat_service`'s `ChatAnswer`/`Citation`
dataclasses on purpose -- those are internal pipeline types (shared with the
eval pipeline), while these are the wire contract. Routes translate between
the two so the API's JSON shape can evolve (versioning, extra fields, ...)
without touching pipeline code, and vice versa.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChatTurnModel(BaseModel):
    """One prior message in the conversation, as sent over the wire."""

    role: Literal["user", "assistant"] = Field(description="Who produced this message.")
    content: str = Field(description="The message text.")


class ChatRequest(BaseModel):
    """Body of a `POST /chat` request: the user's question, plus any prior turns.

    The API stays stateless -- it holds no session store and no conversation
    ids. A client that wants multi-turn behaviour owns its own history and
    replays it here, which keeps the server free to be restarted, scaled out,
    or load-balanced without anything to synchronize.
    """

    query: str = Field(
        ...,
        min_length=1,
        description="The user's question to answer using the indexed corpus.",
        examples=["What is the refund policy?"],
    )
    history: list[ChatTurnModel] = Field(
        default_factory=list,
        description=(
            "Conversation preceding `query`, oldest first. When present, the "
            "question is rewritten into a standalone one before retrieval so "
            "follow-ups ('what about part-time staff?') search for what they "
            "actually mean. Omit for one-shot questions."
        ),
    )


class CitationModel(BaseModel):
    """A single retrieved passage backing the answer, as returned over the wire."""

    chunk_id: str = Field(description="Stable id of the source chunk, e.g. 'handbook.pdf#page=2::chunk0'.")
    document_id: str = Field(description="Id of the source document, e.g. 'handbook.pdf#page=2'.")
    text: str = Field(description="The cited passage's full text.")
    score: float = Field(description="Relevance score in [0, 1] from the most recent retrieval stage.")
    page: int | None = Field(default=None, description="Page number within the source document, if known.")


class ChatResponse(BaseModel):
    """Body of a `POST /chat` response: the generated answer plus its sources."""

    answer: str = Field(description="The model's answer, grounded in the cited passages.")
    citations: list[CitationModel] = Field(
        default_factory=list,
        description="Passages the answer is grounded in, in the order they were presented to the model.",
    )
    rewritten_query: str | None = Field(
        default=None,
        description=(
            "The standalone question retrieval actually ran, when `history` caused "
            "the original to be rewritten. Null when the question was used as-is. "
            "Surface this to end users -- a rewrite that drops a constraint is "
            "otherwise an invisible cause of a wrong-looking answer."
        ),
        examples=["What is the refund policy for part-time staff?"],
    )
    dropped_below_min_score: int = Field(
        default=0,
        description=(
            "Passages that ranked well enough to be returned but scored below "
            "`retrieval.min_score` and were withheld. Non-zero means the answer "
            "rests on fewer sources than the pipeline was willing to consider."
        ),
    )
