"""Pydantic request/response models for the chat API.

Kept separate from `rag.generation.chat_service`'s `ChatAnswer`/`Citation`
dataclasses on purpose -- those are internal pipeline types (shared with the
eval pipeline), while these are the wire contract. Routes translate between
the two so the API's JSON shape can evolve (versioning, extra fields, ...)
without touching pipeline code, and vice versa.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Body of a `POST /chat` request: just the user's question."""

    query: str = Field(
        ...,
        min_length=1,
        description="The user's question to answer using the indexed corpus.",
        examples=["What is the refund policy?"],
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
