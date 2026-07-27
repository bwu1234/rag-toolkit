"""Orchestrates retrieve -> prompt -> generate -> cite for the chat API.

`ChatService` is the seam between the retrieval pipeline (Milestone 5) and
generation: it's the one place that knows how to turn a `Retriever`'s
`ScoredChunk`s into both a grounded prompt and a citation list the API can
return alongside the model's answer. Keeping this orchestration out of the
API route layer means the eval pipeline (Milestone 7) can reuse it directly
without spinning up FastAPI.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from rag.generation.llm import LLMClient
from rag.generation.prompts import SYSTEM_PROMPT, build_rag_prompt
from rag.retrieval.retriever import Retriever
from rag.vectorstore.base import ScoredChunk

logger = logging.getLogger(__name__)

_NO_CONTEXT_ANSWER = (
    "I don't have any indexed information to answer that -- the corpus "
    "index is empty or doesn't contain anything relevant to this question."
)


@dataclass(frozen=True)
class Citation:
    """A single retrieved passage backing an answer, shaped for API responses.

    Deliberately a thin, flat projection of `ScoredChunk` (rather than
    re-exposing it directly) -- the API's citation contract should be able to
    evolve independently of the internal retrieval result type, and callers
    shouldn't need to know about `ScoredChunk.metadata`'s grab-bag shape.
    """

    chunk_id: str
    document_id: str
    text: str
    score: float
    page: int | None = None


@dataclass(frozen=True)
class ChatAnswer:
    """The full result of a chat turn: the model's answer plus its sources."""

    answer: str
    citations: list[Citation] = field(default_factory=list)


def _to_citation(chunk: ScoredChunk) -> Citation:
    page = chunk.metadata.get("page")
    return Citation(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        text=chunk.text,
        score=chunk.score,
        page=page if isinstance(page, int) else None,
    )


class ChatService:
    """Answers a question by retrieving context and asking the LLM to ground its reply in it.

    Two collaborators, both injected as interfaces (`Retriever` is itself a
    thin orchestrator over `EmbeddingModel`/`VectorStore`/`Reranker`; see
    `rag.retrieval.retriever`): nothing here depends on Chroma, Ollama, or any
    concrete adapter, so it's trivial to test with fakes and trivial to point
    at different components via config alone.
    """

    def __init__(self, retriever: Retriever, llm_client: LLMClient) -> None:
        self._retriever = retriever
        self._llm_client = llm_client

    def ask(self, query: str) -> ChatAnswer:
        """Answer `query`, grounded in (and citing) the retrieved context.

        Returns an empty-citation `ChatAnswer` without calling the LLM at all
        when the query is blank or retrieval finds nothing -- both are cases
        where a generated answer could only be a hallucination, and skipping
        the LLM call keeps the no-context path fast and free.
        """

        if not query.strip():
            return ChatAnswer(answer=_NO_CONTEXT_ANSWER, citations=[])

        chunks = self._retriever.retrieve(query)
        if not chunks:
            logger.info("No chunks retrieved for query %r -- skipping generation", query)
            return ChatAnswer(answer=_NO_CONTEXT_ANSWER, citations=[])

        prompt = build_rag_prompt(query, chunks)
        answer = self._llm_client.generate(prompt, system=SYSTEM_PROMPT)
        citations = [_to_citation(chunk) for chunk in chunks]

        logger.info("Answered query %r with %d citation(s)", query, len(citations))
        return ChatAnswer(answer=answer, citations=citations)
