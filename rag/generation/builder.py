"""Wires a fully-configured `ChatService` from a `RagConfig`.

Mirrors `rag.retrieval.builder.build_retriever`: centralizes the plumbing
(build a `Retriever`, build an `LLMClient`, wire them into a `ChatService`) so
the API and any future entrypoint (eval pipeline, CLI `chat` command, ...) get
an identically configured service from one call.
"""

from __future__ import annotations

from rag.config.settings import RagConfig
from rag.generation.chat_service import ChatService
from rag.generation.factory import get_llm_client
from rag.generation.query_rewriter import QueryCondenser
from rag.retrieval.builder import build_retriever


def build_chat_service(config: RagConfig) -> ChatService:
    """Construct a `ChatService` with all components selected per `config`."""

    retriever = build_retriever(config)
    llm_client = get_llm_client(config.llm)

    # The condenser shares the chat client rather than getting its own: same
    # provider, same model, one connection. Building it is free (no I/O), and
    # it only ever runs on a turn that actually has history -- so wiring it in
    # costs nothing for the single-shot callers.
    condenser = (
        QueryCondenser(llm_client, max_history_turns=config.chat.max_history_turns)
        if config.chat.condense_history
        else None
    )

    return ChatService(retriever=retriever, llm_client=llm_client, condenser=condenser)
