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
from rag.retrieval.builder import build_retriever


def build_chat_service(config: RagConfig) -> ChatService:
    """Construct a `ChatService` with all components selected per `config`."""

    retriever = build_retriever(config)
    llm_client = get_llm_client(config.llm)
    return ChatService(retriever=retriever, llm_client=llm_client)
