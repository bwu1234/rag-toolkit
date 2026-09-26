"""Wires a fully-configured `ChatService` from a `RagConfig`.

Mirrors `rag.retrieval.builder.build_retriever`: centralizes the plumbing
(build a `Retriever`, build an `LLMClient`, wire them into a `ChatService`) so
the API and any future entrypoint (eval pipeline, CLI `chat` command, ...) get
an identically configured service from one call.
"""

from __future__ import annotations

from collections.abc import Sequence

from rag.config.settings import RagConfig
from rag.generation.chat_service import ChatService
from rag.generation.crag import DocumentGrader, GroundednessChecker, RetryQueryRewriter
from rag.generation.factory import get_llm_client
from rag.generation.query_rewriter import QueryCondenser
from rag.observability.factory import config_fingerprint
from rag.observability.sink import TurnSink
from rag.observability.usage import MeteredLLMClient
from rag.retrieval.builder import build_retriever


def build_chat_service(
    config: RagConfig,
    corpora: Sequence[str] | None = None,
    *,
    turn_sink: TurnSink | None = None,
) -> ChatService:
    """Construct a `ChatService` with all components selected per `config`.

    `corpora` overrides `config.corpora.active`, selecting which index the
    retriever reads -- see `build_retriever`.

    `turn_sink`, if given, receives a `TurnRecord` for every turn. It's a
    parameter rather than read from `config.observability` here so that only
    the entrypoints people talk to opt in (see `get_turn_sink`); the eval
    runners call this without one and their traffic stays out of the log.
    """

    # Wrapped once, before it's shared, so every component's calls -- not just
    # generation's -- are counted into the turn's `llm_calls` and tokens.
    llm_client = MeteredLLMClient(get_llm_client(config.llm))
    # Share the one client: query expansion (HyDE / multi-query) generates
    # text too, and it should talk to the same daemon over the same connection
    # rather than opening a parallel one.
    retriever = build_retriever(config, llm_client, corpora)

    # The condenser shares the chat client rather than getting its own: same
    # provider, same model, one connection. Building it is free (no I/O), and
    # it only ever runs on a turn that actually has history -- so wiring it in
    # costs nothing for the single-shot callers.
    condenser = (
        QueryCondenser(llm_client, max_history_turns=config.chat.max_history_turns)
        if config.chat.condense_history
        else None
    )

    # Same sharing rule again: the CRAG checks are LLM calls like any other, and
    # each is built only if its own flag is on -- so `grade_documents: false`
    # with `check_groundedness: true` really does skip grading entirely rather
    # than constructing a grader nothing calls.
    crag = config.crag
    grader = DocumentGrader(llm_client) if crag.enabled and crag.grade_documents else None
    retry_rewriter = RetryQueryRewriter(llm_client) if crag.enabled and crag.max_retries else None
    groundedness_checker = GroundednessChecker(llm_client) if crag.enabled and crag.check_groundedness else None

    return ChatService(
        retriever=retriever,
        llm_client=llm_client,
        condenser=condenser,
        grader=grader,
        retry_rewriter=retry_rewriter,
        groundedness_checker=groundedness_checker,
        max_retries=crag.max_retries if crag.enabled else 0,
        max_regenerations=crag.max_regenerations if crag.enabled else 0,
        turn_sink=turn_sink,
        turn_metadata={
            "corpus": config.corpus_selection(corpora).slug,
            "llm": f"{config.llm.provider}:{config.llm.model}",
            "reranker": f"{config.reranker.provider}:{config.reranker.model}",
            "config_fingerprint": config_fingerprint(config),
        },
    )
