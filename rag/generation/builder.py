"""Wires a fully-configured chat responder from a `RagConfig`.

Mirrors `rag.retrieval.builder.build_retriever`: centralizes the plumbing
(build a `Retriever`, build an `LLMClient`, wire them into a `ChatService`, or
the agent's equivalents into an `AgentService`) so the API, UI, CLI and eval
runners get an identically configured responder from one call, whichever
`chat.mode` selects.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from rag.config.settings import RagConfig
from rag.generation.agent import AgentService
from rag.generation.chat_service import ChatResponder, ChatService
from rag.generation.crag import DocumentGrader, GroundednessChecker, RetryQueryRewriter
from rag.generation.factory import get_llm_client
from rag.generation.llm import ToolCallingLLM
from rag.generation.prompts import agent_system_prompt
from rag.generation.query_rewriter import QueryCondenser
from rag.observability.factory import config_fingerprint
from rag.observability.sink import TurnSink
from rag.observability.usage import metered_client
from rag.retrieval.builder import build_retriever
from rag.retrieval.retriever import PassageRetriever
from rag.tools import RagTools

logger = logging.getLogger(__name__)


def build_chat_service(
    config: RagConfig,
    corpora: Sequence[str] | None = None,
    *,
    turn_sink: TurnSink | None = None,
    retriever: PassageRetriever | None = None,
) -> ChatResponder:
    """Construct the chat responder `config.chat.mode` selects, all components per `config`.

    `pipeline` builds a `ChatService`; `agentic` builds an `AgentService`
    (see `build_agent_service`). Both answer through the same `ask()`.

    `corpora` overrides `config.corpora.active`, selecting which index the
    retriever reads -- see `build_retriever`.

    `turn_sink`, if given, receives a `TurnRecord` for every turn. It's a
    parameter rather than read from `config.observability` here so that only
    the entrypoints people talk to opt in (see `get_turn_sink`); the eval
    runners call this without one and their traffic stays out of the log.

    `retriever`, if given, replaces the configured one, and with it expansion,
    routing and web search; everything after retrieval is built as usual. Only
    the eval oracle passes one (`rag.eval.oracle`). The agent searches through
    its own tools, so this raises under `chat.mode: agentic`.
    """

    if config.chat.mode == "agentic":
        if retriever is not None:
            raise ValueError("A replacement retriever applies to chat.mode: pipeline only")
        return build_agent_service(config, corpora, turn_sink=turn_sink)

    # Wrapped once, before it's shared, so every component's calls -- not just
    # generation's -- are counted into the turn's `llm_calls` and tokens.
    llm_client = metered_client(get_llm_client(config.llm))
    # Share the one client: query expansion (HyDE / multi-query) generates
    # text too, and it should talk to the same daemon over the same connection
    # rather than opening a parallel one.
    if retriever is None:
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
        prompt_style=config.chat.prompt,
        turn_metadata={
            "corpus": config.corpus_selection(corpora).slug,
            "llm": f"{config.llm.provider}:{config.llm.model}",
            "reranker": f"{config.reranker.provider}:{config.reranker.model}",
            "config_fingerprint": config_fingerprint(config),
        },
    )


def build_agent_llm(config: RagConfig) -> ToolCallingLLM:
    """The metered, tool-calling client the agent drives: `agent.llm`, else `llm`.

    Its own client even when it falls back to `llm`'s model, because agent
    calls request `agent.num_ctx` and the pipeline's don't. Raises here, at
    build time, when the selected provider's adapter can't call tools -- not on
    the first agent turn.
    """

    llm_config = config.agent.llm or config.llm
    client = metered_client(get_llm_client(llm_config, num_ctx=config.agent.num_ctx))
    if not isinstance(client, ToolCallingLLM):
        source = "agent.llm" if config.agent.llm is not None else "llm (agent.llm is unset)"
        raise ValueError(
            f"The agent needs a model that can call tools, but {source} selects provider "
            f"{llm_config.provider!r}, whose adapter doesn't implement ToolCallingLLM. "
            "Set agent.llm to an ollama or gemini model."
        )
    return client


def build_agent_service(
    config: RagConfig,
    corpora: Sequence[str] | None = None,
    *,
    turn_sink: TurnSink | None = None,
) -> AgentService:
    """Construct the `chat.mode: agentic` responder.

    Two clients, on purpose (plan decision 2): the agent loop runs on
    `agent.llm` (the measured split is a 27b there), and the utility calls --
    query expansion inside retrieval, the groundedness check -- stay on `llm`.
    Nothing is loaded here: `RagTools` builds its retriever on the first search.

    Pipeline-only settings are reported, not silently dropped: the condenser
    and CRAG's grader and retries have no place in the loop, and only CRAG's
    groundedness check carries over.
    """

    crag = config.crag
    if crag.enabled and (crag.grade_documents or crag.max_retries):
        logger.warning(
            "chat.mode is agentic: crag.grade_documents and crag.max_retries don't apply to "
            "the agent and are ignored (only crag.check_groundedness does)"
        )

    utility_client = metered_client(get_llm_client(config.llm))
    selection = config.corpus_selection(corpora)
    registry = config.corpora.registry
    descriptions = [(name, registry[name].description if name in registry else None) for name in selection.names]

    return AgentService(
        build_agent_llm(config),
        RagTools(config=config, llm_client=utility_client),
        system_prompt=agent_system_prompt(
            config.agent.strategy,
            descriptions,
            model_filters=config.agent.model_filters,
            list_documents="rag_list_documents" in config.agent.tools,
        ),
        corpora=selection.names,
        strategy=config.agent.strategy,
        max_tool_calls=config.agent.max_tool_calls,
        timeout_s=config.agent.timeout_s,
        max_passage_chars=config.agent.max_passage_chars,
        max_history_turns=config.chat.max_history_turns,
        model_filters=config.agent.model_filters,
        offered_tools=config.agent.tools,
        groundedness_checker=(
            GroundednessChecker(utility_client) if crag.enabled and crag.check_groundedness else None
        ),
        turn_sink=turn_sink,
        turn_metadata={
            "corpus": selection.slug,
            "mode": f"agentic:{config.agent.strategy}",
            "llm": f"{(config.agent.llm or config.llm).provider}:{(config.agent.llm or config.llm).model}",
            "reranker": f"{config.reranker.provider}:{config.reranker.model}",
            "config_fingerprint": config_fingerprint(config),
        },
    )
