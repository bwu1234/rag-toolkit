"""Wires the `chat.mode: agentic` responder from a `RagConfig`.

The agent half of what `rag.generation.builder` does for the pipeline. Entrypoints
call `rag.chat.build_chat_service`, which picks between the two by `chat.mode`.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from rag.agent.prompts import agent_system_prompt
from rag.agent.service import AgentService
from rag.config.settings import RagConfig
from rag.generation.crag import GroundednessChecker
from rag.llm.base import ToolCallingLLM
from rag.llm.factory import get_llm_client
from rag.observability.factory import config_fingerprint
from rag.observability.sink import TurnSink
from rag.observability.usage import metered_client
from rag.tools import RagTools

logger = logging.getLogger(__name__)


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
            calculator="calculator" in config.agent.tools,
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
