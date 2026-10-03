"""The one place that chooses between the pipeline and the agent.

`chat.mode` selects the responder. This module sits above both `rag.generation`
(the retrieve-then-generate pipeline) and `rag.agent`, which never import each
other's builders, so the API, UI, CLI and eval runners get an identically
configured responder from one call. Both answer through the same `ask()`.
"""

from __future__ import annotations

from collections.abc import Sequence

from rag.agent.builder import build_agent_service
from rag.config.settings import RagConfig
from rag.generation.builder import build_pipeline_service
from rag.generation.chat_service import ChatResponder
from rag.observability.sink import TurnSink
from rag.retrieval.retriever import PassageRetriever


def build_chat_service(
    config: RagConfig,
    corpora: Sequence[str] | None = None,
    *,
    turn_sink: TurnSink | None = None,
    retriever: PassageRetriever | None = None,
) -> ChatResponder:
    """Construct the chat responder `config.chat.mode` selects, all components per `config`.

    `pipeline` builds a `ChatService` (`build_pipeline_service`); `agentic`
    builds an `AgentService` (`build_agent_service`).

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
    return build_pipeline_service(config, corpora, turn_sink=turn_sink, retriever=retriever)
