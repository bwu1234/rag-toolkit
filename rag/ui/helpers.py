"""Pure helper functions for the Streamlit UI.

Kept separate from ``app.py`` so they can be imported and tested without
a running Streamlit server or session state.
"""

from __future__ import annotations

from typing import Any

from rag.config.settings import RagConfig
from rag.events import PipelineEvent
from rag.generation.chat_service import ChatAnswer, Citation
from rag.generation.query_rewriter import ChatTurn


def history_from_messages(messages: list[dict[str, Any]]) -> list[ChatTurn]:
    """Convert Streamlit's session-state message dicts into `ChatTurn`s.

    Lives here rather than in ``app.py`` because it's the one piece of the
    multi-turn path with logic worth testing: session state carries render-only
    keys (``answer``, ``events``) and, on a failed turn, an assistant message
    whose ``content`` is an error banner rather than a real reply. Feeding that
    banner to the condenser would have it rewrite the follow-up around an error
    message, so failed turns are skipped -- an assistant message is only real
    conversation if it came with a ``ChatAnswer``.
    """

    return [
        ChatTurn(role=message["role"], content=message["content"])
        for message in messages
        if message["role"] == "user" or message.get("answer") is not None
    ]


def format_citation_label(citation: Citation, rank: int) -> str:
    """Return a short label like ``'[1] guide.pdf (p.3)'``."""
    page_part = f" (p.{citation.page})" if citation.page is not None else ""
    return f"[{rank}] {citation.document_id}{page_part}"


def format_citation_preview(citation: Citation, max_chars: int = 300) -> str:
    """Return a trimmed, single-line text preview of a citation chunk."""
    text = citation.text.strip().replace("\n", " ")
    if len(text) > max_chars:
        return text[:max_chars] + "…"
    return text


def format_answer_notices(answer: ChatAnswer) -> list[str]:
    """Return the caption lines that belong under an assistant bubble.

    Both notices describe something the pipeline did that the answer text
    can't show: it searched for a different question than the one typed, or it
    withheld passages it retrieved. Rendered next to the answer rather than
    inside the collapsed pipeline trace, because a user re-reading an old turn
    shouldn't have to expand a debug panel to learn either one.
    """

    notices = []
    if answer.rewritten_query:
        notices.append(f"🔄 Searched for: *{answer.rewritten_query}*")
    if answer.search_queries:
        notices.append(f"🔎 Expanded into {len(answer.search_queries)} search queries")
    if answer.dropped_below_min_score:
        notices.append(
            f"🔻 {answer.dropped_below_min_score} passage(s) withheld — scored below the relevance threshold"
        )
    return notices


def format_event_line(event: PipelineEvent) -> str:
    """Return a one-line markdown string for a single pipeline event."""
    timing = f" `({event.elapsed_ms:.0f} ms)`" if event.elapsed_ms is not None else ""
    return f"**{event.stage}** — {event.message}{timing}"


def sidebar_config_summary(config: RagConfig) -> str:
    """Return a multi-line markdown string summarising the active config."""
    lines = [
        f"**Embedder:** `{config.embedding.provider}:{config.embedding.model}`",
        f"**LLM:** `{config.llm.provider}:{config.llm.model}`",
        f"**Reranker:** `{config.reranker.provider}`",
        f"**Retrieval mode:** `{config.retrieval.mode}`",
        f"**Retrieve top-k:** {config.retrieval.top_k} → rerank to {config.retrieval.rerank_top_k}",
        f"**Min score:** {config.retrieval.min_score}",
        f"**Query expansion:** `{config.retrieval.expansion.provider}`",
        f"**Condense history:** `{config.chat.condense_history}`",
        f"**Vector store:** `{config.vector_store.provider}` "
        f"(`{config.vector_store.collection_name}`)",
    ]
    return "\n\n".join(lines)
