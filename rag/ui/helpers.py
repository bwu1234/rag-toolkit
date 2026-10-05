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
from rag.observability.records import Rating


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
    if answer.graded_out:
        notices.append(f"🧹 {answer.graded_out} passage(s) dropped — judged not to answer the question")
    if answer.retry_queries:
        notices.append(
            f"🔁 Retried retrieval {len(answer.retry_queries)} time(s): *{answer.retry_queries[-1]}*"
        )
    # Only the failing verdict is worth a caption. "Grounded" is the expected
    # outcome, and captioning it would train users to skim past the one state
    # that actually needs their attention.
    if answer.grounded is False:
        notices.append("⚠️ This answer failed its groundedness check — treat it as unverified")
    if answer.generation_failure is not None:
        notices.append(
            f"⛔ No answer was generated ({answer.generation_failure}) — this is not a finding about the documents"
        )
    if answer.invalid_citations:
        markers = ", ".join(f"[{n}]" for n in answer.invalid_citations)
        notices.append(f"⚠️ Cites {markers}, which name no passage the model was shown")
    return notices


def format_turn_metrics(answer: ChatAnswer) -> str | None:
    """Return a one-line cost summary: wall time, LLM calls, tokens, and how many passages were cited.

    `None` for a turn that was never measured (no `total_ms`), so a hand-built
    `ChatAnswer` doesn't render a misleading "0 ms".
    """

    if answer.total_ms is None:
        return None
    parts = [f"⏱️ {answer.total_ms / 1000:.1f} s", f"{answer.llm_calls} LLM call(s)"]
    if answer.prompt_tokens is not None or answer.completion_tokens is not None:
        prompt = "?" if answer.prompt_tokens is None else f"{answer.prompt_tokens:,}"
        completion = "?" if answer.completion_tokens is None else f"{answer.completion_tokens:,}"
        parts.append(f"{prompt} → {completion} tokens")
    if answer.citations:
        parts.append(f"cited {len(answer.cited_chunk_ids)} of {len(answer.citations)} passage(s)")
    return " · ".join(parts)


def rating_from_feedback_widget(value: int | None) -> Rating | None:
    """Map `st.feedback("thumbs")`'s value (1 up, 0 down, None cleared) to a stored rating."""

    if value is None:
        return None
    return "up" if value == 1 else "down"


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
        f"**Contextual chunks:** `{config.chunking.contextual.enabled}`",
        f"**CRAG:** `{config.crag.enabled}`"
        + (
            f" (grade=`{config.crag.grade_documents}`, retries={config.crag.max_retries}, "
            f"groundedness=`{config.crag.check_groundedness}`)"
            if config.crag.enabled
            else ""
        ),
        f"**Condense history:** `{config.chat.condense_history}`",
        f"**Vector store:** `{config.vector_store.provider}` "
        f"(`{config.vector_store.collection_name}`)",
    ]
    return "\n\n".join(lines)
