"""Streamlit chat UI for the RAG system.

Run with::

    streamlit run rag/ui/app.py

The app builds a :class:`~rag.generation.chat_service.ChatService` once at
startup (via ``@st.cache_resource``) using the same config.yaml that the CLI
and API use, so all component selection and tuning happens in one place.

Layout
------
* **Sidebar** — shows the active config (embedding model, LLM, reranker,
  retrieval top-k) so it's always clear what's running under the hood.
* **Main area** — chat history rendered as alternating user/assistant bubbles
  with collapsible citation cards below each answer.
* **Input bar** — a text input pinned to the bottom (``st.chat_input``).

Every answer gets a thumbs up/down widget. Ratings go to the same turn log the
chat service writes each turn's record to (``observability.turn_log``), keyed
by the turn's id -- so logged questions with feedback can be mined for new eval
samples later.

Pure helper functions (citation formatting, config summary) live in
``rag/ui/helpers.py`` so they can be tested without a running Streamlit server.
"""

from __future__ import annotations

import logging

import streamlit as st

from rag.config.settings import RagConfig, load_config
from rag.events import PipelineEvent
from rag.chat import build_chat_service
from rag.generation.chat_service import ChatAnswer, ChatResponder
from rag.logging_config import configure_logging
from rag.observability.factory import get_turn_sink
from rag.observability.records import FeedbackRecord
from rag.observability.sink import TurnSink
from rag.ui.helpers import (
    format_answer_notices,
    format_citation_label,
    format_citation_preview,
    format_event_line,
    format_turn_metrics,
    history_from_messages,
    rating_from_feedback_widget,
    sidebar_config_summary,
)

configure_logging()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config (must be the first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="RAG Chat",
    page_icon="🔍",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Cached resources — built once per Streamlit server process
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading config and building chat service…")
def _load_chat_service() -> tuple[ChatResponder, RagConfig, TurnSink | None]:
    config = load_config()
    sink = get_turn_sink(config.observability.turn_log)
    service = build_chat_service(config, turn_sink=sink)
    return service, config, sink


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------


_EXPAND_TRACE_KEY = "expand_trace"


def _init_session() -> None:
    if "messages" not in st.session_state:
        # Each message: {"role": "user"|"assistant", "content": str,
        #                "answer": ChatAnswer | None,
        #                "events": list[PipelineEvent]}
        st.session_state["messages"] = []
    # Seeded before the sidebar checkbox is created so the widget picks this up
    # as its initial value; afterwards the widget owns the key.
    st.session_state.setdefault(_EXPAND_TRACE_KEY, True)


def _trace_expanded() -> bool:
    """Whether pipeline traces should render expanded (sidebar preference)."""
    return bool(st.session_state.get(_EXPAND_TRACE_KEY, True))


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _render_events(events: list[PipelineEvent]) -> None:
    """Render a completed pipeline trace, expanded per the sidebar preference."""
    if not events:
        return
    with st.status(
        f"Pipeline trace ({len(events)} step(s))", state="complete", expanded=_trace_expanded()
    ):
        for event in events:
            st.write(format_event_line(event))


def _render_notices(answer: ChatAnswer) -> None:
    """Render what the pipeline changed (rewritten query, expansion, withheld passages)."""
    for notice in format_answer_notices(answer):
        st.caption(notice)


def _render_search_queries(answer: ChatAnswer) -> None:
    """Render the expanded search queries behind a collapsed expander.

    Collapsed because HyDE emits multi-sentence hypothetical passages that
    would dwarf the answer -- and, being invented, must not read as corpus
    content. The caption above says how many there were; this is for when
    you're debugging why retrieval found what it found.
    """
    if not answer.search_queries:
        return
    with st.expander(f"🔎 {len(answer.search_queries)} search query/queries", expanded=False):
        st.caption("Generated to search with — not corpus content, and not necessarily true.")
        for rank, search_query in enumerate(answer.search_queries, start=1):
            st.markdown(f"**{rank}.** {search_query}")


def _render_metrics(answer: ChatAnswer) -> None:
    """Render the turn's wall time, LLM calls and tokens as a caption."""
    metrics = format_turn_metrics(answer)
    if metrics:
        st.caption(metrics)


def _record_feedback(turn_id: str, sink: TurnSink) -> None:
    """`st.feedback` callback: append the new rating to the turn log.

    Every click is appended (the latest per turn wins when read back) rather
    than updated in place -- the log is append-only. Clearing a rating isn't
    recorded; there's no rating to store.
    """
    rating = rating_from_feedback_widget(st.session_state.get(_feedback_key(turn_id)))
    if rating is None:
        return
    try:
        sink.record_feedback(FeedbackRecord(turn_id=turn_id, rating=rating))
    except Exception:  # noqa: BLE001
        logger.exception("Failed to record feedback for turn %s", turn_id)
        st.toast("⚠️ Couldn't save that rating — see the server log.")


def _feedback_key(turn_id: str) -> str:
    return f"feedback_{turn_id}"


def _render_feedback(answer: ChatAnswer, sink: TurnSink | None) -> None:
    """Render thumbs up/down for a logged turn. Hidden when turn logging is off."""
    if sink is None or answer.turn_id is None:
        return
    st.feedback(
        "thumbs",
        key=_feedback_key(answer.turn_id),
        on_change=_record_feedback,
        args=(answer.turn_id, sink),
    )


def _render_citations(answer: ChatAnswer) -> None:
    """Render collapsible citation cards under an assistant message."""
    if not answer.citations:
        return
    with st.expander(f"📄 {len(answer.citations)} source(s)", expanded=False):
        for rank, citation in enumerate(answer.citations, start=1):
            label = format_citation_label(citation, rank)
            preview = format_citation_preview(citation)
            st.markdown(f"**{label}** — score `{citation.score:.3f}`")
            st.caption(preview)
            if rank < len(answer.citations):
                st.divider()


def _render_history(sink: TurnSink | None) -> None:
    """Re-render all messages from session state."""
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                _render_events(msg.get("events", []))
                if msg.get("answer"):
                    _render_notices(msg["answer"])
                    _render_search_queries(msg["answer"])
                    _render_citations(msg["answer"])
                    _render_metrics(msg["answer"])
                    _render_feedback(msg["answer"], sink)


def _handle_query(query: str, chat_service: ChatResponder, sink: TurnSink | None) -> None:
    """Add the user message, run the pipeline, and append the assistant reply."""
    # Snapshot the conversation *before* appending the current question -- the
    # condenser wants the turns preceding the query, not the query itself.
    history = history_from_messages(st.session_state["messages"])

    # Show and persist the user message immediately.
    st.session_state["messages"].append({"role": "user", "content": query, "answer": None})
    with st.chat_message("user"):
        st.markdown(query)

    # Run the pipeline, streaming each completed stage into a live status
    # block as it happens -- the on_event callback is called synchronously
    # from within chat_service.ask, so this is a real step-by-step trace,
    # not a simulated one.
    with st.chat_message("assistant"):
        events: list[PipelineEvent] = []
        with st.status("Running pipeline…", expanded=True) as status:

            def _on_event(event: PipelineEvent) -> None:
                events.append(event)
                status.update(label=f"Running pipeline… ({event.stage})")
                status.write(format_event_line(event))

            try:
                answer = chat_service.ask(query, history=history, on_event=_on_event)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Error during chat service call")
                status.update(label="Pipeline failed", state="error", expanded=True)
                error_msg = f"⚠️ Error: {exc}"
                st.error(error_msg)
                st.session_state["messages"].append(
                    {"role": "assistant", "content": error_msg, "answer": None, "events": events}
                )
                return

            status.update(
                label=f"Pipeline trace ({len(events)} step(s))",
                state="complete",
                expanded=_trace_expanded(),
            )

        st.markdown(answer.answer)
        _render_notices(answer)
        _render_search_queries(answer)
        _render_citations(answer)
        _render_metrics(answer)
        _render_feedback(answer, sink)

    st.session_state["messages"].append(
        {"role": "assistant", "content": answer.answer, "answer": answer, "events": events}
    )


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------


def main() -> None:
    _init_session()

    # Build (or retrieve cached) chat service.
    try:
        chat_service, config, sink = _load_chat_service()
    except Exception as exc:  # noqa: BLE001
        st.error(
            f"Failed to build chat service: {exc}\n\n"
            "Check that Ollama is running and the vector index is built "
            "(`python -m rag.cli index`)."
        )
        st.stop()

    # Sidebar.
    with st.sidebar:
        st.title("⚙️ Config")
        st.markdown(sidebar_config_summary(config))
        st.divider()
        st.checkbox(
            "Keep pipeline trace expanded",
            key=_EXPAND_TRACE_KEY,
            help="Show each turn's retrieve → rerank → generate steps without having to click in.",
        )
        if st.button("🗑️ Clear chat history"):
            st.session_state["messages"] = []
            st.rerun()

    # Main header.
    st.title("🔍 RAG Chat")
    st.caption(
        "Ask a question — the system retrieves relevant passages and generates a cited answer."
    )

    # Render existing history.
    _render_history(sink)

    # Chat input (pinned to bottom by Streamlit).
    if query := st.chat_input("Ask a question about your documents…"):
        _handle_query(query, chat_service, sink)


main()
