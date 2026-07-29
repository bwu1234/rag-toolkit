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

Pure helper functions (citation formatting, config summary) live in
``rag/ui/helpers.py`` so they can be tested without a running Streamlit server.
"""

from __future__ import annotations

import logging

import streamlit as st

from rag.config.settings import RagConfig, load_config
from rag.events import PipelineEvent
from rag.generation.builder import build_chat_service
from rag.generation.chat_service import ChatAnswer, ChatService
from rag.logging_config import configure_logging
from rag.ui.helpers import (
    format_citation_label,
    format_citation_preview,
    format_event_line,
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
def _load_chat_service() -> tuple[ChatService, RagConfig]:
    config = load_config()
    service = build_chat_service(config)
    return service, config


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------


def _init_session() -> None:
    if "messages" not in st.session_state:
        # Each message: {"role": "user"|"assistant", "content": str,
        #                "answer": ChatAnswer | None,
        #                "events": list[PipelineEvent]}
        st.session_state["messages"] = []


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _render_events(events: list[PipelineEvent]) -> None:
    """Render a completed pipeline trace as a collapsed status block."""
    if not events:
        return
    with st.status(f"Pipeline trace ({len(events)} step(s))", state="complete", expanded=False):
        for event in events:
            st.write(format_event_line(event))


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


def _render_history() -> None:
    """Re-render all messages from session state."""
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                _render_events(msg.get("events", []))
                if msg.get("answer"):
                    _render_citations(msg["answer"])


def _handle_query(query: str, chat_service: ChatService) -> None:
    """Add the user message, run the pipeline, and append the assistant reply."""
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
                answer = chat_service.ask(query, on_event=_on_event)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Error during chat service call")
                status.update(label="Pipeline failed", state="error", expanded=True)
                error_msg = f"⚠️ Error: {exc}"
                st.error(error_msg)
                st.session_state["messages"].append(
                    {"role": "assistant", "content": error_msg, "answer": None, "events": events}
                )
                return

            status.update(label=f"Pipeline trace ({len(events)} step(s))", state="complete", expanded=False)

        st.markdown(answer.answer)
        _render_citations(answer)

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
        chat_service, config = _load_chat_service()
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
        if st.button("🗑️ Clear chat history"):
            st.session_state["messages"] = []
            st.rerun()

    # Main header.
    st.title("🔍 RAG Chat")
    st.caption(
        "Ask a question — the system retrieves relevant passages and generates a cited answer."
    )

    # Render existing history.
    _render_history()

    # Chat input (pinned to bottom by Streamlit).
    if query := st.chat_input("Ask a question about your documents…"):
        _handle_query(query, chat_service)


main()
