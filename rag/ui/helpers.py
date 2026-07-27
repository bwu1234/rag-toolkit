"""Pure helper functions for the Streamlit UI.

Kept separate from ``app.py`` so they can be imported and tested without
a running Streamlit server or session state.
"""

from __future__ import annotations

from rag.config.settings import RagConfig
from rag.generation.chat_service import Citation


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


def sidebar_config_summary(config: RagConfig) -> str:
    """Return a multi-line markdown string summarising the active config."""
    lines = [
        f"**Embedder:** `{config.embedding.provider}:{config.embedding.model}`",
        f"**LLM:** `{config.llm.provider}:{config.llm.model}`",
        f"**Reranker:** `{config.reranker.provider}`",
        f"**Retrieve top-k:** {config.retrieval.top_k} → rerank to {config.retrieval.rerank_top_k}",
        f"**Vector store:** `{config.vector_store.provider}` "
        f"(`{config.vector_store.collection_name}`)",
    ]
    return "\n\n".join(lines)
