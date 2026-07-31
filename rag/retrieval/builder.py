"""Wires a fully-configured `Retriever` from a `RagConfig`.

Construction needs four config sections (`embedding`, `vector_store`,
`reranker`, `retrieval`) plus a resolved `paths.index_dir`. Centralizing that
wiring here means the CLI and the chat API both get an identically configured
retriever from one call, instead of duplicating factory plumbing.

The BM25 sparse index is always opened from disk (cheap: JSON load + lazy
rebuild) so ``retrieval.mode: hybrid`` works without a second code path;
dense mode simply ignores it.
"""

from __future__ import annotations

from rag.config.settings import RagConfig
from rag.embedding.factory import get_embedder
from rag.generation.factory import get_llm_client
from rag.generation.llm import LLMClient
from rag.retrieval.factory import get_query_expander, get_reranker
from rag.retrieval.retriever import Retriever
from rag.retrieval.sparse import BM25Index, bm25_index_path
from rag.vectorstore.factory import get_vector_store


def build_retriever(config: RagConfig, llm_client: LLMClient | None = None) -> Retriever:
    """Construct a `Retriever` with all components selected per `config`.

    `llm_client` is only consulted when `retrieval.expansion.provider` needs
    one; it's a parameter so `build_chat_service` can pass the client it
    already built rather than opening a second one to the same daemon. Left
    unset, a client is constructed on demand -- which costs nothing, since
    `get_llm_client` does no I/O.
    """

    paths = config.paths.resolved()
    embedder = get_embedder(config.embedding)
    vector_store = get_vector_store(config.vector_store, paths.index_dir)
    reranker = get_reranker(config.reranker)
    sparse_index = BM25Index(bm25_index_path(paths.index_dir))

    expander_client = llm_client if llm_client is not None else get_llm_client(config.llm)
    query_expander = get_query_expander(config.retrieval.expansion, expander_client)

    return Retriever(
        embedder=embedder,
        vector_store=vector_store,
        reranker=reranker,
        top_k=config.retrieval.top_k,
        rerank_top_k=config.retrieval.rerank_top_k,
        sparse_index=sparse_index,
        mode=config.retrieval.mode,
        rrf_k=config.retrieval.rrf_k,
        min_score=config.retrieval.min_score,
        query_expander=query_expander,
    )
