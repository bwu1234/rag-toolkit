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
from rag.retrieval.factory import get_reranker
from rag.retrieval.retriever import Retriever
from rag.retrieval.sparse import BM25Index, bm25_index_path
from rag.vectorstore.factory import get_vector_store


def build_retriever(config: RagConfig) -> Retriever:
    """Construct a `Retriever` with all components selected per `config`."""

    paths = config.paths.resolved()
    embedder = get_embedder(config.embedding)
    vector_store = get_vector_store(config.vector_store, paths.index_dir)
    reranker = get_reranker(config.reranker)
    sparse_index = BM25Index(bm25_index_path(paths.index_dir))

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
    )
