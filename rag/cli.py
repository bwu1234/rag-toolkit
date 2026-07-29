"""Command-line entrypoints for the RAG system.

Run with `python -m rag.cli <command>`. Each command configures logging once
and reads paths/parameters from the loaded `RagConfig` -- never hardcoded.
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter

from rag.chunking.chunkers import get_chunker
from rag.config.settings import load_config
from rag.embedding.factory import get_embedder
from rag.generation.builder import build_chat_service
from rag.ingestion.cleaners import clean_documents, clean_text
from rag.ingestion.loaders import load_corpus
from rag.logging_config import configure_logging
from rag.retrieval.builder import build_retriever
from rag.retrieval.sparse import BM25Index, bm25_index_path
from rag.vectorstore.factory import get_vector_store

logger = logging.getLogger(__name__)

# How many chunks to embed and upsert per round trip. Keeps memory and
# per-request payload size bounded regardless of corpus size; independent of
# `OllamaEmbedder`'s own internal batch size, which governs HTTP call size.
_INDEX_BATCH_SIZE = 64


def _cmd_ingest(args: argparse.Namespace) -> None:
    """Load every supported file in the corpus directory and print summary stats.

    This is intentionally a read-only, side-effect-free command at this stage
    -- it exercises the loader + cleaner pipeline end to end so you can sanity
    check extraction quality before chunking/embedding (later milestones) make
    runs more expensive.
    """

    config = load_config(args.config)
    corpus_dir = config.paths.resolved().corpus_dir

    logger.info("Loading corpus from %s", corpus_dir)
    documents = load_corpus(corpus_dir)

    if not documents:
        logger.warning("No documents loaded -- is the corpus directory empty or unsupported?")
        return

    by_type: Counter[str] = Counter(doc.doc_type for doc in documents)
    raw_chars = sum(len(doc.text) for doc in documents)
    cleaned_chars = sum(len(clean_text(doc.text)) for doc in documents)

    print(f"\nLoaded {len(documents)} document(s) from {corpus_dir}")
    print("By type:")
    for doc_type, count in sorted(by_type.items()):
        print(f"  {doc_type:10s} {count}")
    print(f"\nTotal characters -- raw: {raw_chars:,}  cleaned: {cleaned_chars:,}")

    if args.show:
        print(f"\nFirst {args.show} document(s):")
        for doc in documents[: args.show]:
            preview = clean_text(doc.text)[:200].replace("\n", " ")
            print(f"\n[{doc.id}] ({doc.doc_type}, title={doc.metadata.get('title')!r})")
            print(f"  {preview}{'...' if len(preview) == 200 else ''}")


def _cmd_chunk(args: argparse.Namespace) -> None:
    """Run load -> clean -> chunk and print size statistics + sample chunks.

    Like `ingest`, this is read-only and exists to let you tune
    `chunking.chunk_size` / `chunk_overlap` in config.yaml and immediately see
    the effect on chunk counts and sizes -- before paying for embeddings.
    """

    config = load_config(args.config)
    corpus_dir = config.paths.resolved().corpus_dir

    logger.info("Loading corpus from %s", corpus_dir)
    documents = clean_documents(load_corpus(corpus_dir))
    if not documents:
        logger.warning("No documents loaded -- is the corpus directory empty or unsupported?")
        return

    chunker = get_chunker(config.chunking)
    chunks = chunker.chunk(documents)
    if not chunks:
        logger.warning("No chunks produced -- are the documents empty?")
        return

    sizes = [len(c.text) for c in chunks]
    print(f"\n{len(documents)} document(s) -> {len(chunks)} chunk(s)")
    print(
        f"Strategy: {config.chunking.strategy}  "
        f"(chunk_size={config.chunking.chunk_size}, overlap={config.chunking.chunk_overlap})"
    )
    print(f"Chunk size (chars) -- min: {min(sizes)}  max: {max(sizes)}  avg: {sum(sizes) / len(sizes):.0f}")

    if args.show:
        print(f"\nFirst {args.show} chunk(s):")
        for chunk in chunks[: args.show]:
            preview = chunk.text[:160].replace("\n", " ")
            span = f"{chunk.metadata['char_start']}-{chunk.metadata['char_end']}"
            print(f"\n[{chunk.id}] (chars {span}, title={chunk.metadata.get('title')!r})")
            print(f"  {preview}{'...' if len(preview) == 160 else ''}")


def _cmd_index(args: argparse.Namespace) -> None:
    """Run load -> clean -> chunk -> embed -> upsert, building the persistent vector index.

    Unlike `ingest`/`chunk`, this command has side effects (writes to
    `paths.index_dir`) and costs real time/compute (one embedding call per
    chunk batch). Upserting is keyed by `chunk.id`, so re-running after small
    corpus edits is safe and idempotent; pass `--reset` to wipe the collection
    first after a chunking/embedding config change that invalidates old ids or
    vector dimensions.
    """

    config = load_config(args.config)
    paths = config.paths.resolved()

    logger.info("Loading corpus from %s", paths.corpus_dir)
    documents = clean_documents(load_corpus(paths.corpus_dir))
    if not documents:
        logger.warning("No documents loaded -- is the corpus directory empty or unsupported?")
        return

    chunker = get_chunker(config.chunking)
    chunks = chunker.chunk(documents)
    if not chunks:
        logger.warning("No chunks produced -- are the documents empty?")
        return

    embedder = get_embedder(config.embedding)
    store = get_vector_store(config.vector_store, paths.index_dir)
    # Always maintain the BM25 text index alongside the vector store so
    # switching retrieval.mode to hybrid later does not require re-embedding.
    sparse = BM25Index(bm25_index_path(paths.index_dir))

    if args.reset:
        logger.info("Resetting collection %r before indexing", config.vector_store.collection_name)
        store.reset()
        sparse.reset()

    print(f"\n{len(documents)} document(s) -> {len(chunks)} chunk(s) to index")
    print(f"Embedder: {config.embedding.provider}:{config.embedding.model} ({config.embedding.base_url})")
    print(f"Vector store: {config.vector_store.provider} (collection={config.vector_store.collection_name!r}, dir={paths.index_dir})")
    print(f"Sparse index: BM25 ({bm25_index_path(paths.index_dir).name})")

    import hashlib

    for start in range(0, len(chunks), _INDEX_BATCH_SIZE):
        batch = chunks[start : start + _INDEX_BATCH_SIZE]

        # Compute a stable content hash per chunk so we can skip re-embedding
        # chunks whose text hasn't changed since the last index run.
        batch_ids = [c.id for c in batch]
        new_hashes = {c.id: hashlib.sha256(c.text.encode("utf-8")).hexdigest() for c in batch}

        # Ask the store which ids already exist and what metadata they carry.
        existing = store.get_metadatas(batch_ids)

        # Determine which chunks actually changed (or are new) by comparing
        # the stored `content_hash` metadata against the freshly computed one.
        to_update: list[Chunk] = []
        for c in batch:
            stored = existing.get(c.id, {})
            stored_hash = stored.get("content_hash")
            if stored_hash != new_hashes[c.id]:
                # Build a fresh Chunk with an index-time content_hash set in
                # metadata — Chunk is frozen, so create a new instance.
                updated_meta = dict(c.metadata)
                updated_meta["content_hash"] = new_hashes[c.id]
                updated = type(c)(
                    id=c.id,
                    text=c.text,
                    document_id=c.document_id,
                    source=c.source,
                    doc_type=c.doc_type,
                    metadata=updated_meta,
                )
                to_update.append(updated)

        if not to_update:
            print(f"  skipped {len(batch)} unchanged chunk(s)")
            # still mark progress for the CLI user
            continue

        vectors = embedder.embed_documents([chunk.text for chunk in to_update])
        store.upsert(to_update, vectors)
        sparse.upsert(to_update)
        done = min(start + _INDEX_BATCH_SIZE, len(chunks))
        print(f"  embedded + upserted {len(to_update)} (changed) / {done}/{len(chunks)} chunk(s)")

    sparse.flush()
    print(
        f"\nIndex now holds {store.count()} vector chunk(s) "
        f"+ {sparse.count()} BM25 chunk(s) (dimensions={embedder.dimensions})"
    )


def _cmd_retrieve(args: argparse.Namespace) -> None:
    """Run the full retrieve -> rerank pipeline for a query and print ranked results.

    Read-only against the existing index (build one first with `index`). This
    is the command to use while tuning `retrieval.top_k` /
    `retrieval.rerank_top_k` or comparing `reranker.provider: none` vs.
    `cross_encoder` -- you see exactly what the chat API passes to the LLM as
    context, with citations and scores attached.
    """

    config = load_config(args.config)
    retriever = build_retriever(config)

    print(f"\nQuery: {args.query!r}")
    print(
        f"Retrieval: mode={config.retrieval.mode}  top_k={config.retrieval.top_k}  "
        f"rerank_top_k={config.retrieval.rerank_top_k}  "
        f"reranker={config.reranker.provider}  min_score={config.retrieval.min_score}"
        + (f"  rrf_k={config.retrieval.rrf_k}" if config.retrieval.mode == "hybrid" else "")
    )

    outcome = retriever.retrieve(args.query)
    results = outcome.chunks
    if not results:
        if outcome.candidate_count == 0:
            print("\nNo candidates at all -- is the index empty? Build it with `python -m rag.cli index`.")
        else:
            print(
                f"\nNo results above the relevance floor: {outcome.candidate_count} candidate(s) "
                f"retrieved, all {outcome.dropped_below_min_score} reranked result(s) scored below "
                f"min_score={config.retrieval.min_score}. Lower it to see them."
            )
        return

    if outcome.dropped_below_min_score:
        print(
            f"\nWithheld {outcome.dropped_below_min_score} result(s) scoring below "
            f"min_score={config.retrieval.min_score}."
        )

    print(f"\nTop {len(results)} result(s):")
    for rank, chunk in enumerate(results, start=1):
        preview = chunk.text[:200].replace("\n", " ")
        page = chunk.metadata.get("page")
        citation = f"{chunk.document_id}" + (f" (p.{page})" if page is not None else "")
        print(f"\n[{rank}] score={chunk.score:.3f}  {citation}")
        print(f"  {preview}{'...' if len(preview) == 200 else ''}")


def _cmd_chat(args: argparse.Namespace) -> None:
    """Run the full retrieve -> rerank -> generate pipeline for a question and print the answer.

    This is the same orchestration the `/chat` API endpoint uses
    (`rag.generation.chat_service.ChatService`), exercised from the command
    line -- handy for sanity-checking prompts and citations against a live
    Ollama daemon without standing up the API.
    """

    config = load_config(args.config)
    chat_service = build_chat_service(config)

    print(f"\nQuestion: {args.query!r}")
    print(f"LLM: {config.llm.provider}:{config.llm.model} ({config.llm.base_url})")

    result = chat_service.ask(args.query)

    # The CLI is single-shot, so `rewritten_query` is only ever set here if a
    # future caller passes history -- printed anyway so the two things the
    # pipeline does silently are visible from every entrypoint, not just the UI.
    if result.rewritten_query:
        print(f"Retrieved for: {result.rewritten_query!r}")

    print(f"\nAnswer:\n{result.answer}")
    if result.dropped_below_min_score:
        print(
            f"\n({result.dropped_below_min_score} passage(s) withheld below "
            f"min_score={config.retrieval.min_score})"
        )
    if result.citations:
        print(f"\nCitations ({len(result.citations)}):")
        for rank, citation in enumerate(result.citations, start=1):
            label = f"{citation.document_id}" + (f" (p.{citation.page})" if citation.page is not None else "")
            preview = citation.text[:160].replace("\n", " ")
            print(f"\n[{rank}] score={citation.score:.3f}  {label}")
            print(f"  {preview}{'...' if len(preview) == 160 else ''}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag", description="RAG system CLI")
    parser.add_argument("--config", default=None, help="Path to a config YAML (defaults to rag/config/config.yaml)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Load the corpus and print summary statistics")
    ingest.add_argument(
        "--show", type=int, default=0, metavar="N",
        help="Also print a cleaned-text preview of the first N documents",
    )
    ingest.set_defaults(func=_cmd_ingest)

    chunk = subparsers.add_parser("chunk", help="Load, clean, and chunk the corpus and print size statistics")
    chunk.add_argument(
        "--show", type=int, default=0, metavar="N",
        help="Also print a preview of the first N chunks",
    )
    chunk.set_defaults(func=_cmd_chunk)

    index = subparsers.add_parser("index", help="Load, clean, chunk, embed, and upsert the corpus into the vector index")
    index.add_argument(
        "--reset", action="store_true",
        help="Delete the existing collection before indexing (use after a chunking/embedding config change)",
    )
    index.set_defaults(func=_cmd_index)

    retrieve = subparsers.add_parser("retrieve", help="Retrieve and rerank chunks for a query against the existing index")
    retrieve.add_argument("query", help="The question or search query to retrieve chunks for")
    retrieve.set_defaults(func=_cmd_retrieve)

    chat = subparsers.add_parser("chat", help="Ask a question and get a generated, cited answer (retrieve -> rerank -> generate)")
    chat.add_argument("query", help="The question to ask")
    chat.set_defaults(func=_cmd_chat)

    return parser


def main() -> None:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
