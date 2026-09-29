"""Command-line entrypoints for the RAG system.

Run with `python -m rag.cli <command>`. Each command configures logging once
and reads paths/parameters from the loaded `RagConfig` -- never hardcoded.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from collections import Counter

from rag.chunking.context_cache import ContextCache, context_cache_path
from rag.chunking.contextualizer import ChunkContextualizer
from rag.chunking.models import Chunk, content_hash
from rag.config.settings import CorpusSelection, RagConfig, load_config
from rag.embedding.factory import get_embedder
from rag.generation.builder import build_chat_service
from rag.generation.factory import get_llm_client
from rag.index_manifest import (
    INDEX_SECTIONS,
    IndexManifest,
    IndexManifestMismatch,
    index_manifest_path,
    prepare_for_indexing,
    read_index_manifest,
)
from rag.index_report import IndexState, build_report, format_report
from rag.ingestion.cleaners import clean_text
from rag.ingestion.corpora import chunk_selected_corpora, load_selected_corpora
from rag.logging_config import configure_logging
from rag.observability.factory import get_turn_sink, turn_log_path
from rag.observability.sink import read_turn_log, turns_with_feedback
from rag.retrieval.builder import build_retriever
from rag.retrieval.factory import get_sparse_index, sparse_index_path
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
    selection, documents = load_selected_corpora(config, args.corpus)

    if not documents:
        logger.warning("No documents loaded -- is the corpus directory empty or unsupported?")
        return

    by_type: Counter[str] = Counter(doc.doc_type for doc in documents)
    raw_chars = sum(len(doc.text) for doc in documents)
    cleaned_chars = sum(len(clean_text(doc.text)) for doc in documents)

    print(f"\nLoaded {len(documents)} document(s) from corpus {selection.describe()}")
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
    _selection, documents, chunks = chunk_selected_corpora(config, args.corpus)
    if not documents:
        logger.warning("No documents loaded -- is the corpus directory empty or unsupported?")
        return
    if not chunks:
        logger.warning("No chunks produced -- are the documents empty?")
        return

    sizes = [len(c.text) for c in chunks]
    print(f"\n{len(documents)} document(s) -> {len(chunks)} chunk(s)")
    if config.chunking.strategy == "none":
        print("Strategy: none (one chunk per document)")
    else:
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
    chunk batch). It is incremental: unchanged chunks are skipped by content
    hash, and chunks the corpus no longer produces (deleted or shortened
    documents) are removed from both indexes. Changing the embedder,
    contextual settings, carried metadata or header template can't be applied
    incrementally; the index manifest
    refuses the run until `--reset` rebuilds it (see `rag.index_manifest`).
    """

    config = load_config(args.config)
    paths = config.paths.resolved()

    selection, documents, chunks = chunk_selected_corpora(config, args.corpus)
    if not documents:
        logger.warning("No documents loaded -- is the corpus directory empty or unsupported?")
        return
    if not chunks:
        logger.warning("No chunks produced -- are the documents empty?")
        return

    context_cache: ContextCache | None = None
    contextualizer = None
    if config.chunking.contextual.enabled:
        if config.chunking.contextual.cache:
            context_cache = ContextCache(context_cache_path(paths.index_dir))
            if args.clear_context_cache:
                context_cache.clear()
        contextualizer = ChunkContextualizer(
            get_llm_client(config.llm),
            max_document_chars=config.chunking.contextual.max_document_chars,
            max_context_chars=config.chunking.contextual.max_context_chars,
            concurrency=config.chunking.contextual.concurrency,
            cache=context_cache,
            model_name=config.llm.model,
        )
    elif args.clear_context_cache:
        # Honour the flag even when contextual chunking is off, so a stale cache
        # can be cleared without first turning the feature back on.
        ContextCache(context_cache_path(paths.index_dir)).clear()

    embedder = get_embedder(config.embedding)
    store = get_vector_store(
        config.vector_store, selection.index_dir, collection_name=selection.collection_name
    )
    # Always maintain the keyword index alongside the vector store so
    # switching retrieval.mode to hybrid later does not require re-embedding.
    sparse = get_sparse_index(config.sparse_index, selection.index_dir, selection.slug)

    if args.reset:
        logger.info("Resetting collection %r before indexing", selection.collection_name)
        store.reset()
        sparse.reset()

    # Before any write: refuse to extend an index built with a different
    # embedder or contextual settings, which the per-chunk text hash can't see.
    prepare_for_indexing(
        index_manifest_path(selection.index_dir, selection.slug),
        config,
        index_is_empty=store.count() == 0 and sparse.count() == 0,
    )

    print(f"\nCorpus: {selection.describe()}")
    print(f"{len(documents)} document(s) -> {len(chunks)} chunk(s) to index")
    print(f"Embedder: {config.embedding.provider}:{config.embedding.model} ({config.embedding.base_url})")
    print(f"Vector store: {config.vector_store.provider} (collection={selection.collection_name!r}, dir={selection.index_dir})")
    sparse_path = sparse_index_path(config.sparse_index, selection.index_dir, selection.slug)
    print(f"Sparse index: {config.sparse_index.provider} ({sparse_path.name})")
    if contextualizer is not None:
        print(
            f"Contextual chunking: on ({config.llm.provider}:{config.llm.model}) "
            f"-- one LLM call per changed chunk, {contextualizer.concurrency} at a time"
        )
        if context_cache is not None:
            print(f"Context cache: {len(context_cache)} entry(s) at {context_cache.path.name}")

    for start in range(0, len(chunks), _INDEX_BATCH_SIZE):
        batch = chunks[start : start + _INDEX_BATCH_SIZE]

        # Compute a stable content hash per chunk so we can skip re-embedding
        # chunks whose text hasn't changed since the last index run.
        batch_ids = [c.id for c in batch]
        new_hashes = {c.id: content_hash(c.text) for c in batch}

        # Ask the store which ids already exist and what metadata they carry.
        existing = store.get_metadatas(batch_ids)

        # Determine which chunks actually changed (or are new) by comparing
        # the stored `content_hash` metadata against the freshly computed one.
        to_update: list[Chunk] = []
        for c in batch:
            stored = existing.get(c.id, {})
            stored_hash = stored.get("content_hash")
            # Require the chunk to be current in BOTH stores. Chroma persists on
            # write while the BM25 index is flushed once at the end of a run, so
            # an interrupted run leaves chunks in the vector store that never
            # reached the sparse one -- and keying the skip on the vector store
            # alone would strand them there permanently. Observed for real: a
            # killed contextual build left 4,236 vectors against 4,172 BM25
            # chunks, and a re-run "successfully" skipped every one of them.
            if stored_hash != new_hashes[c.id] or not sparse.has_chunk(c.id):
                # Chunk is frozen, so copy it with an index-time content_hash
                # in its metadata. `replace` keeps every other field (the
                # header included) without listing them here.
                updated_meta = {**c.metadata, "content_hash": new_hashes[c.id]}
                to_update.append(dataclasses.replace(c, metadata=updated_meta))

        if not to_update:
            print(f"  skipped {len(batch)} unchanged chunk(s)")
            # still mark progress for the CLI user
            continue

        # Contextualize *after* the change check, not before: this is one LLM
        # call per chunk, and there's no sense paying it for chunks we already
        # know we're skipping. Note the content hash covers `chunk.text` only,
        # so flipping `chunking.contextual.enabled` doesn't invalidate an
        # existing index by itself -- re-index with `--reset` after changing it.
        if contextualizer is not None:
            print(f"  generating context for {len(to_update)} chunk(s)...")
            to_update = contextualizer.contextualize(to_update, documents)

        # Embed the index text (header + context + chunk) while the store keeps
        # `chunk.text` verbatim, so retrieval matches on the enriched string and
        # citations still quote the real source span.
        vectors = embedder.embed_documents([chunk.index_text for chunk in to_update])
        store.upsert(to_update, vectors)
        sparse.upsert(to_update)
        done = min(start + _INDEX_BATCH_SIZE, len(chunks))
        print(f"  embedded + upserted {len(to_update)} (changed) / {done}/{len(chunks)} chunk(s)")

    # Chunk ids are positional (`<doc>::chunk<n>`), so a deleted document leaves
    # all its chunks behind and a shortened one leaves its tail. Anything either
    # index holds that this run didn't produce is stale.
    current_ids = {chunk.id for chunk in chunks}
    stale_vectors = sorted(store.ids() - current_ids)
    stale_sparse = sorted(sparse.ids() - current_ids)
    if stale_vectors or stale_sparse:
        store.delete(stale_vectors)
        sparse.delete(stale_sparse)
        print(f"  removed {max(len(stale_vectors), len(stale_sparse))} stale chunk(s) no longer in the corpus")

    sparse.flush()
    if context_cache is not None:
        context_cache.close()
    print(
        f"\nIndex now holds {store.count()} vector chunk(s) "
        f"+ {sparse.count()} sparse chunk(s) (dimensions={embedder.dimensions})"
    )


def _cmd_index_report(args: argparse.Namespace) -> None:
    """Describe what the configured chunker makes of the corpus, and whether the index matches.

    Read-only, and needs no embedder or LLM: sizes and boundary defects come
    from chunking the corpus the way `index` would, and the index section only
    reads ids, stored hashes and the manifest. Run it before and after a
    chunking change to see what the change did before paying for a reindex.
    """

    config = load_config(args.config)
    selection, documents, chunks = chunk_selected_corpora(config, args.corpus)
    report = build_report(
        corpus=selection.describe(),
        chunking=(
            {
                "strategy": config.chunking.strategy,
                "chunk_size": config.chunking.chunk_size,
                "chunk_overlap": config.chunking.chunk_overlap,
            }
            # `none` ignores the sizes; printing them would suggest they apply.
            if config.chunking.strategy != "none"
            else {"strategy": "none"}
        ),
        documents=documents,
        chunks=chunks,
        min_chars=args.min_chars,
        max_chars=args.max_chars or config.chunking.chunk_size,
        header_template=config.chunking.header.template,
        index=_index_state(config, selection, chunks),
    )
    if args.json:
        print(json.dumps(dataclasses.asdict(report), indent=2))
    else:
        print(format_report(report))


def _index_state(config: RagConfig, selection: CorpusSelection, chunks: list[Chunk]) -> IndexState | None:
    """Compare the built index for `selection` with `chunks`, or None if it isn't built.

    Keyed on the sparse index file because it is always written alongside the vector
    collection, and checking it first keeps this command read-only: opening the
    vector store would create an empty collection where there was none.
    """

    sparse_path = sparse_index_path(config.sparse_index, selection.index_dir, selection.slug)
    if not sparse_path.exists():
        return None
    sparse = get_sparse_index(config.sparse_index, selection.index_dir, selection.slug)
    store = get_vector_store(config.vector_store, selection.index_dir, collection_name=selection.collection_name)

    corpus_ids = {chunk.id for chunk in chunks}
    vector_ids, sparse_ids = store.ids(), sparse.ids()
    stored = store.get_metadatas(sorted(corpus_ids & vector_ids))
    changed = sum(
        1 for chunk in chunks
        if chunk.id in stored and stored[chunk.id].get("content_hash") != content_hash(chunk.text)
    )

    manifest = read_index_manifest(index_manifest_path(selection.index_dir, selection.slug))
    differences = (
        manifest.differences(IndexManifest.from_config(config), sections=INDEX_SECTIONS)
        if manifest is not None
        else []
    )
    return IndexState(
        manifest=dataclasses.asdict(manifest) if manifest is not None else None,
        manifest_differences=differences,
        vector_chunks=len(vector_ids),
        sparse_chunks=len(sparse_ids),
        missing=len(corpus_ids - (vector_ids & sparse_ids)),
        stale=len((vector_ids | sparse_ids) - corpus_ids),
        changed=changed,
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
    retriever = build_retriever(config, corpora=args.corpus)

    print(f"\nQuery: {args.query!r}")
    print(
        f"Retrieval: mode={config.retrieval.mode}  top_k={config.retrieval.top_k}  "
        f"rerank_top_k={config.retrieval.rerank_top_k}  "
        f"reranker={config.reranker.provider}  min_score={config.retrieval.min_score}  "
        f"expansion={config.retrieval.expansion.provider}"
        + (f"  rrf_k={config.retrieval.rrf_k}" if config.retrieval.mode == "hybrid" else "")
    )

    outcome = retriever.retrieve(args.query)
    results = outcome.chunks

    if outcome.search_queries:
        print(f"\nSearched {len(outcome.search_queries)} query/queries:")
        for rank, search_query in enumerate(outcome.search_queries, start=1):
            print(f"  {rank}. {search_query}")
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
    """Answer a question through the configured `chat.mode` and print the answer.

    This is the same responder the `/chat` API endpoint uses (`ChatService`,
    or `AgentService` under `chat.mode: agentic`), exercised from the command
    line -- handy for sanity-checking prompts and citations against a live
    Ollama daemon without standing up the API.
    """

    config = load_config(args.config)
    chat_service = build_chat_service(
        config, corpora=args.corpus, turn_sink=get_turn_sink(config.observability.turn_log)
    )

    print(f"\nQuestion: {args.query!r}")
    print(f"LLM: {config.llm.provider}:{config.llm.model} ({config.llm.base_url})")
    if config.chat.mode == "agentic":
        agent_llm = config.agent.llm or config.llm
        print(f"Agent: {config.agent.strategy} on {agent_llm.provider}:{agent_llm.model}")

    result = chat_service.ask(args.query)

    # The CLI is single-shot, so `rewritten_query` is only ever set here if a
    # future caller passes history -- printed anyway so the two things the
    # pipeline does silently are visible from every entrypoint, not just the UI.
    if result.rewritten_query:
        print(f"Retrieved for: {result.rewritten_query!r}")
    if result.stopped_reason is not None:
        print(
            f"Agent ran {result.tool_calls} search(es) in {result.retrieval_attempts} round(s); "
            f"stopped: {result.stopped_reason}"
        )
        for rank, search_query in enumerate(result.search_queries, start=1):
            print(f"  {rank}. {search_query}")
    elif result.search_queries:
        print(f"Expanded into {len(result.search_queries)} search query/queries:")
        for rank, search_query in enumerate(result.search_queries, start=1):
            print(f"  {rank}. {search_query}")

    if result.retry_queries:
        print(f"Retried {len(result.retry_queries)} time(s) with rewritten queries:")
        for rank, retry_query in enumerate(result.retry_queries, start=1):
            print(f"  {rank}. {retry_query}")

    print(f"\nAnswer:\n{result.answer}")
    if result.grounded is False:
        print(
            "\n!! This answer FAILED its groundedness check -- the checker judged it "
            "to state things the retrieved passages don't support. Treat it as unverified."
        )
    if result.graded_out:
        print(f"\n({result.graded_out} passage(s) dropped by the relevance grader)")
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

    _print_turn_metrics(result.stage_ms, result.total_ms, result.llm_calls,
                        result.prompt_tokens, result.completion_tokens)
    if result.citations:
        print(f"Cited in the answer: {len(result.cited_chunk_ids)} of {len(result.citations)} passage(s)")
    if result.turn_id and config.observability.turn_log.provider != "none":
        print(f"Turn {result.turn_id} logged to {turn_log_path(config.observability.turn_log)}")


def _print_turn_metrics(
    stage_ms: dict[str, float],
    total_ms: float | None,
    llm_calls: int,
    prompt_tokens: int | None,
    completion_tokens: int | None,
) -> None:
    tokens = (
        f", {'?' if prompt_tokens is None else prompt_tokens} prompt -> "
        f"{'?' if completion_tokens is None else completion_tokens} generated tokens"
        if prompt_tokens is not None or completion_tokens is not None
        else ""
    )
    total = f"{total_ms:.0f} ms" if total_ms is not None else "?"
    print(f"\nTurn: {total} total, {llm_calls} LLM call(s){tokens}")
    if stage_ms:
        print("  " + ", ".join(f"{stage} {ms:.0f} ms" for stage, ms in stage_ms.items()))


def _cmd_turns(args: argparse.Namespace) -> None:
    """Print the most recent logged chat turns, with any feedback given on them.

    The read side of Milestone 12's turn log: a quick way to find the turns
    worth turning into eval samples (`--feedback down`) without writing a script.
    """

    config = load_config(args.config)
    turn_log = config.observability.turn_log
    if turn_log.provider != "jsonl":
        print(f"Turn log provider is {turn_log.provider!r}; only 'jsonl' can be read back.")
        return
    path = turn_log_path(turn_log)
    pairs = turns_with_feedback(read_turn_log(path))
    if args.feedback == "any":
        pairs = [(turn, fb) for turn, fb in pairs if fb is not None]
    elif args.feedback in ("up", "down"):
        pairs = [(turn, fb) for turn, fb in pairs if fb is not None and fb.get("rating") == args.feedback]

    print(f"{path}: showing {min(args.show, len(pairs))} of {len(pairs)} matching turn(s)")
    for turn, fb in pairs[-args.show:] if args.show else []:
        rating = f"  feedback={fb['rating']}" if fb else ""
        print(f"\n[{turn.get('timestamp')}] {turn.get('turn_id')}  outcome={turn.get('outcome')}{rating}")
        print(f"  Q: {turn.get('query')!r}")
        if turn.get("rewritten_query"):
            print(f"  searched for: {turn['rewritten_query']!r}")
        answer = (turn.get("answer") or turn.get("error") or "").replace("\n", " ")
        print(f"  A: {answer[:200]}{'...' if len(answer) > 200 else ''}")
        print(f"  shown={len(turn.get('shown_chunk_ids', []))} cited={turn.get('cited_chunk_ids', [])}")
        _print_turn_metrics(turn.get("stage_ms", {}), turn.get("total_ms"), turn.get("llm_calls", 0),
                            turn.get("prompt_tokens"), turn.get("completion_tokens"))
        if fb and fb.get("comment"):
            print(f"  comment: {fb['comment']!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag", description="RAG system CLI")
    parser.add_argument("--config", default=None, help="Path to a config YAML (defaults to rag/config/config.yaml)")

    # `--corpus` lives on a parent parser rather than on the top-level one so it
    # can be written *after* the subcommand (`rag index --corpus edgar`), which
    # is where people reach for it. A top-level flag would have to precede the
    # subcommand, and declaring it in both places would let the subparser's
    # default clobber the value parsed at the top level.
    corpus_args = argparse.ArgumentParser(add_help=False)
    corpus_args.add_argument(
        "--corpus", action="append", default=None, metavar="NAME",
        help=(
            "Corpus to operate on, overriding corpora.active. Repeat to pool several "
            "into one index (e.g. --corpus baseline --corpus edgar), which is how "
            "cross-corpus distractor robustness is measured."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Load the corpus and print summary statistics", parents=[corpus_args])
    ingest.add_argument(
        "--show", type=int, default=0, metavar="N",
        help="Also print a cleaned-text preview of the first N documents",
    )
    ingest.set_defaults(func=_cmd_ingest)

    chunk = subparsers.add_parser("chunk", help="Load, clean, and chunk the corpus and print size statistics", parents=[corpus_args])
    chunk.add_argument(
        "--show", type=int, default=0, metavar="N",
        help="Also print a preview of the first N chunks",
    )
    chunk.set_defaults(func=_cmd_chunk)

    index = subparsers.add_parser("index", help="Load, clean, chunk, embed, and upsert the corpus into the vector index", parents=[corpus_args])
    index.add_argument(
        "--reset", action="store_true",
        help="Delete the existing collection before indexing (use after a chunking/embedding config change)",
    )
    index.add_argument(
        "--clear-context-cache", action="store_true",
        help=(
            "Discard checkpointed chunk contexts and regenerate them. Not needed after a "
            "config change (the cache key covers every input to the call, so those miss "
            "on their own) -- use it to force a rerun on identical inputs, e.g. when "
            "comparing two models."
        ),
    )
    index.set_defaults(func=_cmd_index)

    index_report = subparsers.add_parser(
        "index-report",
        help="Chunk-size, table-split and duplicate stats for the corpus, and whether the index matches it",
        parents=[corpus_args],
    )
    index_report.add_argument(
        "--min-chars", type=int, default=100, metavar="N",
        help="Count chunks shorter than N characters (default: 100)",
    )
    index_report.add_argument(
        "--max-chars", type=int, default=None, metavar="N",
        help="Count chunks longer than N characters (default: chunking.chunk_size)",
    )
    index_report.add_argument("--json", action="store_true", help="Print the report as JSON")
    index_report.set_defaults(func=_cmd_index_report)

    retrieve = subparsers.add_parser("retrieve", help="Retrieve and rerank chunks for a query against the existing index", parents=[corpus_args])
    retrieve.add_argument("query", help="The question or search query to retrieve chunks for")
    retrieve.set_defaults(func=_cmd_retrieve)

    chat = subparsers.add_parser("chat", help="Ask a question and get a generated, cited answer (retrieve -> rerank -> generate)", parents=[corpus_args])
    chat.add_argument("query", help="The question to ask")
    chat.set_defaults(func=_cmd_chat)

    turns = subparsers.add_parser("turns", help="Show recently logged chat turns and their feedback")
    turns.add_argument("--show", type=int, default=10, metavar="N", help="How many of the most recent turns to print")
    turns.add_argument(
        "--feedback", choices=["all", "any", "up", "down"], default="all",
        help="Filter by feedback: all turns, turns with any rating, or only thumbs up/down",
    )
    turns.set_defaults(func=_cmd_turns)

    return parser


def main() -> None:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except IndexManifestMismatch as exc:
        # An expected, user-fixable condition: print the instructions, not a traceback.
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
