"""The retrieval tool surface, defined once and shared by every agent that calls it.

Two consumers read the same `ToolSpec`s: the MCP server (`rag.mcp`), which
serves them to an outside agent, and the in-process agent (Milestone 19),
which advertises them to its own model through `ToolSpec.definition`. A
search-quality fix therefore lands once, and our own agent evals exercise the
exact tool an MCP client gets.

A `ToolSpec` is a name, a description, and a type-annotated handler. The JSON
Schema a caller sees is *derived* from the handler's signature via pydantic
rather than written out by hand, which is what lets the SDK-backed MCP server,
the dependency-free fallback, and the agent advertise byte-identical schemas
without any of them owning the contract.

Only read-only tools live here. Indexing is deliberately absent: it is a
minutes-long, embedding-cost operation that rewrites shared state, and an
autonomous agent should not be able to trigger it as a side effect of
answering a question. Build indexes with `python -m rag.cli index`.

Nothing here imports the `mcp` SDK, so the fallback transport and the agent
never pay for (or require) the optional extra.
"""

from __future__ import annotations

import dataclasses
import difflib
import hashlib
import inspect
import logging
import re
import threading
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Annotated, Any, get_type_hints

from pydantic import Field, create_model

from rag.chunking.chunkers import carried_metadata
from rag.ingestion.models import Document
from rag.config.settings import REPO_ROOT, CorpusSelection, RagConfig, load_config
from rag.events import EventSink
from rag.query_filter import DOCUMENT_ID, QueryFilter, check_filterable
from rag.llm.base import LLMClient, ToolDefinition
from rag.retrieval.builder import build_retriever
from rag.retrieval.retriever import RetrievalResult, Retriever
from rag.retrieval.factory import get_sparse_index, sparse_index_path

logger = logging.getLogger(__name__)

#: Hard ceiling on results per `rag_search` call. Retrievers are cached and
#: built with `rerank_top_k` widened to this, so a per-call `top_k` is served
#: by slicing an already-ranked list -- no rebuild, no mutation of shared
#: state, and taking the best 5 of a 20-deep rerank is identical to a 5-deep
#: one because the reranker returns them sorted.
MAX_RESULTS = 20

#: Per-chunk character budget. Chunks run ~1000 chars and a default search
#: returns five of them; handing an agent the full text of all of them at once
#: is a meaningful bite out of its context for passages it may well discard.
#: Callers that want the whole span ask for it explicitly.
DEFAULT_MAX_CHARS = 1200

#: Documents per `rag_list_documents` call: by default enough for `edgar_md`'s
#: 61 filings in one listing, and at most a bound a BEIR-sized corpus can't
#: blow through. Past it the result says how many were left out.
DEFAULT_LIST_LIMIT = 100
MAX_LIST_LIMIT = 500

#: Characters per `rag_read_document` window. The default is about 1.5k
#: tokens, five search passages' worth: enough to see a whole table or the
#: paragraphs around a hit. The cap fits `edgar_md`'s median filing (~48k)
#: in one call for a caller with the context to spare; the largest (143k)
#: still takes several, and `next_start` says where to continue.
DEFAULT_READ_CHARS = 6000
MAX_READ_CHARS = 50_000

#: `rag_find` matches per call, and the text shown on each side of one.
DEFAULT_FIND_RESULTS = 20
MAX_FIND_RESULTS = 100
FIND_CONTEXT_CHARS = 150


class StaleSourceError(ValueError):
    """A document's text no longer matches the offsets a caller holds for it.

    Raised instead of applying old offsets to new text: a search hit's
    `char_start`/`char_end` come from the index, which can be older than the
    file. A `ValueError`, so the transports and the agent report it as a tool
    error.
    """


def document_version(document: Document) -> str:
    """A digest of the cleaned text offsets index into: changes whenever a window could.

    Covers the text only, not metadata or the loader/cleaner identity the
    source-version contract also names (`docs/milestone-19-plan.md`), so it
    tells two texts apart but doesn't yet pin a whole build.
    """

    return hashlib.sha256(document.text.encode()).hexdigest()[:16]


def _refs(schema: Any) -> set[str]:
    """Names of the `#/$defs/...` definitions `schema` refers to directly."""

    found: set[str] = set()
    if isinstance(schema, dict):
        ref = schema.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            found.add(ref.removeprefix("#/$defs/"))
        for value in schema.values():
            found |= _refs(value)
    elif isinstance(schema, list):
        for value in schema:
            found |= _refs(value)
    return found


@dataclass(frozen=True)
class DescriptionNote:
    """A sentence of a tool's description that only holds when other parts are offered.

    A description that tells the model to call a tool it doesn't have, or to
    pass an argument the caller pinned, steers it toward calls that fail. MCP
    serves every tool and argument, so it gets every note; the agent offers a
    subset and gets only the notes that hold for it.
    """

    text: str
    #: Other tools the sentence names.
    tools: tuple[str, ...] = ()
    #: This tool's arguments the sentence names.
    arguments: tuple[str, ...] = ()


@dataclass(frozen=True)
class ToolSpec:
    """One tool: its wire identity plus the callable behind it.

    `base_description` is the text the tool always carries; `notes` add
    sentences that depend on what else is offered.
    """

    name: str
    base_description: str
    handler: Callable[..., dict[str, Any]]
    notes: tuple[DescriptionNote, ...] = ()

    @property
    def description(self) -> str:
        """The description with every note: what MCP serves, every tool and argument offered."""

        return self.description_for()

    def description_for(
        self, *, offered_tools: Collection[str] | None = None, pinned: Collection[str] = ()
    ) -> str:
        """The description, keeping only the notes whose tools are offered and arguments aren't pinned.

        `offered_tools` None means every tool is offered.
        """

        kept = [
            note.text for note in self.notes
            if (offered_tools is None or set(note.tools) <= set(offered_tools))
            and not set(note.arguments) & set(pinned)
        ]
        return " ".join([self.base_description, *kept])

    @property
    def input_schema(self) -> dict[str, Any]:
        """JSON Schema for this tool's arguments, derived from the handler."""

        return input_schema_for(self.handler)

    @property
    def definition(self) -> ToolDefinition:
        """This tool as the agent's model sees it.

        A wrapper, not a second schema: the parameters are `input_schema`, the
        same dict MCP's `tools/list` serves, and each `ToolCallingLLM` adapter
        turns it into its provider's wire format.
        """

        return ToolDefinition(name=self.name, description=self.description, parameters=self.input_schema)

    def definition_without(self, *pinned: str, offered_tools: Collection[str] | None = None) -> ToolDefinition:
        """`definition`, minus arguments the caller fixes itself.

        A projection of the same schema, not a second one: every remaining
        property is byte-identical to what MCP serves. The agent uses it to
        keep `corpus`, `top_k` and `max_chars` out of the model's hands --
        they belong to the turn (the eval's `--corpus`, the prompt-size
        guards), not to the model. Only optional arguments can be pinned; a
        required one has no value to fall back on.

        The description drops the notes that name a pinned argument or a tool
        outside `offered_tools` (None: every tool), so it never points the
        model at something it can't call.
        """

        schema = self.input_schema
        properties = dict(schema.get("properties", {}))
        required = list(schema.get("required", []))
        for name in pinned:
            if name not in properties:
                raise ValueError(f"{self.name} has no argument {name!r} to pin")
            if name in required:
                raise ValueError(f"{self.name} argument {name!r} is required and can't be pinned")
            del properties[name]
        parameters = {**schema, "properties": properties}
        if "$defs" in schema:
            # Keep only the definitions a remaining property still reaches, so
            # a pinned argument's types don't ride along in every prompt.
            defs: dict[str, Any] = schema["$defs"]
            reached: dict[str, Any] = {}
            pending = _refs(properties)
            while pending:
                name = pending.pop()
                if name not in reached and name in defs:
                    reached[name] = defs[name]
                    pending |= _refs(defs[name])
            parameters["$defs"] = {name: defs[name] for name in defs if name in reached}
            if not parameters["$defs"]:
                del parameters["$defs"]
        description = self.description_for(offered_tools=offered_tools, pinned=pinned)
        return ToolDefinition(name=self.name, description=description, parameters=parameters)


def input_schema_for(handler: Callable[..., Any]) -> dict[str, Any]:
    """Build a JSON Schema for `handler`'s parameters using pydantic.

    The `mcp` SDK does this itself when it registers a function, so the
    fallback transport calls this to arrive at the same schema from the same
    signature instead of maintaining a parallel hand-written copy that would
    quietly rot. `tests/test_mcp.py` asserts the two agree.
    """

    hints = get_type_hints(handler, include_extras=True)
    fields: dict[str, Any] = {}
    for name, parameter in inspect.signature(handler).parameters.items():
        annotation = hints.get(name, Any)
        default = ... if parameter.default is parameter.empty else parameter.default
        fields[name] = (annotation, default)

    model = create_model(f"{handler.__name__}Arguments", **fields)
    schema = model.model_json_schema()
    schema.pop("title", None)
    return schema


def _relative_to_repo(path: Path) -> str:
    """Render a path relative to the repo when it sits inside it.

    Absolute paths leak the developer's home directory into every result and
    cost context for no information; a repo-relative one is also what the user
    would type to open the file.
    """

    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


class RagTools:
    """Lazily-built, cached access to the retrieval pipeline for tool handlers.

    Two properties matter for serving MCP:

    - **Lazy.** Nothing is constructed at import or at server startup. Under
      stdio the client expects an `initialize` response promptly, and eagerly
      importing Chroma and a sentence-transformers cross-encoder would spend
      seconds before the handshake -- so the first `rag_search` pays that cost
      instead, and `rag_list_corpora` never pays it at all.
    - **Cached per corpus selection.** One `Retriever` per selection slug,
      reused across calls. The cross-encoder holds several hundred MB of
      weights once loaded; rebuilding a retriever per call would reload them.

    Handlers run on worker threads (both transports dispatch synchronous tools
    off the event loop), so the cache is guarded by a reentrant lock. Only the
    build is locked -- `Retriever.retrieve` is read-only with respect to the
    retriever, so concurrent searches proceed in parallel.
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        *,
        config: RagConfig | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        """Read config from `config_path` on first use, or take `config` as given.

        The MCP entrypoints pass a path and let the first call load it. The
        agent passes the `RagConfig` its builder already holds, so `--config`
        overlays and in-memory overrides (the eval matrices) reach its searches
        too, instead of a second load from disk quietly reading something else.

        `llm_client` goes to `build_retriever` for query expansion. The agent
        passes its metered client so expansion calls, if expansion is on,
        count toward the turn; MCP leaves it unset.
        """

        if config_path is not None and config is not None:
            raise ValueError("pass config_path or config, not both")
        self._config_path = config_path
        self._config: RagConfig | None = config
        self._llm_client = llm_client
        self._retrievers: dict[str, tuple[CorpusSelection, Retriever]] = {}
        # Cleaned documents per corpus selection, for reading and finding:
        # (selection, fingerprint of the files on disk, documents by id).
        self._documents: dict[str, tuple[CorpusSelection, tuple[Any, ...], dict[str, Document]]] = {}
        # Reentrant: `_retriever_for` holds the lock and reads `.config`,
        # which takes it again.
        self._lock = threading.RLock()

    @property
    def config(self) -> RagConfig:
        with self._lock:
            if self._config is None:
                self._config = load_config(self._config_path)
            return self._config

    def _retriever_for(
        self, corpora: Sequence[str] | None
    ) -> tuple[CorpusSelection, Retriever]:
        config = self.config
        # Raises ValueError naming the available corpora on a typo, which the
        # transports surface as a tool error -- an agent that guesses a corpus
        # name gets told the real ones rather than an empty result set.
        selection = config.corpus_selection(corpora)

        with self._lock:
            cached = self._retrievers.get(selection.slug)
            if cached is None:
                logger.info("Building retriever for corpus selection %s", selection.describe())
                retriever = build_retriever(config, self._llm_client, corpora=selection.names)
                # Widen stage 2 only. Stage-1 `top_k` governs candidate width
                # and fusion, so changing it would change which chunks compete;
                # `rerank_top_k` only governs how deep the sorted output runs.
                retriever.rerank_top_k = max(retriever.rerank_top_k, MAX_RESULTS)
                cached = (selection, retriever)
                self._retrievers[selection.slug] = cached
            return cached

    def retrieve(
        self,
        query: str,
        corpus: str | Sequence[str] | None = None,
        top_k: int | None = None,
        *,
        query_filter: QueryFilter | None = None,
        on_event: EventSink | None = None,
    ) -> tuple[CorpusSelection, RetrievalResult]:
        """Run retrieve -> rerank and return the top `top_k` chunks, unformatted.

        The one retrieval path behind `rag_search`: `search` renders this as
        MCP's JSON, and the agent renders it as numbered passages. Raises
        `ValueError` for arguments a caller got wrong (empty query, `top_k`
        out of range, unknown corpus, a filter on a field chunks don't store),
        so both can report them as tool errors.
        """

        if not query.strip():
            raise ValueError("query must not be empty")

        requested = self.config.retrieval.rerank_top_k if top_k is None else top_k
        if not 1 <= requested <= MAX_RESULTS:
            raise ValueError(f"top_k must be between 1 and {MAX_RESULTS} (got {requested})")

        names = [corpus] if isinstance(corpus, str) else corpus
        selection, retriever = self._retriever_for(names)
        outcome = retriever.retrieve(query, query_filter=query_filter, on_event=on_event)
        return selection, dataclasses.replace(outcome, chunks=outcome.chunks[:requested])

    def search(
        self,
        query: str,
        corpus: str | list[str] | None = None,
        top_k: int | None = None,
        max_chars: int | None = None,
        filters: QueryFilter | None = None,
    ) -> dict[str, Any]:
        """Run retrieve -> rerank and return ranked passages."""

        budget = DEFAULT_MAX_CHARS if max_chars is None else max_chars
        if budget < 1:
            raise ValueError(f"max_chars must be at least 1 (got {budget})")

        selection, outcome = self.retrieve(query, corpus, top_k, query_filter=filters)
        chunks = outcome.chunks

        payload: dict[str, Any] = {
            "query": query,
            "corpora": list(selection.names),
            "pooled": selection.is_pooled,
            "candidate_count": outcome.candidate_count,
            "returned": len(chunks),
            **({"filters": filters.model_dump(exclude_defaults=True)} if filters and not filters.is_empty else {}),
            "results": [
                _result_entry(rank, chunk, budget)
                for rank, chunk in enumerate(chunks, start=1)
            ],
        }
        if outcome.search_queries:
            payload["search_queries"] = outcome.search_queries
        if outcome.dropped_below_min_score:
            payload["dropped_below_min_score"] = outcome.dropped_below_min_score

        # "No results" has two causes an agent must not conflate: an index that
        # was never built, versus a corpus that genuinely has nothing to say.
        # `RetrievalResult` distinguishes them, so pass the distinction on as a
        # actionable hint rather than letting the agent guess from an empty list.
        if not chunks:
            if outcome.candidate_count == 0:
                payload["hint"] = (
                    f"No candidates at all -- the index for {selection.describe()} looks "
                    f"empty or unbuilt. Build it with: python -m rag.cli index "
                    + " ".join(f"--corpus {name}" for name in selection.names)
                )
            else:
                payload["hint"] = (
                    f"{outcome.candidate_count} candidate(s) retrieved but all scored below "
                    f"retrieval.min_score={self.config.retrieval.min_score}. The corpus may "
                    "not cover this question."
                )
        return payload

    def list_documents(
        self,
        corpus: str | Sequence[str] | None = None,
        filters: QueryFilter | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List the selected corpora's documents with their metadata, optionally filtered.

        Reads the documents' front matter from disk -- no embedder, no index,
        no cleaning (31 ms for `edgar_md`'s 61 filings) -- so it answers "which
        companies and periods exist" without ranking anything. A filter
        applies exactly as it does to search: over `document_id` and the
        `chunking.carry_metadata` fields, dates as YYYYMMDD.

        `chars` is the document's length as loaded, before cleaning.

        Pages by `offset` into the matching documents, which are in a stable
        order (corpus, then path); the payload carries `next_offset` while
        more remain. A corpus edited between pages can shift them.

        Raises `ValueError` for a bad argument (unknown corpus, a filter on a
        field chunks don't store, `limit` or `offset` out of range).
        """

        requested = DEFAULT_LIST_LIMIT if limit is None else limit
        if not 1 <= requested <= MAX_LIST_LIMIT:
            raise ValueError(f"limit must be between 1 and {MAX_LIST_LIMIT} (got {requested})")
        if offset < 0:
            raise ValueError(f"offset must be at least 0 (got {offset})")
        config = self.config
        names = [corpus] if isinstance(corpus, str) else (list(corpus) if corpus is not None else None)
        carried_keys = config.chunking.carry_metadata
        if filters is not None and filters.is_empty:
            filters = None
        if filters is not None:
            check_filterable(filters, carried_keys)

        # Imported here: the loaders pull in pypdf, which MCP's stdio handshake
        # shouldn't wait for when no one lists documents.
        from rag.ingestion.corpora import load_selected_corpora

        selection, documents = load_selected_corpora(config, names, clean=False)
        matched = [
            document for document in documents
            if filters is None
            or filters.matches({DOCUMENT_ID: document.id, **carried_metadata(document, carried_keys)})
        ]
        listed = [
            {
                "document_id": document.id,
                **_document_metadata(document, carried_keys),
                "chars": len(document.text),
            }
            for document in matched[offset:offset + requested]
        ]
        payload: dict[str, Any] = {
            "corpora": list(selection.names),
            "total": len(matched),
            "offset": offset,
            "returned": len(listed),
            **({"filters": filters.model_dump(exclude_defaults=True)} if filters is not None else {}),
            "documents": listed,
        }
        end = offset + len(listed)
        if end < len(matched):
            payload["next_offset"] = end
            payload["hint"] = (
                f"Showing documents {offset + 1}-{end} of {len(matched)}. Pass offset={end} for the "
                "next page, or narrow the list with filters (e.g. a ticker or a period_end range)."
            )
        elif offset and not listed:
            payload["hint"] = f"offset {offset} is past the last of {len(matched)} matching documents."
        return payload

    def _documents_for(self, corpus: str | Sequence[str] | None) -> tuple[CorpusSelection, dict[str, Document]]:
        """The selection's documents by id, cleaned: the text chunk offsets index into.

        Loading and cleaning `edgar_md` takes ~0.4 s, which an agent paging
        through a filing would pay on every window, so the result is cached
        per selection. The cache is checked against each file's size and
        mtime on every call (a stat per file, well under a millisecond for
        61), so an edited or re-fetched corpus is reloaded, not served stale.
        """

        config = self.config
        names = [corpus] if isinstance(corpus, str) else (list(corpus) if corpus is not None else None)
        selection = config.corpus_selection(names)
        fingerprint = _fingerprint(selection.document_dirs)
        with self._lock:
            cached = self._documents.get(selection.slug)
            if cached is not None and cached[1] == fingerprint:
                return cached[0], cached[2]
        # Imported here for the same reason as in `list_documents`.
        from rag.ingestion.corpora import load_selected_corpora

        selection, documents = load_selected_corpora(config, list(selection.names), clean=True)
        by_id = {document.id: document for document in documents}
        with self._lock:
            self._documents[selection.slug] = (selection, fingerprint, by_id)
        return selection, by_id

    def read_document(
        self,
        document_id: str,
        corpus: str | Sequence[str] | None = None,
        start: int = 0,
        max_chars: int | None = None,
        *,
        scope: QueryFilter | None = None,
        expect_spans: Sequence[tuple[int, int, str]] = (),
    ) -> dict[str, Any]:
        """Return a window of one document's cleaned text, from `start`.

        Offsets are characters into the same text the chunker split, so a
        search hit's `char_start`/`char_end` point into it directly: read from
        a little before `char_start` to see what surrounds a passage. The
        index may be older than the files, though (`python -m rag.cli
        index-report` says whether it is in sync). `version` is a digest of
        the text read, so a caller can tell two reads of different texts apart.

        Two keyword arguments serve the in-process agent; MCP passes neither:

        - `scope`: the turn's filter. A document outside it is reported
          exactly as an unknown one, near misses drawn only from inside it,
          so a guessed id neither reads nor confirms anything.
        - `expect_spans`: `(char_start, char_end, text)` of search hits from
          this document the caller has already shown. If the current text at
          any of those offsets isn't that hit's text, the index and the file
          disagree and `StaleSourceError` is raised rather than serving a
          window whose offsets mean something else now.

        Raises `ValueError` for an unknown or out-of-scope document (with near
        misses named), or `start`/`max_chars` out of range.
        """

        budget = DEFAULT_READ_CHARS if max_chars is None else max_chars
        if not 1 <= budget <= MAX_READ_CHARS:
            raise ValueError(f"max_chars must be between 1 and {MAX_READ_CHARS} (got {budget})")
        selection, documents = self._documents_for(corpus)
        if scope is not None and not scope.is_empty:
            documents = self._in_scope(documents, scope)
        document = documents.get(document_id)
        if document is None:
            raise ValueError(_unknown_document(document_id, documents, selection))
        length = len(document.text)
        if not 0 <= start < max(length, 1):
            raise ValueError(f"start must be between 0 and {max(length - 1, 0)} for {document_id!r} (got {start})")
        for span_start, span_end, text in expect_spans:
            # The fixed chunker strips its spans, and the structured one puts a
            # split table's header rows in front of later pieces: the current
            # slice, stripped, must still sit inside the hit's text.
            if document.text[span_start:span_end].strip() not in text:
                raise StaleSourceError(
                    f"{document_id!r} has changed since the search index was built: the text at a search "
                    f"result's offsets ({span_start}-{span_end}) is no longer that result's text, so offsets "
                    "from search don't apply to it. Rebuild the index (python -m rag.cli index)."
                )

        end = min(start + budget, length)
        payload: dict[str, Any] = {
            "document_id": document_id,
            "corpora": list(selection.names),
            "source": _relative_to_repo(document.source),
            "doc_type": document.doc_type,
            **_document_metadata(document, self.config.chunking.carry_metadata),
            "version": document_version(document),
            "length": length,
            "start": start,
            "end": end,
            "text": document.text[start:end],
        }
        if end < length:
            payload["next_start"] = end
        return payload

    def _in_scope(self, documents: dict[str, Document], scope: QueryFilter) -> dict[str, Document]:
        """The documents `scope` matches, by the same fields search filters on."""

        carried_keys = self.config.chunking.carry_metadata
        check_filterable(scope, carried_keys)
        return {
            document_id: document for document_id, document in documents.items()
            if scope.matches({DOCUMENT_ID: document_id, **carried_metadata(document, carried_keys)})
        }

    def find(
        self,
        phrase: str,
        corpus: str | Sequence[str] | None = None,
        document_id: str | None = None,
        filters: QueryFilter | None = None,
        max_results: int | None = None,
    ) -> dict[str, Any]:
        """Find every literal occurrence of `phrase` in the selected documents.

        Case-insensitive, and any run of whitespace in `phrase` matches any
        run in the text, so a figure that wraps across a line or a table cell
        still matches. Nothing is ranked or embedded: matches come in document
        order, then position, with offsets `rag_read_document` takes.

        Raises `ValueError` for an empty phrase, an unknown document, a filter
        on a field chunks don't store, or `max_results` out of range.
        """

        words = phrase.split()
        if not words:
            raise ValueError("phrase must not be empty")
        requested = DEFAULT_FIND_RESULTS if max_results is None else max_results
        if not 1 <= requested <= MAX_FIND_RESULTS:
            raise ValueError(f"max_results must be between 1 and {MAX_FIND_RESULTS} (got {requested})")
        carried_keys = self.config.chunking.carry_metadata
        if filters is not None and filters.is_empty:
            filters = None
        if filters is not None:
            check_filterable(filters, carried_keys)

        selection, documents = self._documents_for(corpus)
        if document_id is not None:
            if document_id not in documents:
                raise ValueError(_unknown_document(document_id, documents, selection))
            candidates = [documents[document_id]]
        else:
            candidates = list(documents.values())
        if filters is not None:
            candidates = [
                document for document in candidates
                if filters.matches({DOCUMENT_ID: document.id, **carried_metadata(document, carried_keys)})
            ]

        pattern = re.compile(r"\s+".join(re.escape(word) for word in words), re.IGNORECASE)
        matches: list[dict[str, Any]] = []
        total = 0
        documents_matched = 0
        for document in candidates:
            found = 0
            version: str | None = None
            for match in pattern.finditer(document.text):
                found += 1
                if len(matches) < requested:
                    version = version or document_version(document)
                    matches.append(_find_entry(document, version, match.start(), match.end()))
            total += found
            documents_matched += bool(found)

        payload: dict[str, Any] = {
            "phrase": phrase,
            "corpora": list(selection.names),
            **({"document_id": document_id} if document_id is not None else {}),
            **({"filters": filters.model_dump(exclude_defaults=True)} if filters is not None else {}),
            "documents_searched": len(candidates),
            "documents_matched": documents_matched,
            "total_matches": total,
            "returned": len(matches),
            "matches": matches,
        }
        if total > len(matches):
            payload["hint"] = (
                f"Showing the first {len(matches)} of {total} matches. Narrow with document_id or "
                "filters, or use a longer phrase."
            )
        elif not total:
            payload["hint"] = (
                "No literal match. The wording may differ from the phrase: try a shorter phrase, "
                "or rag_search for the idea rather than the words."
            )
        return payload

    def list_corpora(self) -> dict[str, Any]:
        """Describe every registered corpus and whether it has been indexed."""

        config = self.config
        active = list(config.corpus_selection(None).names)
        registry = config.corpora.registry
        names = sorted(registry) if registry else [config.IMPLICIT_CORPUS_NAME]

        corpora: list[dict[str, Any]] = []
        for name in names:
            selection = config.corpus_selection([name])
            entry: dict[str, Any] = {
                "name": name,
                "active": name in active,
                "documents_dir": _relative_to_repo(selection.document_dirs[0]),
                "documents_present": selection.document_dirs[0].is_dir(),
            }
            description = registry[name].description if registry else None
            if description:
                entry["description"] = " ".join(description.split())

            # Read the sparse index sidecar rather than opening the vector collection:
            # `get_vector_store` uses get_or_create, so probing it here would
            # litter the store with empty collections just from listing. The
            # indexer always writes both, so the sidecar is a faithful signal.
            sidecar = sparse_index_path(config.sparse_index, selection.index_dir, selection.slug)
            if sidecar.exists():
                try:
                    entry["indexed_chunks"] = get_sparse_index(config.sparse_index, selection.index_dir, selection.slug).count()
                except Exception:  # pragma: no cover - corrupt sidecar
                    logger.warning("Could not read sparse index at %s", sidecar, exc_info=True)
                    entry["indexed_chunks"] = None
            else:
                entry["indexed_chunks"] = 0
            entry["indexed"] = bool(entry["indexed_chunks"])
            corpora.append(entry)

        return {
            "corpora": corpora,
            "active": active,
            "retrieval": {
                "mode": config.retrieval.mode,
                "top_k": config.retrieval.top_k,
                "rerank_top_k": config.retrieval.rerank_top_k,
                "reranker": config.reranker.provider,
                "min_score": config.retrieval.min_score,
            },
        }


def _result_entry(rank: int, chunk: Any, max_chars: int) -> dict[str, Any]:
    text = chunk.text
    truncated = len(text) > max_chars
    entry: dict[str, Any] = {
        "rank": rank,
        "score": round(chunk.score, 4),
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        # `source` is a Path; JSON has no such type.
        "source": _relative_to_repo(chunk.source),
        "text": text[:max_chars],
    }
    if truncated:
        entry["truncated"] = True
        entry["full_length"] = len(text)
    page = chunk.metadata.get("page")
    if page is not None:
        entry["page"] = page
    # Where the passage sits in its document's cleaned text: the offsets
    # rag_read_document takes, to read around a hit.
    for key in ("char_start", "char_end"):
        if key in chunk.metadata:
            entry[key] = int(chunk.metadata[key])
    # Present only when contextual chunking generated one at index time.
    if chunk.context:
        entry["context"] = chunk.context
    # Present only when `chunking.header` rendered one: the company and period
    # an agent needs to tell near-identical filings apart.
    if chunk.header:
        entry["header"] = chunk.header
    return entry


def _fingerprint(directories: Sequence[Path]) -> tuple[Any, ...]:
    """Every file under `directories` with its size and mtime: changes when the corpus does."""

    entries: list[Any] = []
    for directory in directories:
        if not directory.is_dir():
            entries.append((str(directory), None))
            continue
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                stat = path.stat()
                entries.append((str(path), stat.st_size, stat.st_mtime_ns))
    return tuple(entries)


def _document_metadata(document: Document, carried_keys: Sequence[str]) -> dict[str, Any]:
    """A document's carried metadata, dates as ISO strings -- what a listing shows."""

    return {
        key: value.isoformat() if isinstance(value, date) else value
        for key, value in document.metadata.items()
        if key in carried_keys
    }


def _unknown_document(document_id: str, documents: dict[str, Document], selection: CorpusSelection) -> str:
    """The error for a `document_id` the selection doesn't hold, naming near misses."""

    close = difflib.get_close_matches(document_id, list(documents), n=3, cutoff=0.6)
    suggestion = f" Did you mean: {', '.join(close)}?" if close else ""
    return (
        f"No document {document_id!r} in {selection.describe()}.{suggestion} "
        "rag_list_documents lists the valid ids."
    )


def _find_entry(document: Document, version: str, start: int, end: int) -> dict[str, Any]:
    """One `rag_find` match: its offsets and the text around it, on one line."""

    before = max(start - FIND_CONTEXT_CHARS, 0)
    after = min(end + FIND_CONTEXT_CHARS, len(document.text))
    context = " ".join(document.text[before:after].split())
    return {
        "document_id": document.id,
        "version": version,
        "start": start,
        "end": end,
        "match": document.text[start:end],
        "context": ("..." if before else "") + context + ("..." if after < len(document.text) else ""),
    }


_SEARCH_DESCRIPTION = """\
Search the indexed document corpus and return the most relevant passages, \
ranked best first. Runs the full retrieval pipeline (hybrid dense + BM25 \
search, then cross-encoder reranking) and returns raw passages with their \
provenance -- it does not generate an answer.

Use this to ground an answer in the user's own documents. Each result names \
the document it came from, so you can cite exactly where a claim came from."""

_SEARCH_READ_NOTE = DescriptionNote(
    "Each result also gives its `char_start`/`char_end` in that document: pass them to "
    "rag_read_document to read what surrounds a passage instead of searching again.",
    tools=("rag_read_document",),
)

_LIST_CORPORA_DESCRIPTION = """\
List the document corpora available to search, with a description of each, \
whether its index has been built, and how many chunks it holds. Also reports \
the active retrieval settings. Call this before rag_search when you are unsure \
what `corpus` to pass, or to check whether a corpus is indexed at all."""


_LIST_DOCUMENTS_DESCRIPTION = """\
List the documents in the corpus with their metadata (on EDGAR filings: \
company, ticker, form, period_end, filed) and length. Nothing is ranked: this \
answers what the corpus contains -- which companies, filings and periods \
exist -- which a search cannot, since search only returns its best matches. \
Use it for questions about a set ("which airlines", "every 10-Q for the June \
2026 quarter"), and to check that something exists before searching for it."""

_LIST_PAGES_NOTE = DescriptionNote(
    "Long listings come in pages: pass the result's `next_offset` as `offset`.",
    arguments=("offset",),
)

_READ_DOCUMENT_DESCRIPTION = """\
Read a window of one document's text, starting at a character offset. Use it \
to expand evidence you already have -- the rest of a table, the paragraphs \
around a search hit (start a little before its `char_start`), or a filing \
read front to back -- without searching again. The result gives `start`, \
`end` and the document's `length`; while more remains it gives `next_start`, \
which continues where this window stopped."""

_FIND_DESCRIPTION = """\
Find every occurrence of an exact phrase -- a name, a figure, a defined term \
-- in the documents, case-insensitively. Unlike rag_search nothing is ranked, \
so a match can't be crowded out by passages that merely share words with it; \
a phrase that appears nowhere returns no matches, which a search can't show. \
Each match gives its document, offsets and the text around it."""

_FIND_READ_NOTE = DescriptionNote(
    "Pass a match's `start` to rag_read_document to read more around it.",
    tools=("rag_read_document",),
)


def build_tool_specs(tools: RagTools) -> list[ToolSpec]:
    """Bind `tools` into the list of tools the MCP transports and the agent serve.

    The handlers are defined here as closures with fully annotated signatures
    because those annotations *are* the published schema -- including the
    `Field(description=...)` text, which is what an agent reads when deciding
    whether a tool applies.
    """

    def rag_search(
        query: Annotated[
            str,
            Field(description="Natural-language question or search phrase to retrieve passages for."),
        ],
        corpus: Annotated[
            str | list[str] | None,
            Field(
                description=(
                    "Corpus name to search, or several to search them as one pooled index. "
                    "Omit to use the configured active corpora. Call rag_list_corpora for valid names."
                )
            ),
        ] = None,
        top_k: Annotated[
            int | None,
            Field(
                ge=1,
                le=MAX_RESULTS,
                description=(
                    f"How many passages to return (1-{MAX_RESULTS}). "
                    "Omit to use the configured retrieval.rerank_top_k."
                ),
            ),
        ] = None,
        max_chars: Annotated[
            int | None,
            Field(
                ge=1,
                description=(
                    f"Truncate each passage to this many characters (default {DEFAULT_MAX_CHARS}). "
                    "Raise it when you need a full passage verbatim; truncated results are "
                    "flagged with `truncated` and `full_length`."
                ),
            ),
        ] = None,
        filters: Annotated[
            QueryFilter | None,
            Field(
                description=(
                    "Restrict the search to passages whose document metadata matches, e.g. "
                    '{"equals": {"ticker": "AAPL"}, "range": {"period_end": {"gte": "2025-01-01", '
                    '"lte": "2025-12-31"}}}. `equals`/`any_of` take strings; `range` takes integers '
                    "or ISO dates (dates are compared as YYYYMMDD). Filterable fields are "
                    "`document_id` plus the corpus's carried metadata (on EDGAR filings: company, "
                    "ticker, form, period_end, filed, accession). Use it when the question names a "
                    "company or period: identical passages from other periods then can't outrank "
                    "the right one. Omit to search everything."
                )
            ),
        ] = None,
    ) -> dict[str, Any]:
        # The `mcp` SDK validates arguments into a `QueryFilter`; the fallback
        # transport passes the raw JSON object. A bad one raises
        # `ValidationError` (a `ValueError`), reported as a tool error.
        if filters is not None and not isinstance(filters, QueryFilter):
            filters = QueryFilter.model_validate(filters)
        return tools.search(query, corpus=corpus, top_k=top_k, max_chars=max_chars, filters=filters)

    def rag_list_documents(
        corpus: Annotated[
            str | list[str] | None,
            Field(
                description=(
                    "Corpus name to list, or several. Omit to use the configured active corpora. "
                    "Call rag_list_corpora for valid names."
                )
            ),
        ] = None,
        filters: Annotated[
            QueryFilter | None,
            Field(
                description=(
                    "List only the documents whose metadata matches, in the same form as "
                    'rag_search\'s filters, e.g. {"any_of": {"form": ["10-Q"]}, "range": '
                    '{"period_end": {"gte": "2026-04-01", "lte": "2026-06-30"}}}. Omit to list everything.'
                )
            ),
        ] = None,
        limit: Annotated[
            int | None,
            Field(
                ge=1,
                le=MAX_LIST_LIMIT,
                description=f"Most documents to return (default {DEFAULT_LIST_LIMIT}, at most {MAX_LIST_LIMIT}).",
            ),
        ] = None,
        offset: Annotated[
            int,
            Field(
                ge=0,
                description=(
                    "How many matching documents to skip, for the next page of a long listing: "
                    "pass the previous result's `next_offset`. Default 0."
                ),
            ),
        ] = 0,
    ) -> dict[str, Any]:
        if filters is not None and not isinstance(filters, QueryFilter):
            filters = QueryFilter.model_validate(filters)
        return tools.list_documents(corpus=corpus, filters=filters, limit=limit, offset=offset)

    def rag_read_document(
        document_id: Annotated[
            str,
            Field(description="The document to read, by the id search results, find matches and listings give it."),
        ],
        start: Annotated[
            int,
            Field(
                ge=0,
                description=(
                    "Character offset to start reading at (default 0, the beginning). Use a search "
                    "result's `char_start` (minus some margin), a find match's `start`, or the "
                    "previous window's `next_start`."
                ),
            ),
        ] = 0,
        max_chars: Annotated[
            int | None,
            Field(
                ge=1,
                le=MAX_READ_CHARS,
                description=f"Window size in characters (default {DEFAULT_READ_CHARS}, at most {MAX_READ_CHARS}).",
            ),
        ] = None,
        corpus: Annotated[
            str | list[str] | None,
            Field(
                description=(
                    "Corpus the document is in, or several. Omit to use the configured active corpora."
                )
            ),
        ] = None,
    ) -> dict[str, Any]:
        return tools.read_document(document_id, corpus=corpus, start=start, max_chars=max_chars)

    def rag_find(
        phrase: Annotated[
            str,
            Field(
                description=(
                    "The exact words to find, e.g. a product name or a figure as printed (\"84.1%\"). "
                    "Case and line breaks are ignored; nothing else is (no stemming or synonyms)."
                )
            ),
        ],
        document_id: Annotated[
            str | None,
            Field(description="Look only in this document. Omit to look in every selected document."),
        ] = None,
        filters: Annotated[
            QueryFilter | None,
            Field(
                description=(
                    "Look only in documents whose metadata matches, in the same form as "
                    'rag_search\'s filters, e.g. {"equals": {"ticker": "DAL"}}. Omit to look everywhere.'
                )
            ),
        ] = None,
        max_results: Annotated[
            int | None,
            Field(
                ge=1,
                le=MAX_FIND_RESULTS,
                description=(
                    f"Most matches to return (default {DEFAULT_FIND_RESULTS}, at most {MAX_FIND_RESULTS}); "
                    "`total_matches` counts them all."
                ),
            ),
        ] = None,
        corpus: Annotated[
            str | list[str] | None,
            Field(description="Corpus name to look in, or several. Omit to use the configured active corpora."),
        ] = None,
    ) -> dict[str, Any]:
        if filters is not None and not isinstance(filters, QueryFilter):
            filters = QueryFilter.model_validate(filters)
        return tools.find(
            phrase, corpus=corpus, document_id=document_id, filters=filters, max_results=max_results
        )

    def rag_list_corpora() -> dict[str, Any]:
        return tools.list_corpora()

    return [
        ToolSpec("rag_search", _SEARCH_DESCRIPTION, rag_search, notes=(_SEARCH_READ_NOTE,)),
        ToolSpec("rag_list_documents", _LIST_DOCUMENTS_DESCRIPTION, rag_list_documents, notes=(_LIST_PAGES_NOTE,)),
        ToolSpec("rag_read_document", _READ_DOCUMENT_DESCRIPTION, rag_read_document),
        ToolSpec("rag_find", _FIND_DESCRIPTION, rag_find, notes=(_FIND_READ_NOTE,)),
        ToolSpec("rag_list_corpora", _LIST_CORPORA_DESCRIPTION, rag_list_corpora),
    ]
