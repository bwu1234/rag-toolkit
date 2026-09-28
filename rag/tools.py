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
import inspect
import logging
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, get_type_hints

from pydantic import Field, create_model

from rag.config.settings import REPO_ROOT, CorpusSelection, RagConfig, load_config
from rag.events import EventSink
from rag.query_filter import QueryFilter
from rag.generation.llm import LLMClient, ToolDefinition
from rag.retrieval.builder import build_retriever
from rag.retrieval.retriever import RetrievalResult, Retriever
from rag.retrieval.sparse import BM25Index, bm25_index_path

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


@dataclass(frozen=True)
class ToolSpec:
    """One tool: its wire identity plus the callable behind it."""

    name: str
    description: str
    handler: Callable[..., dict[str, Any]]

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

    def definition_without(self, *pinned: str) -> ToolDefinition:
        """`definition`, minus arguments the caller fixes itself.

        A projection of the same schema, not a second one: every remaining
        property is byte-identical to what MCP serves. The agent uses it to
        keep `corpus`, `top_k` and `max_chars` out of the model's hands --
        they belong to the turn (the eval's `--corpus`, the prompt-size
        guards), not to the model. Only optional arguments can be pinned; a
        required one has no value to fall back on.
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
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={**schema, "properties": properties},
        )


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

            # Read the BM25 sidecar rather than opening the vector collection:
            # `get_vector_store` uses get_or_create, so probing it here would
            # litter the store with empty collections just from listing. The
            # indexer always writes both, so the sidecar is a faithful signal.
            sidecar = bm25_index_path(selection.index_dir, selection.slug)
            if sidecar.exists():
                try:
                    entry["indexed_chunks"] = BM25Index(sidecar).count()
                except Exception:  # pragma: no cover - corrupt sidecar
                    logger.warning("Could not read BM25 index at %s", sidecar, exc_info=True)
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
    # Present only when contextual chunking generated one at index time.
    if chunk.context:
        entry["context"] = chunk.context
    # Present only when `chunking.header` rendered one: the company and period
    # an agent needs to tell near-identical filings apart.
    if chunk.header:
        entry["header"] = chunk.header
    return entry


_SEARCH_DESCRIPTION = """\
Search the indexed document corpus and return the most relevant passages, \
ranked best first. Runs the full retrieval pipeline (hybrid dense + BM25 \
search, then cross-encoder reranking) and returns raw passages with their \
provenance -- it does not generate an answer.

Use this to ground an answer in the user's own documents. Each result names \
the document it came from, so you can cite exactly where a claim came from."""

_LIST_CORPORA_DESCRIPTION = """\
List the document corpora available to search, with a description of each, \
whether its index has been built, and how many chunks it holds. Also reports \
the active retrieval settings. Call this before rag_search when you are unsure \
what `corpus` to pass, or to check whether a corpus is indexed at all."""


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

    def rag_list_corpora() -> dict[str, Any]:
        return tools.list_corpora()

    return [
        ToolSpec(name="rag_search", description=_SEARCH_DESCRIPTION, handler=rag_search),
        ToolSpec(
            name="rag_list_corpora",
            description=_LIST_CORPORA_DESCRIPTION,
            handler=rag_list_corpora,
        ),
    ]
