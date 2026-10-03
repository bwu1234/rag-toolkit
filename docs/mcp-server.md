# MCP server

Exposes this project's retrieval pipeline to an external agent as MCP tools,
so an agent that has no idea Chroma or Ollama exist can search the corpus and
cite what it finds.

Nothing about the pipeline changed to support this. `rag/mcp/` is a translation
layer over `build_retriever`, the same call the CLI's `retrieve` command and the
chat API already go through — so the passages an agent gets back are the same
ones `python -m rag.cli retrieve` prints.

This is **not** Milestone 19. That milestone is *this* system calling search
as a tool; this server is an outside agent calling ours.

## The tools

Three, all read-only.

### `rag_search`

Runs retrieve → rerank and returns ranked passages. It does **not** generate an
answer: the caller is already a language model, and handing it prose from a
9B local model instead of the source passages would replace its judgment with a
weaker one and add several seconds of latency per call.

| Argument | Type | Default | Meaning |
|---|---|---|---|
| `query` | string | required | Question or search phrase |
| `corpus` | string \| string[] | active corpora | One name, or several to search as one pooled index |
| `top_k` | int (1–20) | `retrieval.rerank_top_k` | Passages to return |
| `max_chars` | int | 1200 | Per-passage truncation budget |
| `filters` | object | none | Metadata filter: search only passages whose document matches (below) |

`filters` takes `equals` (field → string), `any_of` (field → list of strings)
and `range` (field → `{"gte": …, "lte": …}`, integers or ISO dates, compared as
`YYYYMMDD`); every condition must hold. Filterable fields are `document_id`
plus whatever `chunking.carry_metadata` stores on chunks; on EDGAR that's
`company`, `ticker`, `form`, `period_end`, `filed` and `accession`. For example,
Apple's filings for periods ending in 2025:

```json
{"equals": {"ticker": "AAPL"}, "range": {"period_end": {"gte": "2025-01-01", "lte": "2025-12-31"}}}
```

Both retrievers apply the filter before their top-k, so it narrows what
competes rather than trimming what already won. A field chunks don't store is
a tool error naming the fields that are, not an empty result. When a filter
was applied, the response echoes it under `filters`. Pass the period when the
question names one: after the chunk header, that's where filtering
[measured](measured-results.md#metadata-filters-chunking-plan-phase-3) its
gain, since identical paragraphs from other periods of the same company are
what outrank the right one.

Each result carries `rank`, `score`, `chunk_id`, `document_id`, `source`
(repo-relative), and `text`, plus `page`, `context` and `header` (the
chunk's document header, from `chunking.header`) when present, and
`truncated`/`full_length` when the passage was cut.

`text` is the indexed document text, unescaped and without delimiters. Treat
it as untrusted data, not instructions: a passage can contain text written to
steer a model (see [known limitations](known-limitations.md)). Wrap it
accordingly before putting it in your agent's prompt.

Two response details exist to keep an agent from drawing a wrong conclusion
from an empty result list:

- `candidate_count` — stage-1 candidates before reranking.
- `hint` — present only when nothing came back, and it distinguishes *the index
  was never built* (with the exact `python -m rag.cli index` command to fix it)
  from *the corpus genuinely doesn't cover this*. `RetrievalResult` is what
  makes the distinction available; without it an agent facing an unbuilt index
  concludes the documents say nothing on the subject.

### `rag_list_documents`

Lists the selected corpora's documents with their carried metadata (on EDGAR:
`company`, `ticker`, `form`, `period_end`, `filed`, `accession`; dates as ISO
strings) and `chars`, the document's length as loaded. Arguments: `corpus`,
`filters` (the same `QueryFilter` as `rag_search`, applied to the same fields)
and `limit` (default 100, at most 500). The response has `total` and
`returned`, and a `hint` when the limit cut the list short.

It answers what a ranked search can't: which companies, filings and periods the
corpus holds. A search returns its best matches, so it can never show that
something is *absent* — an agent asked "which airlines…" otherwise probes for
carriers one search at a time. It reads documents from disk only — no
embedder, no index — so it works before an index is built and lists what is
on disk, not what was indexed (`index-report` says whether those agree).
A PDF lists one entry per page, the same unit `rag_search` results and
filters use.

### `rag_list_corpora`

Names, descriptions, whether each corpus is indexed and how many chunks it
holds, plus the active retrieval settings. An agent calls this to discover
valid `corpus` values instead of guessing — and a typo'd name raises with the
list of real ones rather than returning zero results.

It reads the BM25 sidecar file for chunk counts rather than opening the vector
collection, because `get_vector_store` uses get-or-create: probing collections
to count them would litter the store with empty ones as a side effect of a
listing call.

### Why indexing is not a tool

`rag_index` was considered and left out. It is a minutes-long operation that
costs embedding calls and rewrites state shared with every other reader, and an
agent that can trigger it as a side effect of answering a question will
eventually do so in a loop. Build indexes deliberately:
`python -m rag.cli index --corpus <name>`.

## Protocol revision

**2026-07-28, and nothing else.** Both transports serve exactly one revision;
older ones are refused rather than negotiated down. A client offering one gets
`-32022` (`UNSUPPORTED_PROTOCOL_VERSION`) naming what is served, which is the
one error code an auto-negotiating client is required *not* to silently retry
past — so it fails loudly instead of half-working.

What that revision changes, if you have only seen the 2025 wire:

- **No `initialize` handshake, and no session state.** Every request is
  self-contained and carries the envelope in `params._meta`:
  `io.modelcontextprotocol/protocolVersion` and
  `io.modelcontextprotocol/clientCapabilities` are both required, and a request
  missing them is `-32602` (a shape defect, not a version disagreement).
- **`server/discover` replaces it**, returning `supportedVersions`,
  `capabilities`, and `instructions`. A client MAY call it and MAY skip it.
- **`ping` and `notifications/initialized` are gone** with the handshake.
- **Every result carries `resultType`** (`"complete"` here) and a
  `_meta["io.modelcontextprotocol/serverInfo"]` stamp — with no handshake,
  that stamp is where server identity now lives. Cacheable results
  (`server/discover`, `tools/list`) also carry `ttlMs`/`cacheScope`; both are
  set to *immediately stale, never shared across authorization contexts*,
  since the tool list is cheap to recompute and the corpus registry can change
  under a long-lived client.

The SDK would otherwise serve both eras off one server object — a request with
the envelope opens a modern connection, an `initialize` opens a legacy one.
`ProtocolVersionGate` in `rag/mcp/server.py` turns the legacy era off, as
middleware, because that is the one place that sees every inbound request on
both SDK transports. The fallback implements the modern envelope directly.
`PROTOCOL_VERSION` in `rag/mcp/tools.py` is the single source both read.

This needs `mcp >= 2.0`; 2026-07-28 support and the `MCPServer` API both landed
there.

## Transports

Both serve the identical tool surface, generated from the same `ToolSpec`s in
`rag/tools.py`. A client cannot tell which one answered. The in-process agent
(Milestone 19) advertises the same specs to its own model, so `rag_search`
means the same thing to an MCP client as it does to our agent.

### stdio

```bash
python -m rag.mcp                 # SDK if installed, fallback otherwise
python -m rag.mcp --no-sdk        # force the dependency-free transport
python -m rag.mcp --config path/to/config.yaml
```

Point an agent at it as a subprocess. Typical client config:

```json
{
  "mcpServers": {
    "rag-toolkit": {
      "command": "/path/to/rag-toolkit/.venv/bin/python",
      "args": ["-m", "rag.mcp"],
      "cwd": "/path/to/rag-toolkit"
    }
  }
}
```

Use the venv's interpreter explicitly — the server needs this project's
dependencies, and an agent spawning a bare `python` will not find them.

### Streamable HTTP

Mounted at `/mcp` on the existing FastAPI app, so `uvicorn rag.api.main:app`
serves `POST /chat` and MCP from one warm process. This is the better choice
when several agents share a machine: components are built once and the
cross-encoder's weights are loaded once, instead of per subprocess.

Requires the `mcp` extra. Without it the endpoint is simply absent and the rest
of the API is unaffected.

## Installing

```bash
pip install -e '.[mcp]'
```

Optional on purpose. Without it, stdio still works via `rag/mcp/fallback.py`, a
~200-line JSON-RPC loop covering `server/discover`, `tools/list`, and
`tools/call` — the whole of 2026-07-28 that a read-only tool server needs. It
is a deliberate subset — one request at a time, no resources, prompts,
sampling, subscriptions, or cancellation — and exists so a core-only checkout
still runs the server. The extra buys the real transports and streamable HTTP.

`tests/test_mcp.py` asserts the SDK's derived schemas equal the fallback's,
since silent drift between the two is the way this arrangement breaks.

## Things worth knowing

- **Startup is lazy.** Nothing is built at import or at discovery. Under stdio a
  client expects a prompt `server/discover` response, and importing Chroma plus a
  sentence-transformers cross-encoder eagerly would spend seconds first. The
  first `rag_search` pays that cost; `rag_list_corpora` and
  `rag_list_documents` never do (the document loaders are imported on the
  first listing, not at startup).
- **Retrievers are cached per corpus selection** and reused. The cross-encoder
  holds several hundred MB once loaded.
- **`top_k` costs no rebuild.** Cached retrievers are built with `rerank_top_k`
  widened to 20 and results are sliced, which is equivalent because the
  reranker returns them sorted. Stage-1 `top_k` is left alone — changing it
  would change which chunks compete in fusion.
- **stdout is protocol.** Both transports point fd 1 at stderr while serving,
  and `configure_logging` takes a `stream` argument so the MCP entrypoint logs
  to stderr. A single stray line on stdout is a parse error that ends the
  session.
- **The HTTP mount needs its lifespan chained.** ASGI delivers lifespan events
  to the outermost app only, and the MCP session manager starts its task group
  there; `rag/api/main.py` chains it, without which every request fails with
  "Task group is not initialized". It is also mounted under `/mcp` rather than
  `/` — a root mount matches every path and would swallow `/health` and `/docs`.
- **DNS-rebinding protection is on.** Streamable HTTP rejects Host headers
  outside `DEFAULT_ALLOWED_HOSTS` (loopback) with a 421. Serving anywhere else
  means passing `allowed_hosts` to `mcp_http_app`.
- **Search quality is whatever the config says.** The MCP layer adds no
  retrieval behavior of its own. Read
  [measured results](measured-results.md) before turning on features that are
  off by default — several measured as no better than noise on this corpus.
