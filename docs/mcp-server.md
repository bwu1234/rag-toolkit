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

Five, all read-only: `rag_search` ranks passages; `rag_list_documents`,
`rag_read_document` and `rag_find` navigate documents without ranking; and
`rag_list_corpora` describes what can be searched.

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
`truncated`/`full_length` when the passage was cut. `char_start`/`char_end`
place the passage in its document's cleaned text: pass them to
`rag_read_document` to read around a hit instead of searching again.

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
`limit` (default 100, at most 500) and `offset` (default 0). The response has
`total`, `offset` and `returned`; while documents remain past this page it
also has `next_offset` (pass it as `offset`) and a `hint`. Pages are in a
stable order (corpus, then path) and paginate the *filtered* list. A corpus
edited between pages can shift them; nothing pins a snapshot.

It answers what a ranked search can't: which companies, filings and periods the
corpus holds. A search returns its best matches, so it can never show that
something is *absent* — an agent asked "which airlines…" otherwise probes for
carriers one search at a time. It reads documents from disk only — no
embedder, no index — so it works before an index is built and lists what is
on disk, not what was indexed (`index-report` says whether those agree).
A PDF lists one entry per page, the same unit `rag_search` results and
filters use.

### `rag_read_document`

Returns a window of one document's text: `document_id` (required), `start`
(character offset, default 0), `max_chars` (default 6000, at most 50,000) and
`corpus`. The response has `start`, `end`, `length`, the window's `text`, the
document's carried metadata, and `next_start` while text remains, so
consecutive calls rebuild the document exactly while its cleaned text remains
unchanged.

The text is the *cleaned* text the chunker split, so a search hit's
`char_start`/`char_end` index straight into it: start a few hundred
characters before `char_start` to see the paragraph or table around a
passage. A chunk's `text` is that span stripped of surrounding whitespace, or,
for the second and later pieces of a split table, the span with the table's
header rows in front. Offsets come from the index and the text from disk, so
they agree only while the index is in sync (`python -m rag.cli index-report`).
An unknown `document_id` is a tool error naming close matches.

There is currently no source-version argument or pinned snapshot spanning
search/read/find calls. Checking `index-report` before a task is useful but
cannot prevent a file changing during it. The planned
[source-version contract](milestone-19-plan.md#source-version-consistency)
will expose source identity and require matching text or an explicit mismatch
when following a hit. Those fields and guarantees are not yet part of this
MCP API. Stable document IDs alone do not prove source-version equality.

### `rag_find`

A literal phrase search: `phrase` (required), `document_id` or `filters` to
narrow it, `max_results` (default 20, at most 100) and `corpus`. Matching is
case-insensitive and any whitespace in the phrase matches any run of
whitespace, so a figure split across a line or table cell still matches;
nothing else is normalized (no stemming, no synonyms), and regex characters
are literal. Matches come in document order, then position, each with
`document_id`, `start`/`end` (offsets for `rag_read_document`), the exact
`match` and about 150 characters of `context` either side. `total_matches`
and `documents_matched` count everything even when `max_results` cut the list.

Use it for an exact name or figure that ranked search can crowd out: hybrid
ranking scores term overlap and the reranker reorders, so "Enflonsia" or a
dollar amount can fall below the top five. It can also show that a phrase
appears *nowhere*, which a search can't. It scans the text linearly: about
20 ms for `edgar_md`'s 3.5M characters once loaded.

Both tools load and clean the selected corpora on first use (about 0.4 s for
`edgar_md`) and cache them per corpus selection, reloading when any file's
size or modification time changes. Like `rag_list_documents`, they read from
disk and need no index or embedder.

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

Served at `http://127.0.0.1:8000/mcp` (no trailing slash) on the existing
FastAPI app, so `uvicorn rag.api.main:app` serves `POST /chat` and MCP from
one warm process. `/mcp/` redirects to `/mcp`; point clients at the URL
without the slash, since a client that won't re-send a POST across a redirect
never reaches the server. On the 2026-07-28 revision the SDK routes HTTP
requests by header, so each POST carries `MCP-Protocol-Version: 2026-07-28`
and an `Mcp-Method` header matching the body's method (a conforming client
sends both). Without them the SDK treats the POST as the 2025 session era and
answers `400 Bad Request: Missing session ID` -- a misleading message for a
missing header, and the first thing to check when a hand-rolled client fails. This is the better choice
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
  first `rag_search` pays that cost; `rag_list_corpora` and the navigation
  tools never do (the document loaders are imported on first use, not at
  startup).
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
- **The HTTP endpoint needs its lifespan chained.** ASGI delivers lifespan
  events to the outermost app only, and the MCP session manager starts its
  task group there; `rag/api/main.py` chains it, without which every request
  fails with "Task group is not initialized".
- **It is a route, not a mount.** `rag/api/main.py` adds the SDK app's one
  route to the API's router. Mounting it at `/mcp` served it at `/mcp/` only
  (`POST /mcp` drew a 307), and mounting it at `/` would match every path and
  swallow `/health` and `/docs`. Copying the route drops the SDK app's
  middleware, so startup refuses if it ever has any (the SDK adds auth
  middleware when auth is configured). `tests/test_mcp.py` drives
  `POST /mcp` over HTTP with the lifespan running.
- **DNS-rebinding protection is on.** Streamable HTTP rejects Host headers
  outside `DEFAULT_ALLOWED_HOSTS` (loopback) with a 421. Serving anywhere else
  means passing `allowed_hosts` to `mcp_http_app`.
- **Search quality is whatever the config says.** The MCP layer adds no
  retrieval behavior of its own. Read
  [measured results](measured-results.md) before turning on features that are
  off by default — several measured as no better than noise on this corpus.
