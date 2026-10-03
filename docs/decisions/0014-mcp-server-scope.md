# 0014 — MCP server: read-only tools, one protocol revision

- **Status:** Accepted
- **Recorded:** 2026-10-03, retroactively, from [MCP server](../mcp-server.md)

## Context

External agents should be able to use this repo's retrieval without
reimplementing it. MCP is the standard way to offer that. The 2026-07-28
protocol revision dropped the `initialize` handshake and session state, and
the SDK can serve both the old and new protocols from one server.

## Decision

- **Tools are read-only:** `rag_search`, `rag_list_documents`,
  `rag_list_corpora`. They are served from `rag/tools.py`, which the internal
  agent also uses ([0009](0009-pipeline-default-agentic-opt-in.md)).
- **Indexing is not a tool.** It takes minutes, costs embedding calls, and
  rewrites state every reader shares. An agent that can trigger it while
  answering a question will eventually do so in a loop.
- **Exactly one protocol revision (2026-07-28).** Older revisions are refused
  with `-32022`, which compliant clients must not silently retry past.
  `ProtocolVersionGate` middleware turns the old protocol off on both SDK
  transports.
- **stdio works without the `mcp` extra** (a fallback implementation).
  Streamable HTTP needs it, mounted at `POST /mcp` on the API.

## Alternatives considered

- **Negotiating down to older revisions.** That means two protocols to test.
  A clear refusal fails loudly instead of half-working.
- **A `rag_index` tool.** Rejected for the reasons above.

## Consequences

- MCP clients get exactly what `cli retrieve` returns. No CRAG or
  condensing, whatever the config says.
- Clients still on the 2025 wire format can't connect until they upgrade.
- The MCP server is not Milestone 19. It serves retrieval outward, while
  Milestone 19 is this system calling search itself.
