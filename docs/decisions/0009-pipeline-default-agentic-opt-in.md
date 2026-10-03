# 0009 — Always-retrieve pipeline by default; agentic search as an opt-in mode

- **Status:** Accepted
- **Recorded:** 2026-10-03, retroactively, from
  [Chat API notes](../milestone-notes.md#chat-api-notes-milestone-6),
  [Milestone 19 plan](../milestone-19-plan.md#decisions) and
  [agentic results](../measured-results.md#agentic-retrieval-milestone-19-phase-4)

## Context

For a corpus Q&A tool, every question is supposed to be about the corpus,
so retrieval can be unconditional. Multi-hop questions (comparisons across
companies or periods) need more than one search, and one retrieve-then-
generate pass can't do that.

## Decision

`chat.mode: pipeline` (the default) retrieves on every non-blank query, after
condensing any history into one standalone question. `chat.mode: agentic`
swaps in `AgentService`. The agent model calls `rag_search` as a tool
(`react` or `planned` strategy), with guards (search cap, duplicate-query
refusal, wall-clock budget, forced final answer), and a passage ledger that
owns citation numbering. Both modes return a `ChatAnswer` via the
`ChatResponder` base, so the API, UI, CLI and eval runners don't change.

Supporting choices:

- The agent has its own model (`agent.llm`), so utility calls don't pay
  27b latency.
- It uses the same tool surface as the MCP server, so a search fix lands once.

## Alternatives considered

- **Agentic as the default.** Measured: with the 27b it closes most of the
  multi-hop gap, but at 1–2 minutes per hard question. The 9b agent gains
  nothing.
- **A router that decides whether to retrieve.** Not needed while every
  question is meant to be corpus-shaped. The agent mode is where that choice
  belongs.
- **Widening `LLMClient` for tool calls.** Rejected for a capability subclass
  ([0001](0001-swappable-interfaces.md)).

## Consequences

- The default is fast and predictable. Hard questions need an explicit mode
  switch, and the caller pays the latency.
- CRAG's grader and retries don't apply in agentic mode. Groundedness checking
  only reports its verdict.
- Metadata filters are pinned by the turn, not chosen by the model, unless
  `agent.model_filters` is turned on (which hasn't been measured).
