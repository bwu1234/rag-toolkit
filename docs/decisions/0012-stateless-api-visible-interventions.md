# 0012 — Stateless API; every silent intervention travels with the answer

- **Status:** Accepted
- **Recorded:** 2026-10-03, retroactively, from
  [Chat API notes](../milestone-notes.md#chat-api-notes-milestone-6)

## Context

The same answer pipeline serves a FastAPI route, a Streamlit UI, a CLI and
the eval runners. The pipeline sometimes does things the answer text can't
show: it searches a rewritten question, drops passages below the floor,
grades passages out, retries, or produces an answer that fails its
groundedness check.

## Decision

- **History is supplied by the caller and never stored on the server.**
  `ChatRequest.history` goes in, and a follow-up is condensed into one
  standalone question that drives both retrieval and generation. One
  `ChatService` is built at startup and shared.
- **Per-turn facts are returned, not kept on the instance.**
  `RetrievalResult` and `ChatAnswer` carry counts and diagnostics
  (`rewritten_query`, `dropped_below_min_score`, `graded_out`, `grounded`,
  ...), because mutable instance state would mix concurrent requests.
- **No context means no LLM call.** A blank query, an empty index, everything
  below the floor, and everything graded out each get their own message,
  because each has a different fix.
- **Wire schemas are separate from internal types**, and only routes
  translate between them.

## Alternatives considered

- **Server-side sessions.** Simpler for clients, but the server stops being
  restartable and horizontally scalable.
- **Passing raw history to the answering model.** Retrieval is stateless, so
  "what about part-time staff?" would be searched literally.
- **Diagnostics only in the debug event stream.** Only callers that pass
  `on_event` would see them.

## Consequences

- Clients carry the conversation.
- Every entrypoint can show what the pipeline did. The UI captions only the
  failing groundedness verdict, since captioning the expected outcome trains
  users to skim past the one state that needs attention.
