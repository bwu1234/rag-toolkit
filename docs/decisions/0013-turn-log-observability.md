# 0013 — One append-only turn log, metered at the client, written only by human-facing entrypoints

- **Status:** Accepted
- **Recorded:** 2026-10-03, retroactively, from
  [observability notes](../milestone-notes.md#observability-notes-milestone-12)

## Context

A turn can call the LLM from up to six places that share one client. Latency
per stage was already in the pipeline events. Real questions and thumbs-down
feedback are the raw material for future eval samples.

## Decision

- **Latency** is summed from the existing `PipelineEvent`s, not from a second
  set of timers that could disagree with the UI trace.
- **LLM calls and tokens** are counted by a `MeteredLLMClient` wrapper, into a
  `ContextVar` meter, because sync routes run in a thread pool against one
  shared service. A missing token count is recorded as `None`, never 0.
- **One JSONL record per turn, failures included.** Feedback is appended as
  its own line and joined to its turn at read time. Each record carries a
  config fingerprint.
- **Only the API, the UI and `cli chat` log.** The eval runners don't, so
  the log can't fill up with the eval set's own questions.
- **Logging never costs the user their answer.** A failed write is swallowed
  with a warning. `POST /feedback` is the exception, since storing the rating
  is its whole purpose.

## Alternatives considered

- **Counting at each call site.** Five component APIs would have changed.
- **Writing feedback into the turn's record.** Feedback arrives later, from a
  different request, and might race with the API and UI sharing one file.
- **OpenTelemetry first.** Planned as a later `TurnSink` adapter. It can't be
  read back, which is why reading is a property of the JSONL store and not
  of the interface.

## Consequences

- `(query, shown, cited)` is logged on every turn as an implicit, model-biased
  relevance signal: useful in aggregate for learning-to-rank, not per turn.
- The turn id on feedback isn't validated (that would mean scanning the file
  on every click). An unknown id simply never joins.
