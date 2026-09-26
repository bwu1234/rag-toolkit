"""Milestone 12: per-turn records, LLM usage metering, and user feedback.

Three pieces, each usable alone:

- `usage` -- `MeteredLLMClient`, which counts every LLM call made during a
  chat turn (and its tokens) without the components making those calls knowing.
- `records` -- the `TurnRecord` / `FeedbackRecord` shapes that get persisted.
- `sink` -- the `TurnSink` interface those records are written through, with a
  local JSONL implementation as the default.
"""
