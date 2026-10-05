"""A turn's deadline, visible to every network call the turn makes without passing it down.

The agent owns a turn's time budget, but the calls that spend it sit several
layers away: the agent's model, the embedder inside a search, the
groundedness checker's model. Threading a `timeout` argument through
`LLMClient.generate`, `EmbeddingModel.embed_query` and everything between
would change every interface and every test fake for the sake of one caller.
Instead the agent opens a `deadline_scope`, and each HTTP adapter asks
`request_timeout()` for the time left when it sends a request -- the same
`ContextVar` pattern `rag.observability.usage` uses for its per-turn meter,
and for the same reason: the API answers concurrent turns on one set of
clients, and a context variable is per thread.

What the bound buys, per backend (`docs/milestone-19-plan.md`, "Execution and
output contracts"):

- **Ollama** (model and embedder): the request's timeout is the time left, so
  httpx closes the connection when it runs out. Ollama cancels a request's
  context when its client disconnects, and both its runners stop generating
  at the next token when that happens (`mlxrunner/pipeline.go`,
  `llm/llama_server.go`), so the GPU work ends too, not only the wait.
- **Gemini**: the same client-side bound, plus pacing and retry waits that
  give up instead of sleeping past the deadline. Whether Google stops
  generating when the connection closes isn't documented; assume the call
  may run to completion server-side and count against quota, which the
  daily counter has already charged.
- **In-process work** (the cross-encoder reranker, Chroma, the sparse index)
  can't be interrupted. The agent bounds it only by not starting a search
  once the time for one is gone (see `AgentService`).

httpx applies a timeout to each phase of a request (connect, write, each
read) rather than to the whole. Both providers answer a non-streaming request
only once it's complete, so the read phase is effectively the whole
generation, and the bound overshoots by at most a connect and a write.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass


class DeadlineExceeded(TimeoutError):
    """A call was refused or cut off because its turn's deadline passed.

    Distinct from an upstream failure (a provider that's down raises
    `RuntimeError`), so the agent can end the turn on the budget that ran out
    rather than report the provider broken. `label` names the scope that
    expired.
    """

    def __init__(self, message: str, *, label: str) -> None:
        super().__init__(message)
        self.label = label


@dataclass(frozen=True)
class _Scope:
    at: float
    label: str
    clock: Callable[[], float]

    def remaining(self) -> float:
        return self.at - self.clock()


_ACTIVE: ContextVar[_Scope | None] = ContextVar("rag_active_deadline", default=None)


@contextmanager
def deadline_scope(
    at: float | None, *, label: str, clock: Callable[[], float] = time.monotonic
) -> Iterator[None]:
    """Bound every adapter call in the block by `at` (on `clock`); `None` adds no bound.

    Nested scopes keep the earlier deadline: an inner scope can tighten the
    bound, never extend it.
    """

    outer = _ACTIVE.get()
    if at is None or (outer is not None and outer.remaining() <= at - clock()):
        yield
        return
    token = _ACTIVE.set(_Scope(at=at, label=label, clock=clock))
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def remaining_s() -> float | None:
    """Seconds left in the active scope (negative once past), or None outside one."""

    scope = _ACTIVE.get()
    return None if scope is None else scope.remaining()


def check(what: str) -> None:
    """Raise `DeadlineExceeded` if the active scope has run out, before starting `what`."""

    scope = _ACTIVE.get()
    if scope is not None and scope.remaining() <= 0:
        raise DeadlineExceeded(f"{scope.label} deadline passed before {what}", label=scope.label)


def check_wait(seconds: float, what: str) -> None:
    """Raise `DeadlineExceeded` rather than wait `seconds` the active scope doesn't have."""

    scope = _ACTIVE.get()
    if scope is not None and seconds >= (remaining := scope.remaining()):
        raise DeadlineExceeded(
            f"{what} needs {seconds:.1f}s; the {scope.label} deadline leaves {max(remaining, 0.0):.1f}s",
            label=scope.label,
        )


def request_timeout(default: float, what: str) -> float:
    """The timeout for one request: `default`, cut to the time left; raises when none is left."""

    scope = _ACTIVE.get()
    if scope is None:
        return default
    remaining = scope.remaining()
    if remaining <= 0:
        raise DeadlineExceeded(f"{scope.label} deadline passed before {what}", label=scope.label)
    return min(default, remaining)


def raise_if_cut_short(what: str, timeout: float, default: float, cause: BaseException) -> None:
    """After a request timed out: raise `DeadlineExceeded` if the deadline set its timeout.

    Decided by the timeout the request was sent with, not by re-reading the
    clock: a request given the deadline's remainder timed out *because* of the
    deadline. One that timed out on the adapter's own `default` is an
    upstream failure, which the caller reports as such; this returns and lets
    it.
    """

    scope = _ACTIVE.get()
    if scope is not None and timeout < default:
        raise DeadlineExceeded(f"{scope.label} deadline reached during {what}", label=scope.label) from cause
