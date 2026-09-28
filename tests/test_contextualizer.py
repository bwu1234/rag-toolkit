"""Tests for contextual chunking (`rag.chunking.contextualizer`).

Hermetic like every other LLM-backed component's tests: `ChunkContextualizer`
is driven by a fake `LLMClient` that records its prompts and returns canned
replies, so prompt assembly, truncation, and every fail-open path are verified
with no Ollama daemon involved.

The load-bearing property under test is the separation `Chunk` maintains:
generated context enriches what gets *indexed* (`index_text`) and never
touches what gets *cited* (`text`).
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from rag.chunking.context_cache import ContextCache, context_cache_path
from rag.chunking.contextualizer import ChunkContextualizer, build_context_prompt
from rag.chunking.models import Chunk
from rag.ingestion.models import Document


def _document(doc_id: str = "api_reference.md", text: str = "ACS API reference. Rate limits apply.") -> Document:
    return Document(
        id=doc_id,
        text=text,
        source=Path(doc_id),
        doc_type="markdown",
        metadata={"title": "API Reference"},
    )


def _chunk(
    chunk_id: str = "api_reference.md::chunk0",
    *,
    document_id: str = "api_reference.md",
    text: str | None = None,
) -> Chunk:
    return Chunk(
        id=chunk_id,
        # Text defaults to something distinct per chunk id. Identical chunk text
        # would produce identical prompts, which both defeats a scripted fake and
        # makes the context cache legitimately collapse two chunks into one entry.
        text=text if text is not None else f"The limit is 1,000 requests per minute. [{chunk_id}]",
        document_id=document_id,
        source=Path(document_id),
        doc_type="markdown",
        metadata={"chunk_index": 0, "title": "API Reference"},
    )


class _FakeLLMClient:
    """Returns one canned reply for every call, recording prompts.

    Deliberately reply-per-call-order free: generation runs on a thread pool, so
    a fake that indexed replies by call count would hand a given chunk a
    different reply depending on thread scheduling. Tests needing per-chunk
    replies use `_ScriptedLLMClient`, which keys off prompt content instead.
    """

    def __init__(self, reply: str = "Context sentence.") -> None:
        self.reply = reply
        self._lock = threading.Lock()
        self.calls: list[tuple[str, str | None]] = []

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        with self._lock:
            self.calls.append((prompt, system))
        return self.reply


class _ScriptedLLMClient:
    """Replies chosen by a marker found in the prompt -- deterministic under concurrency."""

    def __init__(self, replies: dict[str, str], default: str = "Context sentence.") -> None:
        self.replies = replies
        self.default = default
        self._lock = threading.Lock()
        self.calls: list[tuple[str, str | None]] = []

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        with self._lock:
            self.calls.append((prompt, system))
        for marker, reply in self.replies.items():
            if marker in prompt:
                return reply
        return self.default


class _RaisingLLMClient:
    def generate(self, prompt: str, *, system: str | None = None) -> str:
        raise RuntimeError("daemon unreachable")


# ---------------------------------------------------------------------------
# The core contract: context enriches the index, never the citation
# ---------------------------------------------------------------------------


def test_contextualize_attaches_context_without_touching_chunk_text() -> None:
    client = _FakeLLMClient("This excerpt is from the ACS API rate limiting section.")
    contextualizer = ChunkContextualizer(client)  # type: ignore[arg-type]

    chunk = _chunk(text="The limit is 1,000 requests per minute.")
    [result] = contextualizer.contextualize([chunk], [_document()])

    assert result.context == "This excerpt is from the ACS API rate limiting section."
    assert result.text == "The limit is 1,000 requests per minute.", "the cited span must stay verbatim"


def test_index_text_joins_context_and_text_for_indexing() -> None:
    client = _FakeLLMClient("From the ACS rate limiting section.")
    contextualizer = ChunkContextualizer(client)  # type: ignore[arg-type]

    chunk = _chunk(text="The limit is 1,000 requests per minute.")
    [result] = contextualizer.contextualize([chunk], [_document()])

    assert result.index_text == (
        "From the ACS rate limiting section.\n\nThe limit is 1,000 requests per minute."
    )


def test_index_text_without_context_is_exactly_the_chunk_text() -> None:
    chunk = _chunk()

    assert chunk.context is None
    assert chunk.index_text == chunk.text


def test_contextualize_preserves_id_provenance_and_metadata() -> None:
    contextualizer = ChunkContextualizer(_FakeLLMClient())  # type: ignore[arg-type]
    original = _chunk()

    [result] = contextualizer.contextualize([original], [_document()])

    assert result.id == original.id
    assert result.document_id == original.document_id
    assert result.source == original.source
    assert result.doc_type == original.doc_type
    assert result.metadata == original.metadata


def test_contextualize_returns_chunks_in_order() -> None:
    client = _ScriptedLLMClient({"chunk0": "first context", "chunk1": "second context"})
    contextualizer = ChunkContextualizer(client)  # type: ignore[arg-type]
    chunks = [_chunk("doc.md::chunk0", document_id="doc.md"), _chunk("doc.md::chunk1", document_id="doc.md")]

    results = contextualizer.contextualize(chunks, [_document("doc.md")])

    assert [c.id for c in results] == ["doc.md::chunk0", "doc.md::chunk1"]
    assert [c.context for c in results] == ["first context", "second context"]


def test_order_is_preserved_when_replies_arrive_out_of_order() -> None:
    """Results are written back by position, not by completion order.

    The whole risk of running these calls concurrently is that a slow early
    chunk finishes after a fast later one. Here chunk0 is made deliberately
    slower than chunk1, so completion order is the reverse of input order.
    """

    class _ReorderingClient(_ScriptedLLMClient):
        def generate(self, prompt: str, *, system: str | None = None) -> str:
            if "chunk0" in prompt:
                time.sleep(0.05)
            return super().generate(prompt, system=system)

    client = _ReorderingClient({"chunk0": "context zero", "chunk1": "context one"})
    contextualizer = ChunkContextualizer(client, concurrency=2)  # type: ignore[arg-type]
    chunks = [_chunk("doc.md::chunk0", document_id="doc.md"), _chunk("doc.md::chunk1", document_id="doc.md")]

    results = contextualizer.contextualize(chunks, [_document("doc.md")])

    assert [c.id for c in results] == ["doc.md::chunk0", "doc.md::chunk1"]
    assert [c.context for c in results] == ["context zero", "context one"]


def test_concurrency_actually_overlaps_calls() -> None:
    """Verify calls are genuinely in flight together, not just threaded serially."""

    barrier = threading.Barrier(3, timeout=5)

    class _BarrierClient:
        def generate(self, prompt: str, *, system: str | None = None) -> str:
            # Blocks until three calls are simultaneously inside generate();
            # times out and raises if the pool is running them one at a time.
            barrier.wait()
            return "ctx"

    contextualizer = ChunkContextualizer(_BarrierClient(), concurrency=3)  # type: ignore[arg-type]
    chunks = [_chunk(f"doc.md::chunk{i}", document_id="doc.md") for i in range(3)]

    results = contextualizer.contextualize(chunks, [_document("doc.md")])

    assert [c.context for c in results] == ["ctx", "ctx", "ctx"]


def test_concurrency_is_capped_at_the_number_of_pending_chunks() -> None:
    contextualizer = ChunkContextualizer(_FakeLLMClient(), concurrency=64)  # type: ignore[arg-type]
    results = contextualizer.contextualize([_chunk()], [_document()])
    assert results[0].context == "Context sentence."


# ---------------------------------------------------------------------------
# Prompt assembly and bounds
# ---------------------------------------------------------------------------


def test_prompt_carries_both_the_document_and_the_excerpt() -> None:
    client = _FakeLLMClient()
    contextualizer = ChunkContextualizer(client)  # type: ignore[arg-type]

    contextualizer.contextualize([_chunk()], [_document(text="ACS API reference. Rate limits apply.")])

    [(prompt, system)] = client.calls
    assert "ACS API reference. Rate limits apply." in prompt
    assert "The limit is 1,000 requests per minute." in prompt
    assert system is not None and "situate" in system.lower()


def test_document_text_is_truncated_to_max_document_chars() -> None:
    client = _FakeLLMClient()
    contextualizer = ChunkContextualizer(client, max_document_chars=10)  # type: ignore[arg-type]

    contextualizer.contextualize([_chunk()], [_document(text="A" * 500)])

    [(prompt, _system)] = client.calls
    assert "A" * 10 in prompt
    assert "A" * 11 not in prompt


def test_generated_context_is_truncated_to_max_context_chars() -> None:
    client = _FakeLLMClient("B" * 500)
    contextualizer = ChunkContextualizer(client, max_context_chars=20)  # type: ignore[arg-type]

    [result] = contextualizer.contextualize([_chunk()], [_document()])

    assert result.context == "B" * 20


def test_context_is_collapsed_to_one_line_and_unquoted() -> None:
    client = _FakeLLMClient('  "From the\n  rate limiting   section."  ')
    contextualizer = ChunkContextualizer(client)  # type: ignore[arg-type]

    [result] = contextualizer.contextualize([_chunk()], [_document()])

    assert result.context == "From the rate limiting section."


def test_build_context_prompt_delimits_document_and_excerpt() -> None:
    prompt = build_context_prompt("the document", "the excerpt")

    assert "<document>" in prompt and "the document" in prompt
    assert "<excerpt>" in prompt and "the excerpt" in prompt


# ---------------------------------------------------------------------------
# Fail-open paths: a broken contextualizer must never break an index run
# ---------------------------------------------------------------------------


def test_raising_client_leaves_the_chunk_unchanged() -> None:
    contextualizer = ChunkContextualizer(_RaisingLLMClient())  # type: ignore[arg-type]

    [result] = contextualizer.contextualize([_chunk()], [_document()])

    assert result.context is None
    assert result.index_text == result.text


def test_empty_reply_leaves_the_chunk_unchanged() -> None:
    contextualizer = ChunkContextualizer(_FakeLLMClient("   "))  # type: ignore[arg-type]

    [result] = contextualizer.contextualize([_chunk()], [_document()])

    assert result.context is None


def test_chunk_with_no_matching_document_is_left_unchanged() -> None:
    client = _FakeLLMClient()
    contextualizer = ChunkContextualizer(client)  # type: ignore[arg-type]

    [result] = contextualizer.contextualize([_chunk(document_id="missing.md")], [_document("other.md")])

    assert result.context is None
    assert client.calls == [], "no parent document means nothing to generate context from"


def test_one_failure_does_not_stop_the_rest_of_the_batch() -> None:
    # First chunk gets an empty (unusable) reply, second gets a real one.
    client = _ScriptedLLMClient({"chunk0": "", "chunk1": "a real context"})
    contextualizer = ChunkContextualizer(client)  # type: ignore[arg-type]
    chunks = [_chunk("doc.md::chunk0", document_id="doc.md"), _chunk("doc.md::chunk1", document_id="doc.md")]

    results = contextualizer.contextualize(chunks, [_document("doc.md")])

    assert [c.context for c in results] == [None, "a real context"]


def test_one_raising_call_does_not_stop_the_rest_of_the_batch() -> None:
    class _PartiallyRaisingClient(_ScriptedLLMClient):
        def generate(self, prompt: str, *, system: str | None = None) -> str:
            if "chunk0" in prompt:
                raise RuntimeError("daemon unreachable")
            return super().generate(prompt, system=system)

    client = _PartiallyRaisingClient({"chunk1": "a real context"})
    contextualizer = ChunkContextualizer(client, concurrency=2)  # type: ignore[arg-type]
    chunks = [_chunk("doc.md::chunk0", document_id="doc.md"), _chunk("doc.md::chunk1", document_id="doc.md")]

    results = contextualizer.contextualize(chunks, [_document("doc.md")])

    assert [c.context for c in results] == [None, "a real context"]


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def test_cache_round_trips_a_generated_context(tmp_path: Path) -> None:
    cache = ContextCache(context_cache_path(tmp_path))
    client = _FakeLLMClient("cached context")
    contextualizer = ChunkContextualizer(client, cache=cache)  # type: ignore[arg-type]

    first = contextualizer.contextualize([_chunk()], [_document()])
    assert first[0].context == "cached context"
    assert len(client.calls) == 1

    # A second contextualizer over a *fresh* cache instance reads from disk and
    # makes no LLM calls -- this is the resume path after an interrupted run.
    reloaded = ContextCache(context_cache_path(tmp_path))
    second_client = _FakeLLMClient("should not be called")
    second = ChunkContextualizer(second_client, cache=reloaded).contextualize(  # type: ignore[arg-type]
        [_chunk()], [_document()]
    )
    assert second[0].context == "cached context"
    assert second_client.calls == []


def test_cache_misses_when_a_truncation_limit_changes(tmp_path: Path) -> None:
    """A config change must not serve a blurb generated under the old settings."""
    cache = ContextCache(context_cache_path(tmp_path))
    document = _document(text="A" * 500)
    ChunkContextualizer(  # type: ignore[arg-type]
        _FakeLLMClient("old"), cache=cache, max_document_chars=100
    ).contextualize([_chunk()], [document])

    client = _FakeLLMClient("regenerated")
    result = ChunkContextualizer(  # type: ignore[arg-type]
        client, cache=cache, max_document_chars=200
    ).contextualize([_chunk()], [document])

    assert result[0].context == "regenerated"
    assert len(client.calls) == 1


def test_cache_misses_when_the_model_changes(tmp_path: Path) -> None:
    cache = ContextCache(context_cache_path(tmp_path))
    ChunkContextualizer(  # type: ignore[arg-type]
        _FakeLLMClient("from model a"), cache=cache, model_name="model-a"
    ).contextualize([_chunk()], [_document()])

    client = _FakeLLMClient("from model b")
    result = ChunkContextualizer(  # type: ignore[arg-type]
        client, cache=cache, model_name="model-b"
    ).contextualize([_chunk()], [_document()])

    assert result[0].context == "from model b"
    assert len(client.calls) == 1


def test_failures_are_not_cached(tmp_path: Path) -> None:
    """A transient error must not be memoized as 'this chunk has no context'."""
    cache = ContextCache(context_cache_path(tmp_path))
    ChunkContextualizer(_RaisingLLMClient(), cache=cache).contextualize(  # type: ignore[arg-type]
        [_chunk()], [_document()]
    )
    assert len(cache) == 0

    client = _FakeLLMClient("recovered")
    result = ChunkContextualizer(client, cache=cache).contextualize(  # type: ignore[arg-type]
        [_chunk()], [_document()]
    )
    assert result[0].context == "recovered"


def test_cache_survives_a_torn_final_line(tmp_path: Path) -> None:
    """An interrupted run can leave a half-written record; earlier ones still load."""
    path = context_cache_path(tmp_path)
    cache = ContextCache(path)
    ChunkContextualizer(_FakeLLMClient("good"), cache=cache).contextualize(  # type: ignore[arg-type]
        [_chunk()], [_document()]
    )
    cache.close()
    with path.open("a", encoding="utf-8") as f:
        f.write('{"key": "truncated", "cont')

    reloaded = ContextCache(path)
    assert len(reloaded) == 1


def test_clear_empties_the_cache(tmp_path: Path) -> None:
    cache = ContextCache(context_cache_path(tmp_path))
    ChunkContextualizer(_FakeLLMClient("ctx"), cache=cache).contextualize(  # type: ignore[arg-type]
        [_chunk()], [_document()]
    )
    assert len(cache) == 1

    cache.clear()
    assert len(cache) == 0
    assert not context_cache_path(tmp_path).exists()

    client = _FakeLLMClient("regenerated")
    result = ChunkContextualizer(client, cache=cache).contextualize(  # type: ignore[arg-type]
        [_chunk()], [_document()]
    )
    assert result[0].context == "regenerated"
    assert len(client.calls) == 1


def test_empty_chunk_list_makes_no_llm_calls() -> None:
    client = _FakeLLMClient()
    contextualizer = ChunkContextualizer(client)  # type: ignore[arg-type]

    assert contextualizer.contextualize([], [_document()]) == []
    assert client.calls == []
