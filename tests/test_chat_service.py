"""Tests for `ChatService` orchestration.

`ChatService` is tested against fakes for `Retriever` and `LLMClient` --
mirroring how `test_retriever.py` fakes `EmbeddingModel`/`VectorStore`/
`Reranker` -- so the orchestration logic (when to skip generation, how chunks
become citations, what gets passed to the LLM) is verified without a running
Ollama daemon, Chroma index, or cross-encoder model.
"""

from __future__ import annotations

from pathlib import Path

from rag.events import EventSink
from rag.generation.chat_service import ChatService, Citation
from rag.generation.query_rewriter import ChatTurn, QueryCondenser
from rag.retrieval.retriever import RetrievalResult
from rag.vectorstore.base import ScoredChunk


def _scored(chunk_id: str, text: str = "text", score: float = 0.5, page: int | None = None) -> ScoredChunk:
    metadata = {"page": page} if page is not None else {}
    return ScoredChunk(
        chunk_id=chunk_id,
        text=text,
        document_id=f"{chunk_id}-doc",
        source=Path(f"{chunk_id}.md"),
        doc_type="markdown",
        score=score,
        metadata=metadata,
    )


class _FakeRetriever:
    def __init__(
        self,
        results: list[ScoredChunk],
        *,
        candidate_count: int | None = None,
        dropped: int = 0,
    ) -> None:
        self.results = results
        # Default to "the index had exactly what we returned" -- tests that care
        # about the no-context branches set these explicitly.
        self.candidate_count = candidate_count if candidate_count is not None else len(results)
        self.dropped = dropped
        self.queries: list[str] = []

    def retrieve(self, query: str, *, on_event: EventSink | None = None) -> RetrievalResult:
        self.queries.append(query)
        return RetrievalResult(
            chunks=self.results,
            candidate_count=self.candidate_count,
            dropped_below_min_score=self.dropped,
        )


class _FakeLLMClient:
    def __init__(self, reply: str = "a generated answer") -> None:
        self.reply = reply
        self.calls: list[tuple[str, str | None]] = []

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        self.calls.append((prompt, system))
        return self.reply


class _FakeCondenser:
    """Returns a canned rewrite and records what it was asked to condense."""

    def __init__(self, rewrite: str = "a standalone question") -> None:
        self.rewrite = rewrite
        self.calls: list[tuple[str, list[ChatTurn]]] = []

    def condense(self, query: str, history: list[ChatTurn]) -> str:
        self.calls.append((query, history))
        return self.rewrite


def _service(
    results: list[ScoredChunk] | None = None,
    reply: str = "a generated answer",
    condenser: QueryCondenser | None = None,
    *,
    candidate_count: int | None = None,
    dropped: int = 0,
):
    results = results if results is not None else [_scored("a", "Refunds within 30 days.", page=2)]
    retriever = _FakeRetriever(results, candidate_count=candidate_count, dropped=dropped)
    llm_client = _FakeLLMClient(reply)
    service = ChatService(retriever=retriever, llm_client=llm_client, condenser=condenser)
    return service, retriever, llm_client


def test_ask_returns_generated_answer_with_citations_from_retrieved_chunks() -> None:
    chunks = [_scored("a", "Refunds within 30 days.", score=0.9, page=2), _scored("b", "Exchanges allowed.", score=0.7)]
    service, retriever, llm_client = _service(chunks, reply="You can get a refund within 30 days [1].")

    result = service.ask("what is the refund policy")

    assert result.answer == "You can get a refund within 30 days [1]."
    assert result.citations == [
        Citation(chunk_id="a", document_id="a-doc", text="Refunds within 30 days.", score=0.9, page=2),
        Citation(chunk_id="b", document_id="b-doc", text="Exchanges allowed.", score=0.7, page=None),
    ]
    assert retriever.queries == ["what is the refund policy"]


def test_ask_passes_a_grounded_prompt_and_system_prompt_to_the_llm() -> None:
    service, _retriever, llm_client = _service([_scored("a", "Refunds within 30 days.")])

    service.ask("what is the refund policy")

    [(prompt, system)] = llm_client.calls
    assert "Refunds within 30 days." in prompt
    assert "what is the refund policy" in prompt
    assert system is not None and "cite" in system.lower()


def test_ask_on_blank_query_skips_retrieval_and_generation() -> None:
    service, retriever, llm_client = _service()

    result = service.ask("   ")

    assert result.citations == []
    assert result.answer  # some explanatory message
    assert retriever.queries == []
    assert llm_client.calls == []


def test_ask_on_empty_retrieval_skips_generation_and_explains() -> None:
    service, retriever, llm_client = _service(results=[])

    result = service.ask("an unanswerable question")

    assert result.citations == []
    assert result.answer
    assert retriever.queries == ["an unanswerable question"]
    assert llm_client.calls == [], "the LLM should never be called when there's no context to ground it in"


def test_ask_reports_prompt_and_generate_events() -> None:
    service, *_ = _service()

    events = []
    service.ask("a question", on_event=events.append)

    stages = [event.stage for event in events]
    assert stages == ["prompt", "generate"]
    assert all(event.elapsed_ms is not None for event in events)


def test_ask_on_empty_retrieval_reports_no_context_event() -> None:
    service, *_ = _service(results=[])

    events = []
    service.ask("an unanswerable question", on_event=events.append)

    assert [event.stage for event in events] == ["no_context"]


def test_ask_on_blank_query_reports_no_events() -> None:
    service, *_ = _service()

    events = []
    service.ask("   ", on_event=events.append)

    assert events == []


def test_ask_drops_non_integer_page_metadata() -> None:
    chunk = ScoredChunk(
        chunk_id="a", text="t", document_id="a-doc", source=Path("a.md"),
        doc_type="markdown", score=0.5, metadata={"page": "not-a-number"},
    )
    service, *_ = _service(results=[chunk])

    [citation] = service.ask("query").citations

    assert citation.page is None


# ---------------------------------------------------------------------------
# Multi-turn: history and query condensing
# ---------------------------------------------------------------------------


def test_ask_with_history_retrieves_with_the_condensed_query() -> None:
    condenser = _FakeCondenser("What is the refund policy for part-time staff?")
    service, retriever, _llm = _service(condenser=condenser)  # type: ignore[arg-type]

    service.ask(
        "what about part-time staff?",
        history=[
            ChatTurn(role="user", content="what is the refund policy?"),
            ChatTurn(role="assistant", content="Refunds within 30 days."),
        ],
    )

    assert retriever.queries == ["What is the refund policy for part-time staff?"]


def test_ask_with_history_passes_the_original_query_and_history_to_the_condenser() -> None:
    condenser = _FakeCondenser()
    history = [ChatTurn(role="user", content="what is the refund policy?")]
    service, *_ = _service(condenser=condenser)  # type: ignore[arg-type]

    service.ask("what about part-time staff?", history=history)

    assert condenser.calls == [("what about part-time staff?", history)]


def test_ask_prompts_the_llm_with_the_condensed_query_not_the_raw_follow_up() -> None:
    # build_rag_prompt renders passages plus one question and no conversation,
    # so the raw follow-up would leave the model with nothing to resolve it against.
    condenser = _FakeCondenser("What is the refund policy for part-time staff?")
    service, _retriever, llm_client = _service(condenser=condenser)  # type: ignore[arg-type]

    service.ask(
        "what about part-time staff?",
        history=[ChatTurn(role="user", content="what is the refund policy?")],
    )

    [(prompt, _system)] = llm_client.calls
    assert "What is the refund policy for part-time staff?" in prompt


def test_ask_without_history_skips_condensing_entirely() -> None:
    condenser = _FakeCondenser("should never be used")
    service, retriever, _llm = _service(condenser=condenser)  # type: ignore[arg-type]

    service.ask("what is the refund policy?")

    assert condenser.calls == []
    assert retriever.queries == ["what is the refund policy?"]


def test_ask_with_history_but_no_condenser_uses_the_raw_query() -> None:
    service, retriever, _llm = _service(condenser=None)

    service.ask("what about part-time staff?", history=[ChatTurn(role="user", content="earlier")])

    assert retriever.queries == ["what about part-time staff?"]


def test_ask_reports_a_condense_event_before_the_rest_of_the_pipeline() -> None:
    condenser = _FakeCondenser("a standalone question")
    service, *_ = _service(condenser=condenser)  # type: ignore[arg-type]

    events = []
    service.ask(
        "follow-up",
        history=[ChatTurn(role="user", content="earlier")],
        on_event=events.append,
    )

    stages = [event.stage for event in events]
    assert stages == ["condense", "prompt", "generate"]
    assert "a standalone question" in events[0].message


def test_ask_on_blank_query_skips_condensing_even_with_history() -> None:
    condenser = _FakeCondenser()
    service, *_ = _service(condenser=condenser)  # type: ignore[arg-type]

    service.ask("   ", history=[ChatTurn(role="user", content="earlier")])

    assert condenser.calls == []


# ---------------------------------------------------------------------------
# Making the floor and the rewrite explicit in the answer
# ---------------------------------------------------------------------------


def test_empty_index_and_all_filtered_get_different_answers() -> None:
    # The whole point of the min_score floor: "run the indexer" and "your
    # corpus doesn't cover this" are different problems with different fixes.
    empty_index, *_ = _service(results=[], candidate_count=0)
    all_filtered, *_ = _service(results=[], candidate_count=8, dropped=5)

    empty_answer = empty_index.ask("a question").answer
    filtered_answer = all_filtered.ask("a question").answer

    assert empty_answer != filtered_answer
    assert "index" in empty_answer.lower()
    assert "relevant" in filtered_answer.lower()


def test_all_filtered_answer_reports_the_counts() -> None:
    service, *_ = _service(results=[], candidate_count=8, dropped=5)

    answer = service.ask("an off-corpus question")

    assert "8" in answer.answer and "5" in answer.answer
    assert answer.dropped_below_min_score == 5


def test_answer_reports_partial_drops_alongside_a_real_answer() -> None:
    # A silent partial drop is the case a user can't otherwise notice: the
    # answer arrives normally, just grounded in fewer passages.
    service, *_ = _service(candidate_count=6, dropped=3)

    answer = service.ask("a question")

    assert answer.citations, "this is the answering path, not a no-context path"
    assert answer.dropped_below_min_score == 3


def test_answer_reports_no_drops_when_nothing_was_filtered() -> None:
    service, *_ = _service()

    assert service.ask("a question").dropped_below_min_score == 0


def test_answer_reports_the_rewritten_query_when_condensing_changed_it() -> None:
    condenser = _FakeCondenser("What is the refund policy for part-time staff?")
    service, *_ = _service(condenser=condenser)  # type: ignore[arg-type]

    answer = service.ask(
        "what about part-time staff?",
        history=[ChatTurn(role="user", content="what is the refund policy?")],
    )

    assert answer.rewritten_query == "What is the refund policy for part-time staff?"


def test_answer_omits_the_rewritten_query_when_the_condenser_left_it_alone() -> None:
    # An already-standalone question shouldn't render as "rewritten to: <itself>".
    condenser = _FakeCondenser("what is the refund policy?")
    service, *_ = _service(condenser=condenser)  # type: ignore[arg-type]

    answer = service.ask(
        "what is the refund policy?",
        history=[ChatTurn(role="user", content="hello")],
    )

    assert answer.rewritten_query is None


def test_answer_omits_the_rewritten_query_when_no_condensing_ran() -> None:
    service, *_ = _service()

    assert service.ask("a question").rewritten_query is None


def test_no_context_answer_still_reports_the_rewritten_query() -> None:
    # A follow-up that retrieves nothing is exactly when a user most needs to
    # see what was actually searched for.
    condenser = _FakeCondenser("A rewritten standalone question")
    service, *_ = _service(results=[], candidate_count=4, dropped=4, condenser=condenser)  # type: ignore[arg-type]

    answer = service.ask("follow-up", history=[ChatTurn(role="user", content="earlier")])

    assert answer.rewritten_query == "A rewritten standalone question"
    assert answer.dropped_below_min_score == 4


def test_no_context_event_message_says_why_generation_was_skipped() -> None:
    service, *_ = _service(results=[], candidate_count=8, dropped=5)

    events = []
    service.ask("a question", on_event=events.append)

    [no_context] = [e for e in events if e.stage == "no_context"]
    assert "8 candidate(s)" in no_context.message and "below min_score" in no_context.message


def test_blank_query_answer_is_distinct_from_the_no_context_answers() -> None:
    service, *_ = _service()

    blank = service.ask("   ").answer

    assert "question" in blank.lower()
    assert blank != _service(results=[], candidate_count=0)[0].ask("q").answer
