"""Tests for HyDE and multi-query expansion.

Driven by a fake `LLMClient` (mirroring `test_query_rewriter.py`) -- expansion
is prompt assembly plus reply parsing, both verifiable without a daemon. The
fail-open paths matter as much as the happy ones: an expander that raises would
take down every search, so each strategy is tested against an exploding and an
empty-replying client too.
"""

from __future__ import annotations

import pytest

from rag.config.settings import QueryExpansionConfig
from rag.retrieval.expansion import (
    ExpandedQuery,
    HyDEQueryExpander,
    MultiQueryExpander,
    NoOpQueryExpander,
    parse_query_lines,
)
from rag.retrieval.factory import get_query_expander


class _FakeLLMClient:
    def __init__(self, *replies: str) -> None:
        # Cycles through replies so HyDE's num_documents>1 can get distinct ones.
        self.replies = list(replies) or ["a generated reply"]
        self.calls: list[tuple[str, str | None]] = []

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        self.calls.append((prompt, system))
        return self.replies[(len(self.calls) - 1) % len(self.replies)]


class _ExplodingLLMClient:
    def generate(self, prompt: str, *, system: str | None = None) -> str:
        raise RuntimeError("ollama is down")


# ---------------------------------------------------------------------------
# ExpandedQuery
# ---------------------------------------------------------------------------


def test_unchanged_expansion_is_not_flagged_as_expanded() -> None:
    assert ExpandedQuery.unchanged("a question").is_expanded is False


def test_differing_dense_and_sparse_counts_as_expanded() -> None:
    # HyDE's shape: a generated passage for dense, the real question everywhere else.
    expanded = ExpandedQuery(
        dense=["a hypothetical passage"], sparse=["a question"], rerank=["a question"]
    )
    assert expanded.is_expanded is True


def test_all_queries_dedupes_across_dense_and_sparse() -> None:
    expanded = ExpandedQuery(dense=["a", "b"], sparse=["a", "c"], rerank=["a"])
    assert expanded.all_queries() == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# parse_query_lines
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "reply",
    [
        "first question\nsecond question",
        "1. first question\n2. second question",
        "- first question\n- second question",
        "1) first question\n\n2) second question\n",
        '"first question"\n"second question"',
    ],
    ids=["plain", "numbered-dot", "bulleted", "numbered-paren-blanks", "quoted"],
)
def test_parse_query_lines_strips_the_formatting_models_add(reply: str) -> None:
    assert parse_query_lines(reply, limit=5) == ["first question", "second question"]


def test_parse_query_lines_dedupes_case_insensitively() -> None:
    # A duplicate is pure cost: an extra embedding call and a redundant ranked
    # list that skews fusion toward whatever it found.
    assert parse_query_lines("Same Thing\nsame thing\nother", limit=5) == ["Same Thing", "other"]


def test_parse_query_lines_respects_the_limit() -> None:
    assert parse_query_lines("a\nb\nc\nd", limit=2) == ["a", "b"]


def test_parse_query_lines_on_an_empty_reply() -> None:
    assert parse_query_lines("   \n\n  ", limit=3) == []


# ---------------------------------------------------------------------------
# NoOpQueryExpander
# ---------------------------------------------------------------------------


def test_noop_expander_returns_the_query_untouched() -> None:
    expanded = NoOpQueryExpander().expand("a question")
    assert expanded.dense == ["a question"]
    assert expanded.sparse == ["a question"]
    assert expanded.is_expanded is False


# ---------------------------------------------------------------------------
# HyDE
# ---------------------------------------------------------------------------


def test_hyde_embeds_the_generated_passage_and_keeps_the_question_for_bm25() -> None:
    # The asymmetry is the point: BM25 on invented specifics matches words that
    # may appear nowhere in the corpus.
    llm = _FakeLLMClient("Rate limits are enforced at 1,000 requests per minute.")
    expanded = HyDEQueryExpander(llm).expand("what is the rate limit?")  # type: ignore[arg-type]

    assert expanded.dense == [
        "Rate limits are enforced at 1,000 requests per minute.",
        "what is the rate limit?",
    ]
    assert expanded.sparse == ["what is the rate limit?"]


def test_hyde_can_drop_the_original_query_from_the_dense_list() -> None:
    llm = _FakeLLMClient("A generated passage.")
    expanded = HyDEQueryExpander(llm, include_original=False).expand("q")  # type: ignore[arg-type]

    assert expanded.dense == ["A generated passage."]


def test_hyde_generates_one_passage_per_num_documents() -> None:
    llm = _FakeLLMClient("passage one", "passage two", "passage three")
    expanded = HyDEQueryExpander(llm, num_documents=3, include_original=False).expand("q")  # type: ignore[arg-type]

    assert expanded.dense == ["passage one", "passage two", "passage three"]
    assert len(llm.calls) == 3


def test_hyde_collapses_whitespace_in_the_generated_passage() -> None:
    llm = _FakeLLMClient("  A passage\nacross   lines.  ")
    expanded = HyDEQueryExpander(llm, include_original=False).expand("q")  # type: ignore[arg-type]

    assert expanded.dense == ["A passage across lines."]


def test_hyde_falls_back_to_the_original_query_when_generation_raises() -> None:
    expanded = HyDEQueryExpander(_ExplodingLLMClient()).expand("q")  # type: ignore[arg-type]

    assert expanded == ExpandedQuery.unchanged("q")


def test_hyde_falls_back_to_the_original_query_on_an_empty_passage() -> None:
    expanded = HyDEQueryExpander(_FakeLLMClient("   ")).expand("q")  # type: ignore[arg-type]

    assert expanded == ExpandedQuery.unchanged("q")


def test_hyde_keeps_the_passages_it_did_get_when_only_some_generations_fail() -> None:
    llm = _FakeLLMClient("a real passage", "   ")
    expanded = HyDEQueryExpander(llm, num_documents=2, include_original=False).expand("q")  # type: ignore[arg-type]

    assert expanded.dense == ["a real passage"]


def test_hyde_instructs_the_model_to_invent_rather_than_hedge() -> None:
    llm = _FakeLLMClient("a passage")
    HyDEQueryExpander(llm).expand("q")  # type: ignore[arg-type]

    [(_prompt, system)] = llm.calls
    assert system is not None
    assert "invent" in system.lower()


# ---------------------------------------------------------------------------
# Multi-query
# ---------------------------------------------------------------------------


def test_multi_query_puts_the_original_first_then_the_variants() -> None:
    llm = _FakeLLMClient("how is throttling configured?\nwhat are the request caps?")
    expanded = MultiQueryExpander(llm).expand("what is the rate limit?")  # type: ignore[arg-type]

    assert expanded.dense == [
        "what is the rate limit?",
        "how is throttling configured?",
        "what are the request caps?",
    ]


def test_multi_query_gives_both_retrievers_the_same_queries() -> None:
    # Unlike HyDE, a rephrasing is still a real question -- BM25 benefits from
    # the extra vocabulary rather than being poisoned by invented terms.
    llm = _FakeLLMClient("variant one\nvariant two")
    expanded = MultiQueryExpander(llm).expand("original")  # type: ignore[arg-type]

    assert expanded.dense == expanded.sparse


def test_multi_query_does_not_duplicate_the_original_if_the_model_echoes_it() -> None:
    llm = _FakeLLMClient("Original\nsomething else")
    expanded = MultiQueryExpander(llm).expand("original")  # type: ignore[arg-type]

    assert expanded.dense == ["original", "something else"]


def test_multi_query_caps_variants_at_num_queries() -> None:
    llm = _FakeLLMClient("a\nb\nc\nd\ne")
    expanded = MultiQueryExpander(llm, num_queries=2).expand("q")  # type: ignore[arg-type]

    assert expanded.dense == ["q", "a", "b"]


def test_multi_query_falls_back_to_the_original_when_generation_raises() -> None:
    expanded = MultiQueryExpander(_ExplodingLLMClient()).expand("q")  # type: ignore[arg-type]

    assert expanded == ExpandedQuery.unchanged("q")


def test_multi_query_falls_back_to_the_original_on_an_unparseable_reply() -> None:
    expanded = MultiQueryExpander(_FakeLLMClient("\n\n  \n")).expand("q")  # type: ignore[arg-type]

    assert expanded == ExpandedQuery.unchanged("q")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def test_factory_returns_a_noop_expander_without_needing_an_llm() -> None:
    expander = get_query_expander(QueryExpansionConfig(provider="none"), None)
    assert isinstance(expander, NoOpQueryExpander)


@pytest.mark.parametrize("provider", ["hyde", "multi_query"])
def test_factory_builds_the_selected_generating_expander(provider: str) -> None:
    config = QueryExpansionConfig(provider=provider)  # type: ignore[arg-type]
    expander = get_query_expander(config, _FakeLLMClient())  # type: ignore[arg-type]

    expected = {"hyde": HyDEQueryExpander, "multi_query": MultiQueryExpander}[provider]
    assert isinstance(expander, expected)


def test_factory_passes_config_knobs_through() -> None:
    hyde = get_query_expander(
        QueryExpansionConfig(provider="hyde", num_documents=3, include_original=False),
        _FakeLLMClient(),  # type: ignore[arg-type]
    )
    assert hyde.num_documents == 3 and hyde.include_original is False  # type: ignore[attr-defined]

    multi = get_query_expander(
        QueryExpansionConfig(provider="multi_query", num_queries=5),
        _FakeLLMClient(),  # type: ignore[arg-type]
    )
    assert multi.num_queries == 5  # type: ignore[attr-defined]


def test_factory_rejects_a_generating_provider_without_an_llm_client() -> None:
    # A wiring bug should fail at build time, not mid-query.
    with pytest.raises(ValueError, match="needs an LLMClient"):
        get_query_expander(QueryExpansionConfig(provider="hyde"), None)
