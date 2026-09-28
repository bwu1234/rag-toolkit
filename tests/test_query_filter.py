"""Tests for `QueryFilter`: validation, matching, and its Chroma translation (chunking plan, Phase 3)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from rag.query_filter import IntRange, QueryFilter, check_filterable

_APPLE_2024 = {"document_id": "AAPL_10-K_2024-09-28.md", "ticker": "AAPL", "form": "10-K", "period_end": 20240928}


def test_every_condition_must_hold() -> None:
    f = QueryFilter(equals={"ticker": "AAPL"}, any_of={"form": ["10-K", "10-Q"]}, range={"period_end": {"gte": 20240101}})

    assert f.matches(_APPLE_2024)
    assert not f.matches({**_APPLE_2024, "ticker": "MSFT"})
    assert not f.matches({**_APPLE_2024, "form": "8-K"})
    assert not f.matches({**_APPLE_2024, "period_end": 20231230})


def test_a_missing_field_never_matches() -> None:
    assert not QueryFilter(equals={"ticker": "AAPL"}).matches({"document_id": "notes.md"})
    assert not QueryFilter(range={"period_end": {"lte": 20991231}}).matches({"document_id": "notes.md"})


def test_range_bounds_accept_iso_dates() -> None:
    bounds = IntRange(gte="2024-09-28", lte="2025-06-30")  # type: ignore[arg-type]

    assert (bounds.gte, bounds.lte) == (20240928, 20250630)


@pytest.mark.parametrize(
    "bad",
    [{}, {"gte": 20250101, "lte": 20240101}, {"gte": "not-a-date"}],
    ids=["no bounds", "empty range", "unparseable"],
)
def test_invalid_ranges_are_rejected(bad: dict) -> None:
    with pytest.raises(ValidationError):
        IntRange(**bad)


def test_unknown_keys_and_empty_sets_are_rejected() -> None:
    with pytest.raises(ValidationError):
        QueryFilter(where={"ticker": "AAPL"})  # type: ignore[call-arg]
    with pytest.raises(ValidationError, match="at least one value"):
        QueryFilter(any_of={"ticker": []})


def test_chroma_where_uses_and_only_for_two_or_more_clauses() -> None:
    assert QueryFilter().to_chroma_where() is None
    assert QueryFilter(equals={"ticker": "AAPL"}).to_chroma_where() == {"ticker": {"$eq": "AAPL"}}
    assert QueryFilter(equals={"ticker": "AAPL"}, range={"period_end": {"gte": 1, "lte": 2}}).to_chroma_where() == {
        "$and": [{"ticker": {"$eq": "AAPL"}}, {"period_end": {"$gte": 1}}, {"period_end": {"$lte": 2}}]
    }


def test_filtering_on_a_field_chunks_do_not_store_is_an_error() -> None:
    check_filterable(QueryFilter(equals={"document_id": "a.md", "ticker": "AAPL"}), ["ticker"])
    with pytest.raises(ValueError, match="tickr"):
        check_filterable(QueryFilter(equals={"tickr": "AAPL"}), ["ticker"])
