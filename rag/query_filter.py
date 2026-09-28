"""A typed metadata filter that restricts retrieval to matching chunks.

Retrieval without a filter ranks the whole corpus. With one, both stage-1
retrievers rank only the chunks whose stored metadata matches it, *before*
taking their top-k -- so a filter narrows what competes rather than trimming
what already won, and never returns fewer results than an unfiltered query
would when enough matching chunks exist.

The fields a filter can name are the chunk's `document_id` plus whatever
`chunking.carry_metadata` copies onto chunks (on EDGAR: `company`, `ticker`,
`form`, `period_end`, `filed`, `accession`). Dates are stored as `YYYYMMDD`
integers, so a period is a `range`; bounds may be given as ISO dates and are
converted.

A pydantic model rather than an internal dataclass because it is validated at
two wire boundaries -- `POST /chat` and the MCP `rag_search` tool -- and the
same shape travels unchanged from there to the stores.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

#: Always filterable: every chunk stores it, whatever `carry_metadata` says.
DOCUMENT_ID = "document_id"


def _as_int(value: int | date) -> int:
    return int(value.strftime("%Y%m%d")) if isinstance(value, date) else value


class IntRange(BaseModel):
    """An inclusive range on an integer field; either bound may be omitted, not both."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    gte: int | None = Field(default=None, description="Lower bound, inclusive. A date becomes YYYYMMDD.")
    lte: int | None = Field(default=None, description="Upper bound, inclusive. A date becomes YYYYMMDD.")

    @field_validator("gte", "lte", mode="before")
    @classmethod
    def _dates_to_ints(cls, value: Any) -> Any:
        if isinstance(value, str):
            try:
                return _as_int(date.fromisoformat(value))
            except ValueError:
                return value  # left for int validation to reject with its own message
        return _as_int(value) if isinstance(value, date) else value

    @model_validator(mode="after")
    def _has_a_bound(self) -> IntRange:
        if self.gte is None and self.lte is None:
            raise ValueError("a range needs `gte`, `lte`, or both")
        if self.gte is not None and self.lte is not None and self.gte > self.lte:
            raise ValueError(f"range is empty: gte {self.gte} > lte {self.lte}")
        return self

    def contains(self, value: Any) -> bool:
        if not isinstance(value, int) or isinstance(value, bool):
            return False
        return (self.gte is None or value >= self.gte) and (self.lte is None or value <= self.lte)


class QueryFilter(BaseModel):
    """Conditions a chunk's metadata must all satisfy (AND across every entry).

    - `equals`: string field == value, e.g. `{"ticker": "AAPL"}`.
    - `any_of`: string field is one of the values, e.g. `{"form": ["10-K", "10-Q"]}`.
    - `range`: integer field within bounds, e.g. `{"period_end": {"gte": "2025-01-01"}}`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    equals: dict[str, str] = Field(default_factory=dict, description="Field must equal this string.")
    any_of: dict[str, list[str]] = Field(
        default_factory=dict, description="Field must equal one of these strings."
    )
    range: dict[str, IntRange] = Field(
        default_factory=dict, description="Integer field (dates as YYYYMMDD) within these bounds."
    )

    @field_validator("any_of")
    @classmethod
    def _non_empty_sets(cls, value: dict[str, list[str]]) -> dict[str, list[str]]:
        empty = [name for name, options in value.items() if not options]
        if empty:
            raise ValueError(f"any_of needs at least one value for: {', '.join(empty)}")
        return value

    @property
    def fields(self) -> set[str]:
        """Every metadata field this filter names."""
        return set(self.equals) | set(self.any_of) | set(self.range)

    @property
    def is_empty(self) -> bool:
        return not self.fields

    def matches(self, metadata: Mapping[str, Any]) -> bool:
        """Whether a chunk with this flat metadata (including `document_id`) passes."""
        return (
            all(metadata.get(name) == value for name, value in self.equals.items())
            and all(metadata.get(name) in options for name, options in self.any_of.items())
            and all(bounds.contains(metadata.get(name)) for name, bounds in self.range.items())
        )

    def to_chroma_where(self) -> dict[str, Any] | None:
        """The same conditions as a Chroma `where` clause, or `None` when empty."""
        clauses: list[dict[str, Any]] = [{name: {"$eq": value}} for name, value in self.equals.items()]
        clauses += [{name: {"$in": list(options)}} for name, options in self.any_of.items()]
        for name, bounds in self.range.items():
            if bounds.gte is not None:
                clauses.append({name: {"$gte": bounds.gte}})
            if bounds.lte is not None:
                clauses.append({name: {"$lte": bounds.lte}})
        if not clauses:
            return None
        # Chroma rejects `$and` with fewer than two operands.
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}

    def describe(self) -> str:
        """Compact human-readable form for logs and trace events."""
        parts = [f"{name}={value}" for name, value in self.equals.items()]
        parts += [f"{name} in {options}" for name, options in self.any_of.items()]
        parts += [
            f"{name} in [{'' if b.gte is None else b.gte}, {'' if b.lte is None else b.lte}]"
            for name, b in self.range.items()
        ]
        return ", ".join(parts) or "(empty)"


class UnfilterableField(ValueError):
    """A filter names a field chunks don't store -- a caller error, not a pipeline one."""


def check_filterable(query_filter: QueryFilter, carried: list[str] | tuple[str, ...]) -> None:
    """Reject a filter naming a field chunks don't store.

    Without this, a typo (`tickr`) or a field left out of `carry_metadata`
    matches nothing and looks exactly like "the corpus has no answer".

    Raises:
        UnfilterableField: a named field is neither `document_id` nor carried.
    """

    allowed = {DOCUMENT_ID, *carried}
    unknown = sorted(query_filter.fields - allowed)
    if unknown:
        raise UnfilterableField(
            f"Cannot filter on {', '.join(unknown)}: chunks store only "
            f"{', '.join(sorted(allowed))} (see chunking.carry_metadata)."
        )
