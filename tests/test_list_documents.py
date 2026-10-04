"""`RagTools.list_documents` / the `rag_list_documents` tool, over a corpus on disk.

Hermetic: a temporary corpus of Markdown files with front matter, no index and
no embedder -- the listing reads documents only, which is the point of it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.config.settings import ChunkingConfig, CorporaConfig, CorpusConfig, RagConfig
from rag.query_filter import QueryFilter
from rag.tools import MAX_LIST_LIMIT, RagTools, build_tool_specs


def _filing(ticker: str, form: str, period_end: str, body: str = "Text.") -> str:
    return (
        f"---\ncompany: {ticker} Inc.\nticker: {ticker}\nform: {form}\n"
        f"period_end: {period_end}\nfiled: {period_end}\naccession: 0000-{ticker}\n---\n# {ticker}\n\n{body}\n"
    )


@pytest.fixture()
def tools(tmp_path: Path) -> RagTools:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "DAL_10-Q_2026-06-30.md").write_text(_filing("DAL", "10-Q", "2026-06-30", "x" * 50))
    (docs / "DAL_10-K_2025-12-31.md").write_text(_filing("DAL", "10-K", "2025-12-31"))
    (docs / "UAL_10-Q_2026-06-30.md").write_text(_filing("UAL", "10-Q", "2026-06-30"))
    (docs / "notes.txt").write_text("No front matter here.")
    config = RagConfig(
        corpora=CorporaConfig(active=["c"], registry={"c": CorpusConfig(documents_dir=docs)}),
        # The listing shows -- and filters on -- exactly the fields chunks carry.
        chunking=ChunkingConfig(carry_metadata=["title", "company", "ticker", "form", "period_end", "filed"]),
    )
    return RagTools(config=config)


def test_lists_every_document_with_its_metadata_and_dates_as_iso(tools: RagTools) -> None:
    payload = tools.list_documents()

    assert (payload["corpora"], payload["total"], payload["returned"]) == (["c"], 4, 4)
    assert "hint" not in payload
    by_id = {d["document_id"]: d for d in payload["documents"]}
    assert set(by_id) == {"DAL_10-K_2025-12-31.md", "DAL_10-Q_2026-06-30.md", "UAL_10-Q_2026-06-30.md", "notes.txt"}
    dal = by_id["DAL_10-Q_2026-06-30.md"]
    assert {k: dal[k] for k in ("ticker", "form", "period_end")} == {
        "ticker": "DAL", "form": "10-Q", "period_end": "2026-06-30",
    }
    assert dal["chars"] > 50
    assert "accession" not in dal  # in the front matter, but not carried
    # A document without front matter is still listed, with what it has.
    assert set(by_id["notes.txt"]) == {"document_id", "title", "chars"}


def test_a_filter_applies_as_it_does_to_search(tools: RagTools) -> None:
    payload = tools.list_documents(
        filters=QueryFilter(any_of={"form": ["10-Q"]}, range={"period_end": {"gte": "2026-04-01"}})
    )

    assert [d["document_id"] for d in payload["documents"]] == ["DAL_10-Q_2026-06-30.md", "UAL_10-Q_2026-06-30.md"]
    assert payload["filters"] == {"any_of": {"form": ["10-Q"]}, "range": {"period_end": {"gte": 20260401}}}


def test_a_filter_on_document_id_works(tools: RagTools) -> None:
    payload = tools.list_documents(filters=QueryFilter(equals={"document_id": "notes.txt"}))

    assert [d["document_id"] for d in payload["documents"]] == ["notes.txt"]


def test_the_limit_truncates_and_says_how_to_get_the_next_page(tools: RagTools) -> None:
    payload = tools.list_documents(limit=1)

    assert (payload["total"], payload["offset"], payload["returned"], payload["next_offset"]) == (4, 0, 1, 1)
    assert "documents 1-1 of 4" in payload["hint"] and "offset=1" in payload["hint"]


def test_pages_cover_every_document_once_in_a_stable_order(tools: RagTools) -> None:
    pages: list[str] = []
    offset: int | None = 0
    while offset is not None:
        payload = tools.list_documents(limit=3, offset=offset)
        pages += [d["document_id"] for d in payload["documents"]]
        offset = payload.get("next_offset")

    assert pages == [d["document_id"] for d in tools.list_documents()["documents"]]
    assert len(pages) == len(set(pages)) == 4
    assert "hint" not in payload  # the last page


def test_pagination_applies_after_the_filter(tools: RagTools) -> None:
    payload = tools.list_documents(filters=QueryFilter(equals={"ticker": "DAL"}), limit=1, offset=1)

    assert (payload["total"], payload["returned"]) == (2, 1)
    assert payload["documents"][0]["ticker"] == "DAL"
    assert "next_offset" not in payload


def test_an_offset_past_the_end_lists_nothing_and_says_so(tools: RagTools) -> None:
    payload = tools.list_documents(offset=10)

    assert (payload["total"], payload["returned"]) == (4, 0)
    assert "past the last of 4" in payload["hint"]


def test_a_negative_offset_is_an_argument_error(tools: RagTools) -> None:
    with pytest.raises(ValueError, match="offset must be at least 0"):
        tools.list_documents(offset=-1)


@pytest.mark.parametrize("limit", [0, MAX_LIST_LIMIT + 1])
def test_a_limit_out_of_range_is_an_argument_error(tools: RagTools, limit: int) -> None:
    with pytest.raises(ValueError, match="limit must be between"):
        tools.list_documents(limit=limit)


def test_a_filter_on_a_field_chunks_do_not_store_is_an_error(tools: RagTools) -> None:
    with pytest.raises(ValueError, match="Cannot filter on sector"):
        tools.list_documents(filters=QueryFilter(equals={"sector": "airlines"}))


def test_an_unknown_corpus_is_an_error(tools: RagTools) -> None:
    with pytest.raises(ValueError):
        tools.list_documents(corpus="nope")


def test_the_tool_validates_a_raw_filter_and_documents_every_argument(tools: RagTools) -> None:
    spec = next(s for s in build_tool_specs(tools) if s.name == "rag_list_documents")

    payload = spec.handler(filters={"equals": {"ticker": "UAL"}})
    assert spec.handler(limit=1, offset=3)["documents"][0]["document_id"] == "notes.txt"

    assert [d["document_id"] for d in payload["documents"]] == ["UAL_10-Q_2026-06-30.md"]
    for name, prop in spec.input_schema["properties"].items():
        assert prop.get("description"), f"{name} has no description"
    assert spec.input_schema.get("required", []) == []
