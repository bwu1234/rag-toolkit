"""`rag_read_document` and `rag_find`, over a corpus on disk.

Hermetic like `test_list_documents.py`: a temporary corpus, no index and no
embedder. Both tools read the cleaned text the chunker splits, so the tests
pin that offsets from the chunker index into what `rag_read_document` serves.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from rag.config.settings import ChunkingConfig, CorporaConfig, CorpusConfig, RagConfig
from rag.ingestion.corpora import chunk_selected_corpora
from rag.query_filter import QueryFilter
from rag.tools import (
    DEFAULT_READ_CHARS,
    FIND_CONTEXT_CHARS,
    MAX_FIND_RESULTS,
    MAX_READ_CHARS,
    RagTools,
    StaleSourceError,
    build_tool_specs,
)

_DAL_BODY = (
    "Delta's fuel expense fell.\n\n| Item | 2026 |\n|---|---|\n| Aircraft fuel and\nrelated taxes | 84.1% |\n\n"
    + "Filler sentence about operations. " * 400
    + "\n\nEnflonsia is not mentioned by airlines, but AIRCRAFT FUEL is, twice."
)


def _filing(ticker: str, period_end: str, body: str) -> str:
    return (
        f"---\ncompany: {ticker} Inc.\nticker: {ticker}\nform: 10-Q\n"
        f"period_end: {period_end}\nfiled: {period_end}\n---\n# {ticker}\n\n{body}\n"
    )


@pytest.fixture()
def corpus(tmp_path: Path) -> Path:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "DAL_10-Q_2026-06-30.md").write_text(_filing("DAL", "2026-06-30", _DAL_BODY))
    (docs / "UAL_10-Q_2026-06-30.md").write_text(_filing("UAL", "2026-06-30", "United bought aircraft fuel."))
    return docs


@pytest.fixture()
def config(corpus: Path) -> RagConfig:
    return RagConfig(
        corpora=CorporaConfig(active=["c"], registry={"c": CorpusConfig(documents_dir=corpus)}),
        chunking=ChunkingConfig(carry_metadata=["title", "company", "ticker", "form", "period_end"]),
    )


@pytest.fixture()
def tools(config: RagConfig) -> RagTools:
    return RagTools(config=config)


# --------------------------------------------------------------------------
# rag_read_document
# --------------------------------------------------------------------------


def test_reads_a_default_window_and_says_where_to_continue(tools: RagTools) -> None:
    payload = tools.read_document("DAL_10-Q_2026-06-30.md")

    assert (payload["start"], payload["end"]) == (0, DEFAULT_READ_CHARS)
    assert payload["length"] > DEFAULT_READ_CHARS
    assert payload["next_start"] == DEFAULT_READ_CHARS
    assert len(payload["text"]) == DEFAULT_READ_CHARS
    assert (payload["ticker"], payload["period_end"]) == ("DAL", "2026-06-30")


def test_consecutive_windows_rebuild_the_document_exactly(tools: RagTools) -> None:
    text, start = "", 0
    while True:
        payload = tools.read_document("DAL_10-Q_2026-06-30.md", start=start, max_chars=4000)
        text += payload["text"]
        if "next_start" not in payload:
            break
        start = payload["next_start"]

    assert len(text) == payload["length"] == payload["end"]
    assert text == tools.read_document("DAL_10-Q_2026-06-30.md", max_chars=MAX_READ_CHARS)["text"]


@pytest.mark.parametrize("strategy", ["fixed", "structured"])
def test_chunk_offsets_index_into_the_text_it_serves(config: RagConfig, strategy: str) -> None:
    """A search hit's `char_start`/`char_end` are only useful if they point here."""

    config = config.model_copy(update={"chunking": config.chunking.model_copy(update={"strategy": strategy})})
    tools = RagTools(config=config)
    _, _, chunks = chunk_selected_corpora(config, None)
    assert chunks
    for chunk in chunks:
        start, end = chunk.metadata["char_start"], chunk.metadata["char_end"]
        window = tools.read_document(chunk.document_id, start=start, max_chars=end - start)
        # The fixed chunker strips its spans; the structured one puts a split
        # table's header rows in front of each later piece.
        assert window["text"].strip() in chunk.text


def test_an_unknown_document_names_near_misses(tools: RagTools) -> None:
    with pytest.raises(ValueError, match=r"Did you mean: DAL_10-Q_2026-06-30\.md.*rag_list_documents"):
        tools.read_document("DAL_10Q_2026-06-30.md")


@pytest.mark.parametrize("kwargs", [{"start": -1}, {"start": 10**6}, {"max_chars": 0}, {"max_chars": MAX_READ_CHARS + 1}])
def test_out_of_range_windows_are_argument_errors(tools: RagTools, kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError, match="must be between"):
        tools.read_document("UAL_10-Q_2026-06-30.md", **kwargs)


def test_an_edited_corpus_is_reloaded_not_served_stale(tools: RagTools, corpus: Path) -> None:
    path = corpus / "UAL_10-Q_2026-06-30.md"
    assert "bought" in tools.read_document(path.name)["text"]

    path.write_text(_filing("UAL", "2026-06-30", "United sold its hedges."))
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))  # same-second writes

    assert "sold its hedges" in tools.read_document(path.name)["text"]


def test_the_version_names_the_text_and_changes_with_it(tools: RagTools, corpus: Path) -> None:
    path = corpus / "UAL_10-Q_2026-06-30.md"
    before = tools.read_document(path.name)
    assert before["version"] == tools.read_document(path.name, start=3)["version"]
    assert before["version"] == tools.find("United")["matches"][0]["version"]
    assert before["source"].endswith("UAL_10-Q_2026-06-30.md")

    path.write_text(_filing("UAL", "2026-06-30", "United sold its hedges."))
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))

    assert tools.read_document(path.name)["version"] != before["version"]


def test_a_document_outside_the_scope_reads_as_unknown(tools: RagTools) -> None:
    dal = QueryFilter(equals={"ticker": "DAL"})

    assert tools.read_document("DAL_10-Q_2026-06-30.md", scope=dal)["ticker"] == "DAL"
    with pytest.raises(ValueError) as outside:
        tools.read_document("UAL_10-Q_2026-06-30.md", scope=dal)
    with pytest.raises(ValueError) as absent:
        tools.read_document("AAL_10-Q_2026-06-30.md", scope=dal)
    # The same error as an id that doesn't exist, near misses drawn only from inside the scope.
    assert str(outside.value).replace("UAL", "AAL") == str(absent.value)
    assert "Did you mean: DAL_10-Q_2026-06-30.md?" in str(outside.value)


def test_spans_the_text_still_holds_read_and_changed_ones_are_stale(config: RagConfig, corpus: Path) -> None:
    tools = RagTools(config=config)
    _, _, chunks = chunk_selected_corpora(config, None)
    ual = [c for c in chunks if c.document_id == "UAL_10-Q_2026-06-30.md"]
    spans = [(c.metadata["char_start"], c.metadata["char_end"], c.text) for c in ual]

    assert tools.read_document("UAL_10-Q_2026-06-30.md", expect_spans=spans)["start"] == 0

    path = corpus / "UAL_10-Q_2026-06-30.md"
    path.write_text(_filing("UAL", "2026-06-30", "Delta never bought anything."))
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))

    with pytest.raises(StaleSourceError, match="changed since the search index was built"):
        tools.read_document("UAL_10-Q_2026-06-30.md", expect_spans=spans)


def test_documents_are_loaded_once_per_selection(tools: RagTools, monkeypatch: pytest.MonkeyPatch) -> None:
    import rag.ingestion.corpora as corpora

    loads: list[object] = []
    real = corpora.load_selected_corpora
    monkeypatch.setattr(corpora, "load_selected_corpora", lambda *a, **k: loads.append(a) or real(*a, **k))

    tools.read_document("UAL_10-Q_2026-06-30.md")
    tools.find("fuel")

    assert len(loads) == 1


# --------------------------------------------------------------------------
# rag_find
# --------------------------------------------------------------------------


def test_finds_case_insensitively_across_line_breaks_in_document_order(tools: RagTools) -> None:
    payload = tools.find("aircraft fuel")

    assert (payload["total_matches"], payload["documents_matched"], payload["documents_searched"]) == (3, 2, 2)
    matches = payload["matches"]
    assert [m["document_id"] for m in matches] == ["DAL_10-Q_2026-06-30.md"] * 2 + ["UAL_10-Q_2026-06-30.md"]
    assert [m["match"] for m in matches[:2]] == ["Aircraft fuel", "AIRCRAFT FUEL"]
    assert matches[0]["start"] < matches[1]["start"]
    assert "hint" not in payload


def test_match_offsets_read_back_as_the_match(tools: RagTools) -> None:
    for match in tools.find("aircraft fuel")["matches"]:
        window = tools.read_document(match["document_id"], start=match["start"], max_chars=match["end"] - match["start"])
        assert window["text"] == match["match"]


def test_context_is_bounded_single_line_and_marks_elisions(tools: RagTools) -> None:
    match = tools.find("Enflonsia")["matches"][0]

    assert "\n" not in match["context"]
    assert match["context"].startswith("...") and not match["context"].endswith("...")
    assert len(match["context"]) <= 2 * FIND_CONTEXT_CHARS + len("Enflonsia") + 3


def test_regex_characters_are_literal(tools: RagTools) -> None:
    assert tools.find("84.1%")["total_matches"] == 1
    assert tools.find("84.1.")["total_matches"] == 0


def test_no_match_says_what_to_try(tools: RagTools) -> None:
    payload = tools.find("jet fuel hedging")

    assert payload["total_matches"] == 0
    assert "rag_search" in payload["hint"]


def test_scoping_by_document_or_filter(tools: RagTools) -> None:
    assert tools.find("aircraft fuel", document_id="UAL_10-Q_2026-06-30.md")["total_matches"] == 1
    assert tools.find("aircraft fuel", filters=QueryFilter(equals={"ticker": "DAL"}))["total_matches"] == 2
    with pytest.raises(ValueError, match="No document"):
        tools.find("fuel", document_id="nope.md")
    with pytest.raises(ValueError, match="Cannot filter on sector"):
        tools.find("fuel", filters=QueryFilter(equals={"sector": "airlines"}))


def test_max_results_caps_matches_but_total_counts_all(tools: RagTools) -> None:
    payload = tools.find("filler", max_results=5)

    assert (payload["returned"], payload["total_matches"]) == (5, 400)
    assert "first 5 of 400" in payload["hint"]


@pytest.mark.parametrize("kwargs", [{"phrase": "  "}, {"phrase": "x", "max_results": MAX_FIND_RESULTS + 1}])
def test_bad_find_arguments_are_errors(tools: RagTools, kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        tools.find(**kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# As tools
# --------------------------------------------------------------------------


def test_handlers_validate_raw_arguments_and_document_every_argument(tools: RagTools) -> None:
    specs = {spec.name: spec for spec in build_tool_specs(tools)}
    read, find = specs["rag_read_document"], specs["rag_find"]

    whole = read.handler(document_id="UAL_10-Q_2026-06-30.md")["text"]
    assert read.handler(document_id="UAL_10-Q_2026-06-30.md", start=2, max_chars=5)["text"] == whole[2:7]
    assert find.handler(phrase="aircraft fuel", filters={"equals": {"ticker": "UAL"}})["total_matches"] == 1
    assert read.input_schema["required"] == ["document_id"]
    assert find.input_schema["required"] == ["phrase"]
    for spec in (read, find):
        for name, prop in spec.input_schema["properties"].items():
            assert prop.get("description"), f"{spec.name}.{name} has no description"
