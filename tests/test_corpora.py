"""Tests for the multi-corpus registry (`rag.config.settings.CorpusSelection`).

The registry exists to let the same corpora be indexed **isolated** (each on its
own) and **pooled** (several into one collection), because the gap between those
two numbers is the cross-corpus interference cost -- how much a corpus of
plausible-but-wrong neighbours degrades retrieval.  That comparison is only
meaningful if the two indexes coexist rather than overwriting each other, so the
property under test is that a selection deterministically names its own storage.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.ingestion.corpora import load_selected_corpora
from rag.config.settings import CorporaConfig, CorpusConfig, RagConfig, VectorStoreConfig
from rag.retrieval.sparse import BM25_INDEX_FILENAME, bm25_index_path


def _config(**corpora: str) -> RagConfig:
    return RagConfig(
        corpora=CorporaConfig(
            active=sorted(corpora),
            registry={name: CorpusConfig(documents_dir=Path(d)) for name, d in corpora.items()},
        ),
        vector_store=VectorStoreConfig(provider="chroma", collection_name="rag_corpus"),
    )


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def test_no_registry_falls_back_to_paths_corpus_dir() -> None:
    """A config predating the registry must keep working unchanged."""
    selection = RagConfig().corpus_selection()

    assert selection.names == ("default",)
    assert selection.document_dirs[0].name == "documents"
    assert not selection.is_pooled


def test_active_selects_from_the_registry() -> None:
    config = _config(
        baseline="data/corpora/baseline/documents", edgar="data/corpora/edgar/documents"
    )
    config.corpora.active = ["edgar"]

    selection = config.corpus_selection()

    assert selection.names == ("edgar",)
    assert selection.document_dirs[0].name == "documents"


def test_explicit_names_override_active() -> None:
    config = _config(baseline="data/corpora/baseline/documents", edgar="data/edgar")
    config.corpora.active = ["baseline"]

    assert config.corpus_selection(["edgar"]).names == ("edgar",)


def test_unknown_corpus_name_raises_rather_than_indexing_nothing() -> None:
    """A typo must not silently produce an empty index and an all-zero eval run."""
    config = _config(baseline="data/corpora/baseline/documents")

    with pytest.raises(ValueError, match="Unknown corpus name"):
        config.corpus_selection(["edgr"])


def test_unknown_name_error_lists_what_is_available() -> None:
    config = _config(baseline="data/corpora/baseline/documents", edgar="data/edgar")

    with pytest.raises(ValueError, match="baseline, edgar"):
        config.corpus_selection(["nope"])


def test_empty_registry_with_active_names_still_resolves_the_implicit_corpus() -> None:
    config = RagConfig(corpora=CorporaConfig(active=[], registry={}))
    assert config.corpus_selection().names == ("default",)


# ---------------------------------------------------------------------------
# Naming: isolated and pooled indexes must not collide
# ---------------------------------------------------------------------------


def test_isolated_and_pooled_selections_get_different_collections() -> None:
    config = _config(baseline="data/corpora/baseline/documents", edgar="data/edgar")

    isolated = config.corpus_selection(["edgar"]).collection_name
    pooled = config.corpus_selection(["baseline", "edgar"]).collection_name

    assert isolated != pooled, "pooling must not overwrite the isolated index"
    assert isolated == "rag_corpus__edgar"
    assert pooled == "rag_corpus__baseline+edgar"


def test_isolated_and_pooled_selections_get_different_bm25_files() -> None:
    config = _config(baseline="data/corpora/baseline/documents", edgar="data/edgar")

    isolated = config.corpus_selection(["edgar"])
    pooled = config.corpus_selection(["baseline", "edgar"])

    assert bm25_index_path(isolated.index_dir, isolated.slug) != bm25_index_path(
        pooled.index_dir, pooled.slug
    )


def test_selection_order_does_not_change_the_index_name() -> None:
    """--corpus a --corpus b and --corpus b --corpus a are the same index."""
    config = _config(baseline="data/corpora/baseline/documents", edgar="data/edgar")

    assert (
        config.corpus_selection(["edgar", "baseline"]).collection_name
        == config.corpus_selection(["baseline", "edgar"]).collection_name
    )


def test_repeated_name_is_deduped() -> None:
    config = _config(baseline="data/corpora/baseline/documents")
    selection = config.corpus_selection(["baseline", "baseline"])

    assert selection.names == ("baseline",)
    assert not selection.is_pooled


def test_pooled_flag_and_description() -> None:
    config = _config(baseline="data/corpora/baseline/documents", edgar="data/edgar")

    assert not config.corpus_selection(["edgar"]).is_pooled
    assert "isolated" in config.corpus_selection(["edgar"]).describe()

    pooled = config.corpus_selection(["baseline", "edgar"])
    assert pooled.is_pooled
    assert "pooled" in pooled.describe()


def test_bm25_path_without_a_slug_keeps_the_unqualified_filename() -> None:
    assert bm25_index_path(Path("/idx")).name == BM25_INDEX_FILENAME
    assert bm25_index_path(Path("/idx"), "edgar").name == "bm25_index__edgar.json"


def test_document_dirs_are_absolute() -> None:
    config = _config(edgar="data/corpora/edgar/documents")
    assert all(d.is_absolute() for d in config.corpus_selection().document_dirs)


# ---------------------------------------------------------------------------
# Pooled loading: id collisions must be loud, not silent
# ---------------------------------------------------------------------------


def _corpus_at(root: Path, name: str, filenames: list[str]) -> Path:
    directory = root / name
    directory.mkdir(parents=True)
    for filename in filenames:
        (directory / filename).write_text(f"# {filename}\n\nContent of {name}/{filename}.\n")
    return directory


def _pooled_config(tmp_path: Path, **dirs: Path) -> RagConfig:
    return RagConfig(
        corpora=CorporaConfig(
            active=sorted(dirs),
            registry={name: CorpusConfig(documents_dir=d) for name, d in dirs.items()},
        )
    )


def test_pooling_distinct_documents_loads_all_of_them(tmp_path: Path) -> None:
    config = _pooled_config(
        tmp_path,
        alpha=_corpus_at(tmp_path, "alpha", ["a.md"]),
        beta=_corpus_at(tmp_path, "beta", ["b.md"]),
    )

    selection, documents = load_selected_corpora(config, ["alpha", "beta"])

    assert selection.is_pooled
    assert sorted(d.id for d in documents) == ["a.md", "b.md"]


def test_pooling_corpora_that_share_a_filename_raises(tmp_path: Path) -> None:
    """Both would produce Document.id 'faq.md', so chunk ids would collide and
    the vector store would silently upsert one over the other."""
    config = _pooled_config(
        tmp_path,
        alpha=_corpus_at(tmp_path, "alpha", ["faq.md"]),
        beta=_corpus_at(tmp_path, "beta", ["faq.md"]),
    )

    with pytest.raises(ValueError, match="appears in both corpus"):
        load_selected_corpora(config, ["alpha", "beta"])


def test_same_filename_is_fine_when_corpora_are_used_in_isolation(tmp_path: Path) -> None:
    config = _pooled_config(
        tmp_path,
        alpha=_corpus_at(tmp_path, "alpha", ["faq.md"]),
        beta=_corpus_at(tmp_path, "beta", ["faq.md"]),
    )

    assert len(load_selected_corpora(config, ["alpha"])[1]) == 1
    assert len(load_selected_corpora(config, ["beta"])[1]) == 1
