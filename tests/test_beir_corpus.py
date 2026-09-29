"""Tests for BEIR corpora as named corpora (public benchmarks plan, phase 1).

Network-free: the fetcher's extraction and the converter run on a tiny
synthetic dataset with its own manifest, built in ``tmp_path``.
"""

from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from beir_to_eval_set import ConversionError, convert  # noqa: E402
from fetch_beir import FetchError, expected_files, extract  # noqa: E402

from rag.chunking.chunkers import WholeDocumentChunker, get_chunker  # noqa: E402
from rag.config.settings import ChunkingConfig, CorporaConfig, CorpusConfig, RagConfig  # noqa: E402
from rag.eval.dataset import MODE_QRELS, EvalDataset, EvalSample  # noqa: E402
from rag.eval.relevance import judge_ranking  # noqa: E402
from rag.ingestion.corpora import chunk_selected_corpora  # noqa: E402
from rag.ingestion.loaders import BeirCorpusLoader, beir_passage_text, load_corpus  # noqa: E402
from rag.ingestion.models import Document  # noqa: E402

CORPUS = [
    {"_id": "d1", "title": "Statins", "text": "Statins lower  LDL.", "metadata": {"url": "x"}},
    {"_id": "d2", "title": "", "text": "No title here."},
    {"_id": "d3", "title": "", "text": ""},
]
QUERIES = [{"_id": "q1", "text": "Do statins work?"}, {"_id": "q2", "text": "Untitled?"}]
QRELS = "query-id\tcorpus-id\tscore\nq1\td1\t2\nq1\td2\t0\nq2\td2\t1\n"


def _jsonl(rows: list[dict]) -> str:
    return "".join(json.dumps(r) + "\n" for r in rows)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Loader and text contract
# ---------------------------------------------------------------------------


def test_passage_text_joins_title_and_body_and_drops_an_empty_title() -> None:
    assert beir_passage_text("Statins", "Lower LDL.") == "Statins Lower LDL."
    assert beir_passage_text("", "Lower LDL.") == "Lower LDL."


def test_loader_keeps_ids_verbatim_and_puts_the_title_in_the_text(tmp_path: Path) -> None:
    path = tmp_path / "corpus.jsonl"
    path.write_text(_jsonl(CORPUS), encoding="utf-8")

    docs = BeirCorpusLoader().load(path, corpus_root=tmp_path)

    assert [d.id for d in docs] == ["d1", "d2", "d3"]
    assert docs[0].text == "Statins Statins lower  LDL."
    assert docs[0].metadata == {"title": "Statins"}
    assert docs[1].text == "No title here."
    # The empty passage is still loaded; the chunker decides what to index.
    assert docs[2].text == ""
    assert {d.doc_type for d in docs} == {"beir"}


@pytest.mark.parametrize(
    "rows, message",
    [
        ([{"_id": "a", "text": "x"}, {"_id": "a", "text": "y"}], "duplicate _id"),
        ([{"text": "no id"}], "needs '_id' and 'text'"),
    ],
)
def test_loader_refuses_malformed_passages(tmp_path: Path, rows: list[dict], message: str) -> None:
    path = tmp_path / "corpus.jsonl"
    path.write_text(_jsonl(rows), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        BeirCorpusLoader().load(path, corpus_root=tmp_path)


def test_load_corpus_dispatches_jsonl_to_the_beir_loader(tmp_path: Path) -> None:
    (tmp_path / "corpus.jsonl").write_text(_jsonl(CORPUS[:2]), encoding="utf-8")

    assert [d.id for d in load_corpus(tmp_path)] == ["d1", "d2"]


# ---------------------------------------------------------------------------
# Identity chunker and the cleaning bypass
# ---------------------------------------------------------------------------


def _doc(doc_id: str, text: str) -> Document:
    return Document(id=doc_id, text=text, source=Path("/c/corpus.jsonl"), doc_type="beir", metadata={"title": "T"})


def test_whole_document_chunker_keeps_each_document_as_one_unaltered_chunk() -> None:
    long_text = "  word " * 1000  # far past any fixed chunk size, with edge whitespace
    chunks = WholeDocumentChunker(carry_metadata=["title"]).chunk([_doc("a", long_text), _doc("b", "short")])

    assert [c.id for c in chunks] == ["a::chunk0", "b::chunk0"]
    assert chunks[0].text == long_text
    assert chunks[0].metadata == {"title": "T", "chunk_index": 0, "char_start": 0, "char_end": len(long_text)}


def test_whole_document_chunker_skips_empty_documents() -> None:
    chunks = WholeDocumentChunker().chunk([_doc("a", ""), _doc("b", "   "), _doc("c", "x")])

    assert [c.document_id for c in chunks] == ["c"]


def test_get_chunker_selects_whole_document_for_strategy_none() -> None:
    assert isinstance(get_chunker(ChunkingConfig(strategy="none")), WholeDocumentChunker)


def _bench_config(tmp_path: Path, *, clean: bool) -> RagConfig:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "corpus.jsonl").write_text(_jsonl([{"_id": "p", "title": "", "text": "exam-\nple  text"}]), encoding="utf-8")
    return RagConfig(
        corpora=CorporaConfig(active=["bench"], registry={"bench": CorpusConfig(documents_dir=raw, clean=clean)}),
        chunking=ChunkingConfig(strategy="none"),
    )


def test_corpus_with_clean_false_is_indexed_exactly_as_loaded(tmp_path: Path) -> None:
    _selection, documents, chunks = chunk_selected_corpora(_bench_config(tmp_path, clean=False), ["bench"])

    assert documents[0].text == "exam-\nple  text"
    assert chunks[0].text == "exam-\nple  text"


def test_corpus_with_clean_true_is_still_cleaned(tmp_path: Path) -> None:
    _selection, documents, _chunks = chunk_selected_corpora(_bench_config(tmp_path, clean=True), ["bench"])

    assert documents[0].text == "example text"


# ---------------------------------------------------------------------------
# Graded document labels and the qrels mode
# ---------------------------------------------------------------------------


def _qrels_sample(**overrides: object) -> dict:
    return {
        "id": "q1",
        "query": "Do statins work?",
        "expected_doc_ids": ["d9", {"id": "d1", "grade": 2}, {"id": "d2", "grade": 0}],
        "matching_mode": MODE_QRELS,
        **overrides,
    }


def test_graded_doc_ids_credit_positive_grades_and_keep_zero_grades() -> None:
    sample = EvalSample.from_dict(_qrels_sample())

    assert sample.expected_doc_ids == ["d9", "d1"]
    assert (sample.doc_grade("d9"), sample.doc_grade("d1"), sample.doc_grade("d2")) == (1, 2, 0)
    assert sample.doc_grade("unjudged") == 0
    assert sample.matching_mode == MODE_QRELS


def test_graded_doc_ids_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "set.json"
    EvalDataset.from_dicts([_qrels_sample(benchmark="beir-x/test")]).save(path)

    reloaded = json.loads(path.read_text(encoding="utf-8"))[0]
    assert reloaded["expected_doc_ids"] == ["d9", {"id": "d1", "grade": 2}, {"id": "d2", "grade": 0}]
    assert reloaded["benchmark"] == "beir-x/test"


def test_legacy_string_doc_ids_are_unchanged() -> None:
    sample = EvalSample.from_dict({"id": "s", "query": "q", "expected_doc_ids": ["a.md", "b.md"]})

    assert sample.expected_doc_ids == ["a.md", "b.md"]
    assert sample.doc_grades == {}
    assert sample.to_dict()["expected_doc_ids"] == ["a.md", "b.md"]


@pytest.mark.parametrize(
    "doc_ids, message",
    [
        (["d1", {"id": "d1", "grade": 2}], "listed twice"),
        ([{"id": "d1", "grade": -1}], "negative grade"),
        ([{"grade": 1}], "'id' key"),
    ],
)
def test_malformed_doc_labels_are_refused(doc_ids: list, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        EvalSample.from_dict(_qrels_sample(expected_doc_ids=doc_ids))


def test_a_qrels_sample_may_hold_only_non_relevant_judgments() -> None:
    """trec_eval -c counts such a query at 0, so the schema must be able to hold it."""
    sample = EvalSample.from_dict(_qrels_sample(expected_doc_ids=[{"id": "d1", "grade": 0}]))

    assert sample.expected_doc_ids == [] and sample.doc_grades == {"d1": 0}


def test_a_qrels_sample_needs_some_judgment() -> None:
    with pytest.raises(ValueError, match="needs judged expected_doc_ids"):
        EvalSample.from_dict(_qrels_sample(expected_doc_ids=[]))


def test_a_document_mode_sample_still_needs_a_credited_document() -> None:
    with pytest.raises(ValueError, match="needs expected_doc_ids"):
        EvalSample.from_dict(_qrels_sample(expected_doc_ids=[{"id": "d1", "grade": 0}], matching_mode="document"))


def test_qrels_samples_are_refused_by_chunk_level_judging() -> None:
    with pytest.raises(NotImplementedError, match="rag.eval.qrels"):
        judge_ranking(EvalSample.from_dict(_qrels_sample()), [])


# ---------------------------------------------------------------------------
# Fetcher extraction and the converter, on a synthetic pinned dataset
# ---------------------------------------------------------------------------


def _files() -> dict[str, str]:
    return {
        "toy/corpus.jsonl": _jsonl(CORPUS),
        "toy/queries.jsonl": _jsonl(QUERIES),
        "toy/qrels/test.tsv": QRELS,
    }


def _manifest(files: dict[str, str], **qrels_counts: int) -> dict:
    return {
        "name": "beir-toy",
        "inventory": {
            "corpus": {"sha256": _sha(files["toy/corpus.jsonl"])},
            "queries": {"sha256": _sha(files["toy/queries.jsonl"])},
            "qrels": {
                "test": {
                    "sha256": _sha(files["toy/qrels/test.tsv"]),
                    "judged_queries": qrels_counts.get("judged_queries", 2),
                    "judgments": qrels_counts.get("judgments", 3),
                }
            },
        },
    }


def _zip(tmp_path: Path, files: dict[str, str]) -> Path:
    archive = tmp_path / "toy.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for name, text in files.items():
            zf.writestr(name, text)
        zf.writestr("__MACOSX/._corpus.jsonl", "junk")
    return archive


def _fetched(tmp_path: Path, **qrels_counts: int) -> Path:
    files = _files()
    root = tmp_path / "beir-toy"
    root.mkdir()
    manifest = _manifest(files, **qrels_counts)
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    extract(_zip(tmp_path, files), expected_files(manifest, root))
    return root


def test_extract_lays_out_passages_apart_from_queries_and_qrels(tmp_path: Path) -> None:
    root = _fetched(tmp_path)

    assert sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()) == [
        "documents/corpus.jsonl",
        "manifest.json",
        "source/qrels/test.tsv",
        "source/queries.jsonl",
    ]


def test_extract_refuses_a_file_that_does_not_match_the_manifest(tmp_path: Path) -> None:
    files = _files()
    manifest = _manifest(files)
    tampered = {**files, "toy/qrels/test.tsv": QRELS.replace("\t2\n", "\t1\n")}

    with pytest.raises(FetchError, match="does not match"):
        extract(_zip(tmp_path, tampered), expected_files(manifest, tmp_path / "out"))
    assert not (tmp_path / "out" / "source" / "qrels" / "test.tsv").exists()


def test_extract_refuses_an_archive_missing_a_pinned_file(tmp_path: Path) -> None:
    files = _files()
    manifest = _manifest(files)
    del files["toy/queries.jsonl"]

    with pytest.raises(FetchError, match="lacks toy/queries.jsonl"):
        extract(_zip(tmp_path, files), expected_files(manifest, tmp_path / "out"))


def test_convert_writes_one_graded_qrels_sample_per_judged_query(tmp_path: Path) -> None:
    dataset = convert(_fetched(tmp_path), "test")

    first, second = dataset.samples
    assert (first.id, first.query) == ("q1", "Do statins work?")
    assert first.to_dict()["expected_doc_ids"] == [{"id": "d1", "grade": 2}, {"id": "d2", "grade": 0}]
    assert first.matching_mode == MODE_QRELS
    assert first.extra == {"benchmark": "beir-toy/test"}
    assert second.expected_doc_ids == ["d2"]


def test_convert_refuses_counts_that_disagree_with_the_inventory(tmp_path: Path) -> None:
    with pytest.raises(ConversionError, match="manifest inventory"):
        convert(_fetched(tmp_path, judgments=4), "test")


def test_convert_refuses_an_unknown_split(tmp_path: Path) -> None:
    with pytest.raises(ConversionError, match="no 'dev' split"):
        convert(_fetched(tmp_path), "dev")


@pytest.mark.parametrize(
    "qrels, message",
    [
        (QRELS + "q1\td1\t1\n", "twice"),
        (QRELS + "q1\tmissing\t1\n", "not in the corpus"),
        (QRELS + "q404\td1\t1\n", "no text"),
        (QRELS + "q2\td1\thigh\n", "not an integer"),
        ("qid\tdid\tscore\n", "header"),
    ],
)
def test_convert_refuses_malformed_qrels(tmp_path: Path, qrels: str, message: str) -> None:
    root = _fetched(tmp_path)
    path = root / "source" / "qrels" / "test.tsv"
    path.write_text(qrels, encoding="utf-8")
    # Re-pin, so the check under test is the validation, not the hash.
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest["inventory"]["qrels"]["test"]["sha256"] = _sha(qrels)
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ConversionError, match=message):
        convert(root, "test")


def test_convert_refuses_unpinned_input(tmp_path: Path) -> None:
    root = _fetched(tmp_path)
    (root / "source" / "queries.jsonl").write_text(_jsonl(QUERIES[:1]), encoding="utf-8")

    with pytest.raises(ConversionError, match="does not match the sha256"):
        convert(root, "test")


def test_convert_keeps_a_query_judged_only_non_relevant(tmp_path: Path) -> None:
    qrels = QRELS + "q2\td1\t0\n"
    root = _fetched(tmp_path, judgments=4)
    (root / "source" / "qrels" / "test.tsv").write_text(qrels.replace("q2\td2\t1\n", ""), encoding="utf-8")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest["inventory"]["qrels"]["test"].update(sha256=_sha(qrels.replace("q2\td2\t1\n", "")), judgments=3)
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    second = convert(root, "test").samples[1]

    assert (second.id, second.expected_doc_ids, second.doc_grades) == ("q2", [], {"d1": 0})
