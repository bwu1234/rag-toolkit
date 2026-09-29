"""Tests for the BEIR phase-0 records (public benchmarks plan).

Network-free: the inventory runs on a tiny synthetic dataset, and the pinned
reference scores are checked for agreement between the manifests (what
phase 3 will read) and the protocol doc (what people read).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from inventory_beir import inventory  # noqa: E402

DATASETS = {"FiQA-2018": "fiqa", "SciFact": "scifact", "NFCorpus": "nfcorpus"}


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_inventory_reports_edge_cases(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "corpus.jsonl",
        [
            {"_id": "1", "title": "", "text": "first"},
            {"_id": "2", "title": "", "text": ""},
            {"_id": "3", "title": "T", "text": "third"},
        ],
    )
    _write_jsonl(
        tmp_path / "queries.jsonl",
        [{"_id": "1", "text": "q one"}, {"_id": "9", "text": "q nine"}],
    )
    (tmp_path / "qrels").mkdir()
    (tmp_path / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\n1\t2\t1\n1\t3\t2\n9\t1\t0\n", encoding="utf-8"
    )

    result = inventory(tmp_path)

    corpus = result["corpus"]
    assert corpus["documents"] == 3
    assert corpus["empty_title"] == 2
    assert corpus["empty_title_and_text_ids"] == ["2"]
    assert result["queries"]["ids_that_are_doc_ids"] == 1
    test = result["qrels"]["test"]
    assert test["grade_histogram"] == {"0": 1, "1": 1, "2": 1}
    # Query 9 is judged, but only with grade 0: it has no positive.
    assert test["judged_queries_without_positive"] == ["9"]
    assert test["mean_positive_per_query"] == 2.0
    assert test["judgments_on_empty_documents"] == 1
    assert test["judged_queries_that_are_doc_ids"] == 1


def _doc_table() -> dict[str, dict[str, float]]:
    text = (REPO / "docs" / "beir-reference-protocol.md").read_text(encoding="utf-8")
    rows: dict[str, dict[str, float]] = {}
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip("|").split("|")]
        if cells and cells[0] in DATASETS and len(cells) == 6 and re.fullmatch(r"0\.\d{4}", cells[2]):
            bm_ndcg, bm_r, bge_ndcg, bge_r = map(float, cells[2:])
            rows[DATASETS[cells[0]]] = {
                "queries": float(cells[1].replace(",", "")),
                "bm25-flat/nDCG@10": bm_ndcg,
                "bm25-flat/R@100": bm_r,
                "bge-base-en-v1.5.faiss/nDCG@10": bge_ndcg,
                "bge-base-en-v1.5.faiss/R@100": bge_r,
            }
    return rows


@pytest.mark.parametrize("name", sorted(DATASETS.values()))
def test_manifest_matches_protocol_doc(name: str) -> None:
    manifest = json.loads(
        (REPO / "data" / "corpora" / f"beir-{name}" / "manifest.json").read_text(encoding="utf-8")
    )
    expected = _doc_table()[name]
    runs = manifest["reference"]["runs"]
    for run in ("bm25-flat", "bge-base-en-v1.5.faiss"):
        for metric in ("nDCG@10", "R@100"):
            assert runs[run][metric] == expected[f"{run}/{metric}"], (run, metric)
    assert manifest["inventory"]["qrels"]["test"]["judged_queries"] == expected["queries"]
    assert manifest["reference"]["tolerance_abs"] == 0.0005
    assert len(manifest["archive"]["sha256"]) == 64
