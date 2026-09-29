"""Inventory an unpacked BEIR dataset for its provenance manifest.

Phase 0 of docs/public-benchmarks-plan.md: pin what the pinned archive
actually contains before anything is converted, so later phases can check
their loader and converter output against these numbers.

    python scripts/inventory_beir.py path/to/scifact [--zip path/to/scifact.zip]

Reads `corpus.jsonl`, `queries.jsonl` and every `qrels/*.tsv`; prints one JSON
object with counts, the qrels grade histogram per split, and the edge cases
the text contract and scoring protocol have to decide (empty titles/texts,
duplicate ids, query ids that are also document ids, qrels pointing at
missing documents or queries, judged queries with no positive grade). Stdlib
only, read-only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl_ids(path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    ids: list[str] = []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            ids.append(str(row["_id"]))
            rows.append(row)
    return ids, rows


def _duplicates(ids: list[str]) -> list[str]:
    return sorted(i for i, n in Counter(ids).items() if n > 1)


def inventory(root: Path) -> dict[str, Any]:
    doc_ids, docs = _read_jsonl_ids(root / "corpus.jsonl")
    query_ids, queries = _read_jsonl_ids(root / "queries.jsonl")
    doc_set, query_set = set(doc_ids), set(query_ids)

    empty_ids = sorted(
        str(d["_id"])
        for d in docs
        if not d.get("title", "").strip() and not d.get("text", "").strip()
    )
    empty_set = set(empty_ids)
    corpus: dict[str, Any] = {
        "documents": len(doc_ids),
        "duplicate_ids": _duplicates(doc_ids),
        "empty_title": sum(1 for d in docs if not d.get("title", "").strip()),
        "missing_title_key": sum(1 for d in docs if "title" not in d),
        "empty_text": sum(1 for d in docs if not d.get("text", "").strip()),
        "empty_title_and_text": len(empty_ids),
        "empty_title_and_text_ids": empty_ids[:50],
        "text_with_leading_or_trailing_whitespace": sum(
            1 for d in docs if d.get("text", "") != d.get("text", "").strip()
        ),
        "text_with_newline": sum(1 for d in docs if "\n" in d.get("text", "")),
        "max_chars": max((len(d.get("title", "")) + len(d.get("text", "")) for d in docs), default=0),
        "sha256": _sha256(root / "corpus.jsonl"),
    }

    splits: dict[str, Any] = {}
    for qrels_path in sorted((root / "qrels").glob("*.tsv")):
        pairs: list[tuple[str, str]] = []
        grades: Counter[int] = Counter()
        positive_counts: Counter[str] = Counter()
        with qrels_path.open(encoding="utf-8", newline="") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader)
            for row in reader:
                if not row:
                    continue
                qid, did, score = row[0], row[1], int(row[2])
                pairs.append((qid, did))
                grades[score] += 1
                if score > 0:
                    positive_counts[qid] += 1
        judged = {q for q, _ in pairs}
        positive = set(positive_counts)
        splits[qrels_path.stem] = {
            "header": header,
            "judgments": len(pairs),
            "judged_queries": len(judged),
            "queries_with_positive": len(positive),
            "judged_queries_without_positive": sorted(judged - positive)[:50],
            "grade_histogram": {str(g): n for g, n in sorted(grades.items())},
            "mean_positive_per_query": round(
                sum(positive_counts.values()) / len(positive), 3
            ) if positive else 0.0,
            "duplicate_pairs": len(pairs) - len(set(pairs)),
            "qrels_docs_missing_from_corpus": len({d for _, d in pairs} - doc_set),
            "qrels_queries_missing_from_queries": sorted(judged - query_set)[:50],
            "judged_queries_that_are_doc_ids": len(judged & doc_set),
            "self_judgments": sum(1 for q, d in pairs if q == d),
            "judgments_on_empty_documents": sum(1 for _, d in pairs if d in empty_set),
            "sha256": _sha256(qrels_path),
        }

    return {
        "corpus": corpus,
        "queries": {
            "total": len(query_ids),
            "duplicate_ids": _duplicates(query_ids),
            "empty_text": sum(1 for q in queries if not q.get("text", "").strip()),
            "ids_that_are_doc_ids": len(query_set & doc_set),
            "unjudged_in_any_split": len(
                query_set - set().union(*(_judged(root / "qrels" / f"{s}.tsv") for s in splits))
            ),
            "sha256": _sha256(root / "queries.jsonl"),
        },
        "qrels": splits,
    }


def _judged(path: Path) -> set[str]:
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        return {row[0] for row in reader if row}


def main() -> None:
    parser = argparse.ArgumentParser(description="Inventory an unpacked BEIR dataset.")
    parser.add_argument("root", type=Path, help="unpacked dataset directory")
    parser.add_argument("--zip", type=Path, help="archive to hash alongside")
    args = parser.parse_args()
    result = inventory(args.root)
    if args.zip:
        result = {"archive_sha256": _sha256(args.zip), **result}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
