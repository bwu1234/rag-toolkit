#!/usr/bin/env python
"""Convert a fetched BEIR split's qrels into an eval set.

Public benchmarks plan, phase 1. Reads ``source/queries.jsonl`` and
``source/qrels/<split>.tsv`` (written by ``scripts/fetch_beir.py``) and writes
``data/eval/beir_<name>_<split>.json``: one sample per judged query, with

* ``id`` -- the BEIR query id, verbatim (so the reference ``--remove-query``
  rule, "drop a hit whose docid equals the query id", can be applied later);
* ``query`` -- the query text, verbatim;
* ``expected_doc_ids`` -- every judged document with its original grade, in
  the graded form ``{"id": ..., "grade": 2}`` (bare string for grade 1).
  Grade-0 judgments are kept and give no credit;
* ``matching_mode: "qrels"`` -- scored the ``trec_eval`` way in phase 2, and
  refused by the runner until then.

The input files must match the hashes pinned in the manifest, and malformed
data fails the run rather than being dropped: duplicate query ids or
judgments, a qrels query with no text, a qrels document not in the corpus, a
non-integer or negative grade. A query judged only non-relevant is kept (it
scores 0 and counts, as under ``trec_eval -c``). The output is gitignored and regenerated from
the pinned archive and this script.

Usage::

    python scripts/beir_to_eval_set.py scifact test
    python scripts/beir_to_eval_set.py nfcorpus dev --out /tmp/nf_dev.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from fetch_beir import dataset_name, sha256_of  # noqa: E402

from rag.eval.dataset import MODE_QRELS, EvalDataset, EvalSample  # noqa: E402
from rag.logging_config import configure_logging  # noqa: E402

logger = logging.getLogger(__name__)

QRELS_HEADER = ["query-id", "corpus-id", "score"]


class ConversionError(ValueError):
    """The input is missing, unpinned or malformed."""


def _verified(path: Path, pinned: str) -> Path:
    if not path.exists():
        raise ConversionError(f"{path} is missing; run scripts/fetch_beir.py first")
    if sha256_of(path) != pinned:
        raise ConversionError(f"{path} does not match the sha256 pinned in the manifest")
    return path


def read_queries(path: Path) -> dict[str, str]:
    queries: dict[str, str] = {}
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "_id" not in row or "text" not in row:
                raise ConversionError(f"{path.name}:{n}: a query needs '_id' and 'text'")
            qid = str(row["_id"])
            if qid in queries:
                raise ConversionError(f"{path.name}:{n}: duplicate query id {qid!r}")
            queries[qid] = str(row["text"])
    return queries


def read_corpus_ids(path: Path) -> set[str]:
    with path.open(encoding="utf-8") as f:
        return {str(json.loads(line)["_id"]) for line in f if line.strip()}


def read_qrels(path: Path) -> dict[str, dict[str, int]]:
    """query id -> {document id: grade}, in file order."""
    qrels: dict[str, dict[str, int]] = {}
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if header != QRELS_HEADER:
            raise ConversionError(f"{path.name}: header {header!r}, expected {QRELS_HEADER!r}")
        for n, row in enumerate(reader, start=2):
            if not row:
                continue
            if len(row) != 3:
                raise ConversionError(f"{path.name}:{n}: expected 3 columns, got {len(row)}")
            qid, doc_id, raw_grade = row
            try:
                grade = int(raw_grade)
            except ValueError:
                raise ConversionError(f"{path.name}:{n}: grade {raw_grade!r} is not an integer") from None
            if grade < 0:
                raise ConversionError(f"{path.name}:{n}: negative grade {grade}")
            judged = qrels.setdefault(qid, {})
            if doc_id in judged:
                raise ConversionError(f"{path.name}:{n}: query {qid!r} judges document {doc_id!r} twice")
            judged[doc_id] = grade
    return qrels


def convert(root: Path, split: str) -> EvalDataset:
    """Build the eval set for ``split`` of the BEIR corpus at ``root``."""
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    inventory = manifest["inventory"]
    if split not in inventory["qrels"]:
        raise ConversionError(
            f"{manifest['name']} has no {split!r} split; it has {', '.join(inventory['qrels'])}"
        )
    queries = read_queries(_verified(root / "source" / "queries.jsonl", inventory["queries"]["sha256"]))
    corpus_ids = read_corpus_ids(_verified(root / "documents" / "corpus.jsonl", inventory["corpus"]["sha256"]))
    qrels = read_qrels(_verified(root / "source" / "qrels" / f"{split}.tsv", inventory["qrels"][split]["sha256"]))

    unknown_queries = sorted(set(qrels) - set(queries))
    if unknown_queries:
        raise ConversionError(f"qrels name {len(unknown_queries)} query id(s) with no text: {unknown_queries[:5]}")
    unknown_docs = sorted({d for judged in qrels.values() for d in judged} - corpus_ids)
    if unknown_docs:
        raise ConversionError(f"qrels name {len(unknown_docs)} document id(s) not in the corpus: {unknown_docs[:5]}")
    no_positive = sorted(q for q, judged in qrels.items() if not any(g > 0 for g in judged.values()))
    if no_positive:
        # Kept, not dropped: trec_eval -c counts a query judged only
        # non-relevant, at 0, so dropping it would raise every mean.
        logger.warning(
            "%d judged query(ies) have no positive grade; kept, and they score 0: %s",
            len(no_positive),
            no_positive[:5],
        )

    benchmark = f"{manifest['name']}/{split}"
    samples = [
        EvalSample.from_dict(
            {
                "id": qid,
                "query": text,
                "expected_doc_ids": [
                    doc_id if grade == 1 else {"id": doc_id, "grade": grade}
                    for doc_id, grade in qrels[qid].items()
                ],
                "matching_mode": MODE_QRELS,
                "benchmark": benchmark,
            }
        )
        # queries.jsonl order, restricted to the split's judged queries.
        for qid, text in queries.items()
        if qid in qrels
    ]

    expected = inventory["qrels"][split]
    judgments = sum(len(s.expected_doc_ids) + sum(1 for g in s.doc_grades.values() if g == 0) for s in samples)
    if len(samples) != expected["judged_queries"] or judgments != expected["judgments"]:
        raise ConversionError(
            f"converted {len(samples)} queries / {judgments} judgments; the manifest inventory "
            f"records {expected['judged_queries']} / {expected['judgments']}"
        )
    return EvalDataset(samples=samples)


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert a fetched BEIR split's qrels into an eval set.")
    parser.add_argument("name", help="Dataset: fiqa, scifact or nfcorpus (the beir- prefix is optional).")
    parser.add_argument("split", help="qrels split: train, dev or test (as the dataset provides).")
    parser.add_argument("--out", type=Path, default=None, help="Output path (default data/eval/beir_<name>_<split>.json).")
    args = parser.parse_args()
    configure_logging()

    name = dataset_name(args.name)
    root = REPO / "data" / "corpora" / f"beir-{name}"
    out = args.out or REPO / "data" / "eval" / f"beir_{name}_{args.split}.json"
    try:
        dataset = convert(root, args.split)
    except (ConversionError, FileNotFoundError) as exc:
        logger.error("%s", exc)
        return 1
    dataset.save(out)
    logger.info("Wrote %d qrels sample(s) to %s", len(dataset), out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
