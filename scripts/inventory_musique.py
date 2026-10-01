#!/usr/bin/env python
"""Inventory the unpacked MuSiQue v1.0 release for its provenance manifest.

Phase A of the MuSiQue follow-on in docs/public-benchmarks-plan.md: pin what
the archive contains before anything is converted, so the converter can be
checked against these numbers, and settle the two questions pooling raises:

* **Paragraph ids.** MuSiQue paragraphs carry only a per-question index. The
  converter will id them ``musique-<sha256(title + "\\n" + text)[:16]>`` and
  dedupe identical ones; this counts the collapse and checks that no two
  different paragraphs share an id.
* **MuSiQue-Full under pooling.** Each unanswerable question is a contrast
  copy of an answerable one with supporting evidence withheld from its
  context. In a corpus pooled over the split, the withheld paragraph comes
  back through the answerable twin's context. This counts the pairs where it
  does, which decides whether Full can be run pooled at all.

    python scripts/inventory_musique.py path/to/data [--zip path/to/musique_v1.0.zip]

``path/to/data`` is the archive's ``data/`` directory. Prints one JSON object.
Stdlib only, read-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

SPLITS = ("ans_v1.0_train", "ans_v1.0_dev", "ans_v1.0_test", "full_v1.0_train", "full_v1.0_dev", "full_v1.0_test")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def paragraph_id(paragraph: dict[str, Any]) -> str:
    """The converter's id: a pure function of title and text, so identical paragraphs collapse."""
    content = f"{paragraph['title']}\n{paragraph['paragraph_text']}"
    return "musique-" + hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def _load(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _histogram(values: list[Any]) -> dict[str, int]:
    return {str(k): n for k, n in sorted(Counter(values).items(), key=lambda kv: str(kv[0]))}


def inventory_split(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ids = [paragraph_id(p) for r in rows for p in r["paragraphs"]]
    content: dict[str, str] = {}
    collisions = 0
    for r in rows:
        for p in r["paragraphs"]:
            text = f"{p['title']}\n{p['paragraph_text']}"
            if content.setdefault(paragraph_id(p), text) != text:
                collisions += 1
    labelled = [r for r in rows if r.get("answer")]
    # A decomposition step points at its supporting paragraph by per-question idx.
    bad_support_idx = 0
    for r in labelled:
        by_idx = {p["idx"]: p for p in r["paragraphs"]}
        for step in r.get("question_decomposition", []):
            idx = step.get("paragraph_support_idx")
            if idx is not None and not by_idx.get(idx, {}).get("is_supporting", False):
                bad_support_idx += 1
    supporting = [paragraph_id(p) for r in rows for p in r["paragraphs"] if p.get("is_supporting")]
    return {
        "questions": len(rows),
        "duplicate_question_ids": sorted(i for i, n in Counter(r["id"] for r in rows).items() if n > 1)[:10],
        "hop_types": _histogram([r["id"].split("__")[0] for r in rows]),
        "paragraphs_per_question": _histogram([len(r["paragraphs"]) for r in rows]),
        "supporting_per_question": _histogram([sum(bool(p.get("is_supporting")) for p in r["paragraphs"]) for r in rows]),
        "decomposition_steps": _histogram([len(r.get("question_decomposition", [])) for r in rows]),
        "answerable": _histogram([r.get("answerable") for r in rows]),
        "labelled": len(labelled),
        "with_answer_aliases": sum(1 for r in rows if r.get("answer_aliases")),
        "paragraph_slots": len(ids),
        "unique_paragraphs": len(set(ids)),
        "unique_supporting_paragraphs": len(set(supporting)),
        "id_collisions": collisions,
        "empty_paragraph_text": sum(1 for r in rows for p in r["paragraphs"] if not p["paragraph_text"].strip()),
        "max_paragraph_chars": max(len(p["paragraph_text"]) for r in rows for p in r["paragraphs"]),
        "decomposition_support_idx_not_supporting": bad_support_idx,
    }


def full_pooling_leak(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Pairs whose unanswerable copy gets its withheld paragraphs back from the pooled split."""
    twins: dict[str, dict[bool, dict[str, Any]]] = {}
    for r in rows:
        twins.setdefault(r["id"], {})[bool(r["answerable"])] = r
    pool = {paragraph_id(p) for r in rows for p in r["paragraphs"]}
    pairs = withheld = restored = 0
    for pair in twins.values():
        if True not in pair or False not in pair:
            continue
        pairs += 1
        missing = (
            {paragraph_id(p) for p in pair[True]["paragraphs"] if p["is_supporting"]}
            - {paragraph_id(p) for p in pair[False]["paragraphs"]}
        )
        if missing:
            withheld += 1
            restored += missing <= pool
    return {"pairs": pairs, "unanswerable_missing_supporting": withheld, "all_withheld_present_in_pool": restored}


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--zip", type=Path, default=None, help="The archive, to record its hash")
    args = parser.parse_args()

    out: dict[str, Any] = {"files": {}, "splits": {}}
    if args.zip:
        out["archive"] = {"sha256": _sha256(args.zip), "bytes": args.zip.stat().st_size}
    loaded: dict[str, list[dict[str, Any]]] = {}
    for split in SPLITS:
        path = args.data_dir / f"musique_{split}.jsonl"
        out["files"][path.name] = {"sha256": _sha256(path), "bytes": path.stat().st_size}
        loaded[split] = _load(path)
        out["splits"][split] = inventory_split(loaded[split])
    singlehop = args.data_dir / "dev_test_singlehop_questions_v1.0.json"
    out["files"][singlehop.name] = {"sha256": _sha256(singlehop), "bytes": singlehop.stat().st_size}

    train = {paragraph_id(p) for r in loaded["ans_v1.0_train"] for p in r["paragraphs"]}
    dev = loaded["ans_v1.0_dev"]
    dev_ids = {paragraph_id(p) for r in dev for p in r["paragraphs"]}
    dev_supporting = {paragraph_id(p) for r in dev for p in r["paragraphs"] if p["is_supporting"]}
    train_questions = {r["question"] for r in loaded["ans_v1.0_train"]}
    out["ans_dev_vs_train"] = {
        "dev_paragraphs_in_train": len(dev_ids & train),
        "dev_supporting_paragraphs_in_train": len(dev_supporting & train),
        "dev_questions_verbatim_in_train": sum(r["question"] in train_questions for r in dev),
    }
    out["full_dev_pooling_leak"] = full_pooling_leak(loaded["full_v1.0_dev"])
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
