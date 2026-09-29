#!/usr/bin/env python
"""Score identical rankings with this repo's qrels evaluator and with ``trec_eval``, and compare.

Public benchmarks plan, phase 2 (and phase 3, over the reference runs). For
each TREC run file, this scores the file itself twice: with
:mod:`rag.eval.qrels` against a qrels eval set, and with a ``trec_eval``
binary (``-c -q -m ndcg_cut.10 -m recall.100``) against a TREC qrels file.
Both evaluators read the same bytes, so any difference is the evaluators'.

Agreement is required per query and in aggregate to within 0.0001 after
rounding ours to ``trec_eval``'s 4 printed decimals, the tolerance frozen in
``docs/beir-reference-protocol.md``. Exit status 1 on any excess.

``trec_eval`` is a reproduction dependency, never a repo dependency. Build the
pinned version from source::

    git clone --depth 1 --branch v9.0.4 https://github.com/usnistgov/trec_eval.git
    make -C trec_eval

Usage::

    python scripts/trec_eval_parity.py --trec-eval trec_eval/trec_eval \\
        --eval-set data/eval/beir_scifact_test.json \\
        --qrels qrels.beir-v1.0.0-scifact.test.txt  runs/scifact/final.trec runs/scifact/stage1.trec

Without ``--qrels`` the TREC qrels are written from the eval set, which checks
the scorer but not the conversion; pass the reference qrels to check both.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from rag.eval.dataset import EvalDataset  # noqa: E402
from rag.eval.qrels import METRICS, qrels_grades, read_run, score_rankings  # noqa: E402

#: trec_eval measure name -> ours.
MEASURES = {"ndcg_cut_10": "nDCG@10", "recall_100": "R@100"}
TOLERANCE = 1e-4


def run_trec_eval(binary: Path, qrels: Path, run: Path) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """Per-query and ``all`` values from ``trec_eval -c -q``."""
    out = subprocess.run(
        [str(binary), "-c", "-q", "-m", "ndcg_cut.10", "-m", "recall.100", str(qrels), str(run)],
        capture_output=True, text=True, check=True,
    ).stdout
    per_query: dict[str, dict[str, float]] = {}
    overall: dict[str, float] = {}
    for line in out.splitlines():
        measure, qid, value = line.split()
        if measure in MEASURES:
            target = overall if qid == "all" else per_query.setdefault(qid, {})
            target[MEASURES[measure]] = float(value)
    return per_query, overall


def write_trec_qrels(dataset: EvalDataset, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for sample in dataset:
            for doc_id, grade in qrels_grades(sample).items():
                f.write(f"{sample.id} 0 {doc_id} {grade}\n")


def compare(binary: Path, dataset: EvalDataset, qrels: Path, run: Path) -> float:
    """Print the comparison for one run file; return the largest difference found."""
    ours = score_rankings(list(dataset), read_run(run))
    ours_by_query = {q.query_id: q.values for q in ours.per_query}
    theirs, theirs_all = run_trec_eval(binary, qrels, run)

    unknown = sorted(set(theirs) - set(ours_by_query))
    if unknown:
        raise SystemExit(f"{run.name}: trec_eval scored queries the eval set lacks: {unknown[:5]}")
    worst = 0.0
    for qid, values in theirs.items():
        for name, value in values.items():
            worst = max(worst, abs(round(ours_by_query[qid][name], 4) - value))
    print(f"{run}")
    for name in METRICS:
        diff = abs(round(ours.means[name], 4) - theirs_all[name])
        worst = max(worst, diff)
        print(f"  {name:<8} ours {ours.means[name]:.6f}  trec_eval {theirs_all[name]:.4f}  |diff| {diff:.4f}")
    print(
        f"  {len(theirs)} queries compared per query; "
        f"{len(ours_by_query) - len(theirs)} declared but absent from the run (0 under -c); "
        f"max |diff| {worst:.4f}"
    )
    return worst


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare qrels scores with trec_eval on identical run files.")
    parser.add_argument("runs", nargs="+", type=Path, help="TREC run files.")
    parser.add_argument("--trec-eval", required=True, type=Path, help="Path to a trec_eval binary (9.0.4 pinned).")
    parser.add_argument("--eval-set", required=True, type=Path, help="A qrels eval set (matching_mode 'qrels').")
    parser.add_argument("--qrels", type=Path, default=None, help="TREC qrels for trec_eval (default: from the eval set).")
    args = parser.parse_args()

    dataset = EvalDataset.load(args.eval_set)
    with tempfile.TemporaryDirectory() as tmp:
        qrels = args.qrels
        if qrels is None:
            qrels = Path(tmp) / "qrels.txt"
            write_trec_qrels(dataset, qrels)
        worst = max(compare(args.trec_eval, dataset, qrels, run) for run in args.runs)
    verdict = "PASS" if worst <= TOLERANCE else "FAIL"
    print(f"{verdict}: largest difference {worst:.4f} (tolerance {TOLERANCE})")
    return 0 if worst <= TOLERANCE else 1


if __name__ == "__main__":
    raise SystemExit(main())
