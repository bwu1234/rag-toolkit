"""Does the qrels scorer match trec_eval on awkward rankings? (public benchmarks plan, Phase 2).

FROZEN RECORD, not maintained tooling (`scripts/trec_eval_parity.py` is the
maintained check). Real rankings rarely exercise the edge cases, so this
builds rankings that do and scores them with both evaluators:

- NFCorpus's real graded test qrels (grades 1 and 2), and a synthetic qrels
  set with grades 0-3, including queries judged only non-relevant;
- per query, a rotating mix of 150-deep lists with heavy score ties (scores
  rounded to 0.1), 7-deep and 1-deep lists, a 120-deep tied list, a 60-deep
  list, and a query missing from the run entirely; every list mixes judged
  and unjudged documents.

Then it checks that the comparison has power: with the document-id
tie-break removed from `trec_order`, parity must fail.

Result (2026-09-29, trec_eval 9.0.4 built from tag v9.0.4, d63f5c4b):
NFCorpus 323 queries (53 missing from the run, 13,827 tied adjacent pairs)
and synthetic 200 queries (33 missing, 8,585 tied pairs) both agree to
0.0000 per query and in aggregate. The mutant fails with a per-query
difference of 1.0. A first version of the synthetic check disagreed by
0.0039 in the mean because the schema could not hold a query judged only
non-relevant; trec_eval -c counts those at 0, and the schema now does too.

    python scripts/experiments/2026-09-qrels-parity/stress_parity.py \\
        path/to/trec_eval path/to/qrels.beir-v1.0.0-nfcorpus.test.txt
"""

import random
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import rag.eval.qrels as qrels_module  # noqa: E402
from rag.eval.dataset import MODE_QRELS, EvalDataset, EvalSample  # noqa: E402
from rag.eval.qrels import document_ranking, write_run  # noqa: E402
from trec_eval_parity import TOLERANCE, compare  # noqa: E402


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    judged: dict[str, dict[str, int]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        qid, _, doc_id, grade = line.split()
        judged.setdefault(qid, {})[doc_id] = int(grade)
    return judged


def dataset_from(judged: dict[str, dict[str, int]]) -> EvalDataset:
    return EvalDataset(
        samples=[
            EvalSample.from_dict(
                {
                    "id": qid,
                    "query": qid,
                    "expected_doc_ids": [d if g == 1 else {"id": d, "grade": g} for d, g in docs.items()],
                    "matching_mode": MODE_QRELS,
                }
            )
            for qid, docs in judged.items()
        ]
    )


def awkward_rankings(judged: dict[str, dict[str, int]], seed: int) -> dict:
    rng = random.Random(seed)
    pool = [f"u{i}" for i in range(5000)]
    rankings = {}
    for i, (qid, docs) in enumerate(judged.items()):
        kind = i % 6
        if kind == 5:
            continue  # missing from the run
        depth = {0: 150, 1: 7, 2: 1, 3: 120, 4: 60}[kind]
        candidates = list(docs) + rng.sample(pool, 200)
        rng.shuffle(candidates)
        pairs = [(d, round(rng.random(), 1) if kind in (0, 3) else rng.random()) for d in dict.fromkeys(candidates)]
        pairs.sort(key=lambda p: -p[1])
        rankings[qid] = document_ranking(pairs[:depth])
    return rankings


def check(binary: Path, qrels: Path, seed: int, tmp: Path) -> float:
    judged = load_qrels(qrels)
    run = tmp / f"{qrels.stem}.{seed}.trec"
    write_run(run, awkward_rankings(judged, seed), tag="stress")
    return compare(binary, dataset_from(judged), qrels, run)


def main() -> int:
    binary, nfcorpus = Path(sys.argv[1]), Path(sys.argv[2])
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        synthetic = tmp / "synthetic.qrels"
        rng = random.Random(7)
        synthetic.write_text(
            "".join(
                f"s{q} 0 u{d} {rng.choice([0, 0, 1, 2, 3])}\n"
                for q in range(200)
                for d in rng.sample(range(3000), rng.randint(1, 40))
            ),
            encoding="utf-8",
        )
        worst = max(check(binary, nfcorpus, 1, tmp), check(binary, synthetic, 2, tmp))

        original = qrels_module.trec_order
        qrels_module.trec_order = lambda docs: sorted(docs, key=lambda d: d.score, reverse=True)
        try:
            mutant = check(binary, nfcorpus, 1, tmp)
        finally:
            qrels_module.trec_order = original
    print(f"parity {worst:.4f}; mutant without the id tie-break {mutant:.4f}")
    return 0 if worst <= TOLERANCE < mutant else 1


if __name__ == "__main__":
    raise SystemExit(main())
