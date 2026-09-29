"""How much of the exact top k does Chroma's HNSW return, per `ef_search`? (choosing `vector_store.hnsw_ef_search`).

FROZEN RECORD; results are in docs/measured-results.md ("HNSW `ef_search`").
Public benchmarks phase 3 found Chroma's default (`ef_search` 100) returning
88.6% of the exact top 100 on FiQA. This measures that recall, and query
latency, over a grid of values, on any built index:

- **Label-free.** Recall is against exact inner-product search over the
  vectors stored in the same collection, so no relevance label is read and
  the choice can't be tuned to a test set. Query texts come from an eval set.
- **One process per value.** Chroma reads `ef_search` when a process first
  loads the collection's HNSW segment; a later `modify()` in that process is
  ignored. So each value runs in a fresh subprocess that sets it before its
  first query.
- The collection is left at `--restore` (default 100, Chroma's default)
  afterwards, since `modify()` persists.

    python scripts/experiments/2026-09-hnsw-ef-search/ann_recall.py --config rag/config/beir_bge.yaml \\
        --corpus beir-fiqa --eval-set data/eval/beir_fiqa_test.json --k 100 --out fiqa.json
    python scripts/experiments/2026-09-hnsw-ef-search/ann_recall.py --corpus edgar \\
        --eval-set data/eval/edgar_eval_set.json --eval-set data/eval/edgar_underspecified_set.json --k 20
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import chromadb  # noqa: E402

from rag.config.settings import load_config  # noqa: E402
from rag.embedding.factory import get_embedder  # noqa: E402
from rag.eval.dataset import EvalDataset  # noqa: E402

GRID = (100, 200, 400, 800, 1600)


def open_collection(config_path: str | None, corpus: str) -> chromadb.Collection:
    selection = load_config(config_path).corpus_selection([corpus])
    return chromadb.PersistentClient(path=str(selection.index_dir)).get_collection(selection.collection_name)


def prepare(args: argparse.Namespace, work: Path) -> int:
    """Stored vectors, query vectors and the exact top k, saved for the workers."""
    collection = open_collection(args.config, args.corpus)
    ids: list[str] = []
    rows = []
    for offset in range(0, collection.count(), 5000):
        page = collection.get(include=["embeddings"], limit=5000, offset=offset)
        ids += page["ids"]
        rows.append(np.asarray(page["embeddings"], dtype=np.float32))
    vectors = np.concatenate(rows)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)  # cosine space

    queries = [s.query for path in args.eval_set for s in EvalDataset.load(path)]
    embedder = get_embedder(load_config(args.config).embedding)
    qvecs = np.asarray([embedder.embed_query(q) for q in queries], dtype=np.float32)
    qvecs /= np.linalg.norm(qvecs, axis=1, keepdims=True)
    exact = np.argsort(-(qvecs @ vectors.T), axis=1)[:, : args.k]
    np.savez(work / "prep.npz", qvecs=qvecs, exact=np.asarray(ids)[exact])
    return len(queries)


def worker(args: argparse.Namespace, work: Path) -> None:
    prep = np.load(work / "prep.npz")
    collection = open_collection(args.config, args.corpus)
    collection.modify(configuration={"hnsw": {"ef_search": args.worker}})  # before the first query
    latencies, shares = [], []
    for qvec, exact in zip(prep["qvecs"], prep["exact"]):
        start = time.perf_counter()
        got = collection.query(query_embeddings=[qvec.tolist()], n_results=args.k, include=[])["ids"][0]
        latencies.append((time.perf_counter() - start) * 1000)
        shares.append(len(set(got) & set(exact)) / len(exact))
    shares_top10 = []
    for qvec, exact in zip(prep["qvecs"], prep["exact"]):
        got = collection.query(query_embeddings=[qvec.tolist()], n_results=min(10, args.k), include=[])["ids"][0]
        shares_top10.append(len(set(got) & set(exact[:10])) / len(exact[:10]))
    print(json.dumps({
        "ef_search": args.worker,
        f"recall@{args.k}": sum(shares) / len(shares),
        "recall@10": sum(shares_top10) / len(shares_top10),
        "queries_below_1": sum(s < 1 for s in shares),
        "median_ms": statistics.median(latencies),
        "p95_ms": float(np.percentile(latencies, 95)),
    }))


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--config")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--eval-set", action="append", required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--grid", type=int, nargs="+", default=list(GRID))
    parser.add_argument("--restore", type=int, default=100)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--worker", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--work", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker is not None:
        worker(args, args.work)
        return

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        n_queries = prepare(args, work)
        base = [sys.executable, __file__, "--corpus", args.corpus, "--k", str(args.k), "--work", str(work)]
        base += ["--config", args.config] if args.config else []
        for path in args.eval_set:
            base += ["--eval-set", path]
        rows = []
        for ef in [*args.grid, args.restore]:
            out = subprocess.run([*base, "--worker", str(ef)], capture_output=True, text=True, check=True).stdout
            if ef == args.restore and len(rows) == len(args.grid):
                break  # the last run only restores the collection's setting
            rows.append(json.loads(out.strip().splitlines()[-1]))
            print(json.dumps(rows[-1]), flush=True)
    record = {"config": args.config, "corpus": args.corpus, "k": args.k, "queries": n_queries,
              "eval_sets": args.eval_set, "restored_ef_search": args.restore, "rows": rows}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
