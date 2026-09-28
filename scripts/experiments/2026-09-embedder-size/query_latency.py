"""Per-query embedding latency for each embedder size (chunking plan, Phase 1b).

FROZEN RECORD, not maintained tooling: the script behind the latency column of
"Embedder size" in docs/measured-results.md, committed so the number can be
rerun. run_matrix.py's `elapsed_s` times a whole eval run, reranker included,
so it can't separate what the embedder adds to a query. This times
`embed_query` alone, one question at a time as a live query would arrive.

Run from the repo root with the models pulled in Ollama. Each model is warmed
up first so the load from disk isn't counted; nothing else should be using
Ollama during the run.

Usage: python scripts/experiments/2026-09-embedder-size/query_latency.py
"""
import json
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from rag.config.settings import load_config  # noqa: E402
from rag.embedding.factory import get_embedder  # noqa: E402
from rag.eval.dataset import EvalDataset  # noqa: E402

SIZES = ("0.6b", "4b", "8b")
EVAL_SET = REPO / "data/eval/edgar_eval_set.json"
WARMUP = 5


def main() -> None:
    queries = [sample.query for sample in EvalDataset.load(EVAL_SET)]
    results = {}
    for size in SIZES:
        config = load_config(REPO / f"data/eval/config_embedder_{size}.yaml")
        embedder = get_embedder(config.embedding)
        for query in queries[:WARMUP]:
            embedder.embed_query(query)
        timings_ms = []
        for query in queries:
            started = time.perf_counter()
            embedder.embed_query(query)
            timings_ms.append((time.perf_counter() - started) * 1000)
        timings_ms.sort()
        results[config.embedding.model] = {
            "n": len(timings_ms),
            "median_ms": round(statistics.median(timings_ms), 1),
            "p95_ms": round(timings_ms[int(0.95 * (len(timings_ms) - 1))], 1),
            "mean_ms": round(statistics.fmean(timings_ms), 1),
        }
        print(config.embedding.model, results[config.embedding.model], flush=True)
    out = REPO / "data/eval/results/probe_2026-09_embedder_query_latency.json"
    out.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
