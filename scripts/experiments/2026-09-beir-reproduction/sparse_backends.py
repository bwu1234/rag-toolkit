"""This repo's BM25 backends on the BEIR test splits, beside the Lucene reference (public benchmarks plan, phase 3).

FROZEN RECORD of the phase-3 backend diagnostic; results are in
docs/measured-results.md and docs/public-benchmarks-plan.md ("Phase 3 as
built"). There is no BM25-only retrieval mode (`retrieval.mode` is `dense` or
`hybrid`), so this drives the sparse backends directly: the repo's own loader
and `chunking.strategy: none` produce the chunks (rag/config/beir.yaml), each
backend indexes them through `get_sparse_index`, and every test query is
ranked to depth 1,000, the reference depth. Rankings go through the same
`document_ranking` (self-match removal on) and `score_rankings` the qrels
runner uses.

No embedder is involved, so nothing is re-embedded. Backend parameters are the
shipped ones; nothing is tuned on test.

    python scripts/experiments/2026-09-beir-reproduction/sparse_backends.py scifact nfcorpus fiqa

Writes data/benchmarks/repo/<dataset>/sparse-<provider>/{index/, run.trec, scores.json}.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from rag.config.settings import SparseIndexConfig, load_config  # noqa: E402
from rag.eval.dataset import EvalDataset  # noqa: E402
from rag.eval.qrels import document_ranking, score_rankings, write_run  # noqa: E402
from rag.ingestion.corpora import chunk_selected_corpora  # noqa: E402
from rag.logging_config import configure_logging  # noqa: E402
from rag.retrieval.factory import get_sparse_index  # noqa: E402

DEPTH = 1000
PROVIDERS = ("bm25", "sqlite_fts5")


def run(dataset: str) -> None:
    config = load_config(REPO / "rag" / "config" / "beir.yaml")
    _, documents, chunks = chunk_selected_corpora(config, [f"beir-{dataset}"])
    samples = list(EvalDataset.load(REPO / "data" / "eval" / f"beir_{dataset}_test.json"))
    reference = json.loads((REPO / "data" / "corpora" / f"beir-{dataset}" / "manifest.json").read_text())
    published = reference["reference"]["runs"]["bm25-flat"]
    print(f"{dataset}: {len(documents)} documents, {len(chunks)} chunks, {len(samples)} test queries")

    for provider in PROVIDERS:
        out = REPO / "data" / "benchmarks" / "repo" / dataset / f"sparse-{provider}"
        index = get_sparse_index(SparseIndexConfig(provider=provider), out / "index", slug=f"beir-{dataset}")
        start = time.monotonic()
        if index.count() != len(chunks):
            index.reset()
            index.upsert(chunks)
            index.flush()
        index_seconds = time.monotonic() - start

        start = time.monotonic()
        rankings = {}
        short = 0
        for sample in samples:
            results = index.query(sample.query, top_k=DEPTH)
            short += len(results) < 100
            rankings[sample.id] = document_ranking(
                ((r.document_id, r.score) for r in results), query_id=sample.id, remove_query=True
            )
        query_seconds = time.monotonic() - start

        scores = score_rankings(samples, rankings)
        write_run(out / "run.trec", rankings, tag=f"rag-{provider}")
        record = {
            "dataset": dataset,
            "provider": provider,
            "depth": DEPTH,
            "chunks": len(chunks),
            "queries": len(samples),
            "queries_under_100_results": short,
            "self_matches_removed": sum(r.self_matches_removed for r in rankings.values()),
            "index_seconds": round(index_seconds, 1),
            "query_seconds": round(query_seconds, 1),
            "means": scores.means,
            "reference_bm25_flat": {m: published[m] for m in scores.means},
            "difference": {m: scores.means[m] - published[m] for m in scores.means},
            "per_query": {q.query_id: q.values for q in scores.per_query},
        }
        (out / "scores.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(
            f"  {provider:12} "
            + "  ".join(f"{m} {v:.4f} (ref {published[m]:.4f}, {v - published[m]:+.4f})" for m, v in scores.means.items())
            + f"  [{query_seconds:.0f}s query, {short} short]",
            flush=True,
        )


if __name__ == "__main__":
    configure_logging()
    for name in sys.argv[1:]:
        run(name.removeprefix("beir-"))
