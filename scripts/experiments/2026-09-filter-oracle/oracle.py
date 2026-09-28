"""Oracle metadata filtering on EDGAR (chunking plan, Phase 3, first step).

FROZEN RECORD, not maintained tooling: the script behind "Metadata filtering
oracle" in docs/measured-results.md, committed so those numbers can be rerun.
It runs before any interface change, as the plan asks, so it reaches into
private attributes (`Retriever._vector_store`, `BM25Index._records`, the Chroma
collection) and will break when Phase 3's `QueryFilter` lands. Don't extend it.

Each sample's retrieval is restricted to documents chosen from its own labels:

- ``none``    -- no restriction. Must reproduce the shipped rows exactly, or
                 the harness itself is changing results.
- ``company`` -- every filing from the expected filing's company, all periods.
                 The realistic filter: a caller names the company, not the period.
- ``filing``  -- exactly ``expected_doc_ids``. The ceiling for perfect filtering.

Both stages filter before their top-k, never after: Chroma through a native
`where` clause, BM25 by scoring everything and keeping the allowed documents'
chunks. Filtering after top-k would silently shrink the candidate pool, which
is the failure the plan's interface rule exists to prevent.

Run from the repo root with the shipped `edgar` index built:
    python scripts/experiments/2026-09-filter-oracle/oracle.py
Results go to data/eval/results/probe_2026-09_filter_oracle.json.
"""
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from rag.config.settings import load_config  # noqa: E402
from rag.eval.dataset import EvalDataset  # noqa: E402
from rag.eval.retrieval_eval import run_retrieval_eval  # noqa: E402
from rag.ingestion.corpora import chunk_selected_corpora  # noqa: E402
from rag.logging_config import configure_logging  # noqa: E402
from rag.retrieval.builder import build_retriever  # noqa: E402
from rag.retrieval.retriever import RetrievalResult, Retriever  # noqa: E402
from rag.vectorstore.base import ScoredChunk  # noqa: E402

SETS = ("eval", "period", "underspecified")
MODES = ("none", "company", "filing")
OUT = REPO / "data/eval/results/probe_2026-09_filter_oracle.json"


class OracleRetriever:
    """Sets the allowed documents for each query, then retrieves with filtered stores."""

    def __init__(self, retriever: Retriever, allowed_by_query: dict[str, set[str] | None]) -> None:
        self._retriever = retriever
        self._allowed_by_query = allowed_by_query
        self.allowed: set[str] | None = None
        store = retriever._vector_store
        sparse = retriever._sparse_index
        assert sparse is not None, "the shipped config is hybrid"
        dense_query, sparse_query = store.query, sparse.query

        def filtered_dense(embedding: list[float], top_k: int) -> list[ScoredChunk]:
            if self.allowed is None:
                return dense_query(embedding, top_k)
            result = store._collection.query(  # type: ignore[attr-defined]
                query_embeddings=[embedding],
                n_results=top_k,
                where={"document_id": {"$in": sorted(self.allowed)}},
                include=["documents", "metadatas", "distances"],
            )
            return [
                store._to_scored_chunk(cid, text, meta, dist)  # type: ignore[attr-defined]
                for cid, text, meta, dist in zip(
                    result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
                )
            ]

        def filtered_sparse(query: str, top_k: int) -> list[ScoredChunk]:
            if self.allowed is None:
                return sparse_query(query, top_k)
            everything = sparse_query(query, sparse.count())
            kept = [c for c in everything if c.document_id in self.allowed][:top_k]
            # Re-normalize within the kept set, as BM25Index.query does for its own.
            if kept:
                scores = [c.score for c in kept]
                lo, span = min(scores), max(scores) - min(scores)
                kept = [replace(c, score=(c.score - lo) / span if span else 1.0) for c in kept]
            return kept

        store.query = filtered_dense  # type: ignore[method-assign]
        sparse.query = filtered_sparse  # type: ignore[method-assign]

    def retrieve(self, query: str, **kwargs: Any) -> RetrievalResult:
        self.allowed = self._allowed_by_query[query]
        return self._retriever.retrieve(query, **kwargs)


def main() -> None:
    configure_logging()
    config = load_config()
    _selection, documents, corpus_chunks = chunk_selected_corpora(config, ["edgar"])
    ticker = {d.id: d.metadata["ticker"] for d in documents}
    by_ticker: dict[str, set[str]] = {}
    for doc_id, t in ticker.items():
        by_ticker.setdefault(t, set()).add(doc_id)

    results: dict[str, dict[str, Any]] = {}
    for name in SETS:
        dataset = EvalDataset.load(REPO / f"data/eval/edgar_{name}_set.json")
        for mode in MODES:
            allowed: dict[str, set[str] | None] = {}
            for sample in dataset:
                expected = set(sample.expected_doc_ids)
                if mode == "none":
                    allowed[sample.query] = None
                elif mode == "filing":
                    allowed[sample.query] = expected
                else:
                    allowed[sample.query] = set().union(*(by_ticker[ticker[d]] for d in expected))
            oracle = OracleRetriever(build_retriever(config, corpora=["edgar"]), allowed)
            report = run_retrieval_eval(dataset, oracle, corpus_chunks=corpus_chunks)  # type: ignore[arg-type]
            results.setdefault(name, {})[mode] = {
                "hit_rate": round(report.overall.mean_hit_rate, 4),
                "ndcg": round(report.overall.mean_ndcg, 4),
                "mean_allowed_docs": round(
                    sum(len(a) for a in allowed.values() if a) / len(dataset), 2
                ) if mode != "none" else None,
                "samples": {
                    r.sample_id: {
                        "hit_rate": r.hit,
                        "ndcg": round(r.ndcg, 4),
                        **({"kind": r.kind} if r.kind is not None else {}),
                    }
                    for r in report.sample_results
                },
            }
            print(name, mode, results[name][mode]["hit_rate"], results[name][mode]["ndcg"], flush=True)
            OUT.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
