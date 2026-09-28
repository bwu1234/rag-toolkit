"""Can a per-filing index pick the right filing? (chunking plan, Phase 3b, first step).

FROZEN RECORD, not maintained tooling. Before building routing into `Retriever`,
measure the router alone: one record per filing, ranked by BM25, dense and
their RRF fusion, scored by where each sample's expected filing lands. If the
right filing isn't near the top, filtering chunk retrieval to the router's
picks can only lose.

Record texts compared:
- ``header``      -- the Phase 2 chunk header, as the plan specifies.
- ``header+date`` -- the same plus the period end spelled as questions say it
                     ("February 15, 2026"), since the header's ISO date shares
                     only digits with a question's wording.

Run from the repo root (needs Ollama for the embedder):
    python scripts/experiments/2026-09-doc-routing/router_probe.py
"""

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from rank_bm25 import BM25Okapi  # noqa: E402

from rag.chunking.chunkers import render_header  # noqa: E402
from rag.config.settings import load_config  # noqa: E402
from rag.embedding.factory import get_embedder  # noqa: E402
from rag.ingestion.corpora import load_selected_corpora  # noqa: E402
from rag.retrieval.sparse import tokenize  # noqa: E402

SETS = ("eval", "period", "underspecified")
OUT = REPO / "data/eval/results/probe_2026-09_doc_router.json"


def spelled(value: object) -> str:
    d = value if isinstance(value, date) else date.fromisoformat(str(value))
    return f"{d:%B} {d.day}, {d.year}"


def main() -> None:
    config = load_config()
    template = config.chunking.header.template
    assert template
    _, documents = load_selected_corpora(config, ["edgar"])
    ids = [d.id for d in documents]
    texts = {
        "header": [render_header(template, d) or "" for d in documents],
    }
    texts["header+date"] = [
        f"{h}; period ended {spelled(d.metadata['period_end'])}" for h, d in zip(texts["header"], documents)
    ]
    embedder = get_embedder(config.embedding)

    samples = {}
    for name in SETS:
        raw = json.loads((REPO / f"data/eval/edgar_{name}_set.json").read_text())
        samples[name] = raw["samples"] if isinstance(raw, dict) else raw
    queries = sorted({s["query"] for rows in samples.values() for s in rows})
    qvec = dict(zip(queries, np.array([embedder.embed_query(q) for q in queries])))

    report: dict = {}
    per_sample: dict = {}
    for variant, recs in texts.items():
        dvec = np.array(embedder.embed_documents(recs))
        dvec /= np.linalg.norm(dvec, axis=1, keepdims=True)
        bm25 = BM25Okapi([tokenize(t) for t in recs])
        for name, rows in samples.items():
            ranks: dict[str, list[int]] = {"bm25": [], "dense": [], "rrf": []}
            top_cos, hit1 = [], []
            for s in rows:
                q = qvec[s["query"]] / np.linalg.norm(qvec[s["query"]])
                cos = dvec @ q
                bm = bm25.get_scores(tokenize(s["query"]))
                orders = {"dense": list(np.argsort(-cos)), "bm25": list(np.argsort(-bm, kind="stable"))}
                rrf = np.zeros(len(ids))
                for order in orders.values():
                    for r, i in enumerate(order, 1):
                        rrf[i] += 1 / (60 + r)
                orders["rrf"] = list(np.argsort(-rrf))
                expected = set(s["expected_doc_ids"])
                for method, order in orders.items():
                    rank = next(r for r, i in enumerate(order, 1) if ids[i] in expected)
                    ranks[method].append(rank)
                top_cos.append(float(cos.max()))
                hit1.append(ranks["rrf"][-1] == 1)
                per_sample.setdefault(variant, {}).setdefault(name, []).append(
                    {"id": s["id"], "tier": s.get("tier"), "rank_rrf": ranks["rrf"][-1],
                     "rank_dense": ranks["dense"][-1], "rank_bm25": ranks["bm25"][-1],
                     "top_cos": float(cos.max()),
                     "bm25_top2": sorted(map(float, bm), reverse=True)[:2],
                     "agree": bool(orders["dense"][0] == orders["bm25"][0]),
                     "top_bm25": ids[orders["bm25"][0]], "top_ids": [ids[i] for i in orders["rrf"][:3]]}
                )
            summary = {}
            for method, rs in ranks.items():
                a = np.array(rs)
                summary[method] = {f"R@{m}": round(float((a <= m).mean()), 3) for m in (1, 2, 3, 5, 10)}
                summary[method]["MRR"] = round(float((1 / a).mean()), 3)
            report.setdefault(variant, {})[name] = summary
            print(f"{variant:12} {name:15} n={len(rows):3}", {m: v for m, v in summary.items()})
    OUT.write_text(json.dumps({"summary": report, "samples": per_sample}, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
