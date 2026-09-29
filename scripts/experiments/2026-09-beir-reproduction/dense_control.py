"""Where does this repo's BGE dense run differ from the Faiss reference? (public benchmarks plan, phase 3).

FROZEN RECORD of the phase-3 dense diagnostic; results are in
docs/public-benchmarks-plan.md ("Phase 3 as built"). The plan's diagnosis
order is scoring, then representation, then encoder, then ranking backend.
Scoring is settled by scripts/trec_eval_parity.py. This separates the rest
with exact-search controls, so that no production Faiss provider is needed:

1. **Document vectors**: cosine between each vector stored in this repo's
   Chroma collection (rag/config/beir_bge.yaml) and the reference Faiss vector
   for the same docid. Compared by id, never by row: the orders differ.
2. **Exact search** (numpy inner product, the Faiss `IndexFlatIP` operation) at
   depth 1,000, scored by `rag.eval.qrels`:
   - `ref-docs`: this repo's query vectors x the reference document vectors.
     Matching the published run means the query side reproduces.
   - `our-docs`: this repo's query vectors x this repo's document vectors.
     Its difference from `ref-docs` is the document encoding.
   - The saved Chroma run (``retrieval_eval --save-run``) against `our-docs`
     is then what approximate (HNSW) search costs; `chroma_ann_recall` is
     the share of the exact top k that Chroma also returned.
   - With ``--cpu-docs``, `cpu-docs`: the same documents re-encoded on the
     CPU (the index is built on Apple-silicon MPS when available), which
     tells device arithmetic apart from a recipe difference.

    python scripts/experiments/2026-09-beir-reproduction/dense_control.py [--cpu-docs] scifact [nfcorpus fiqa]

Needs the reference index (Pyserini downloads it to ~/.cache/pyserini/indexes
in scripts/reproduce_beir_reference.py), the built beir_bge index, and the
saved run under data/benchmarks/repo/<dataset>/bge-dense/.
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import chromadb  # noqa: E402

from rag.config.settings import load_config  # noqa: E402
from rag.embedding.factory import get_embedder  # noqa: E402
from rag.eval.dataset import EvalDataset  # noqa: E402
from rag.eval.qrels import DocRanking, document_ranking, read_run, score_rankings, write_run  # noqa: E402
from rag.logging_config import configure_logging  # noqa: E402

DEPTH = 1000
INDEX_CACHE = Path.home() / ".cache" / "pyserini" / "indexes"


def read_flat_index(path: Path) -> np.ndarray:
    """A Faiss IndexFlat file without Faiss (as in the phase-0 recipe probe)."""
    raw = path.read_bytes()
    assert raw[:4] in (b"IxFI", b"IxF2"), raw[:4]
    (d,) = struct.unpack_from("<i", raw, 4)
    (n,) = struct.unpack_from("<q", raw, 8)
    (metric,) = struct.unpack_from("<i", raw, 33)
    (size,) = struct.unpack_from("<q", raw, 37)
    assert metric == 0 and size == n * d
    return np.frombuffer(raw, dtype="<f4", count=size, offset=45).reshape(n, d)


def exact_run(queries: np.ndarray, qids: list[str], docs: np.ndarray, dids: list[str]) -> dict[str, DocRanking]:
    rankings = {}
    scores = queries @ docs.T
    for row, qid in enumerate(qids):
        top = np.argpartition(-scores[row], DEPTH)[:DEPTH] if len(dids) > DEPTH else np.arange(len(dids))
        rankings[qid] = document_ranking(
            ((dids[i], float(scores[row, i])) for i in top), query_id=qid, remove_query=True
        )
    return rankings


def top10_changes(a: dict[str, DocRanking], b: dict[str, DocRanking]) -> int:
    return sum(a[q].doc_ids[:10] != b.get(q, DocRanking(())).doc_ids[:10] for q in a)


def ann_recall(exact: dict[str, DocRanking], approx: dict[str, DocRanking], k: int) -> float:
    """Mean share of the exact top-k that approximate search also returned in its top k."""
    shares = [
        len(set(exact[q].doc_ids[:k]) & set(approx.get(q, DocRanking(())).doc_ids[:k])) / len(exact[q].doc_ids[:k])
        for q in exact
    ]
    return sum(shares) / len(shares)


def run(dataset: str, cpu_docs: bool) -> None:
    config = load_config(REPO / "rag" / "config" / "beir_bge.yaml")
    manifest = json.loads((REPO / "data" / "corpora" / f"beir-{dataset}" / "manifest.json").read_text())
    published = manifest["reference"]["runs"]["bge-base-en-v1.5.faiss"]
    samples = list(EvalDataset.load(REPO / "data" / "eval" / f"beir_{dataset}_test.json"))
    out = REPO / "data" / "benchmarks" / "repo" / dataset / "bge-dense"

    (ref_dir,) = INDEX_CACHE.glob(f"faiss-flat.beir-v1.0.0-{dataset}.bge-base-en-v1.5.*")
    ref_vecs = read_flat_index(ref_dir / "index")
    ref_ids = (ref_dir / "docid").read_text().split()

    collection = chromadb.PersistentClient(path=str(REPO / config.paths.index_dir)).get_collection(
        f"{config.vector_store.collection_name}__beir-{dataset}"
    )
    # Paged: one `get` of FiQA's 57,600 rows exceeds SQLite's variable limit.
    our_ids: list[str] = []
    texts: list[str] = []
    rows = []
    for offset in range(0, collection.count(), 5000):
        page = collection.get(include=["embeddings", "metadatas", "documents"], limit=5000, offset=offset)
        our_ids += [str(m["document_id"]) for m in page["metadatas"] or []]
        texts += page["documents"] or []
        rows.append(np.asarray(page["embeddings"], dtype=np.float32))
    our_vecs = np.concatenate(rows)
    assert len(set(our_ids)) == len(our_ids), "strategy: none should give one chunk per document"

    ref_row = {d: i for i, d in enumerate(ref_ids)}
    cos = np.einsum("ij,ij->i", our_vecs, ref_vecs[[ref_row[d] for d in our_ids]])
    only_ref = sorted(set(ref_ids) - set(our_ids))

    embedder = get_embedder(config.embedding)
    qids = [s.id for s in samples]
    queries = np.asarray([embedder.embed_query(s.query) for s in samples], dtype=np.float32)

    runs = {
        "ref-docs": exact_run(queries, qids, ref_vecs, ref_ids),
        "our-docs": exact_run(queries, qids, our_vecs, our_ids),
        "chroma": read_run(out / "stage1.trec"),
    }
    cpu_cos = None
    if cpu_docs:
        from sentence_transformers import SentenceTransformer

        cpu_model = SentenceTransformer(config.embedding.model, revision=config.embedding.revision, device="cpu")
        cpu_vecs = cpu_model.encode(texts, batch_size=32, normalize_embeddings=True)
        cpu_cos = np.einsum("ij,ij->i", cpu_vecs, ref_vecs[[ref_row[d] for d in our_ids]])
        runs["cpu-docs"] = exact_run(queries, qids, cpu_vecs, our_ids)
    for name in ("ref-docs", "our-docs"):
        write_run(out / f"exact-{name}.trec", runs[name], tag=f"exact-{name}")

    means = {name: score_rankings(samples, r).means for name, r in runs.items()}
    record = {
        "dataset": dataset,
        "device": str(embedder._sentence_transformer.device),  # type: ignore[attr-defined]
        "documents": {"ours": len(our_ids), "reference": len(ref_ids), "only_in_reference": len(only_ref)},
        "doc_vector_cosine": {
            "min": float(cos.min()),
            "p001": float(np.quantile(cos, 0.001)),
            "median": float(np.median(cos)),
            "below_0.9999": int((cos < 0.9999).sum()),
        },
        "published": {m: published[m] for m in ("nDCG@10", "R@100")},
        "means": means,
        "top10_changes": {
            "our-docs vs ref-docs": top10_changes(runs["ref-docs"], runs["our-docs"]),
            "chroma vs our-docs": top10_changes(runs["our-docs"], runs["chroma"]),
            **({"cpu-docs vs ref-docs": top10_changes(runs["ref-docs"], runs["cpu-docs"])} if cpu_docs else {}),
        },
    }
    record["chroma_ann_recall"] = {f"@{k}": ann_recall(runs["our-docs"], runs["chroma"], k) for k in (10, 100)}
    if cpu_cos is not None:
        record["cpu_doc_vector_cosine"] = {"min": float(cpu_cos.min()), "median": float(np.median(cpu_cos))}
    (out / "control.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    configure_logging()
    args = sys.argv[1:]
    cpu = "--cpu-docs" in args
    for name in (a for a in args if a != "--cpu-docs"):
        run(name.removeprefix("beir-"), cpu)
