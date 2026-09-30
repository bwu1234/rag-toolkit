#!/usr/bin/env python
"""Measure the query-time stack on BEIR: public benchmarks plan, phase 4.

Runs a fixed set of retrieval variants over a BEIR split, saves each run with
its provenance, and compares them as the frozen comparison family in
``docs/beir-phase4-protocol.md``. Retrieval only: no LLM, no
judge, nothing leaves the machine.

Variants differ from their baseline in one factor, as ``run_matrix.py``'s do,
but the family has more than one baseline (hybrid is compared with dense,
the reranker with hybrid), so this is its own runner rather than a
``run_matrix`` axis. All variants use the benchmark depths: 100 candidates
per retriever, 100 after fusion (the reranker's pool), 10 final results.

Every run is written to ``data/benchmarks/phase4/<dataset>/<split>/<variant>/``:
``stage1.trec``, ``final.trec``, ``scores.json`` (per-query scores and
per-stage latency, from ``retrieval_eval``) and ``provenance.json`` (code
revision, eval-set/corpus/qrels/index/config hashes, model identities,
library versions, timings). A run already on disk is skipped unless
``--rerun``.

Usage::

    python scripts/run_beir_stack.py --dataset fiqa --split dev
    python scripts/run_beir_stack.py --dataset scifact --split test --variant bge-dense
    python scripts/run_beir_stack.py --compare --split test     # family table, runs nothing
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import platform
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from run_matrix import QWEN3_RETRIEVAL_INSTRUCTION, apply_overrides  # noqa: E402

from rag.config.settings import RagConfig, load_config  # noqa: E402
from rag.eval.dataset import EvalDataset, EvalSample  # noqa: E402
from rag.eval.paired import PairedDifference, compare_by_id  # noqa: E402
from rag.eval.retrieval_eval import STAGE_1, STAGE_FINAL, run_qrels_eval, save_qrels_run  # noqa: E402
from rag.index_manifest import index_manifest_path  # noqa: E402
from rag.logging_config import configure_logging  # noqa: E402
from rag.retrieval.builder import build_retriever  # noqa: E402

logger = logging.getLogger(__name__)

RESULTS_DIR = REPO / "data" / "benchmarks" / "phase4"
DATASETS = ("fiqa", "nfcorpus", "scifact")
#: SciFact has no dev split: it is a transfer check, run on test only.
SPLITS = {"fiqa": ("dev", "test"), "nfcorpus": ("dev", "test"), "scifact": ("test",)}

#: Benchmark depths, the same for every variant (plan: "a shared stage-1
#: budget of at least 100 ... at least 10 final results"). With no reranker
#: the final ranking is stage 1 cut to 10, so nDCG@10 is the retriever's own.
CANDIDATE_DEPTH = 100
FINAL_DEPTH = 10

BGE_CONFIG = "rag/config/beir_bge.yaml"
QWEN_CONFIG = "rag/config/beir.yaml"
HYBRID = {"retrieval.mode": "hybrid"}
RERANK = {"retrieval.mode": "hybrid", "reranker.provider": "cross_encoder"}


@dataclass(frozen=True)
class Variant:
    """A config file plus the overrides that make this variant."""

    name: str
    config: str
    overrides: dict[str, Any] = field(default_factory=dict)
    note: str = ""


VARIANTS: dict[str, Variant] = {
    v.name: v
    for v in (
        Variant("bge-dense", BGE_CONFIG, note="reference encoder, dense only"),
        Variant("bge-hybrid", BGE_CONFIG, HYBRID, note="BGE dense + bm25, RRF k=60"),
        Variant("bge-hybrid-rerank", BGE_CONFIG, RERANK, note="+ shipped cross-encoder over the fused 100"),
        Variant("qwen-dense", QWEN_CONFIG, note="shipped embedder, dense only"),
        Variant("qwen-hybrid", QWEN_CONFIG, HYBRID, note="descriptive; not in the family"),
        Variant("qwen-hybrid-rerank", QWEN_CONFIG, RERANK, note="the shipped stack at benchmark depths"),
        Variant(
            "qwen-dense-instruct",
            QWEN_CONFIG,
            {"embedding.query_instruction": QWEN3_RETRIEVAL_INSTRUCTION},
            note="measured-off query instruction, re-run",
        ),
    )
}


@dataclass(frozen=True)
class Comparison:
    candidate: str
    baseline: str
    question: str


#: The confirmatory family, frozen before any test run (see the protocol doc).
#: Changing it after test results are seen makes every reading exploratory.
FAMILY: tuple[Comparison, ...] = (
    Comparison("bge-hybrid", "bge-dense", "Does BM25 fusion help dense retrieval?"),
    Comparison("bge-hybrid-rerank", "bge-hybrid", "Does the cross-encoder help over hybrid?"),
    Comparison("qwen-dense", "bge-dense", "Shipped embedder vs the reference encoder, dense only"),
    Comparison("qwen-hybrid-rerank", "qwen-dense", "What the shipped stack adds over its dense leg"),
    Comparison("qwen-dense-instruct", "qwen-dense", "Does Qwen's query instruction help here?"),
)
FAMILY_ALPHA = 0.05
#: Grouped intervals are used only when grouping is informative: the largest
#: group of related queries holds at most this share of the set.
MAX_GROUP_SHARE = 0.05
#: Smallest nDCG@10 difference worth acting on.
WORTHWHILE = 0.01


def eval_set_path(dataset: str, split: str) -> Path:
    return REPO / "data" / "eval" / f"beir_{dataset}_{split}.json"


def run_dir(dataset: str, split: str, variant: str) -> Path:
    return RESULTS_DIR / dataset / split / variant


def reranks(name: str) -> bool:
    return VARIANTS[name].overrides.get("reranker.provider") == "cross_encoder"


def variant_config(variant: Variant) -> RagConfig:
    config = apply_overrides(load_config(REPO / variant.config), variant.overrides)
    config.retrieval.top_k = CANDIDATE_DEPTH
    config.retrieval.rerank_top_k = FINAL_DEPTH
    return config


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()


#: The code this process runs, read once at import. Recording HEAD per run
#: instead was wrong in phase 4: another session switched this checkout's
#: branch mid-run, and 18 of 21 test runs recorded a commit they never loaded.
LOADED_CODE = {"commit": _git("rev-parse", "HEAD"), "dirty": _git("status", "--porcelain", "--untracked-files=no")}


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _ollama_digest(config: RagConfig) -> str | None:
    import httpx

    try:
        tags = httpx.get(f"{config.embedding.base_url}/api/tags", timeout=10).json()
    except httpx.HTTPError:
        return None
    return next((m["digest"] for m in tags.get("models", []) if m["name"] == config.embedding.model), None)


def _hf_snapshot(model: str) -> str | None:
    """The cached Hugging Face commit a model name resolved to, if it is cached."""
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(model, local_files_only=True)).name
    except Exception:  # noqa: BLE001 -- provenance is best effort; a miss is recorded as null
        return None


def code_provenance() -> dict[str, str]:
    """The loaded commit and tree state, plus HEAD now, warning if they differ."""
    head_at_run = _git("rev-parse", "HEAD")
    if head_at_run != LOADED_CODE["commit"]:
        logger.warning(
            "HEAD moved to %s since this process loaded %s; the run uses the loaded code",
            head_at_run[:7],
            LOADED_CODE["commit"][:7],
        )
    return {**LOADED_CODE, "head_at_run": head_at_run}


def provenance(variant: Variant, config: RagConfig, dataset: str, split: str) -> dict[str, Any]:
    corpus = f"beir-{dataset}"
    manifest_path = REPO / "data" / "corpora" / corpus / "manifest.json"
    inventory = json.loads(manifest_path.read_text(encoding="utf-8"))["inventory"]
    resolved = config.model_dump(mode="json")
    index_manifest = index_manifest_path(Path(config.paths.index_dir), corpus)
    embedding: dict[str, Any] = {
        "provider": config.embedding.provider,
        "model": config.embedding.model,
        "revision": config.embedding.revision,
        "query_instruction": config.embedding.query_instruction,
    }
    if config.embedding.provider == "ollama":
        embedding["ollama_digest"] = _ollama_digest(config)
    reranker: dict[str, Any] = {"provider": config.reranker.provider}
    if config.reranker.provider == "cross_encoder":
        reranker |= {"model": config.reranker.model, "hf_snapshot": _hf_snapshot(config.reranker.model)}
    try:
        import torch

        device = "mps" if torch.backends.mps.is_available() else "cpu"
    except ImportError:
        device = None
    return {
        "variant": variant.name,
        "config_file": variant.config,
        "overrides": variant.overrides,
        "depths": {
            "per_retriever": config.retrieval.top_k,
            "fused": config.retrieval.top_k,
            "reranker_pool": config.retrieval.top_k if config.reranker.provider != "none" else 0,
            "final": config.retrieval.rerank_top_k,
        },
        "code": code_provenance(),
        "eval_set": {"path": str(eval_set_path(dataset, split).relative_to(REPO)), "sha256": _sha256(eval_set_path(dataset, split))},
        "corpus": {"name": corpus, "corpus_sha256": inventory["corpus"]["sha256"], "qrels_sha256": inventory["qrels"][split]["sha256"]},
        "index": {
            "dir": str(config.paths.index_dir),
            "manifest": json.loads(index_manifest.read_text(encoding="utf-8")) if index_manifest.exists() else None,
        },
        "config_sha256": hashlib.sha256(json.dumps(resolved, sort_keys=True).encode()).hexdigest(),
        "config": resolved,
        "models": {"embedding": embedding, "reranker": reranker},
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "device": device,
            "libraries": {p: _version(p) for p in ("torch", "sentence-transformers", "transformers", "chromadb", "rank-bm25")},
        },
    }


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def run(dataset: str, split: str, names: list[str], *, rerun: bool) -> None:
    path = eval_set_path(dataset, split)
    if not path.exists():
        raise SystemExit(f"{path} is missing; run scripts/beir_to_eval_set.py {dataset} {split}")
    eval_set = EvalDataset.load(path)
    for name in names:
        out = run_dir(dataset, split, name)
        if (out / "provenance.json").exists() and not rerun:
            logger.info("%s/%s/%s already on disk; skipping (--rerun to repeat)", dataset, split, name)
            continue
        variant = VARIANTS[name]
        config = variant_config(variant)
        record = provenance(variant, config, dataset, split)
        logger.info("Running %s on %s/%s (%d queries)", name, dataset, split, len(eval_set))
        started = datetime.now(timezone.utc)
        clock = time.perf_counter()
        retriever = build_retriever(config, corpora=[f"beir-{dataset}"])
        report = run_qrels_eval(eval_set, retriever)
        record["timing"] = {
            "started": started.isoformat(timespec="seconds"),
            "wall_s": round(time.perf_counter() - clock, 1),
            "latency_ms": report.latency_summary(),
        }
        save_qrels_run(
            report,
            out,
            tag=name,
            settings={"eval_set": record["eval_set"]["path"], "config": variant.config, "variant": name},
        )
        (out / "provenance.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        final, stage1 = report.scores[STAGE_FINAL].means, report.scores[STAGE_1].means
        logger.info(
            "%s: nDCG@10 %.4f, R@100 %.4f, %.0f s", name, final["nDCG@10"], stage1["R@100"], record["timing"]["wall_s"]
        )


# ---------------------------------------------------------------------------
# Comparing
# ---------------------------------------------------------------------------


def query_groups(samples: list[EvalSample]) -> dict[str, str]:
    """Group queries that share a relevant document (connected components).

    Claims written from one abstract (SciFact) are related questions, not
    independent draws. Grades >= 1 only: a shared non-relevant judgment says
    nothing about how the queries were written.
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for sample in samples:
        q = find(f"q:{sample.id}")
        for doc in sample.expected_doc_ids:
            if sample.doc_grade(doc) >= 1:
                parent[find(f"d:{doc}")] = q
    return {s.id: find(f"q:{s.id}") for s in samples}


def load_scores(dataset: str, split: str, variant: str) -> dict[str, Any] | None:
    path = run_dir(dataset, split, variant) / "scores.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _per_query(scores: dict[str, Any], stage: str, metric: str) -> dict[str, float]:
    return {q: v[metric] for q, v in scores["per_query"][stage].items()}


def _cell(d: PairedDifference) -> str:
    return f"{d.mean_diff:+.4f} [{d.ci_low:+.4f}, {d.ci_high:+.4f}]"


def _reading(adjusted: PairedDifference) -> str:
    if adjusted.ci_low > 0:
        return "improved" + ("" if adjusted.mean_diff >= WORTHWHILE else " (below 0.01)")
    if adjusted.ci_high < 0:
        return "worse" + ("" if adjusted.mean_diff <= -WORTHWHILE else " (below 0.01)")
    if -WORTHWHILE < adjusted.ci_low and adjusted.ci_high < WORTHWHILE:
        return "no worthwhile effect"
    return "not shown"


def compare(split: str) -> str:
    datasets = [d for d in DATASETS if split in SPLITS[d]]
    confirmatory = split == "test"
    m = len(FAMILY) * len(datasets)
    level = 1 - FAMILY_ALPHA / m
    lines = [f"# Phase 4, {split} split", ""]
    if confirmatory:
        lines.append(
            f"Family of {m} comparisons (nDCG@10); Bonferroni level {level:.5f}. "
            "'Adjusted' is that interval; the reading is taken from it."
        )
    else:
        lines.append("Dev split: exploratory. 95% intervals, no multiplicity adjustment, no readings.")
    for dataset in datasets:
        eval_set = list(EvalDataset.load(eval_set_path(dataset, split)))
        groups = query_groups(eval_set)
        largest = max(Counter(groups.values()).values())
        grouped = largest / len(eval_set) <= MAX_GROUP_SHARE
        group_count = len(set(groups.values()))
        scores = {v: s for v in VARIANTS if (s := load_scores(dataset, split, v)) is not None}
        lines += ["", f"## {dataset} ({len(eval_set)} queries)", ""]
        lines.append(
            f"Related-query groups (shared relevant document): {group_count}, largest {largest}. "
            + ("Intervals are cluster-robust over them." if grouped and largest > 1 else "")
            + ("Intervals treat queries as independent." if largest == 1 else "")
            + (
                "" if grouped else "Too connected to resample by group: intervals treat queries as "
                "independent, which understates their width by an unknown amount."
            )
        )
        lines += ["", "| Variant | nDCG@10 | R@100 | ms p50 / p95 | rerank ms p50 |", "|---|---|---|---|---|"]
        for name, s in scores.items():
            lat = s.get("latency_ms", {})
            total = lat.get("total", {})
            rerank = lat.get("rerank", {}).get("p50")
            lines.append(
                f"| {name} | {s['means'][STAGE_FINAL]['nDCG@10']:.4f} | {s['means'][STAGE_1]['R@100']:.4f} | "
                f"{total.get('p50', 0):.0f} / {total.get('p95', 0):.0f} | "
                f"{f'{rerank:.0f}' if rerank is not None and reranks(name) else ''} |"
            )
        columns = ["Comparison", "ΔnDCG@10 (95%)", "W/L", "sign p"]
        columns += ["Adjusted", "Reading"] if confirmatory else []
        columns += ["ΔR@100 (95%)"]
        lines += ["", "| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
        for c in FAMILY:
            if c.candidate not in scores or c.baseline not in scores:
                lines.append(f"| {c.candidate} vs {c.baseline} | (not run) |")
                continue
            kwargs: dict[str, Any] = {"groups": groups} if grouped else {}
            base, cand = scores[c.baseline], scores[c.candidate]
            ndcg = compare_by_id(_per_query(base, STAGE_FINAL, "nDCG@10"), _per_query(cand, STAGE_FINAL, "nDCG@10"), **kwargs)
            recall = compare_by_id(_per_query(base, STAGE_1, "R@100"), _per_query(cand, STAGE_1, "R@100"), **kwargs)
            row = f"| {c.candidate} vs {c.baseline} | {_cell(ndcg)} | {ndcg.wins}/{ndcg.losses} | {ndcg.p_value:.2g} |"
            if confirmatory:
                adjusted = compare_by_id(
                    _per_query(base, STAGE_FINAL, "nDCG@10"),
                    _per_query(cand, STAGE_FINAL, "nDCG@10"),
                    confidence=level,
                    **kwargs,
                )
                row += f" [{adjusted.ci_low:+.4f}, {adjusted.ci_high:+.4f}] | {_reading(adjusted)} |"
            lines.append(row + f" {_cell(recall)} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 4: the query-time stack on BEIR.")
    parser.add_argument("--dataset", choices=DATASETS, action="append", help="Repeatable; default all with the split.")
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--variant", choices=list(VARIANTS), action="append", help="Repeatable; default all.")
    parser.add_argument("--rerun", action="store_true", help="Repeat runs already on disk.")
    parser.add_argument("--compare", action="store_true", help="Render the comparison table only.")
    args = parser.parse_args()
    configure_logging()

    if args.compare:
        table = compare(args.split)
        (RESULTS_DIR / f"{args.split}.md").parent.mkdir(parents=True, exist_ok=True)
        (RESULTS_DIR / f"{args.split}.md").write_text(table, encoding="utf-8")
        print(table)
        return 0
    datasets = args.dataset or [d for d in DATASETS if args.split in SPLITS[d]]
    for dataset in datasets:
        if args.split not in SPLITS[dataset]:
            logger.error("%s has no %s split", dataset, args.split)
            return 1
        run(dataset, args.split, args.variant or list(VARIANTS), rerun=args.rerun)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
