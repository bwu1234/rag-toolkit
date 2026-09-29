#!/usr/bin/env python
"""Run a retrieval-eval matrix over config variants and record the results.

Milestone 11 exists because contextual chunking, CRAG, query expansion and
`reranker.aggregate` all shipped functionally verified and numerically
unmeasured.  This is the harness that fixes that.

Design
------
**One factor at a time**, not a full factorial.  A full grid over the
interesting axes is dozens of runs, most of which answer nothing: what you
actually want to know is "does turning this one knob help, holding everything
else at the shipped defaults".  Interactions matter, but they are worth spending
runs on only after the main effects are known.

**Retrieval eval first.**  It needs no LLM at all -- just embeddings and
(optionally) a cross-encoder -- so the whole matrix runs in minutes against one
index.  Answer eval and contextual index builds cost hours and belong in a
second pass; serializing the cheap measurements behind them wastes a night.

**Every result carries its per-sample scores**, so a variant is compared with
the baseline question by question (:mod:`rag.eval.paired`) rather than by
eyeballing two means.  The table shows Δ hit and Δ NDCG with a 95% interval on
the *paired* difference, and the hit column's win/loss count -- the questions
the two runs disagree on, which is all a comparison can learn from.

**Every result carries its config.**  A metric with no config attached is
unreproducible, so each record stores the corpus, the eval set, the resolved
overrides, and a fingerprint over the settings that actually affect retrieval.
Two runs with the same fingerprint are comparable; two without are not.

Usage
-----
    python scripts/run_matrix.py --corpus edgar --eval-set data/eval/edgar_eval_set.json
    python scripts/run_matrix.py --only baseline,reranker   # a subset of axes
    python scripts/run_matrix.py --list                     # show variants, run nothing
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.config.settings import RagConfig, load_config  # noqa: E402
from rag.eval.dataset import EvalDataset, EvalSample  # noqa: E402
from rag.eval.metrics import wilson_interval  # noqa: E402
from rag.eval.paired import compare_by_id, format_difference  # noqa: E402
from rag.eval.retrieval_eval import run_retrieval_eval  # noqa: E402
from rag.ingestion.corpora import chunk_selected_corpora  # noqa: E402
from rag.ingestion.models import Document  # noqa: E402
from rag.logging_config import configure_logging  # noqa: E402
from rag.query_filter import QueryFilter  # noqa: E402
from rag.retrieval.builder import build_retriever  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_RESULTS_DIR = Path("data/eval/results")

# Config paths whose values change what retrieval returns. The fingerprint
# covers exactly these, so a result can be matched to the settings that produced
# it -- and so two runs that differ only in, say, log level compare equal.
FINGERPRINTED = (
    "chunking.chunk_size",
    "chunking.chunk_overlap",
    "chunking.contextual.enabled",
    "chunking.contextual.max_context_chars",
    "embedding.model",
    "retrieval.mode",
    "retrieval.top_k",
    "retrieval.rerank_top_k",
    "retrieval.rrf_k",
    "retrieval.min_score",
    "retrieval.expansion.provider",
    "retrieval.expansion.num_queries",
    "reranker.provider",
    "reranker.model",
    "reranker.aggregate",
)

# Fingerprinted only once set, so results recorded before these fields existed
# keep their fingerprint while the field sits at its default (`None`/`False`).
FINGERPRINTED_WHEN_SET = (
    "embedding.query_instruction",
    "chunking.header.template",
    "reranker.include_header",
    "retrieval.document_routing.top_m",
)

# The chunk header data/eval/config_header.yaml indexes EDGAR with.
HEADER_TEMPLATE = "{company} ({ticker}) {form}, period ended {period_end}"

# The default retrieval task from the Qwen3-Embedding model card.
QWEN3_RETRIEVAL_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"


@dataclass
class Variant:
    """One point in the matrix: a name, an axis, and the overrides to apply."""

    name: str
    axis: str
    overrides: dict[str, Any] = field(default_factory=dict)
    #: Set when a variant needs something the default index does not have.
    requires: str = ""
    #: Retrieve each sample with a metadata filter derived from its own labels:
    #: "company" (the expected filing's ticker) or "company+period" (ticker and
    #: period_end). What a caller who names those would pass; see `derive_filters`.
    filters: str = ""


# The matrix. `baseline` is the shipped default and every other variant differs
# from it in exactly one setting, so a difference is attributable.
#
# Deliberately NOT included here: `chunking.contextual.enabled`, which needs its
# own index build (~1.9h on this corpus) rather than a config flip, and belongs
# to the second pass.
VARIANTS: list[Variant] = [
    Variant("baseline", "baseline", {}),

    # Does keyword search earn its place, or is dense retrieval enough?
    Variant("mode=dense", "mode", {"retrieval.mode": "dense"}),
    Variant("mode=hybrid", "mode", {"retrieval.mode": "hybrid"}),

    # The cross-encoder costs ~3,500 pairs per eval run here. Worth it?
    Variant("reranker=none", "reranker", {"reranker.provider": "none"}),
    Variant("reranker=cross_encoder", "reranker", {"reranker.provider": "cross_encoder"}),

    # min_score is hand-tuned and its meaningful range shifts with the reranker;
    # too high refuses answerable questions, too low does nothing.
    Variant("min_score=0.0", "min_score", {"retrieval.min_score": 0.0}),
    Variant("min_score=0.1", "min_score", {"retrieval.min_score": 0.1}),
    Variant("min_score=0.3", "min_score", {"retrieval.min_score": 0.3}),

    # Does a bigger candidate set help, or is the loss in reranking?
    Variant("rerank_top_k=10", "rerank_top_k", {"retrieval.rerank_top_k": 10}),
    Variant("rerank_top_k=20", "rerank_top_k", {"retrieval.rerank_top_k": 20}),

    # Stage-1 ceiling: with rerank_top_k == top_k the reranker only reorders, it
    # never filters, so `recall` here is what vector+BM25 retrieval could deliver
    # at best. The gap between this and a variant that reranks down to 5 is the
    # cost of reranking; the gap between this and 1.0 is what retrieval never
    # found at all. Those are different problems with different fixes.
    Variant("stage1_top_k=20", "stage1",
            {"retrieval.top_k": 20, "retrieval.rerank_top_k": 20}),
    Variant("stage1_top_k=50", "stage1",
            {"retrieval.top_k": 50, "retrieval.rerank_top_k": 50}),
    Variant("stage1_top_k=100", "stage1",
            {"retrieval.top_k": 100, "retrieval.rerank_top_k": 100}),

    # Qwen3-Embedding's query-side instruction (chunking plan, Phase 1), with the
    # model card's default retrieval task. It changes only query vectors, so no
    # reindex. The stage-1 variant pairs with `stage1_top_k=20`: the misses this
    # targets are chunks stage 1 ranks below 20, before the reranker sees them.
    Variant("query_instruction=retrieval", "query_instruction",
            {"embedding.query_instruction": QWEN3_RETRIEVAL_INSTRUCTION}),
    Variant("query_instruction=retrieval stage1_top_k=20", "query_instruction",
            {"embedding.query_instruction": QWEN3_RETRIEVAL_INSTRUCTION,
             "retrieval.top_k": 20, "retrieval.rerank_top_k": 20}),
    # The embedder's own effect, with no BM25 list for fusion to lean on. Pairs
    # with `mode=dense`.
    Variant("query_instruction=retrieval mode=dense", "query_instruction",
            {"embedding.query_instruction": QWEN3_RETRIEVAL_INSTRUCTION,
             "retrieval.mode": "dense"}),

    # Embedder size (chunking plan, Phase 1b), Q8_0 throughout. Each size has its
    # own index (data/eval/config_embedder_*.yaml builds them), so the variant
    # points `paths.index_dir` at it as well as naming the model. Same three
    # pairs as the query instruction: shipped, stage-1 ceiling, dense-only.
    *[
        Variant(f"embedder={size}{suffix}", "embedder",
                {"embedding.model": model, "paths.index_dir": f"data/index_emb-{size}", **extra},
                requires="index")
        for size, model in (("4b", "qwen3-embedding:4b-q8_0"), ("8b", "qwen3-embedding:8b-q8_0"))
        for suffix, extra in (
            ("", {}),
            (" stage1_top_k=20", {"retrieval.top_k": 20, "retrieval.rerank_top_k": 20}),
            (" mode=dense", {"retrieval.mode": "dense"}),
        )
    ],
    # Phase 1's instruction verdict was measured on 0.6b only; a different
    # checkpoint may respond differently. Pairs with `embedder=4b` and
    # `embedder=4b mode=dense`, the size that led at the shipped config.
    Variant("embedder=4b query_instruction=retrieval", "embedder",
            {"embedding.model": "qwen3-embedding:4b-q8_0", "paths.index_dir": "data/index_emb-4b",
             "embedding.query_instruction": QWEN3_RETRIEVAL_INSTRUCTION}, requires="index"),
    Variant("embedder=4b query_instruction=retrieval mode=dense", "embedder",
            {"embedding.model": "qwen3-embedding:4b-q8_0", "paths.index_dir": "data/index_emb-4b",
             "embedding.query_instruction": QWEN3_RETRIEVAL_INSTRUCTION,
             "retrieval.mode": "dense"}, requires="index"),

    # Deterministic chunk header (chunking plan, Phase 2): "{company} ({ticker})
    # {form}, period ended {period_end}" indexed ahead of each chunk, from the
    # index data/eval/config_header.yaml builds. Same three pairs as the
    # embedder, plus the reranker scoring the header too (pairs with
    # `header=on`), the plan's open question about what the cross-encoder sees.
    # The plain pipeline with every Phase 2 setting pinned off, for pairing now
    # that `baseline` (the shipped default) has the header on. It reads the
    # header-free index `rag/config/vanilla.yaml` builds: the same chunks and
    # embedder as the shipped index, no header.
    Variant("header=off", "header",
            {"paths.index_dir": "data/index_vanilla",
             "chunking.header.template": None,
             "reranker.include_header": False},
            requires="index"),
    *[
        Variant(f"header=on{suffix}", "header",
                {"paths.index_dir": "data/index_header",
                 "chunking.header.template": HEADER_TEMPLATE,
                 # Pinned: the shipped default turned this on after these rows
                 # were measured, and `header=on` means the header alone.
                 "reranker.include_header": False, **extra},
                requires="index")
        for suffix, extra in (
            ("", {}),
            (" stage1_top_k=20", {"retrieval.top_k": 20, "retrieval.rerank_top_k": 20}),
            (" mode=dense", {"retrieval.mode": "dense"}),
            (" rerank_header", {"reranker.include_header": True}),
        )
    ],

    # Metadata filtering (chunking plan, Phase 3), at the shipped config. Each
    # sample is filtered by what its own labels say a caller would name. The
    # oracle before the interface existed found "company" adds nothing after
    # Phase 2 and the headroom is in the period; these measure the real
    # `QueryFilter` path end to end.
    Variant("filters=company", "filters", {}, filters="company"),
    Variant("filters=company+period", "filters", {}, filters="company+period"),

    # Document routing (chunking plan, Phase 3b): the filter above, but chosen
    # by ranking one record per filing instead of taken from labels. Routes only
    # when BM25 and dense agree on the top filing. `filters=company+period` is
    # its ceiling.
    Variant("routing=top1", "routing", {"retrieval.document_routing.top_m": 1}),
    Variant("routing=top2", "routing", {"retrieval.document_routing.top_m": 2}),

    # Sparse backend: SQLite FTS5 in place of rank_bm25. Same tokens, slightly
    # different BM25 (k1 1.2 vs 1.5, IDF floor), so the keyword leg ranks
    # differently. Needs `fts5_index__<slug>.sqlite3` built for the corpus.
    # The rerank_top_k=20 pair reranks nothing, so it compares stage 1 alone
    # against the existing `rerank_top_k=20` row.
    Variant("sparse=sqlite_fts5", "sparse", {"sparse_index.provider": "sqlite_fts5"},
            requires="index"),
    Variant("sparse=sqlite_fts5 rerank_top_k=20", "sparse",
            {"sparse_index.provider": "sqlite_fts5", "retrieval.rerank_top_k": 20},
            requires="index"),

    # The payoff question the stage-1 axis raises: retrieval can surface the
    # right chunk far more often with a bigger candidate pool, but that is only
    # useful if the reranker promotes it into the handful the LLM actually sees.
    # These hold rerank_top_k at the shipped 5 and vary only the pool.
    Variant("pool=20", "pool", {"retrieval.top_k": 20}),
    Variant("pool=50", "pool", {"retrieval.top_k": 50}),
    Variant("pool=100", "pool", {"retrieval.top_k": 100}),

    # Reranker models, compared like for like. `min_score` is pinned to 0.0
    # throughout: the floor's meaning depends on the score distribution of
    # whichever model produced it, so leaving it at the value hand-tuned for
    # MiniLM would confound "is this reranker better" with "does this reranker
    # happen to score above an arbitrary threshold".
    Variant("rr=minilm-L6", "reranker_model",
            {"reranker.model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
             "retrieval.min_score": 0.0}, requires="weights"),
    Variant("rr=bge-base", "reranker_model",
            {"reranker.model": "BAAI/bge-reranker-base",
             "retrieval.min_score": 0.0}, requires="weights"),
    Variant("rr=bge-v2-m3", "reranker_model",
            {"reranker.model": "BAAI/bge-reranker-v2-m3",
             "retrieval.min_score": 0.0}, requires="weights"),
    # The official Qwen/Qwen3-Reranker-* is generative -- it scores by comparing
    # "yes"/"no" token logits behind an instruction template, which is not a
    # sequence-classification head and does not load as a CrossEncoder. This is
    # the seq-cls conversion of the same weights, which does.
    Variant("rr=qwen3-0.6b", "reranker_model",
            {"reranker.model": "tomaarsen/Qwen3-Reranker-0.6B-seq-cls",
             "retrieval.min_score": 0.0}, requires="weights"),
    # Same weights, scored behind the instruction template Qwen3-Reranker was
    # trained with. Kept as a separate variant rather than replacing the one
    # above because the difference between them IS the finding: "loads as a
    # CrossEncoder" and "works as a CrossEncoder" are not the same claim.
    Variant("rr=qwen3-0.6b +prompt", "reranker_model",
            {"reranker.model": "tomaarsen/Qwen3-Reranker-0.6B-seq-cls",
             "reranker.query_prefix":
                 "<Instruct>: Given a web search query, retrieve relevant passages "
                 "that answer the query\n<Query>: {query}",
             "reranker.document_prefix": "<Document>: {document}",
             "retrieval.min_score": 0.0}, requires="weights"),

    # The payoff test. MiniLM could not exploit a larger candidate pool -- hit
    # rate went DOWN from 0.770 to 0.759 as top_k rose 20 -> 100, while stage 1
    # was handing it the right chunk 97.7% of the time. If a stronger reranker
    # converts that pool into results, the ceiling is reranker quality; if it
    # also flattens, the loss is somewhere else entirely.
    Variant("rr=minilm-L6 pool=100", "reranker_pool",
            {"reranker.model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
             "retrieval.top_k": 100, "retrieval.min_score": 0.0}, requires="weights"),
    Variant("rr=bge-base pool=100", "reranker_pool",
            {"reranker.model": "BAAI/bge-reranker-base",
             "retrieval.top_k": 100, "retrieval.min_score": 0.0}, requires="weights"),
    Variant("rr=bge-v2-m3 pool=100", "reranker_pool",
            {"reranker.model": "BAAI/bge-reranker-v2-m3",
             "retrieval.top_k": 100, "retrieval.min_score": 0.0}, requires="weights"),
    Variant("rr=qwen3-0.6b pool=100", "reranker_pool",
            {"reranker.model": "tomaarsen/Qwen3-Reranker-0.6B-seq-cls",
             "retrieval.top_k": 100, "retrieval.min_score": 0.0}, requires="weights"),

    # Expansion needs an LLM call per query, so it is the slow end of this
    # matrix -- and it is non-deterministic, so a single run is a data point,
    # not a result.
    Variant("expansion=hyde", "expansion", {"retrieval.expansion.provider": "hyde"},
            requires="llm"),
    Variant("expansion=multi_query", "expansion",
            {"retrieval.expansion.provider": "multi_query"}, requires="llm"),
    Variant("aggregate=mean", "aggregate",
            {"retrieval.expansion.provider": "multi_query", "reranker.aggregate": "mean"},
            requires="llm"),
]


def derive_filters(kind: str, documents: list[Document]) -> Callable[[EvalSample], QueryFilter | None]:
    """A per-sample `QueryFilter` built from the sample's `expected_doc_ids`.

    Labels stand in for a caller here: "company" is what someone naming only
    the company would pass, "company+period" someone naming both. The filter
    is shaped the way a caller would write it (a ticker and a `period_end`
    range), not as a list of document ids, so it also admits any other filing
    that shares them. A sample with no expected documents gets no filter.
    """
    metadata = {d.id: d.metadata for d in documents}

    def for_sample(sample: EvalSample) -> QueryFilter | None:
        expected = [metadata[doc_id] for doc_id in sample.expected_doc_ids if doc_id in metadata]
        if not expected:
            return None
        tickers = sorted({m["ticker"] for m in expected})
        if kind == "company":
            return QueryFilter(any_of={"ticker": tickers})
        if kind == "company+period":
            periods = [m["period_end"] for m in expected]
            return QueryFilter(any_of={"ticker": tickers}, range={"period_end": {"gte": min(periods), "lte": max(periods)}})
        raise ValueError(f"Unknown derived filter kind: {kind!r}")

    return for_sample


def apply_overrides(config: RagConfig, overrides: dict[str, Any]) -> RagConfig:
    """Return a deep copy of `config` with dotted-path values replaced."""
    updated = config.model_copy(deep=True)
    for path, value in overrides.items():
        target: Any = updated
        parts = path.split(".")
        for part in parts[:-1]:
            target = getattr(target, part)
        setattr(target, parts[-1], value)
    return updated


def read_path(config: RagConfig, path: str) -> Any:
    target: Any = config
    for part in path.split("."):
        target = getattr(target, part)
    return target


def fingerprint(config: RagConfig) -> tuple[str, dict[str, Any]]:
    """Hash the retrieval-relevant settings, and return them alongside the hash."""
    settings = {path: read_path(config, path) for path in FINGERPRINTED}
    settings.update(
        {
            path: value
            for path in FINGERPRINTED_WHEN_SET
            if (value := read_path(config, path)) is not None and value is not False
        }
    )
    if config.sparse_index.provider != "bm25":
        # Fingerprinted only off its default, so rows recorded before the
        # setting existed (all on bm25) keep their fingerprint.
        settings["sparse_index.provider"] = config.sparse_index.provider
    if config.retrieval.document_routing.top_m is not None:
        # The template shapes routing only while routing is on.
        settings["retrieval.document_routing.record_template"] = config.retrieval.document_routing.record_template
    serialized = json.dumps({k: str(v) for k, v in settings.items()}, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12], settings


def run_variant(
    variant: Variant, base: RagConfig, dataset: EvalDataset, corpora: list[str] | None
) -> dict[str, Any]:
    config = apply_overrides(base, variant.overrides)
    digest, settings = fingerprint(config)
    if variant.filters:
        # Derived filters change what every sample retrieves, so they are part
        # of what makes two runs comparable.
        settings["eval.derived_filters"] = variant.filters
        digest = hashlib.sha256(json.dumps({k: str(v) for k, v in settings.items()}, sort_keys=True).encode()).hexdigest()[:12]

    logger.info("[%s] %s", variant.name, variant.overrides or "(shipped defaults)")
    # Re-chunked per variant, since a variant may override chunking; it costs
    # about a second, against minutes for the retrieval itself.
    _selection, documents, corpus_chunks = chunk_selected_corpora(config, corpora)
    filters_for = derive_filters(variant.filters, documents) if variant.filters else None
    started = time.monotonic()
    retriever = build_retriever(config, corpora=corpora)
    report = run_retrieval_eval(dataset, retriever, corpus_chunks=corpus_chunks, filters_for=filters_for)
    elapsed = time.monotonic() - started

    summary = report.overall

    def metrics(s: Any) -> dict[str, float]:
        return {
            "hit_rate": round(s.mean_hit_rate, 4),
            "recall": round(s.mean_recall, 4),
            "precision": round(s.mean_precision, 4),
            "mrr": round(s.mrr, 4),
            "ndcg": round(s.mean_ndcg, 4),
        }

    return {
        "variant": variant.name,
        "axis": variant.axis,
        "overrides": variant.overrides,
        **({"filters": variant.filters} if variant.filters else {}),
        "fingerprint": digest,
        "settings": {k: str(v) for k, v in settings.items()},
        "num_samples": summary.num_samples,
        "elapsed_s": round(elapsed, 1),
        "metrics": metrics(summary),
        # Per-kind figures (the underspecified tier's implicit / paraphrase),
        # which the chunking plan reports separately rather than averaged.
        "by_kind": {
            k.label: {"num_samples": k.num_samples, "metrics": metrics(k)} for k in report.by_kind
        },
        "recall_by_k": {str(k): round(v, 4) for k, v in sorted(summary.recall_by_k.items())},
        # Samples that can't score under this variant's chunking, whatever
        # retrieval does. Two variants with different counts differ in what
        # they *could* find, not only in what they did.
        "unmatchable_spans": [
            {"sample_id": u.sample_id, "span": u.span} for u in report.unmatchable_spans or []
        ],
        # Per-sample scores, keyed by sample id, for paired comparison. Without
        # them a later reader can compare means but never test a difference.
        # Keys match `metrics`, whose values are the means of these.
        "samples": {
            r.sample_id: {
                "hit_rate": r.hit,
                "recall": round(r.recall, 4),
                "mrr": round(r.rr, 4),
                "ndcg": round(r.ndcg, 4),
                **({"kind": r.kind} if r.kind is not None else {}),
                **({"routed_to": r.routed_to} if r.routed_to else {}),
            }
            for r in report.sample_results
        },
    }


def paired_delta(
    result: dict[str, Any], baseline: dict[str, Any], metric: str, *, binary: bool = False
) -> str:
    """Table cell for `metric`, candidate minus baseline, paired by sample id.

    Rows recorded before per-sample scores were stored get the bare difference
    of means, marked so it is not read as tested.
    """
    delta = result["metrics"][metric] - baseline["metrics"][metric]
    if "samples" not in result or "samples" not in baseline:
        return f"{delta:+.3f} (no CI)"
    try:
        diff = compare_by_id(
            {sid: s[metric] for sid, s in baseline["samples"].items()},
            {sid: s[metric] for sid, s in result["samples"].items()},
        )
    except ValueError:
        return f"{delta:+.3f} (unpaired)"
    return format_difference(diff, binary=binary)


def hit_interval(result: dict[str, Any], kind: str | None = None) -> str:
    """95% Wilson interval on a row's hit rate (or one kind's), from its per-sample scores."""
    samples = [
        s for s in result.get("samples", {}).values() if kind is None or s.get("kind") == kind
    ]
    if not samples:
        return "—"
    low, high = wilson_interval(sum(1 for s in samples if s["hit_rate"]), len(samples))
    return f"[{low:.3f}, {high:.3f}]"


def render_table(results: list[dict[str, Any]]) -> str:
    """Markdown table, grouped by axis, with paired deltas against the baseline."""
    baseline = next((r for r in results if r["variant"] == "baseline"), None)
    lines = [
        "| variant | hit | hit 95% CI | recall | prec | MRR | NDCG | Δ hit [95% CI] | Δ NDCG [95% CI] "
        "| unmatch. | s |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    last_axis = None
    for result in results:
        if last_axis is not None and result["axis"] != last_axis:
            lines.append("| | | | | | | | | | | |")
        last_axis = result["axis"]
        m = result["metrics"]
        if baseline and result["variant"] != "baseline":
            hit_str = paired_delta(result, baseline, "hit_rate", binary=True)
            ndcg_str = paired_delta(result, baseline, "ndcg")
        else:
            hit_str = ndcg_str = "—"
        unmatchable = result.get("unmatchable_spans")
        unmatchable_str = "—" if unmatchable is None else str(len(unmatchable))
        lines.append(
            f"| `{result['variant']}` | {m['hit_rate']:.3f} | {hit_interval(result)} | {m['recall']:.3f} | "
            f"{m['precision']:.3f} | {m['mrr']:.3f} | {m['ndcg']:.3f} | {hit_str} | "
            f"{ndcg_str} | {unmatchable_str} | {result['elapsed_s']:.0f} |"
        )
    with_kinds = [r for r in results if r.get("by_kind")]
    if with_kinds:
        lines += [
            "",
            "By kind, each reported on its own (the rows above mix them):",
            "",
            "| variant | kind | n | hit | hit 95% CI | NDCG | Δ hit [95% CI] |",
            "|---|---|---|---|---|---|---|",
        ]
        for result in with_kinds:
            for kind, k in result["by_kind"].items():
                delta = "—"
                if baseline and result["variant"] != "baseline" and kind in baseline.get("by_kind", {}):
                    delta = paired_delta(
                        _only_kind(result, kind), _only_kind(baseline, kind), "hit_rate", binary=True
                    )
                lines.append(
                    f"| `{result['variant']}` | {kind} | {k['num_samples']} | "
                    f"{k['metrics']['hit_rate']:.3f} | {hit_interval(result, kind)} | "
                    f"{k['metrics']['ndcg']:.3f} | {delta} |"
                )
    lines += [
        "",
        "`hit 95% CI` is a Wilson interval on that rate alone, the noise floor of one "
        "run on this many questions. Δ is variant minus `baseline`, paired by sample. `*` marks a 95% interval "
        "that excludes zero; `W/L` counts the questions the variant gained / lost and "
        "`p` is McNemar's exact test on them -- trust it over the CI when W+L is small. "
        "`(no CI)` rows predate per-sample scores and need a re-run to be tested. "
        "`unmatch.` counts expected spans no chunk contains under that variant's "
        "chunking (`—` predates the count).",
    ]
    return "\n".join(lines)


def _only_kind(result: dict[str, Any], kind: str) -> dict[str, Any]:
    """A row narrowed to one kind's samples and metrics, for a per-kind paired delta."""
    return {
        "metrics": result["by_kind"][kind]["metrics"],
        "samples": {sid: s for sid, s in result["samples"].items() if s.get("kind") == kind},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a retrieval-eval config matrix.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--corpus", action="append", default=None, metavar="NAME")
    parser.add_argument("--eval-set", type=Path, default=Path("data/eval/edgar_eval_set.json"))
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--only", default=None,
                        help="Comma-separated axes to run (e.g. baseline,reranker,min_score)")
    parser.add_argument("--variant", action="append", default=None, metavar="NAME",
                        help="Run only these variants by name (repeatable). Narrower than --only.")
    parser.add_argument("--merge", action="store_true",
                        help="Merge into the existing results file, keyed by variant name, "
                             "instead of replacing it -- so adding one model does not mean "
                             "re-running an hour of already-measured variants.")
    parser.add_argument("--skip-llm", action="store_true",
                        help="Skip variants needing an LLM call per query (expansion)")
    parser.add_argument("--list", action="store_true", help="List variants and exit")
    parser.add_argument("--label", default="", help="Free-text label stored with the results")
    args = parser.parse_args()

    configure_logging()

    variants = VARIANTS
    if args.only:
        wanted = {a.strip() for a in args.only.split(",")}
        variants = [v for v in variants if v.axis in wanted]
    if args.variant:
        wanted_names = set(args.variant)
        variants = [v for v in variants if v.name in wanted_names]
        unknown = wanted_names - {v.name for v in VARIANTS}
        if unknown:
            logger.error("Unknown variant(s): %s", ", ".join(sorted(unknown)))
            return 1
    if args.skip_llm:
        variants = [v for v in variants if v.requires != "llm"]

    if args.list:
        for v in variants:
            marker = " (needs LLM)" if v.requires == "llm" else ""
            derived = f" [filters derived: {v.filters}]" if v.filters else ""
            print(f"  {v.axis:<14} {v.name:<26} {v.overrides or '(defaults)'}{derived}{marker}")
        return 0

    if not args.eval_set.exists():
        logger.error("Eval set not found: %s", args.eval_set)
        return 1

    base = load_config(args.config)
    dataset = EvalDataset.load(args.eval_set)
    selection = base.corpus_selection(args.corpus)
    logger.info(
        "Matrix: %d variant(s), %d sample(s), corpus %s",
        len(variants), len(dataset), selection.describe(),
    )

    args.results_dir.mkdir(parents=True, exist_ok=True)
    stem = f"retrieval_{selection.slug}_{args.eval_set.stem}"
    destination = args.results_dir / f"{stem}.json"

    # Seed from the existing file when merging, so a run that adds one model to
    # an already-measured matrix keeps the rest.
    accumulated: dict[str, dict[str, Any]] = {}
    if args.merge and destination.exists():
        accumulated = {r["variant"]: r for r in json.loads(destination.read_text())["results"]}
        logger.info("Merging into %d existing variant(s) in %s", len(accumulated), destination)

    def flush() -> list[dict[str, Any]]:
        """Persist everything measured so far. Called after every variant.

        Writing once at the end loses the whole run to an interruption -- which
        is exactly what happened on the first Qwen run: the cheap variant had
        finished, the expensive one was still going, and both were lost. A
        variant here costs minutes to hours, so results are checkpointed for the
        same reason chunk contexts are.
        """
        order = {v.name: i for i, v in enumerate(VARIANTS)}
        ordered = sorted(accumulated.values(), key=lambda r: order.get(r["variant"], 999))
        record = {
            "label": args.label,
            "corpus": selection.describe(),
            "collection": selection.collection_name,
            "eval_set": str(args.eval_set),
            "num_samples": len(dataset),
            "results": ordered,
        }
        destination.write_text(json.dumps(record, indent=2) + "\n")
        (args.results_dir / f"{stem}.md").write_text(
            f"# Retrieval matrix — {selection.describe()}\n\n"
            f"- Eval set: `{args.eval_set}` ({len(dataset)} samples)\n"
            f"- Collection: `{selection.collection_name}`\n\n{render_table(ordered)}\n"
        )
        return ordered

    ordered: list[dict[str, Any]] = flush() if accumulated else []
    for variant in variants:
        accumulated[variant.name] = run_variant(variant, base, dataset, args.corpus)
        ordered = flush()
        logger.info("Checkpointed %d variant(s) to %s", len(ordered), destination)

    print(f"\n{render_table(ordered)}\n")
    print(f"Wrote {args.results_dir / stem}.json / .md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
