# BEIR phase 4 protocol (frozen)

Phase 4 of the [public benchmarks plan](public-benchmarks-plan.md): the
shipped query-time stages measured on FiQA, NFCorpus and SciFact. This page
records what was decided **before any phase-4 run touched a test split**. It
was committed ahead of the test runs, so its git history dates the freeze.
Scoring protocol, pins and the test-access log are in the
[BEIR reference protocol](beir-reference-protocol.md). Results are in
[measured results](measured-results.md#beir-query-time-stack-public-benchmarks-plan-phase-4).
Every test run executed the frozen commit `0cda7fd`. 18 of the 21 provenance
files record a later HEAD because another session switched the checkout's
branch mid-run; the results section explains why that changed nothing.

Runner: `scripts/run_beir_stack.py`. Its `VARIANTS` and `FAMILY` constants
are the machine-readable copy of this page. A change to either after the test
runs makes the affected readings exploratory.

## Variants

Every variant runs at the **benchmark depths**: 100 candidates per retriever,
100 after RRF fusion (also the reranker's pool), 10 final results. nDCG@10 is
read from the final ranking and R@100 from stage 1. This is a benchmark
configuration, not the shipped operating point: the shipped pipeline fuses
20 candidates and keeps 5. That point is not measured here, because nDCG@10
over 5 results measures a different output policy.

| Variant | Config | Differs from its config by |
|---|---|---|
| `bge-dense` | `beir_bge.yaml` | — (BGE-base, reference encoder, pinned revision) |
| `bge-hybrid` | `beir_bge.yaml` | `retrieval.mode: hybrid` |
| `bge-hybrid-rerank` | `beir_bge.yaml` | hybrid, `reranker.provider: cross_encoder` |
| `qwen-dense` | `beir.yaml` | — (`qwen3-embedding:0.6b`, the shipped embedder) |
| `qwen-hybrid` | `beir.yaml` | hybrid (descriptive only, not in the family) |
| `qwen-hybrid-rerank` | `beir.yaml` | hybrid + cross-encoder: the shipped stack |
| `qwen-dense-instruct` | `beir.yaml` | `embedding.query_instruction`: the model card's web-search task |

Shared settings, all at shipped values: sparse backend `bm25` (rank_bm25),
`rrf_k` 60, reranker `BAAI/bge-reranker-v2-m3` (Hugging Face snapshot recorded
per run), `min_score` 0, no expansion, no routing. `hnsw_ef_search` is 1600, which gives
≥ 0.998 of the exact dense top 100. FiQA dense rows keep a residual HNSW gap of
about 0.0017 nDCG@10 ([HNSW `ef_search`](measured-results.md#hnsw-ef_search)).
`--remove-query` applies to every variant, as the reference runs used it.

**Reranker input cap: 512 tokens** per (query, passage) pair
(`reranker.max_length`, set in `beir.yaml`). The shipped pipeline never
exceeds about 300 tokens: 1,000-character chunks plus their header. At the
model's own 8,192-token limit, scoring 100 FiQA passages from its long tail
(up to 17k characters, 3,848 tokens) exhausted memory on dev, and the run was
killed. The cap truncates 5.3% of FiQA passages, 16.8% of NFCorpus and
15.9% of SciFact. Its effect was measured on NFCorpus dev, which the full
length ran on before the failure. For `bge-hybrid-rerank`, the cap minus the
full length is **−0.0019 nDCG@10 [−0.0059, +0.0021]**, 51 wins and 47 losses
over 324 queries. That is inside the ±0.01 band, so no worthwhile effect. The
cap halves the median rerank time (4.5 s against 8.9 s) and cuts p95 from
22.8 s to 4.8 s. Peak memory on FiQA dev with the cap was 3.7 GB.

**Nothing was tuned.** Every parameter is shared across datasets and set to
its shipped value, except the reranker cap above. The dev splits (FiQA 500 queries, NFCorpus 324) were used
only to check the runner end to end, check repeatability and measure cost,
and to set the reranker cap. The cap was the one decision made on dev,
and it was made on memory, not on scores. Repeatability: two variants re-run on
NFCorpus dev gave identical per-query nDCG@10 and R@100. Their rankings
differed only where float-level variation in the query encoders (Ollama; MPS)
swapped near-tied documents. Dev scores are reported as exploratory and chose nothing.
SciFact has no dev split and is a transfer check on test.

## Comparison family

Five comparisons on each of three datasets: **15 confirmatory tests**, all on
nDCG@10.

| # | Candidate | Baseline | Question |
|---|---|---|---|
| 1 | `bge-hybrid` | `bge-dense` | Does BM25 fusion help dense retrieval? |
| 2 | `bge-hybrid-rerank` | `bge-hybrid` | Does the cross-encoder help over hybrid? |
| 3 | `qwen-dense` | `bge-dense` | The shipped embedder against the reference encoder, dense only |
| 4 | `qwen-hybrid-rerank` | `qwen-dense` | What the shipped stack adds over its own dense leg |
| 5 | `qwen-dense-instruct` | `qwen-dense` | Does Qwen's query instruction help on these queries? |

1–2 are the stage ablation under a controlled encoder. 4 is the plan's
required shipped combination: Qwen dense-only can't show how Qwen interacts
with fusion and reranking. 5 re-runs a feature that is off because it measured
no better than noise on EDGAR. It is cheap (query time only, no LLM) and
these queries resemble the model card's retrieval task.

**Multiplicity.** Bonferroni over the 15. The reading uses the paired
interval at level `1 − 0.05/15` (99.67%, z = 2.935). The 95% interval is
shown beside it.

**Readings**, from the adjusted interval on the mean nDCG@10 difference:

| Adjusted interval | Reading |
|---|---|
| entirely above 0 | *improved*; flagged "below 0.01" when the point estimate is under the smallest worthwhile effect |
| entirely below 0 | *worse*, with the same flag |
| inside (−0.01, +0.01) | *no worthwhile effect* |
| otherwise | *not shown* (neither a win nor evidence of no effect) |

- **Smallest worthwhile effect: 0.01 nDCG@10.** Below it, a difference is
  within FiQA's residual HNSW gap (0.0017) plus ordinary rank noise on a few
  hundred queries.
- **Acceptable regressions: none unreported.** The shipped stack
  (comparison 4) "transfers" only if no dataset reads *worse*. A *worse* on
  any dataset is reported as a regression, whatever the others show. For
  comparison 1, an R@100 loss beyond 0.01 is reported too, although R@100 is
  secondary.
- **Latency is reported, not gated.** Reranking 100 passages costs 3 s
  (FiQA) to 4.5 s (NFCorpus) per query at the median on this machine (M2,
  MPS, dev), far outside an interactive budget. A reranker gain here is a quality reading at a depth the shipped
  pipeline doesn't use. The interactive budget stays the shipped one,
  about 1.1 s to rerank 20 candidates on EDGAR.

**Secondary metric.** R@100 on stage 1, with 95% intervals and no
adjustment. For comparison 2 its difference is zero by construction: the
reranker sees the same fused 100.

**Sign test.** Wins, losses and the exact sign-test p-value are shown per
query. They test direction, not the mean, and no reading uses them.

## Dependence between queries

The interval assumes independent queries. Available provenance was checked
before any run. The grouping below reads the qrels only, not any score. Queries
that share a relevant document form a group (connected components):

| Dataset | Queries | Groups | Largest | Treatment |
|---|---|---|---|---|
| FiQA test | 648 | 648 | 1 | No shared relevant document: independent |
| SciFact test | 300 | 247 | 4 | Claims written from one abstract: **cluster-robust** interval over the 247 groups |
| NFCorpus test | 323 | 6 | 318 | One component: NFCorpus labels propagate through citations, so nearly every query shares documents with another. No usable grouping. Intervals treat queries as independent and **understate their width by an unknown amount** |

The rule in the runner: group when the largest group holds at most 5% of the
queries, else fall back and state the limitation. No EDGAR-style
company/filing groups are invented.

## What is not run

- **Query expansion and CRAG** (the plan's optional re-runs). Both add LLM
  calls per query and are stochastic, so they would need repeats. They are
  scoped out of this pass, not measured.
- **Chunking-time features.** BEIR passages bypass chunking.
- **The shipped depths** (20 / 5), as above.
- **`bm25` vs `sqlite_fts5` in the hybrid.** Phase 3 reported the backends'
  differences. The shipped backend is `bm25`.

## Test access

Each test run happens once, at a committed and clean tree (recorded in its
`provenance.json`), after this page is committed. A rerun is allowed only
for a failure unrelated to outcomes, such as a crash, and is logged in the
[test-access log](beir-reference-protocol.md#test-access-log).
