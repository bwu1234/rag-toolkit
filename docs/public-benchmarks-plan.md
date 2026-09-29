# Public benchmarks plan

Status: proposed, 2026-09-28. Nothing below is implemented. Sizes and query
counts come from the BEIR README; the reference scores are to be copied from
Pyserini's reproduction docs in phase 0, not quoted from memory.

Every number in [measured results](measured-results.md) comes from a corpus
this repo extracted, chunked and embedded itself, graded by questions written
from those same chunks. The [evaluation rigor plan](evaluation-rigor-plan.md)
names the consequence: the results support engineering decisions on EDGAR and
nothing wider. This plan adds a second evidence source that shares none of
that lineage: public corpora that arrive **already split into passages, with
independently authored relevance labels (qrels) and published reference
scores**.

It serves two purposes, in this order:

1. **Validate the harness and retrieval integration separately.** Score the
   same saved rankings with our evaluator and the reference evaluator, then
   reproduce pinned published retrieval runs. Agreement on identical rankings
   checks scoring; reproducing retrieval also checks data and encoder setup.
   Aggregate score agreement alone does not prove either implementation is
   correct, so retain fixtures and per-query comparisons.
2. **Measure the query-time stages on outside data.** Once the harness
   reproduces a known number, run the shipped hybrid + reranker pipeline on the
   same corpora and report the paired difference. That gives an external
   reading on the defaults, scoped to these tasks.

## What this does and does not measure

Because the corpus arrives as predefined retrieval units, this suite bypasses
the production extraction, cleaning, splitting, header and contextual chunking
stages. A new JSONL loader and identity chunker are still exercised and must
preserve the benchmark representation. The bypassed stages are where most
EDGAR findings came from. The suite also measures the configured
embedder only against queries of a different style from the EDGAR ones. This
plan complements the EDGAR suite and replaces nothing in it.

| Stage | Exercised? |
|---|---|
| Production extraction, cleaning, splitting, headers, contextual chunking | No: preserve the benchmark's retrieval units |
| BEIR JSONL loading, title/body serialization, identity chunking | Yes: validate ids and text against the pinned input |
| Embedding | Yes, but for reproduction it is pinned to the reference encoder |
| Dense, BM25 and hybrid retrieval, reranking | Yes: the point of phase 4 |
| Generation and answer judging | Not by BEIR, which has no reference answers (see [deferred work](#deferred-pre-embedded-vectors-and-answer-eval)) |

## Datasets

Starting set, chosen for size (each indexes locally in one sitting) and for
relevance to this repo:

| Dataset | Passages | Test queries | Qrels | Why |
|---|---|---|---|---|
| FiQA-2018 | 57K | 648 | ~2.6 relevant per query; inventory grade values in phase 0 | Financial questions: the closest public neighbour to EDGAR |
| SciFact | 5K | 300 | Binary, ~1.1 relevant per query | Small, and a different domain, so it catches overfitting to finance |
| NFCorpus | 3.6K | 323 | Graded, ~38 relevant per query | Many relevant passages per query, which exercises recall and nDCG differently |

Each is distributed by BEIR as one zip of `corpus.jsonl`, `queries.jsonl` and
available split files under `qrels/` (not every dataset has all three splits).
Larger BEIR sets (NQ 2.7M, HotpotQA 5.2M) are out of scope: see the deferred
section.

FiQA and NFCorpus ship a **dev** split. Any tuning (fusion weights,
`rerank_top_k`, thresholds) happens on dev. Freeze the complete comparison
family, configurations and acceptance rules before inspecting test outcomes;
multiple predeclared comparisons may share the test split. SciFact has train
and test but no dev split; this plan uses its test split only as a transfer
check, never for tuning.

Phase 3 necessarily uses published test results for reproduction. Keep a log
of test access and distinguish corrections justified by the pinned protocol
from changes chosen because they improve relevance scores. Freeze phase-4
choices on dev before inspecting test failure cases. If test outcomes inform
a feature, parameter or model choice, label subsequent results adaptive and
use fresh confirmation data for a new generalization claim. Public test sets
also do not establish that pretrained models have never seen these examples.

## Phase 0: pin the references

- For each dataset, record Pyserini's published nDCG@10 and R@100 for BM25
  (flat) and BGE-base-en-v1.5 (Faiss flat) from its reproduction docs. Pin the
  Pyserini commit the numbers came from.
- Pin the full protocol, including the dataset/split, qrels checksum, query
  population, run depth, evaluation command and evaluator version. Record
  handling of ties, missing queries, unjudged documents and identical query
  and document ids. Pyserini's reference commands include `--remove-query`;
  preserve that rule where the selected run uses it.
- Record exact title/body concatenation (separator, empty-title handling and
  whitespace), query instruction, model and tokenizer revisions, pooling,
  normalization, truncation/max length, precision and library versions.
  The model name alone does not identify an encoder configuration.
- For BM25, pin Lucene/Anserini versions, analyzer, stemming/stopword behavior,
  fields, BM25 parameters and tie handling. For dense retrieval, pin the
  similarity function and exact Faiss flat search settings. Keep commands
  and environment details sufficient to recreate both reference engines in
  an isolated reproduction environment.
- Record dataset source URL, release identifier where available, license,
  archive SHA-256, split inventory, actual counts and qrels grade histogram
  in its manifest. As with EDGAR, the data is gitignored and fetched, never
  committed; manifests and the conversion recipe are versioned.
- Set acceptance tolerances for **both nDCG@10 and R@100** before test
  inspection. The original ±0.005 nDCG@10 proposal is provisional, not an
  allowance to absorb a different retrieval algorithm. Use development runs
  or documented reference reproducibility to justify tolerances, and do not
  widen them after observing a test miss. Identical-ranking scoring has the
  stricter phase-2 numerical agreement requirement.

Exit: a table of both reference metrics for all six dataset/system pairs,
with commit-pinned sources, commands, protocol records and frozen tolerances.
Use the [Pyserini BEIR reproduction page](https://castorini.github.io/pyserini/2cr/beir.html)
and [BEIR dataset inventory](https://github.com/beir-cellar/beir/wiki/Datasets-available)
as starting points; choose the exact reference run before filling the table.

## Phase 1: pre-split corpora as named corpora

No new dependencies. `httpx`, `zipfile`, `json` and `csv` cover it.

- **Fetcher** `scripts/fetch_beir.py NAME`: downloads the zip, checks it against
  a sha256 in `data/corpora/beir-<name>/manifest.json`, and unpacks to
  `data/corpora/beir-<name>/documents/`. It follows the `fetch_edgar.py`
  pattern, so `manifest.json` is the provenance record.
- **JSONL loader and text contract**: load only `corpus.jsonl` as documents;
  queries and qrels must not enter the index. One `Document` per corpus line
  with `id = _id` taken verbatim, so qrels match ids with no mapping table.
  Preserve title in metadata and explicitly form the indexed/encoded text
  from title and body using the phase-0 recipe. Metadata alone does not put
  the title into embeddings, BM25 or reranker input. Use the same declared
  representation across phase-4 variants. This respects the no-namespacing
  rule in `rag/ingestion/corpora.py`: a pooled selection that collides raises, and
  BEIR corpora are only ever evaluated isolated.
- **Cleaning bypass**: add an explicit configuration path that preserves
  benchmark text. `chunk_selected_corpora` currently calls `clean_documents`
  unconditionally; an identity chunker alone cannot disable dehyphenation
  or whitespace rewriting. Include the text recipe and bypass settings in
  index provenance and require rebuilding when they change.
- **Identity chunker** (`chunking.provider: none`, via the `add-provider`
  skill): each document becomes one chunk. Without it, the default 1,000-char
  chunker splits long passages, and one passage then fills several top-10
  slots, which BEIR scoring does not allow. `index-report` should show zero
  split documents.
- **Eval-set converter** `scripts/beir_to_eval_set.py`: writes
  `data/eval/beir_<name>_<split>.json` with an explicit qrels scoring mode,
  carrying original grades and stable query ids (phase 2 adds the schema).
  Validate duplicate ids, split query membership and qrels document references;
  fail on malformed data rather than silently dropping records. Preserve
  zero judgments without counting them as relevant. The output is gitignored
  and regenerated from the pinned zip and versioned converter.
- **Registry entries** `beir-fiqa`, `beir-scifact` and `beir-nfcorpus`, plus
  `rag/config/beir.yaml` (`base: vanilla.yaml`, identity chunker, cleaning
  bypass, header/contextual chunking off, explicit retrieval depths).

Exit: `python -m rag.cli --config rag/config/beir.yaml index --corpus beir-scifact`
builds, and `index-report` with the same config shows one chunk per document
and a synced index. Counts, ids and serialized text agree with the source;
record disk/index size. Phases 1 and 2 can proceed in parallel after agreeing
the qrels schema; converter integration requires that schema to be available.

## Phase 2: standard retrieval metrics

Today's document-mode scoring cannot be compared with published numbers.
`_judge_by_document` in `rag/eval/relevance.py` grades every chunk from an
expected document as 1 and takes `[1] * k` as the ideal ranking. BEIR follows
`trec_eval`: graded gains per **distinct** document, and an ideal ranking
built from the qrels' grades.

There is also a gain-formula mismatch: `rag/eval/metrics.py` currently uses
`2**grade - 1`, while BEIR's `ndcg_cut` uses the grade directly. See the
[BEIR evaluator](https://github.com/beir-cellar/beir/blob/main/beir/retrieval/evaluation.py)
and [trec_eval implementation](https://github.com/usnistgov/trec_eval/blob/master/m_ndcg_cut.c).

- Accept graded document labels, mirroring the span object form:
  `expected_doc_ids: ["id", {"id": "...", "grade": 2}]`. A bare string keeps
  `DEFAULT_GRADE`, so existing sets are unaffected.
- Add an explicit qrels mode, leaving legacy document/span scoring unchanged.
  For nDCG@10, use linear gains `grade / log2(rank + 1)` and an ideal ranking
  built from all positive qrels grades, sorted and truncated at 10. Missing
  ranks contribute zero; a list shorter than 10 still has a defined score.
- R@100 is the number of distinct positively judged documents in the first
  100 results divided by all positively judged documents for the query.
  Unjudged and grade-zero documents receive no relevance credit. Match the
  pinned evaluator's policy for queries with no positive judgments.
- Require unique document ids in saved benchmark rankings; each document
  occupies one rank and receives credit once. Reject duplicate-id run files.
  Any adapter deduplication must retain the first occurrence, happen before
  cutoffs and be recorded. Follow the reference tie/self-match policy.
- Report fixed **nDCG@10** and **R@100** over the full declared query set,
  including empty or short returned lists. Do not reuse `recall_by_k`'s
  current behavior of omitting queries that returned fewer than k results.
- Expose candidate depth and final output depth explicitly. Adding `--top-k`
  alone is insufficient: `retrieval_eval` currently scores `retrieved.chunks`
  after truncation to `rerank_top_k`. Save stage-1 rankings for R@100 and final
  rankings for nDCG@10, with metric names identifying the stage. Baseline
  reproduction must retain at least 100 results without final truncation.
- Add hand-computed fixtures covering mixed grades, short/empty lists,
  unjudged results, ties and duplicate rejection. Cross-check identical saved
  rankings with a pinned `pytrec_eval` or `ir_measures` environment, including
  graded NFCorpus queries. These remain reproduction/test dependencies.

Exit: fixtures pass, and both metrics agree per query and in aggregate to
4 decimal places on identical rankings. Retain unrounded values to diagnose
differences. Run the same check over the complete reference runs in phase 3.

## Phase 3: validate scoring and reproduce the reference runs (the gate)

- Implement the `sentence_transformers` embedding provider. `EmbeddingConfig`
  already accepts `provider: sentence_transformers`, but
  `rag/embedding/factory.py` raises on it, so today that config value passes
  validation and fails at runtime. The library is already a dependency (the
  cross-encoder reranker uses it), so this adds no dependency. BGE takes a
  plain-prefix query instruction, not the Qwen `Instruct:` format the Ollama
  adapter builds, so the prefix format must be per-adapter.
- Recreate the pinned Pyserini Lucene BM25 and exact Faiss BGE runs for all
  three test splits in an isolated environment. Save full rankings and score
  them with both the reference evaluator and our qrels evaluator. Compare
  against both phase-0 metrics and inspect per-query disagreements.
  Record the exact reference index identity/checksum, including whether it
  was rebuilt or downloaded. A downloaded index validates reference search
  and scoring; local encoding is checked separately against the pinned recipe.
- Run this repo's BM25-only and BGE dense-only integrations on the same data.
  The shipped sparse backend uses regex tokenization and `rank_bm25` with
  modified IDF handling; the optional SQLite backend has its own BM25
  semantics. Neither is automatically equivalent to Lucene. Chroma uses
  HNSW, whereas the selected Faiss reference performs exact search.
- Diagnose differences in order: identical-ranking scoring, corpus/query
  representation, encoder outputs/settings, then ranking backend. Use an
  exact-search control over our BGE vectors to distinguish encoder differences
  from approximate-search loss. This can be a reproduction utility without
  adding a production Faiss provider. Record model/runtime/hardware settings
  and separate repeat variability from systematic protocol differences.

Exit requires **both** identical-ranking evaluator agreement on all reference
runs and reproduction of all six reference runs within the phase-0 tolerances
for both metrics. Unexplained discrepancies block phase 4. A score miss is
investigated; it is not automatically evidence of an evaluator bug.

For the repo backends, report differences from the reference separately.
Documented analyzer/BM25 or approximate-search differences may proceed to
phase 4 once scoring and integration checks pass; do not claim that those
backends reproduced the reference. Do not tune their parameters on test to
force agreement. Record the gate evidence in [measured results](measured-results.md)
with links to rankings, qrels, commands and immutable protocol records.

## Phase 4: measure the shipped query-time stack

With the harness validated, use the `measure-change` skill for paired runs on
each dataset:

- Dense (BGE-base) vs. hybrid (dense + BM25 fusion) vs. hybrid + cross-encoder
  reranker, using this repo's backends. Freeze a shared stage-1 budget of at
  least 100 for R@100 and return at least 10 final results when available.
  nDCG@10 remains defined for shorter lists, but truncating at the shipped
  five results would measure a different output policy. Record per-retriever
  depth, fused depth, reranker pool size and final depth separately.
- The shipped embedder (`qwen3-embedding:0.6b`) vs. BGE-base, both dense-only,
  as a transfer reading on the default embedder choice.
- Optional: re-run a measured-off query-time feature (query expansion, CRAG
  retrieval gating) where a dataset plausibly needs it. Chunking-time features
  cannot be measured here at all.

The BGE ablation measures pipeline stages under a controlled encoder. Include
the actual Qwen + hybrid + reranker combination if drawing conclusions about
the shipped stack; Qwen dense-only cannot establish its interaction with
fusion/reranking. Label the selected candidate/output depths as a benchmark
configuration (for example, 100 candidates and 10 outputs). Any run at shipped
depths is a separate operating point.

Before test evaluation, record the main metric, smallest worthwhile effect,
acceptable regressions, latency budget and comparison family. Tune only on
FiQA/NFCorpus dev, and state whether parameters are shared or dataset-specific;
freeze a transfer configuration for SciFact. Declare how multiple confirmatory
comparisons are handled. Optional feature sweeps are exploratory unless
included in the frozen family; repeat stochastic expansion/gating runs and
report run spread separately from variation across queries.

Report paired deltas per dataset with the uncertainty `rag/eval/paired.py`
produces: a normal interval over per-query differences and a separate sign
test on wins/losses. The sign test does not test the mean nDCG difference.
State the assumption that query units are independent; inspect available
provenance for related queries. Where meaningful dependencies exist, predefine
groups and use grouped resampling, or state the interval's limitation. Do not
invent EDGAR-style company/filing groups for these datasets.

Save per-query scores, both ranking stages, corpus/qrels/config/model hashes,
code revision, timings and environment details, following the provenance
requirements of the [eval harness plan](eval-harness-plan.md). These artifacts
are required even if its full orchestration is not yet shipped. Report latency
and index cost alongside quality, and word conclusions as the rigor plan
requires. For example: "hybrid + reranker improved nDCG@10 by X on FiQA test; no effect
shown on SciFact". Don't average across datasets into one headline.

## Deferred: pre-embedded vectors and answer eval

**Importing published vectors into the production pipeline** (Cohere's BEIR
and MS MARCO v2.1 embeddings, Pyserini's Faiss indexes, FlashRAG's e5 index
over Wikipedia-2018). Using reference indexes in phase 3's isolated
reproduction environment does not require a production import feature.
Measure local embedding time and memory in phase 3 before deciding whether
skipping encoding justifies the integration work:

- Queries need the same encoder. For Cohere vectors that means a paid API on
  every new query.
- The index manifest must record an embedder this config can reproduce.
  Otherwise `check_queryable` either refuses the run, or passes it for the
  wrong reason.
- For million-passage corpora, measure disk, memory, build time and search
  latency against the local budget. Corpus size alone does not establish a
  hard Chroma limit. Importing a published Faiss index would require a Faiss
  `VectorStore` adapter and dependency; other storage choices need their own
  capacity assessment.

Revisit only if a benchmark we need is too large to embed locally.

**Answer eval.** BEIR has no reference answers. Candidates, each with a
catch to settle before starting:

- FlashRAG's NQ and HotpotQA with gold answers. Their corpus is 21M
  passages, so this depends on the pre-embedded decision above.
- FinanceBench or FinDER. Nearest to the EDGAR workload, but distributed
  as raw filings, so our chunking is back in the loop.
- The reference-free metrics of
  [Milestone 22](backlog.md#milestone-22--reference-free-eval-metrics).

Public QA sets are probably in the generator's training data, so answer-pass
rates on them can come from memory rather than retrieval. Any such result
needs a closed-book control (the same questions with no retrieved context)
reported beside it.

## Costs and risks

- **Cost**: local execution with no planned API spend. Record data/model/index
  disk usage, peak memory and build/run time. The isolated reference environment
  adds Pyserini/Lucene/Faiss and evaluator setup costs without making them
  production dependencies.
- **Metric and protocol mismatch**: phase 2 checks scoring semantics; phase 3
  checks complete reference runs and diagnoses repo integration differences.
  Do not publish a phase-4 number before those gates pass.
- **Overfitting to public sets**: tune on dev only, and keep SciFact as
  transfer-only.
- **Scope creep toward leaderboard chasing**: the goal is an independent check
  on this system's retrieval, not a BEIR ranking.

## Delivery order

| Phase | Output | Depends on |
|---|---|---|
| 0 | Reference table with sources | — |
| 1 | Fetcher, text-preserving JSONL path, identity chunker, converter, registry/config | 0; converter integration uses phase-2 schema |
| 2 | Explicit qrels mode, linear-gain nDCG@10 / R@100, stage rankings, evaluator checks | Schema agreed with 1; fixtures can proceed in parallel |
| 3 | `sentence_transformers` embedder; reference reproduction, scoring parity, backend diagnostics | 0, 1, 2 |
| 4 | Paired measurements recorded in measured results | 3 |

Tracked under [Milestone 27](backlog.md#milestone-27--eval-coverage-and-judge-reliability)
as the "broaden the corpus" step of the rigor plan, limited to the query-time
stages.
