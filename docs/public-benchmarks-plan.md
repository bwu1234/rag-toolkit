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

1. **Validate the harness.** Run plain dense retrieval with the same encoder a
   published leaderboard used, and require the same score. That is the only
   check available on whether `retrieval_eval` computes what it claims to; a
   self-authored eval set cannot provide it.
2. **Measure the query-time stages on outside data.** Once the harness
   reproduces a known number, run the shipped hybrid + reranker pipeline on the
   same corpora and report the paired difference. That gives an external
   reading on the defaults, scoped to these tasks.

## What this does and does not measure

Because the corpus arrives pre-split, the chunker, cleaners, loaders, chunk
header and contextual chunking are **bypassed**, not tested. Those stages are
where most EDGAR findings came from. The suite also measures the configured
embedder only against queries of a different style from the EDGAR ones. This
plan complements the EDGAR suite and replaces nothing in it.

| Stage | Exercised? |
|---|---|
| Loading, cleaning, chunking, headers, contextual chunking | No: passages arrive pre-split |
| Embedding | Yes, but for reproduction it is pinned to the reference encoder |
| Dense, BM25 and hybrid retrieval, reranking | Yes: the point of phase 4 |
| Generation and answer judging | Not by BEIR, which has no reference answers (see [deferred work](#deferred-pre-embedded-vectors-and-answer-eval)) |

## Datasets

Starting set, chosen for size (each indexes locally in one sitting) and for
relevance to this repo:

| Dataset | Passages | Test queries | Qrels | Why |
|---|---|---|---|---|
| FiQA-2018 | 57K | 648 | Graded, ~2.6 relevant per query | Financial questions: the closest public neighbour to EDGAR |
| SciFact | 5K | 300 | Binary, ~1.1 relevant per query | Small, and a different domain, so it catches overfitting to finance |
| NFCorpus | 3.6K | 323 | Graded, ~38 relevant per query | Many relevant passages per query, which exercises recall and nDCG differently |

Each is distributed by BEIR as one zip of `corpus.jsonl`, `queries.jsonl` and
`qrels/{train,dev,test}.tsv`. Larger BEIR sets (NQ 2.7M, HotpotQA 5.2M) are
out of scope: see the deferred section.

FiQA and NFCorpus ship a **dev** split. Any tuning (fusion weights,
`rerank_top_k`, thresholds) happens on dev. The test split is scored once per
frozen comparison, following the rigor plan's holdout discipline. SciFact has
no dev split, so it is used only as a transfer check, never for tuning.

## Phase 0: pin the references

- For each dataset, record Pyserini's published nDCG@10 and R@100 for BM25
  (flat) and BGE-base-en-v1.5 (Faiss flat) from its reproduction docs. Pin the
  Pyserini commit the numbers came from.
- Record exactly how Pyserini formed the document text it encoded (title and
  body concatenation) and the BGE query instruction it used. Reproduction
  fails on either difference without any error.
- Record each dataset's license in its manifest. As with EDGAR, the data is
  gitignored and fetched, never committed.

Exit: a table of reference scores in this doc, with its source link.

## Phase 1: pre-split corpora as named corpora

No new dependencies. `httpx`, `zipfile`, `json` and `csv` cover it.

- **Fetcher** `scripts/fetch_beir.py NAME`: downloads the zip, checks it against
  a sha256 in `data/corpora/beir-<name>/manifest.json`, and unpacks to
  `data/corpora/beir-<name>/documents/`. It follows the `fetch_edgar.py`
  pattern, so `manifest.json` is the provenance record.
- **JSONL loader**: one `Document` per line with `id = _id` taken verbatim, so
  qrels match document ids with no mapping table. `title` goes into
  `metadata`. This respects the no-namespacing rule in
  `rag/ingestion/corpora.py`: a pooled selection that collides raises, and
  BEIR corpora are only ever evaluated isolated.
- **Identity chunker** (`chunking.provider: none`, via the `add-provider`
  skill): each document becomes one chunk. Without it, the default 1,000-char
  chunker splits long passages, and one passage then fills several top-10
  slots, which BEIR scoring does not allow. `index-report` should show zero
  split documents.
- **Eval-set converter** `scripts/beir_to_eval_set.py`: writes
  `data/eval/beir_<name>_<split>.json` in `matching_mode: document`, carrying
  the qrels grades (phase 2 adds the field). The output is gitignored and
  regenerated from the pinned zip.
- **Registry entries** `beir-fiqa`, `beir-scifact` and `beir-nfcorpus`, plus
  `rag/config/beir.yaml` (`base: vanilla.yaml`, identity chunker, header off).

Exit: `rag.cli index --corpus beir-scifact` builds, and `index-report` shows
one chunk per document and a synced index.

## Phase 2: standard retrieval metrics

Today's document-mode scoring cannot be compared with published numbers.
`_judge_by_document` in `rag/eval/relevance.py` grades every chunk from an
expected document as 1 and takes `[1] * k` as the ideal ranking. BEIR follows
`trec_eval`: graded gains per **distinct** document, and an ideal ranking
built from the qrels' grades.

- Accept graded document labels, mirroring the span object form:
  `expected_doc_ids: ["id", {"id": "...", "grade": 2}]`. A bare string keeps
  `DEFAULT_GRADE`, so existing sets are unaffected.
- Add a qrels scoring mode that credits each document once (at its first
  rank) and builds the ideal ranking from sorted qrels grades truncated at k.
  Report **nDCG@10** and **R@100**, BEIR's headline pair, alongside the
  existing metrics.
- Add `--top-k` to `retrieval_eval`, since R@100 needs 100 results and the
  pipeline default is 20.
- Test against a hand-computed fixture and cross-check a handful of queries
  with `pytrec_eval` or `ir_measures` in a scratch environment, not as a
  dependency.

Exit: unit tests pass, and the scratch cross-check agrees to 4 decimal places.

## Phase 3: reproduce the published baseline (the gate)

- Implement the `sentence_transformers` embedding provider. `EmbeddingConfig`
  already accepts `provider: sentence_transformers`, but
  `rag/embedding/factory.py` raises on it, so today that config value passes
  validation and fails at runtime. The library is already a dependency (the
  cross-encoder reranker uses it), so this adds no dependency. BGE takes a
  plain-prefix query instruction, not the Qwen `Instruct:` format the Ollama
  adapter builds, so the prefix format must be per-adapter.
- Run BM25-only and dense-only (BGE-base) on the three test splits.

Exit criterion: within **±0.005 nDCG@10** of each phase-0 reference; the
tolerance is to be confirmed once run-to-run variance is seen. A miss is a bug
in the harness, the text formatting or the metric, and it is fixed before any
phase-4 number is recorded. Record the reproduced numbers in
[measured results](measured-results.md) as a harness check, not as a result
about this system.

## Phase 4: measure the shipped query-time stack

With the harness validated, use the `measure-change` skill for paired runs on
each dataset:

- Dense (BGE-base) vs. hybrid (dense + BM25 fusion) vs. hybrid + cross-encoder
  reranker. `rerank_top_k` must be at least 10 so nDCG@10 is defined. Recall
  is read from the first stage.
- The shipped embedder (`qwen3-embedding:0.6b`) vs. BGE-base, both dense-only,
  as a transfer reading on the default embedder choice.
- Optional: re-run a measured-off query-time feature (query expansion, CRAG
  retrieval gating) where a dataset plausibly needs it. Chunking-time features
  cannot be measured here at all.

Report paired deltas with the uncertainty `rag/eval/paired.py` already
produces, per dataset, and word conclusions as the rigor plan requires. For
example: "hybrid + reranker improved nDCG@10 by X on FiQA test; no effect
shown on SciFact". Don't average across datasets into one headline.

## Deferred: pre-embedded vectors and answer eval

**Importing published vectors** (Cohere's BEIR and MS MARCO v2.1 embeddings,
Pyserini's Faiss indexes, FlashRAG's e5 index over Wikipedia-2018). At the
sizes above, embedding locally takes minutes to tens of minutes, to be
measured in phase 3. So the value of skipping it is small, and the costs are
real:

- Queries need the same encoder. For Cohere vectors that means a paid API on
  every new query.
- The index manifest must record an embedder this config can reproduce.
  Otherwise `check_queryable` either refuses the run, or passes it for the
  wrong reason.
- The corpora where precomputed vectors matter (2.7M to 113M passages) exceed
  what a local Chroma collection is sized for. They would need a Faiss
  `VectorStore` adapter, which is a new dependency.

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

- **Cost**: all local. The downloads are free, with no API spend. Disk and
  index size are to be recorded per dataset in phase 1.
- **Metric mismatch** is the main risk, and phase 3 exists to catch it. Do not
  publish a phase-4 number before phase 3 passes.
- **Overfitting to public sets**: tune on dev only, and keep SciFact as
  transfer-only.
- **Scope creep toward leaderboard chasing**: the goal is an independent check
  on this system's retrieval, not a BEIR ranking.

## Delivery order

| Phase | Output | Depends on |
|---|---|---|
| 0 | Reference table with sources | — |
| 1 | Fetcher, JSONL loader, identity chunker, eval-set converter, registry entries | 0 |
| 2 | Graded document labels, qrels nDCG@10 / R@100, `--top-k` | — (parallel with 1) |
| 3 | `sentence_transformers` embedder; reproduction within tolerance | 1, 2 |
| 4 | Paired measurements recorded in measured results | 3 |

Tracked under [Milestone 27](backlog.md#milestone-27--eval-coverage-and-judge-reliability)
as the "broaden the corpus" step of the rigor plan, limited to the query-time
stages.
