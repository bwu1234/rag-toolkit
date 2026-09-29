# Public benchmarks plan

Status: proposed 2026-09-28. Phase 0 done 2026-09-29: the reference scores,
pins, protocol and frozen tolerances are in
[BEIR reference protocol](beir-reference-protocol.md), and per-dataset
provenance is in `data/corpora/beir-<name>/manifest.json`. Phase 1 done
2026-09-29 ([what shipped](#phase-1-as-built)). Phase 2 done 2026-09-29
([what shipped](#phase-2-as-built)). Phases 3 onward are not implemented. Sizes and query counts below come from the BEIR README; the
manifests hold the counts measured from the pinned archives.

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
**Met 2026-09-29**, recorded in [BEIR reference protocol](beir-reference-protocol.md).
Phase 0 also found three things later phases must handle: FiQA's 38 empty
documents (one judged relevant on test; the reference BM25 index skips them),
FiQA query ids that collide with document ids (so `--remove-query` can change
FiQA rankings), and a document-side BGE recipe that Pyserini never wrote
down (recovered from the vectors: CLS pooling, 512 tokens).
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

### Phase 1 as built

Shipped 2026-09-29. Where it differs from the list above, the difference is
deliberate:

- **`chunking.strategy: none`**, not `chunking.provider: none`: chunkers are
  selected by `strategy`. `WholeDocumentChunker` keeps each document as one
  chunk with its text unaltered (not even stripped). A document with no text
  yields no chunk, as with the fixed chunker. For FiQA that drops the 38 empty
  passages, which is what the reference BM25 index does; the reference Faiss
  index keeps them, a difference of at most one relevant document on one of
  648 test queries (see the [protocol](beir-reference-protocol.md)).
- **Text contract**: `BeirCorpusLoader` handles `*.jsonl` and builds the text
  with `beir_passage_text`: `title + " " + text`, or the body alone for an
  empty title. This is the recipe of the reference BGE vectors. The title
  goes into `Chunk.text` itself, so embeddings, BM25 and the reranker all see
  it. It is also carried as metadata.
- **Cleaning bypass** is a per-corpus registry flag, `clean: false`, not a
  config-file switch. That way every config (`config.yaml` included) indexes
  a benchmark corpus the same way.
- **Provenance**: neither the text recipe nor the cleaning flag is added to
  the index manifest. Both change `Chunk.text`, so the incremental indexer's
  content hash re-embeds every affected chunk. That is the same reason chunk
  sizes are left out of the manifest. Refusing to index until `--reset` would
  add nothing that the content hash does not already catch.
- **Schema, agreed now so the converter could ship**: `expected_doc_ids`
  accepts `{"id", "grade"}` objects (grade 0 kept, never credited), and
  `matching_mode: "qrels"` marks converted samples. `judge_ranking` refuses
  qrels samples until phase 2 builds their scoring, so a converted set cannot
  be scored by the legacy document mode by mistake.
- **Layout**: `scripts/fetch_beir.py` writes passages to `documents/` and
  queries and qrels to `source/` (gitignored), checking the archive and
  every extracted file against the manifest's hashes.
  `scripts/beir_to_eval_set.py` re-checks those hashes, validates the data,
  and checks its counts against the manifest inventory.
- **`index-report`** now prints `Split documents`, the count this phase's
  exit criterion checks, and leaves `chunk_size` out of the settings line
  when `strategy: none` ignores it.

#### Phase 1 exit

Met 2026-09-29 on SciFact, embedded with the shipped `qwen3-embedding:0.6b`
through Ollama. BGE comes in phase 3.

- `python -m rag.cli --config rag/config/beir.yaml index --corpus beir-scifact`
  built 5,183 vector and 5,183 BM25 chunks in 373 s wall time (this machine,
  local Ollama). The CLI process peaked at 350 MB resident; Ollama's memory is
  separate and was not measured. The index is 111 MB on disk
  (`data/index_beir/`).
- `index-report` with the same config: 5,183 documents, 5,183 chunks,
  0 split documents, 0 zero-chunk documents, 0 duplicates. Index 0 missing,
  0 stale, 0 changed, manifest matches, in sync.
- Against the source, every `corpus.jsonl` `_id` is a document id with
  exactly one chunk, and both the chunk text and the text stored in Chroma
  equal `beir_passage_text(title, text)` for all 5,183 passages.
- `beir_to_eval_set.py scifact test` wrote 300 qrels samples with 339
  judgments, matching the manifest inventory.

Not yet exercised: FiQA and NFCorpus indexing. FiQA is 57,638 passages,
past the ~10^4 chunks where the default `rank_bm25` sparse index degrades
(`SparseIndexConfig`). Expect it to be slow, and measure it before phase 4
relies on it.

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

### Phase 2 as built

Shipped 2026-09-29.

- **`rag/eval/qrels.py`** holds the pure scoring: `ndcg_cut` (linear gain,
  ideal from every positive grade, cut at k), `recall_cut` (distinct
  documents graded ≥ 1), `trec_order` (score descending, then document id
  descending), `document_ranking` (first occurrence per document, optional
  self-match removal, then `trec_order`), and TREC run-file I/O. The reader
  refuses a document ranked twice, and the writer uses `repr` scores so a
  file reads back as the identical floats.
- **Policy for queries without positives**, matched to the reference: a
  query judged only non-relevant is a valid qrels sample. It scores 0 on
  both metrics and counts toward the mean, as under `trec_eval -c`.
  `beir_to_eval_set.py` now keeps such queries (with a warning) instead of
  refusing them. None of the pinned test splits has one.
- **Runner**: `retrieval_eval` sends a qrels set to `run_qrels_eval`. It reads
  the stage-1 ranking from the new `RetrievalResult.candidates` (post-fusion,
  pre-rerank) and the final ranking from `chunks`. It reports nDCG@10 and
  R@100 for both stages, marks nDCG@10 on the final ranking and R@100 on
  stage 1 as the reported figures, and counts collapsed duplicate chunks,
  removed self-matches and short lists. `--candidate-depth` and
  `--final-depth` override `retrieval.top_k` and `rerank_top_k`. The run is
  refused if stage 1 is shallower than 100 or the final ranking shallower
  than 10. `--save-run DIR` writes `stage1.trec`, `final.trec` and unrounded
  `scores.json`. The reference `--remove-query` rule is on by default;
  `--keep-self-matches` turns it off. A set that mixes qrels with other modes
  is refused.
- **Cross-check against `trec_eval` 9.0.4 itself**, built from its release
  tag, not `pytrec_eval` or `ir_measures`: it is the pinned evaluator,
  whereas those wrap it. `scripts/trec_eval_parity.py` scores the same run
  file with both evaluators and fails above 0.0001. Phase 3 reuses it on
  the reference runs.
- **Not wired**: `scripts/run_matrix.py` still calls the legacy path, which
  refuses qrels samples. Phase 4 connects it.

#### Phase 2 exit

Met 2026-09-29. `tests/test_qrels_eval.py` has hand-computed fixtures for
mixed grades, the linear-vs-exponential gain difference, short and empty
lists, unjudged and grade-0 documents, the ideal cut at k, ties, duplicate
collapsing, self-match removal, run-file round trips and malformed files, and
queries missing from a run. Every identical-ranking comparison below agrees
to **0.0000**, per query and in aggregate:

| Rankings | Queries | Notes |
|---|---|---|
| SciFact test, this repo's dense run (stage 1 and final) | 300 | Real rankings; reference qrels from `castorini/eval` |
| NFCorpus test, this repo's dense run (stage 1 and final) | 323 | Real rankings over graded (1–2) qrels |
| NFCorpus test, randomised awkward rankings | 323 | 53 missing from the run, 13,827 tied adjacent pairs, 1- and 7-deep lists |
| Synthetic qrels, grades 0–3 | 200 | Includes queries judged only non-relevant; 33 missing, 8,585 ties |

The randomised check has power: with the document-id tie-break removed, it
fails at a per-query difference of 1.0
(`scripts/experiments/2026-09-qrels-parity/stress_parity.py`). Its first
version also caught the policy gap above: 0.0039 in the mean from four
zero-only queries the schema could not yet hold.

These are scoring checks, not results. The runs use the shipped
`qwen3-embedding:0.6b`, not BGE. On SciFact, that run's nDCG@10 of 0.6789
equals the BM25 reference to four places by coincidence; it reproduces
nothing, and phase 3 is where reproduction happens.

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

## Follow-on: multi-hop and agentic retrieval on MuSiQue

Phases 0–4 score one ranking per single-need query and grade no answers, so
they cannot measure [Milestone 19](milestone-19-plan.md). Its only multi-hop
evidence today is `data/eval/edgar_multihop_set.json`: 34 questions built by
pairing samples written from this repo's own chunks, which is the lineage
problem this plan exists to break. This section adds an outside multi-hop set
with gold evidence and gold answers. It is proposed, not scheduled, and
starts only after the phase-3 gate passes.

### Why MuSiQue

[MuSiQue](https://github.com/StonyBrookNLP/musique)
([paper](https://arxiv.org/abs/2108.00573)) has about 25K 2–4 hop questions
built from Wikipedia paragraphs under CC BY 4.0. Each example carries what
the Milestone 19 matrix needs:

| Field | Use here |
|---|---|
| `paragraphs` with `is_supporting` | Corpus units and paragraph-level gold evidence |
| `question_decomposition` (2–4 steps, each tied to a supporting paragraph) | Hop count for stratification, per-hop evidence recall, and a reference for `planned` sub-queries |
| `answer`, `answer_aliases` | Deterministic answer EM/F1, no judge needed |
| `answerable` (MuSiQue-Full only) | Refusal behavior (see the caveat below) |

The authors built it to resist disconnected reasoning, reporting a 30-point
F1 drop for a single-hop model. That matters because a set that one retrieval
round can solve cannot show an agent's gain. 2WikiMultiHopQA has a similar
shape and is the fallback if MuSiQue fails phase A.

### What it does and does not measure

- **Measures:** whether extra retrieval rounds or decomposition recover
  evidence that one round misses (union evidence recall per hop), and whether
  the final answer is right (EM/F1 against aliases). It also measures
  refusal only if the Full caveat is resolved.
- **Closed, pooled corpus, not open-domain.** The corpus is the union of the
  selected split's paragraphs, far smaller than Wikipedia. Each paragraph was
  chosen as a hard distractor for its own question, and in the pool every
  other question's paragraphs become extra distractors. Report results as
  "MuSiQue-Ans dev, pooled corpus", and never compare them with open-domain
  leaderboard numbers.
- **Not the production chunker.** It reuses the phase-1 identity path, as
  BEIR does.
- **Probably in training data.** The dataset has been public since 2021.
  Every answer row needs a closed-book control beside it (see
  [answer eval](#deferred-pre-embedded-vectors-and-answer-eval)).

### Phase A: pin and inspect

- Pin the release: source URL, archive SHA-256, licence and per-split counts
  in `data/corpora/musique-ans/manifest.json`. The figures above come from the
  paper abstract, the repo README and a Hugging Face mirror; phase A replaces
  them with counts from the pinned files.
- Inventory paragraphs per question, the hop-count distribution, how often
  identical paragraphs recur across questions, and how many supporting
  paragraphs each question has.
- **Splits.** Test labels are hidden behind a leaderboard, so **dev is the
  confirmatory set**. Tuning (agent prompts, search caps, fusion depths) uses a
  fixed, recorded slice of train with its own pooled corpus. Freeze the
  comparison family on that slice before any dev run, under the same
  test-access rules as the BEIR phases.
- **MuSiQue-Full caveat.** Per the paper, unanswerable questions are contrast
  versions of answerable ones, made by withholding supporting evidence from
  that question's context. Pooling can put the withheld paragraph back through
  its answerable twin's context, which makes an "unanswerable" question
  answerable. Verify this in phase A. If it holds, run Full only with
  per-question corpora (20 paragraphs each, one index per question or a
  per-question metadata filter) or leave it out. Until then the pooled
  experiment is MuSiQue-Ans only.

Exit: a manifest with counts, a split and access log, and a written decision
on Full.

### Phase B: corpus and eval-set conversion

Builds on phase 1's JSONL loader, cleaning bypass and identity chunker, and
phase 2's qrels scoring. No new dependencies.

- **Paragraph ids.** MuSiQue paragraphs carry only a per-question index, not
  a global id. Assign `musique-<sha256(title + "\n" + text)[:16]>` and dedupe
  identical paragraphs into one `Document`, recording the collapse count. The
  id is a pure function of content, so the converter is reproducible, and the
  no-namespacing rule still holds because the corpus is only ever evaluated
  isolated.
- **Text recipe.** Title plus body with the same pinned serialization rules as
  phase 0. MuSiQue has no reference encoder, so pin the one the benchmark
  config chooses.
- **Converter** `scripts/musique_to_eval_set.py` writes
  `data/eval/musique_ans_<split>.json` in the multi-hop set's shape, so
  `multihop_eval` runs it unchanged where the shapes agree:
  - `expected_doc_ids`: the supporting paragraph ids, with binary grades for
    qrels scoring;
  - `expected_spans`: the full supporting paragraph texts. Under the identity
    chunker, evidence recall's span matching then reduces to paragraph
    presence;
  - `expected_answer` plus aliases;
  - `hops` and `parts`: one part per decomposition step, holding that step's
    supporting paragraph and intermediate answer. Parts feed **per-hop
    evidence recall only**. MuSiQue grades the final answer, so do not score
    intermediate answers with the per-entity completeness judge, whose rubric
    assumes every part is a user-facing sub-answer.
  - Validate that every supporting paragraph resolves to an id in the pooled
    corpus, and fail on malformed records rather than dropping them.
- **Registry and config.** Add `musique-ans-dev` and `musique-ans-train-tune`
  and reuse `rag/config/beir.yaml`, or a `base:` child of it.

Exit: both indexes build, `index-report` shows one chunk per paragraph and a
synced index, and the converter round-trips counts and ids against the pinned
files.

### Phase C: scoring

- **Answer EM/F1 with aliases**, ported from the official evaluation script
  (pin its commit) and checked for agreement to 4 decimal places on a
  fixture of predictions scored by the reference script. It is deterministic,
  so the judge's calibration does not bear on the primary metric.
  An LLM-judge semantic-equivalence score may be reported beside it as
  secondary, labelled with the judge. Answers are long-form while EM expects a
  short span, so add an answer-extraction step: either a fixed final-line
  format in the benchmark prompt or a recorded extraction rule. Validate it on
  the train slice, because extraction failures would read as retrieval
  failures.
- **Evidence:** union evidence recall overall and per hop, plus phase 2's
  R@100 on the first retrieval round. The gap between the two is the part an
  agent can recover.
- **Cost:** searches per turn, cap-hit rate, LLM calls, tokens and
  wall-clock, as the Milestone 19 matrix already reports.
- Stratify every metric by hop count (2/3/4). Pool across hops only with the
  strata shown.

### Phase D: the Milestone 19 rows on outside data

Run the [Milestone 19 matrix](milestone-19-plan.md) variants that matter for
a default decision: `pipeline / 9b`, `oracle / 9b`, `agentic react / 9b`,
`agentic planned / 9b` and `agentic react / 27b`, plus `closed-book / 9b`.
The oracle row feeds the supporting paragraphs directly.

- **Sample size.** Agent runs take hours locally (Milestone 19 phase 4). Run
  the pipeline and closed-book rows on the full dev set. Run agentic rows on a
  hop-stratified random sample, sized beforehand from train-slice variance
  with `rag/eval/paired.py` for the smallest worthwhile EM difference. Record
  the seed and sample ids.
- Report paired per-question deltas against `pipeline / 9b` with the
  `paired.py` uncertainty, per hop stratum. The question to answer is the
  same as the EDGAR one: does the agent beat the pipeline on multi-hop
  questions by more than noise, and at what latency?
- **How it feeds the default decision.** MuSiQue is supporting evidence for
  the Milestone 19 default-flip criterion, not a replacement for it. That
  criterion also needs the single-hop and refusal sets held within noise,
  which MuSiQue-Ans does not measure. A MuSiQue gain that does not appear on
  EDGAR multi-hop points to a domain difference to investigate; it does not
  justify a flip.

Cost: local only, no API spend. Record disk, index build time and total agent
wall-clock.

## Optional: paired BEIR queries (own-authored questions)

A cheaper, weaker complement to MuSiQue. It builds multi-part questions over
the phase-1 BEIR corpora by **pairing two existing BEIR queries**, the way
`edgar_multihop_set.json` pairs verified EDGAR samples. It answers a narrower
question than MuSiQue: does the agent still help outside financial filings?
Build it only if MuSiQue results leave that question open.

### What stays independent and what does not

| Part | Source | Independent? |
|---|---|---|
| Corpus and passages | BEIR, via phase 1 | Yes |
| Gold evidence | Union of the two source queries' positive qrels | Yes, per source query |
| The combined question | Us | **No**: the same lineage problem as the EDGAR sets |
| The gold answer | Derived from source labels where they exist (SciFact below), otherwise none | Only where derived |

Report every result from this set as **"paired BEIR queries, own-authored
questions"**. It never counts toward the rigor plan's independent evidence,
and it has no published reference score to validate it.

### Known weaknesses

- **Comparison, not chains.** These corpora do not link one passage to the
  next through shared entities, so pairing yields questions that name both
  targets. A single hybrid search can often retrieve both, which understates
  whatever an agent contributes. Measure this before reading anything into
  agentic rows: report the pipeline's first-round union evidence recall on
  the set. If one round already retrieves most of both halves, the set cannot
  show an agent's gain, and building further on it wastes the run time.
- **Qrels coverage.** BEIR's labels judge each source query, not the combined
  question. The wording of the combined question can pull in passages that
  neither source query's pool judged. Score evidence strictly against the
  inherited qrels and label unjudged hits as unjudged, not as misses. The
  set's scores are then a lower bound.
- **Artificial composition.** Joining two unrelated needs into one question
  reads unlike real user questions. Pair only within a shared topic (below)
  and keep the join template fixed.

### Construction

- **Pair source.** Draw both queries from the same split. Use tuning splits
  for pairs that inform choices, and confirmatory splits only for frozen
  runs, following the test-access rules in [Datasets](#datasets). Never pair
  across datasets, and never pair a query with itself or with a query sharing
  a positive passage, since the two halves must need different evidence.
- **Topic matching.** Pair queries whose texts are near neighbours under a
  fixed, recorded encoder, so the combined question has a plausible shared
  subject. Record the encoder, threshold and seed. The encoder must not be
  the embedder under test, or pairing would favour it.
- **Fixed templates, no free-form rewriting.** Combine the two query texts
  with a small, versioned set of templates ("Answer both: … and …"; for
  SciFact, "Which of these claims does the literature support: …?").
  An LLM rewrite would read more naturally but bring back authored wording.
  If a later version adopts rewrites, keep the template version as a paired
  control.
- **Derived answers (SciFact only).** The original
  [SciFact release](https://github.com/allenai/scifact/blob/master/doc/data.md)
  labels each claim's evidence `SUPPORT` or `CONTRADICT`. A claim pair then
  has a gold answer computed from independent labels (which claims are
  supported, contradicted, or lack evidence), with no authored answer text.
  First verify that BEIR's SciFact query and corpus ids map one-to-one onto
  the original claim and document ids, and which original split BEIR's test
  split came from. FiQA and NFCorpus have no reference answers, so their
  pairs are **evidence-only**.
- **Converter** `scripts/beir_pairs_to_eval_set.py` writes
  `data/eval/beir_<name>_pairs_<split>.json` in the multi-hop shape: one
  `part` per source query with its qrels, so `multihop_eval`'s union evidence
  recall and per-part recall apply unchanged. Gitignored, and regenerated
  from the pinned zip, the template version and the recorded seed.

### Measurement

Run it after MuSiQue phase D, with the same variants on a smaller sample. The
headline is per-part union evidence recall (inherited qrels only), and for
SciFact also derived-answer accuracy with a closed-book control. Report it
per dataset, never averaged into MuSiQue or EDGAR numbers.

## Follow-on: unanswerable questions on CRAG

Every BEIR test query has at least one relevant passage, and MuSiQue-Ans is
answerable by construction. Nothing in this plan yet checks whether the system
declines when its documents do not support an answer. The only refusal
evidence today is `data/eval/edgar_refusal_set.json`: 15 hand-written EDGAR
negatives, scored by the refusal judge in `rag/eval/answer_eval.py`. This
section adds an outside set with questions the system should decline, scored
so that declining beats guessing. It is proposed, not scheduled, and needs the
phase-3 gate.

### Why CRAG

[CRAG](https://github.com/facebookresearch/CRAG)
([paper](https://arxiv.org/abs/2406.04744)) is Meta's RAG benchmark: 4,409
questions across five domains and eight question types, each shipped with the
real web search results retrieved for it.

- **Labelled invalid questions.** 525 questions (12%) are `false_premise`,
  such as asking for an album that does not exist. Their gold answer is
  "invalid question".
- **Answers that are often simply absent.** Task 1 gives each question 5 pages
  sampled at random from its top-10 search results, so the answer is
  frequently not in them. The system should then decline rather than guess.
- **Scoring that rewards declining.** CRAG scores an accurate answer 1, a
  missing one ("I don't know") 0 and an incorrect one −1. It reports
  truthfulness as accuracy minus hallucination rate, so a system that guesses
  when unsure loses points. The paper's baselines show why that matters: its
  best straightforward RAG setup reached 43.6% accuracy with a 30.1%
  hallucination rate.
- **A public test split.** Validation 30%, public test 30% (1,335 questions),
  private test 40% held out.

**Licence: CC BY-NC 4.0.** Non-commercial use only. That fits this repo's
evaluation use, but it rules out redistributing derived data. As with every
corpus here, the data is fetched and gitignored, never committed.

### Alternatives considered

| Candidate | Why not the primary |
|---|---|
| MuSiQue-Full | Already planned above. Its unanswerable questions may break when paragraphs are pooled (see the MuSiQue phase A caveat) |
| SQuAD 2.0 | Crowdworkers wrote unanswerable questions against a single paragraph. It tests reading comprehension, questions share much of their paragraph's wording, and a question unanswerable from its paragraph may be answerable elsewhere in its article once pooled |
| CRUMQs ([arXiv 2510.11956](https://arxiv.org/abs/2510.11956)) | Unanswerable multi-hop questions, but built over NeuCLIR and TREC RAG 2025, collections far beyond the local budget |
| UAEval4RAG ([arXiv 2412.12300](https://arxiv.org/abs/2412.12300)), RefusalBench ([arXiv 2510.10390](https://arxiv.org/abs/2510.10390)) | LLM-generated questions, the same lineage problem as writing our own |

### What it does and does not measure

- **Measures:** declining false-premise questions, declining when the answer
  is not in the retrieved pages, and the cost side: refusing questions the
  pages do answer.
- **The production pipeline is back in the loop.** CRAG pages are raw HTML,
  not pre-split passages. There is no HTML loader today (`rag/ingestion/loaders.py`
  reads PDF, Markdown and text), and `scripts/fetch_edgar.py`'s standard-library
  converter is written for SEC filings. Generic web pages need their own
  extraction step. That exercises more of the real pipeline than BEIR does,
  but poor extraction can drop an answer and make a correct refusal look like
  a success. Audit extraction on a sample before reading refusal numbers.
- **Weak retrieval signal.** With 5 pages per question, retrieval chooses
  among only a few pages' chunks. Task 3's 50 pages are a later option if
  retrieval should matter more.
- **Answers that change over time.** CRAG labels each question `static`,
  `slow-changing`, `fast-changing` or `real-time`, and gold answers are tied
  to `query_time`. Keep `static` and `slow-changing`, pass `query_time` into
  the prompt for the latter, and exclude the rest, whose answers depend on
  CRAG's mock APIs.
- **Not measured:** CRAG's knowledge-graph mock APIs (Tasks 2 and 3).
  This repo has no tool for them.
- **Probably in training data.** The dataset has been public since 2024, so
  report a closed-book row beside every answer row. The paper's own LLM-only
  baseline shows how much a model answers from memory.

### Phase A: pin and inspect

- Pin the Task 1 file: source URL, archive SHA-256, licence and counts per
  `split`, `question_type` and `static_or_dynamic`, in
  `data/corpora/crag-t1/manifest.json`. The counts above come from the paper;
  phase A replaces them with counts from the pinned file, and confirms the
  `split` values and the exact false-premise gold string.
- **Splits.** Tune on validation, confirm on public test, and never touch the
  private test. Freeze the comparison family on validation first, under the
  test-access rules in [Datasets](#datasets).
- **Answer-present label.** Only false-premise questions carry an explicit
  "decline" label. For the rest, derive whether a question's pages contain its
  gold answer by normalised matching of `answer` and `alt_ans` against the
  extracted page text. Hand-audit a random sample to estimate the heuristic's
  error rate. It splits answerable questions into answer-present, where
  declining is over-refusal, and answer-absent, where declining is correct.
  Report it as a derived label with its measured error rate.

Exit: a manifest with counts, the kept question population and its
exclusions, and the audited answer-present label.

### Phase B: corpus and per-question scope

- **HTML extraction** to Markdown, behind the existing loader interface, using
  the standard-library `html.parser` as `fetch_edgar.py` does. No new
  dependency without a measured extraction gap. Pages go through the normal
  cleaner and chunker, not the identity path. Pin the extractor version in
  index provenance.
- **One index, per-question scope.** Index every kept page with
  `interaction_id` in its metadata, and retrieve each question only within its
  own pages through a metadata filter. Pooling without the filter would do to
  CRAG what the MuSiQue-Full caveat describes: another question's pages could
  answer a question whose own pages cannot. `retrieval_eval` already takes a
  per-sample `filters_for` hook. `answer_eval` and the agent's search tool
  need the same, and the agent must not be able to widen or drop the filter.
- **Converter** `scripts/crag_to_eval_set.py` writes
  `data/eval/crag_t1_<split>.json`. False-premise questions use a new tier and
  the rest use the answer tier, each with `interaction_id`, `query_time`,
  `static_or_dynamic`, the answer-present label and gold answers with
  aliases. Validate every record, and fail on malformed ones rather than
  dropping them.

Exit: the index builds, `index-report` shows chunk health for web pages, and a
filtered retrieval for any question returns only that question's chunks.

### Phase C: scoring

- **CRAG's three-way grade** per answer: accurate (1), missing (0), incorrect
  (−1). Report accuracy, hallucination rate, missing rate and truthfulness,
  following CRAG's definitions. The grader is `eval.judge`, never the
  generator. CRAG's own auto-eval also uses an LLM judge, so calibrate ours
  against a hand-labelled sample before any comparison.
- **False premise needs its own rubric.** The existing refusal judge asks
  whether the system declined because its documents cannot answer. A
  false-premise question calls for pointing out the false assumption, and a
  plain "I don't know" is a weaker, different outcome. Grade it as: correctly
  rejects the premise, declines without naming it, or answers as if the
  premise were true (incorrect).
- **Report by population, never pooled into one number:** false premise,
  answer-absent, answer-present. Over-refusal on answer-present is the
  counterweight: a system that declines everything would score well on the
  first two.

### Phase D: measurements

- `pipeline / 9b` vs. `closed-book / 9b` on the full kept population.
- The measured-off corrective-RAG checks (`crag:` in config: passage grading
  and a groundedness check). They share the acronym with this dataset but are
  unrelated to it, and they exist to catch unsupported answers. This
  set can show whether it lowers hallucination on answer-absent questions
  without raising over-refusal on answer-present ones. A re-measurement, per
  the measured-off rule: it stays off unless this and the EDGAR sets agree.
- The Milestone 19 agentic rows, on a stratified sample sized from validation
  variance. An agent that keeps searching may hallucinate more when the
  answer is absent, which the EDGAR refusal set is too small to show.
- Paired per-question deltas with `rag/eval/paired.py` uncertainty, per
  population.

Cost: local only, no API spend. HTML pages are larger than BEIR passages, so
record disk, extraction and index time.

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

- MuSiQue, drafted above as the
  [multi-hop follow-on](#follow-on-multi-hop-and-agentic-retrieval-on-musique):
  small enough to embed locally, and scored by EM/F1 rather than a judge.
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
| MuSiQue A–D (follow-on) | Pooled MuSiQue-Ans corpus, EM/F1 scorer, Milestone 19 rows on outside data | 3; agentic rows also need Milestone 19 phase 3 |
| Paired BEIR queries (optional) | Own-authored paired questions over the BEIR corpora; SciFact answers derived from its labels | 1, 2; run after MuSiQue D |
| CRAG A–D (follow-on) | Task 1 web pages with per-question scope, HTML extraction, three-way grading, refusal measured by population | 3; agentic rows also need Milestone 19 phase 3 |

Tracked under [Milestone 27](backlog.md#milestone-27--eval-coverage-and-judge-reliability)
as the "broaden the corpus" step of the rigor plan, limited to the query-time
stages.
