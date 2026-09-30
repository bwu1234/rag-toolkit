# BEIR reference protocol

Phase 0 of the [public benchmarks plan](public-benchmarks-plan.md), recorded
2026-09-29. It pins the published numbers phase 3 must reproduce and the
protocol that produced them, from primary sources: Pyserini's and Anserini's
source at pinned commits, the pinned evaluation data, the archives themselves,
and one recipe probe against the reference vectors. Per-dataset values
(checksums, counts, index identities, exact commands) are in
`data/corpora/beir-<name>/manifest.json`. This page holds what the three
datasets share and why the tolerances are what they are.

Phase 3 (2026-09-29) reproduced all six reference runs exactly, in the
environment below; see [phase 3 as built](public-benchmarks-plan.md#phase-3-as-built).

## Reference scores

Test split, all judged test queries, `trec_eval -c`. Copied from
`pyserini/2cr/beir.yaml` at the pinned commit; unchanged between the 2.4.0
release tag and `cbee418` (2026-09-27 main).

| Dataset | Queries | BM25 flat nDCG@10 | BM25 flat R@100 | BGE-base Faiss nDCG@10 | BGE-base Faiss R@100 |
|---|---|---|---|---|---|
| FiQA-2018 | 648 | 0.2361 | 0.5395 | 0.4065 | 0.7415 |
| SciFact | 300 | 0.6789 | 0.9253 | 0.7408 | 0.9667 |
| NFCorpus | 323 | 0.3218 | 0.2457 | 0.3735 | 0.3368 |

Anserini's own SciFact regression page (ONNX query encoding over a Lucene
flat index of Anserini's BGE vectors) lists the same 0.7408 / 0.9667, a
second route to the dense SciFact row.

## Pins

| What | Pin |
|---|---|
| Pyserini | release 2.4.0, commit `fbfe965fce10960029c7c616165c657704b63050`; PyPI sdist sha256 `45979fa5…0a303541` |
| Anserini (search, bundled jar) | 2.3.0, tag commit `7ed69af0daf02be3c0cf17f0324c2954cec4c989`; Lucene 10.5.0; Java 21 |
| Evaluator | `trec_eval` 9.0.4, the binary bundled in the Anserini 2.3.0 jar, called via `python -m pyserini.eval.trec_eval` |
| Topics and qrels | `castorini/eval` commit `0b4acbd929edd11edfd16250457fb70ff69e9b4f` (Pyserini's `EVAL_COMMIT`) |
| Prebuilt index metadata | `castorini/prebuilt-indexes` commit `367560b4de7d9c3486f666dcc2df7783ca7758f2` (Pyserini's `PREBUILT_INDEXES_COMMIT`) |
| BM25 indexes | built 2022-11-16 at Anserini `505594b6`, `BeirFlatCollection`, `DefaultLuceneDocumentGenerator`; md5 per manifest |
| BGE indexes | Faiss flat, dated 2024-01-07; md5 per manifest |
| BGE model | `BAAI/bge-base-en-v1.5`; revision `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a` reproduced the stored vectors (weights unchanged since the 2023-12-18 safetensors commit) |
| Datasets | BEIR v1.0.0 zips from `public.ukp.informatik.tu-darmstadt.de`; md5 matches the BEIR wiki; sha256 per manifest |

The reference qrels are the archive's `qrels/test.tsv` in TREC format with
identical judgments, and the reference topics are exactly the judged test
query ids with text identical to `queries.jsonl`. Checked for all three.

## Scoring protocol

- **Population**: every query in the test qrels (FiQA 648, SciFact 300,
  NFCorpus 323). Every judged test query has at least one positive grade.
  `-c` averages over all of them, so a query missing from a run scores 0.
  Without `-c`, `trec_eval` silently drops it. A query judged only
  non-relevant would also count, at 0; `rag.eval.qrels` does the same.
- **Depth**: `--hits 1000`, then `--remove-query` drops any hit whose docid
  equals the query id. A ranking can hold fewer than 1,000 hits.
- **`--remove-query` is live on FiQA.** Query and document ids share one
  numeric namespace there: 55 test queries (621 of all queries) have an id
  that is also a document id. This is an id collision, not the ArguAna case
  where the query is itself a corpus document. None of those pairs is judged,
  so removal never drops a relevant document; it only moves the ones below
  it up one rank. Reproduction must
  apply the rule, and phase 4 should apply it to every variant so they stay
  comparable. SciFact and NFCorpus have no collisions, so there the rule
  does nothing.
- **Gain and relevance**: `ndcg_cut.10` uses the qrels grade as linear gain
  (Järvelin–Kekäläinen with log2 discount). `recall.100` counts grade ≥ 1 as
  relevant (the `-l` default). NFCorpus test grades are 1 (11,758) and
  2 (576); FiQA and SciFact are binary. No test split has grade-0 rows.
- **Unjudged documents** are non-relevant. No `-remove-unjudged`, no judged@k.
- **Ties**: `trec_eval` ignores the rank column and re-sorts each query by
  score descending, then docid in *descending* `strcmp` order
  (`form_res_rels.c`, `comp_sim_docno`). A phase-2 evaluator that scores a
  list in its given order will disagree on tied scores unless it applies the
  same sort.
- **Numbers are printed to 4 decimals**, both aggregate and per query (`-q`).

## Text and encoder recipes

**BM25 flat** (`BeirFlatCollection` at `505594b6`): one `contents` field,
`title + "\n" + text`. Anserini's `DefaultEnglishAnalyzer`: `StandardTokenizer`,
English possessive filter, lowercase, Lucene's default English stopword set,
Porter stemmer. BM25 k1 = 0.9, b = 0.4 (`SearchCollection.Args` defaults; the
2CR command sets neither). `DefaultLuceneDocumentGenerator` skips a document whose
contents trim to empty, so the FiQA index holds **57,600** documents against
the archive's 57,638 (below).

**BGE-base Faiss, documents**: `title + " " + text` (Pyserini
`AutoDocumentEncoder`, `add_sep=False`), CLS pooling, L2-normalised,
truncated at 512 tokens, fp32. Pyserini does not document the command that
built these indexes, and its NFCorpus tutorial uses `--pooling mean`, so the
recipe was recovered from the vectors:
`scripts/experiments/2026-09-beir-encoder-recipe/bge_recipe_probe.py`
re-encoded eight SciFact documents (148 to 973 tokens) under 12 candidate
recipes. Only CLS / 512 / title + text matched, at cosine ≥ 0.999999 on every
sample. The next best, CLS / 256, fell to 0.974 on documents longer than 256
tokens. The BERT tokenizer treats `" "` and `"\n"` alike, so the separator
cannot be distinguished and does not matter for this model. FiQA and NFCorpus
are assumed to share the recipe (same build date and naming); phase 3 checks
vectors on all three. The index's docid order differs from `corpus.jsonl`
order, so compare by id, never by row.

**BGE-base Faiss, queries** (2CR command plus `pyserini/search/faiss`
defaults): `"Represent this sentence for searching relevant passages:" + " " +
query` (the prefix applies to every BEIR set except Quora and ArguAna),
`--encoder-class auto`, CLS pooling (the `--pooling` default), `--l2-norm`, fp32
(no `--fp16`), truncation at the tokenizer's 512-token maximum.

**Dense search**: Faiss `IndexFlat` with inner-product metric (metric type 0
in the index header), d = 768, exact. On unit vectors this is cosine.

## Dataset findings that bind later phases

From `scripts/inventory_beir.py` over the pinned archives; full output is in
each manifest under `inventory`.

| | FiQA | SciFact | NFCorpus |
|---|---|---|---|
| Documents | 57,638 | 5,183 | 3,633 |
| Empty title | all 57,638 | 0 | 0 |
| Empty title *and* text | 38 | 0 | 0 |
| Splits | train / dev / test | train / test | train / dev / test |
| Test judgments | 1,706 | 339 | 12,334 |
| Positives per test query | 2.63 | 1.13 | 38.19 |
| Test query ids that are also doc ids | 55 | 0 | 0 |

- **FiQA's 38 empty documents.** One test judgment (query `5206` →
  document `117276`), 2 dev and 35 train judgments point at them. Lucene
  skips them, so the reference BM25 run can never retrieve that one; the
  Faiss index includes them. Phase 1 decided: the loader keeps them, and
  `chunking.strategy: none` gives them no chunk. So both of this repo's
  indexes hold the BM25 reference's document set, which differs from the
  Faiss reference's by at most one relevant document on one of 648 queries. Phase 3 records the difference; it does not affect reproduction
  in the isolated reference environment.
- **FiQA has no titles**, so the title/body join yields a leading separator.
  Both reference encoders strip that, and the phase-1 serializer should too.
- **No duplicate ids, no qrels pointing at missing documents or queries, no
  duplicate judgments, no self-judgments** in any split.
- **Grades**: NFCorpus upstream has three levels; BEIR's test split carries
  grades 1 and 2 only. Phase 2's qrels mode must keep grade 2 as gain 2.

## Tolerances (frozen)

Set before any retrieval run on any test split. Do not widen after a miss.

| Check | Metric | Tolerance |
|---|---|---|
| Phase 2: our evaluator vs `trec_eval` on identical rankings | nDCG@10 and R@100, aggregate and per query | ≤ 0.0001 absolute (`trec_eval` prints 4 decimals); tie order must follow the rule above. Checked by `scripts/trec_eval_parity.py` |
| Phase 3: each of the six reference runs recreated in the isolated environment | nDCG@10 and R@100 | ≤ 0.0005 absolute, each cell |

Why 0.0005: it is Pyserini 2.4.0's own reproduction bar. Its 2CR checker marks
an exact match OK, a difference within 0.0005 "OKish", and anything else a
failure. Pyserini main has since tightened OKish to < 0.0002. A downloaded
BM25 index and exact Faiss search should land exactly. Re-encoded BGE vectors
reproduced to cosine ≥ 0.999999, so floating-point differences between
machines are the only expected source of rank swaps.

The plan's provisional ±0.005 nDCG@10 is **retired**, not reused. It never
applied to the repo's own backends. Per phase 3, `rank_bm25`, the SQLite BM25
and Chroma HNSW are different algorithms from Lucene and exact Faiss: their
differences from the reference are reported as differences, with no
tolerance and no claim to have reproduced anything.

## Test-access log

Plan rule: log test access and keep protocol corrections apart from choices
made because they raise scores.

| Date | Access | Outcome-bearing? |
|---|---|---|
| 2026-09-29 | Test qrels and topics inventoried: counts, grade histogram, id collisions, empty-document judgments; compared with the reference qrels and topics | No: no retrieval was run and no score was computed |
| 2026-09-29 | Eight SciFact *corpus* documents encoded to recover the document recipe | No: corpus only, no queries |
| 2026-09-29 | Phase 3: the six reference runs recreated and scored on all three test splits; `trec_eval` parity over them | Yes, but nothing was chosen from them: the commands are the pinned 2CR ones, unchanged. Every cell matched exactly |
| 2026-09-29 | Phase 3: this repo's `bm25` and `sqlite_fts5` backends and its BGE dense integration (Chroma), with exact-search controls, on all three test splits | Yes. Shipped parameters, reported as differences; no parameter, feature or model was chosen from them |
| 2026-09-29 | `hnsw_ef_search`: the test query *texts* embedded to measure Chroma's recall against exact search; then the BGE dense run re-scored at the chosen 1600 | Recall reads no labels, and it chose the value. The re-scoring read labels only to report the result, after the value was fixed |
| 2026-09-29 | Phase 4: test qrels read to group related queries (shared relevant documents) and query texts inspected for duplicates, before the family was frozen | No: no retrieval was run and no score was computed. The grouping rule is in the [phase-4 protocol](beir-phase4-protocol.md#dependence-between-queries) |
| 2026-09-29 | Phase 4: the frozen family's seven variants, one run each, on all three test splits, after the protocol was committed (`0cda7fd`) | Yes. Nothing was chosen from them: no rerun, no parameter change, no default changed. Reported in [measured results](measured-results.md#beir-query-time-stack-public-benchmarks-plan-phase-4) |

## Recreating the reference environment

Isolated from this repo. Pyserini 2.4.0 needs Python ≥ 3.12, Java 21 and a
separately installed `faiss-cpu`. It pulls `torch`, `transformers` 5.x and
`pyjnius`, none of which becomes a repo dependency:

```bash
python3.12 -m venv .venv-pyserini && . .venv-pyserini/bin/activate
pip install pyserini==2.4.0 faiss-cpu
```

The per-dataset search commands in each manifest (`reference.runs.*.command`)
are the 2CR commands with `${dataset}`, threads and batch sizes substituted.
Each is followed by the manifest's two `eval_commands`. `--threads` and
`--batch-size` do not change results. Record the `faiss-cpu`, `torch` and
`transformers` versions and the hardware used: the 2CR table does not pin
them.

`scripts/reproduce_beir_reference.py` runs them and records all of that.
Two things the 2CR page doesn't say, found in phase 3 on an M2 Mac:

- **Java.** Homebrew's `openjdk@21` works, but `JAVA_HOME` must be
  `$(brew --prefix openjdk@21)/libexec/openjdk.jdk/Contents/Home`, where
  `pyjnius` can find `libjli`; the keg root is not enough.
- **OpenMP.** `faiss-cpu` and `torch` each bundle a `libomp`, and the dense
  search dies with SIGSEGV in `__kmp_suspend_64` unless `OMP_NUM_THREADS=1`.
  Results don't depend on thread count.

## Not pinned, and why

- **Upstream Faiss/torch/transformers versions behind the 2024-01-07
  vectors.** They are not published. The probe shows current versions
  reproduce the vectors to 1e-6, which is the property that matters.
- **The Lucene version that wrote the 2022 BM25 indexes.** They are read
  by Lucene 10.5.0 at search time. A downloaded index settles this; a
  rebuilt one would be checked against the manifest's document count and the
  published scores.
- **Licensing is noted, not cleared.** The Hugging Face BeIR cards say CC BY-SA
  4.0 for all three. Upstream: SciFact claims CC BY 4.0 with S2ORC abstracts
  under ODC-By 1.0; FiQA states no license; NFCorpus is "free to use for
  academic purposes" and refers other uses to NutritionFacts.org's terms. The
  data is fetched for local evaluation only and never committed or
  redistributed, the same as EDGAR.
