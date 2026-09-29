# Known limitations / roadmap

- **The eval sets are development benchmarks with limited generalization.**
  The main EDGAR set is nearly saturated (97.7% retrieval hit), and 173 of
  174 questions have one expected answer span. Its questions were generated
  from existing chunks; the underspecified and multi-hop sets reuse its
  facts. Repeated tuning on these sets is not held-out validation. The
  [evaluation rigor plan](evaluation-rigor-plan.md) specifies fresh grouped
  holdouts, task coverage, corpus diversity, and acceptance criteria.
- **Dense search is approximate.** Chroma's HNSW index returns most, not
  all, of the exact nearest neighbours. `vector_store.hnsw_ef_search` sets the
  search beam: 400 as shipped, which makes EDGAR's dense top 20 exact, and
  1600 in `beir.yaml`, which leaves FiQA at 99.8% of the exact top 100
  ([HNSW `ef_search`](measured-results.md#hnsw-ef_search)). A much larger
  corpus needs the recall re-measured (`scripts/experiments/2026-09-hnsw-ef-search/ann_recall.py`,
  which needs no labels). Chroma fixes the value for the lifetime of a
  process, so a sweep needs one process per value.
- **Hybrid fusion weights both lists equally.** RRF has no per-list weight,
  so when BM25 is much weaker than dense retrieval it drags good dense
  results down. On BEIR FiQA (natural-language questions, little exact-term
  signal) hybrid scored 0.069 nDCG@10 below dense alone, and only the
  reranker recovered it
  ([BEIR query-time stack](measured-results.md#beir-query-time-stack-public-benchmarks-plan-phase-4)).
  EDGAR's tickers and periods are why hybrid is the default. A corpus without
  that signal should measure dense-only before assuming hybrid helps.
- **A fresh EDGAR fetch doesn't reproduce the corpus.** The committed MD&A
  selection rules (`extract_mda`) postdate the fetch: they select nothing for
  8 of the 61 filings (LUV, TGT and CVX, whose MD&A opens with a table) and a
  longer span for 2 JNJ 10-Ks. Re-running `fetch_edgar.py` would build 53
  documents and invalidate eval labels. Use `--cache-raw`, which pins the
  corpus on disk and warns about the drift, and don't re-fetch until
  `extract_mda` is fixed to match
  ([chunking plan, Phase 4](chunking-indexing-plan.md#phase-4--recover-structure-at-parse-time-23-days--done-2026-09-29)).
- **Some EDGAR documents run past MD&A.** COST 10-Ks continue into Part III,
  CVX 10-Ks include risk factors, and JNJ 10-Qs include Part II. The eval
  sets were labeled against this text, so it stays, but it adds off-topic
  distractor chunks that a clean MD&A corpus wouldn't have.
- **Answer pass does not establish citation support.** The answer judge sees
  a reference answer rather than retrieved evidence. A wrong-period passage
  with identical wording can yield a passing answer, and a manual multi-hop
  audit found a false pass on period attribution. Judge calibration and
  evidence-aware scoring remain planned work.
- **The measured-off verdicts rest mostly on questions that name their
  subject.** Contextual chunking (Milestone 9), CRAG (Milestone 10) and query
  expansion were measured on the generated EDGAR set, where every question
  names its company and period, without demonstrating a benefit sufficient
  to enable them under those experiments' conditions
  ([measured results](measured-results.md)). The deterministic chunk header
  has since replaced contextual chunking as the way to put document identity
  into chunks. CRAG and expansion have not been re-measured on the
  `underspecified` tier. Their current results do not establish that they
  cannot help other question distributions.
- The retry rewriter has the same domain-drift failure mode HyDE does, and it's
  severe: on this corpus "How do I raise my throttling ceiling?" was rewritten
  to "Increase maximum CPU frequency limits via BIOS configuration" — a fluent,
  confident rewrite of a completely different question. Unlike HyDE, there's no
  `include_original` safety net, because a retry only happens once the original
  has already failed. The bound on the damage is `max_retries` and the fact that
  generation never answers the rewrite.
- The contextualizer's blurbs on `qwen3.5:9b-mlx` mostly begin "This excerpt…"
  despite the system prompt forbidding it. Harmless — the identifying terms are
  still there and that's what's being indexed — but it wastes a few tokens of
  the `max_context_chars` budget on every chunk.
- On the served path, the grader, the retry rewriter, the groundedness
  checker, the condenser, the expanders, and the answering model are all the
  *same* `llm` model. A groundedness check is only as good as the model
  performing it, and a model checking output shaped like its own has an
  obvious blind spot. Only the eval judge (`eval.judge`) and the agent's loop
  (`agent.llm`) can run on a different model today; the runtime checker has
  no setting of its own.
- CRAG's latency is not visible in `retrieval.min_score`-style tuning: enabling
  `grade_documents` multiplies the per-turn LLM calls by roughly
  `rerank_top_k`, and there's no batching or concurrency in the grader (one
  sequential call per passage, chosen for parse reliability over speed).
  Concurrent grading is the obvious optimization and nothing in the design
  prevents it.
- Contextual chunking's cost still scales linearly with corpus size, and it is
  the binding constraint on the Milestone 11 matrix. Concurrency and
  checkpointing (see `docs/milestone-notes.md`) take the EDGAR corpus from
  ~5.9h to ~4.0h measured (not the ~1.9h first projected from a
  prefix-cache-flattered benchmark) and make an interrupted run resumable, but
  ~4h per *fresh* contextual config is still an overnight-scale job. Higher
  parallelism than a local Ollama offers — or a hosted provider — is the next
  lever, not more client threads.
- Changing anything under `chunking.contextual` still needs a full `--reset`
  (enforced by the index manifest) rather than re-contextualizing
  incrementally, because the per-chunk content hash covers `chunk.text` only.
  The context cache keeps that rebuild from re-paying for blurbs whose inputs
  didn't change.
- Chunking is character-based fixed-size with overlap. On EDGAR, 543 chunks
  (12.8%) start partway through a table, separated from its header row
  (`index-report` counts them). A structure-aware chunker is planned in the
  [chunking plan](chunking-indexing-plan.md) (Phases 4–5); semantic chunking
  was dropped from it on the evidence there. Token-aware sizing waits for an
  embedder with a hard token limit.
- **The chunk header needs typed document metadata.** A document missing any
  field `chunking.header.template` names gets no header, and on EDGAR that
  metadata comes only from the fetcher's YAML front matter. PDFs and
  plain-text files have no front matter, so a non-EDGAR corpus gets no header
  from the shipped template (`index-report` shows the count).
- **Metadata filters come only from the caller.** `POST /chat` and MCP
  `rag_search` take `filters`. The UI and `cli chat` don't, nothing extracts
  them from question text (Milestone 20), and the agent's `filters` argument
  is pinned to the turn's rather than chosen by the model (Milestone 19). The
  measured gain from a period filter is therefore unavailable to most
  traffic.
- **`data/corpora/baseline` is too small to evaluate against, and that is why the EDGAR
  corpus exists.** At 8 documents / 26,429 characters / 31 chunks, `top_k: 20`
  already retrieves ~65% of the corpus, so hit rate and recall@k are saturated
  before any config knob is touched. `data/corpora/edgar` (61 SEC filings,
  3.6M chars, 4,236 chunks — built by `scripts/fetch_edgar.py`, documents
  gitignored) brings that to 0.47%, which is the point. Keep the small corpus as
  a control in the corpus × config matrix; do not read a metric from it alone.
- The EDGAR fetcher's MD&A extraction is **content heuristics, not a filing
  parser**, tuned against a sample of filers. It fails *closed* — 14 of 75
  requested filings were skipped and logged rather than partially indexed,
  because a silently wrong corpus invalidates every metric computed against it.
  Banks (JPM/BAC/GS) and some pharma 10-Ks fold statement tables and acronym
  glossaries into MD&A and are systematically excluded; the corpus is therefore
  sector-skewed away from financials until Milestone 14 provides a real parser.
  Per-document `digit=`/`pipe=` stats print on every run — audit them.
- Expansion can't help *stage 1* at the old corpus size (9 chunks): `top_k: 20`
  means every query already retrieves every chunk, so fusion has nothing to
  recover. All the observable effect at this size comes from reranking against
  the expanded queries — which does change outcomes ("Can I get my erased files
  back?" goes from 0 results to 1 with `multi_query` + `aggregate: max`), but
  surfaced the right *document* and the wrong *chunk* on the run inspected.
- Expansion makes retrieval **non-deterministic**: rephrasings and HyDE passages
  differ per run, so the same query can score differently and cross the
  `min_score` floor on one run and not the next. Worth remembering when a result
  seems to change for no reason, and a reason to hold expansion fixed while
  tuning anything else.
- HyDE's failure mode is domain drift, and it's severe on ambiguous queries: on
  this corpus, "How do I raise my throttling ceiling?" produced a confident
  hypothetical passage about reactor coolant loops and terminal code 99-DELTA.
  `include_original: true` (the default) exists precisely so a generation that
  wanders can't sink the search.
- `retrieval.min_score` is 0.0 (off): the hand-tuned value measured inert,
  and there's no calibration step to derive one from `bge-reranker-v2-m3`'s
  score distribution. Its meaningful range shifts with the reranker, so an
  inherited threshold can silently reject everything. Until one is derived,
  nothing stops the model from being handed five irrelevant passages.
- Query condensing costs an extra LLM round trip on every turn that has
  history, and the rewrite is only as good as the local model. The rewritten
  query is reported back on `ChatAnswer`/`ChatResponse` and shown in the UI, so
  a bad rewrite is at least visible — but nothing detects or corrects one.
- In `pipeline` mode, conversation history is never shown to the *answering*
  model, only to the condenser. Questions whose answer depends on the thread
  rather than on the corpus ("summarize what you just told me") aren't served
  by this design. `chat.mode: agentic` passes history to the model, but it
  is unmeasured and off by default.
- The cross-encoder reranker (`BAAI/bge-reranker-v2-m3`, the default) costs
  ~1.1 s/query on Apple Silicon and loads several hundred MB of weights; it
  will be slower on CPU-only hosts. A pure-LLM reranker behind the same
  `Reranker` interface is untested, and Qwen3-Reranker has no working
  adapter ([measured results](measured-results.md#reranker-models)).
- Stale-chunk removal trusts the corpus as loaded. A file whose loader raises
  (e.g. a corrupt PDF, logged and skipped) produces no chunks that run, so its
  existing chunks are removed along with genuinely deleted documents, and come
  back only once it loads again. An *empty* load aborts the run before
  anything is removed.
- The index manifest cannot vouch for indexes built before it existed: the
  first run on one adopts the current config with a warning, so an index
  already mixing two embedders stays mixed until `--reset`.
- `build_rag_prompt` has no defense against prompt injection — retrieved
  passage text or a user's raw query containing something like "ignore
  previous instructions" is concatenated straight into the prompt with no
  delimiter distinguishing untrusted content from the system instructions.
  Wrapping passages and the query in explicit delimiter tags is a cheap first
  step; nothing is done today (planned in Milestone 28). The agent path
  shares the passage formatting (`format_passage`) and the same exposure.
- **The turn log grows without bound and is not rotated.** `data/logs/turns.jsonl`
  gets one line of several KB per turn, since every event, attempt and answer
  is inlined. It's gitignored local data, but nothing prunes it. It also stores
  queries and answers verbatim, which matters if a deployment ever stops being
  single-user and local.
- `llm_calls` and token counts cover only calls made through the
  `MeteredLLMClient` that `build_chat_service` installs, and only on the thread
  running the turn. Nothing on the query path spawns threads today. A future
  concurrent grader would need to propagate the context
  (`contextvars.copy_context`) or its calls would go uncounted.
- `cited_chunk_ids` trusts the model's `[n]` markers. A small model that cites
  `[1]` by habit, or that forgets to cite at all, yields a misleading or empty
  implicit judgment. Read those in aggregate, not per turn.
- **Document routing (`retrieval.document_routing`) matches wording, not
  meaning.** It routes when a question's words match a filing's record: the
  company name and the period end spelled out. "Q3 2026", "last year" or a
  fiscal-quarter name won't match, so those questions fall back to unfiltered
  retrieval, which is safe. A month that matches the wrong year's filing is
  not safe: "the March 2023 quarter" routed to a 2026 filing and lost its
  answer. Records are built in memory from the BM25 index on the first routed
  query. That's fine at 61 filings, but a corpus of thousands would want them
  persisted at index time. A document missing a field the record template
  names can never be routed to (the router logs how many).
