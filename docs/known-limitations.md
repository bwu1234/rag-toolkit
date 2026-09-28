# Known limitations / roadmap

- **Neither Milestone 9 nor 10 is measured on this corpus yet.** Both are off by
  default and both were verified functionally (see `docs/milestone-notes.md`)
  rather than evaluated. `data/eval/eval_set.json` is now real — 43 samples
  against documents that exist in `data/corpora/baseline/documents/` — so
  retrieval and answer eval finally produce signal, but no before/after numbers
  have been recorded for either feature. Run `retrieval_eval` with
  `chunking.contextual` on vs. off, and `answer_eval` with `crag` on vs. off,
  before recommending either.
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
- The grader, the retry rewriter, the groundedness checker, the condenser, the
  expanders, and the answering model are all the *same* local 9b model. A
  groundedness check is only as good as the model performing it, and a model
  checking output shaped like its own has an obvious blind spot. A larger or
  simply different judge model would be a real improvement and needs no
  interface change — only a second `LLMClient` in the builder.
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
- Chunking is character-based fixed-size with overlap; token-aware and
  structure-aware/semantic chunking are deferred until the end-to-end
  pipeline is proven (both fit behind the existing `Chunker` interface).
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
- `retrieval.min_score` is tuned by hand against the eval set; there's no
  calibration step, and its meaningful range shifts with the reranker. Set it
  too high and answerable questions get refused; too low and it does nothing.
- Query condensing costs an extra LLM round trip on every turn that has
  history, and the rewrite is only as good as the local model. The rewritten
  query is reported back on `ChatAnswer`/`ChatResponse` and shown in the UI, so
  a bad rewrite is at least visible — but nothing detects or corrects one.
- Conversation history is never shown to the *answering* model, only to the
  condenser. Questions whose answer depends on the thread rather than on the
  corpus ("summarize what you just told me") aren't served by this design.
- Reranking is opt-in via config (`reranker.provider: none` is still the
  default in `config.yaml`); switch to `cross_encoder` to enable it. A
  pure-LLM reranker (reusing the existing `LLMClient`/Ollama setup) remains
  a possible lighter-weight alternative behind the same `Reranker` interface.
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
  step; nothing is done today.
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
