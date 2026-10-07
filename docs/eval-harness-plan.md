# Eval harness plan

How eval runs are identified, stored, scored and compared, where that falls
short today, and the phased work to fix it. It is groundwork for
[Milestone 23](backlog.md#milestone-23--eval-regression-gate-in-ci) (the CI
gate needs a committed baseline with per-sample scores) and for the parts of
[Milestone 27](backlog.md#milestone-27--eval-coverage-and-judge-reliability)
that re-judge or repeat runs (judge calibration, generator repeats). The storage
migration preserves existing metric definitions and historical results.
Judge calibration and repeat analysis are part of Phase 4; dataset coverage
and an untouched confirmation set are explicit companion requirements below.
Phases 4a and 4c add separate answer-quality metrics and conversational evals;
these extend the reports without redefining historical pass/fail scores.
Completing the storage migration alone does not establish measurement quality.

## What already works, and stays

- **Ground truth.** Verbatim `expected_spans` with grades, alternatives and
  `matching_mode`, `label_fixes` recorded on each sample, and the freeze rule
  ([Chunking plan, Phase 0](chunking-indexing-plan.md#phase-0--measurement-groundwork-35-days--done-2026-09-27)).
  Existing single-turn datasets remain readable. Optional answer-point and
  review annotations extend them; conversations use a separately versioned
  schema and dataset, described below.
- **Statistics.** Paired deltas with 95% CIs, and McNemar's exact test on hit
  and pass (`rag/eval/paired.py`). Wilson intervals as each tier's noise floor.
- **Crash safety.** Per-sample checkpoints (`rag/eval/checkpoint.py`),
  refused when their fingerprint doesn't match the current run. That
  fingerprint covers the effective config, the judge, the dataset and the code.
- **Config-level swapping.** `apply_overrides` and `base:` inheritance already
  make a component swap a config change, not a code change.

## Where it falls short

1. **The results fingerprint has drifted from the config.** `run_matrix.py`
   hashes an allowlist (`FINGERPRINTED`, `FINGERPRINTED_WHEN_SET`). Checked
   against `RagConfig` on 2026-09-28, the list leaves out, among others:
   `chunking.strategy`, `chunking.carry_metadata`, `reranker.query_prefix`,
   `reranker.document_prefix`, `retrieval.expansion.num_documents`,
   `retrieval.expansion.include_original`, `retrieval.web_search.*`,
   `embedding.provider` and `embedding.dimensions`. Two runs differing only
   in one of these get the same fingerprint and look comparable. This gets
   worse soon: chunking plan Phase 5 adds a second `chunking.strategy`, and
   runs with each strategy would share a fingerprint. An allowlist fails
   open, because every new field is left out until someone remembers it.
2. **A results file holds one row per variant, and a rerun replaces it.**
   Each `retrieval_<slug>_<set>.json` and `answer_<slug>.json` holds one row
   per variant name, and a rerun or `--merge` replaces the row. The only
   history is git. `results_chunking/` exists solely because re-running
   `crag=off` into the default file would have replaced the 40-sample row
   that Milestone 19 phase 4 pairs against. A new directory was the only way
   to pin a baseline.
3. **Results don't record what produced them.** A row has no commit, dataset
   hash, timestamp or index state, and the retrieval matrix doesn't record
   the judge either. After the 7 label fixes in `fac3d54`, only the commit
   message tells you which rows predate them.
4. **Raw outputs are discarded, so rescoring means rerunning.** The retrieval
   matrix keeps per-sample metrics, but not which chunks came back. Single-hop
   answer eval keeps pass/fail and `evidence_retrieved`, but not the answer
   or the judge's reply. Phase 0 step 3 noted it couldn't read any
   answer-side verdict by hand for this reason. Any change to a metric, a
   label or the judge therefore means regenerating everything. For answer
   eval that is tens of minutes per variant per set.
5. **Experiments are defined in Python, in two places.** Variants are hard-coded
   `VARIANTS` lists in `run_matrix.py` and `run_answer_matrix.py`, and both
   define `header=on` separately. A variant needing its own index also needs
   its own `data/eval/config_*.yaml` with a hand-picked `index_dir`, plus a
   manual `index --reset` before the matrix runs.

## Target design

### Dataset storage and experiment tracking

Keep three responsibilities distinct:

| Layer | Storage | Role |
|---|---|---|
| Reviewed questions, evidence labels, rubrics and splits | JSON/JSONL versioned in Git | Canonical benchmark definitions, reviewed changes and frozen releases |
| Prompts, answers, retrieved passages, judgments and score revisions | Immutable run artifacts | Canonical execution evidence for offline rescoring and reproduction |
| Searchable run metadata, metrics and comparisons | Pilot local MLflow with SQLite and a local artifact store | Derived tracking and visualization that can be rebuilt from the artifacts |

A database does not improve benchmark validity by itself. Keep our current
dataset files as the authoring and review source for now. Reconsider database
authoring when concurrent editors, review assignments, adjudication queues or
continuous production-example collection make files cumbersome. Even then,
publish an immutable dataset snapshot for each benchmark release and record
its hash in every run; editing a live table must never redefine an old result.

**Pilot MLflow before building the custom viewer.** The expanded plan now
requires enough tracking and comparison functionality to justify testing an
existing UI. Use a local tracking server, SQLite for MLflow metadata and a
local artifact store. Native MLflow evaluation-dataset management requires a
SQL backend, but evaluation can also accept dictionaries or DataFrames; the
pilot does not require moving the canonical dataset into its database.

Integrate through an optional export adapter over completed run artifacts.
Preserve our run IDs, dataset/selection hashes, score revisions, judge attempt
IDs and baseline references in the tracker. Make re-export idempotent and
record the mapping to tracker IDs. New score revisions produce distinct tracked
evaluations instead of overwriting published results. Export per-sample inputs,
outputs, evidence and scores into the tracker's evaluation views, not only
aggregate numbers and opaque attachments. Keep credentials out of artifacts.
The normal runner, scorer and CLI report must still work without MLflow installed
or running; a tracking failure must not invalidate a completed local run.

The tracker is a searchable presentation of the canonical artifacts, with no
bidirectional editing of benchmark labels or frozen results. A UI annotation
intended as a label fix must enter the normal review/versioning workflow first.
Do not maintain a second custom SQLite schema beside MLflow for the same data.

**The harness remains responsible for valid comparisons.** Retain span and
document matching, refusal rubrics, calibration, repeats, revision compatibility
and paired statistics in our code. Export paired deltas, confidence intervals,
wins/losses and p-values with the exact compared revisions. Dashboard colors or
aggregate deltas do not decide whether to change a default. If a tracker allows
comparison of different datasets or scorers, our report must still refuse it
under Comparison eligibility; a UI warning or dataset intersection does not
override that policy.

**W&B Weave is the alternative to evaluate if collaboration becomes the main
priority.** Weave provides per-example evaluation comparisons and can log
predictions/scores from an existing runner through `EvaluationLogger`. Assess
it against the same artifact-export and comparison requirements, including
deployment and account needs, rather than rewriting the harness around its
execution framework. Weave, rather than the general model-training dashboard,
is the relevant W&B product for this RAG workflow. Start with the local MLflow
pilot; no service adoption or dataset migration is implied by this plan.

### A run folder that is only ever added to

A **run** is one config against an exact selection of eval samples at one
stage. Its manifest, dataset snapshot and raw outputs are frozen when it
completes. Later scoring and judging add immutable revisions:

```
data/eval/runs/20260928T140211Z-retrieval-edgar_eval_set-3f9c2a1b/
  manifest.json     # what produced it (below); status: running | complete
  dataset.json      # original full dataset snapshot, including extra fields
  outputs.jsonl     # first line: fingerprint header; then one line per sample
  scores/
    <score_revision>.json
  datasets/         # additional label snapshots used by later revisions
    <dataset_hash>.json
  judgments/        # answer runs only; repeated judgments are retained
    <judge_hash>/<attempt_id>.jsonl
```

- **The folder name is for humans, the manifest is the source of truth.**
  The name is timestamp, stage, eval set and the first 8 characters of the
  config hash. Reports select runs by manifest fields (experiment, variant,
  config hash, dataset hash), never by parsing folder names. Folders are
  flat, not nested by experiment, because one run can belong to several
  comparisons.
- **`manifest.json`** records:
  - the experiment name and variant name;
  - the full effective config (the dump itself, not only its hash) and
    `config_hash`;
  - `index_hash`, `corpus_hash`, and `index_build_id` (see below);
  - the eval set's path and `dataset_hash`, ordered selected sample IDs,
    `selection_hash`, and the selection method and `--limit`;
  - `code_version` (the git commit plus a dirty-tree digest, from
    `checkpoint.code_version`);
  - the corpus selection and collection;
  - the generator and resolved model revision or digest where available;
    each judgment attempt records its own judge provenance;
  - repeat group and repeat number where applicable;
  - start and end times, and the host.
- **`outputs.jsonl` is the run's raw record.** For retrieval, each line holds
  the ranked results (`chunk_id`, `document_id`, score, chunk text), the
  query filter used, `routed_to`, and the latency. For answers, it holds the
  answer text, latency, token counts, and the actual passages presented to
  the generator: text, chunk and document IDs, order, and relevant metadata
  and headers. IDs alone cannot reproduce evidence scores after re-indexing.
  Multi-hop and agent runs additionally retain each search's query, filters
  and ranked results, distinguishing the union of retrieved evidence from
  passages included in the final generation prompt. Preserve the existing
  single-hop prompt-evidence and multi-hop retrieved-evidence definitions;
  label them explicitly in scores and the viewer.
  For every generation call, retain the fully rendered system prompt and
  ordered messages, supplied history, tool definitions and tool results,
  effective sampling settings, and raw model response. This includes query
  condensation and retries, not only the final answer call. Capture the
  model-facing payload after prompt assembly; context IDs or a template name
  alone cannot reconstruct it. Exclude credentials and transport headers.
  It doubles as the checkpoint: a restarted run appends to it after the
  existing fingerprint-header check, which replaces `.partial/`.
- **Scores are derived and versioned.** Each `scores/<score_revision>.json`
  identifies its source output digest, dataset and selection hashes, scorer
  version, scoring settings, and judgment attempt IDs where applicable.
  Rescoring creates a revision without replacing historical scores. Store
  the label snapshot used by each revision so the result is self-contained.
- **Judging is its own step, with retained attempts.** An answer run generates
  once. `judge_hash` includes provider, model and available model revision,
  generation settings, rubric/prompt, verdict parser version, and reference
  label hash. Each attempt records the source output digest, exact rendered
  judge inputs, verdicts and full replies. A new judge, changed reference,
  changed prompt, or repeated judgment adds an artifact. Repeating the same
  judge uses a new `attempt_id`, allowing judge variance to be measured.
- **Completion is checked separately for outputs, judgments and scores.**
  Partial artifacts cannot enter aggregate reports. Validate exact expected
  sample coverage and unique IDs before marking an artifact complete; retain
  explicit failed, skipped and unparseable outcomes and their denominators.

### Provenance and identity

- **`config_hash`** hashes the whole effective `RagConfig`, minus an explicit
  **denylist** of fields proven irrelevant to the evaluated behavior, with
  a reason for each exclusion. New fields count by default. Do not blanket
  exclude endpoints, timeouts or concurrency: these can change the backend,
  failures or measured latency. Keep the full config even for excluded
  fields. An overly broad hash causes unnecessary reruns; an overly narrow
  one silently reuses stale work. Hash equality is a reuse condition, not
  the rule for whether two experimental variants can be compared.
- **`index_hash`** identifies the index recipe: corpus selection, all chunker
  settings (including strategy, size and overlap), headers, carried metadata,
  embedding settings, and contextual model settings and prompt. Extend the
  existing manifest: today it omits chunk sizes, and `check_queryable` checks
  only embedding settings. It does not already validate this full recipe.
- **`corpus_hash` and `index_build_id`** identify the actual data. Hash source
  document contents and metadata, including loader/cleaning provenance, and
  record an immutable build identity with the stored chunk/content inventory,
  generated contexts and vector/index artifacts or their digests. Rebuilding
  contextual outputs can change the index even with the same recipe and
  corpus. Validate these identities before reuse or resume; a changed corpus
  or index build invalidates both. Record resolved model revisions where
  available rather than relying solely on mutable model names.
- **Serving evidence identity.** Consume Milestone 19's planned
  [source-version contract](milestone-19-plan.md#source-version-consistency):
  each shown passage/read window/find snippet needs its document version,
  cleaned-text identity, offsets and exact rendered text, including synthetic
  table headers or truncation markers. Preserve the mapping to the index build
  when a search supplied the hit. Retain mismatch errors as outcomes; never
  rescore an old trajectory by silently reading current files. Ingestion owns
  identity production and shared tools own enforcement; this harness records
  and validates their artifacts. Offline provenance alone does not guarantee
  that a live turn used mutually consistent sources.
- **A test keeps the classification complete.** It walks every leaf of
  `RagConfig` and fails unless each one is hashed or on the denylist with a
  reason. A separate schema-coverage assertion detects added leaves and
  requires an explicit classification decision; checking only that every
  field falls into the default hashed category would be tautological.
- **`dataset_hash`** is a SHA-256 of canonical JSON of the complete samples,
  including extra fields. In particular `tier` selects the refusal rubric,
  `parts` and `conclusion` define multi-hop judgments, and `kind` defines
  report slices. Include span grades and alternatives. Preserve a full
  snapshot, plus exact selected IDs and their hash, so a limited run cannot
  satisfy a request for the full set. Reject duplicate sample IDs.
  Hash required answer points, review/rubric versions, and conversation inputs
  and execution mode when present. Changing supplied history is an input change.
- **Label fixes and input changes have different recovery paths.** A changed
  query or other execution input requires fresh outputs for affected samples.
  A span/doc-label fix permits offline evidence rescoring; a changed reference
  answer, tier, part or conclusion requires re-judging stored answers. Refuse
  to reuse judgments against changed references. Report changed samples and
  fields, and bring both variants onto the same scoring dataset revision
  before pairing them.

### Comparison eligibility

Require the same evaluated sample IDs and inputs, scoring dataset revision,
metric definitions/settings, and (for answers) judge protocol. Pin the exact
score revisions and judgment attempts used. Different config hashes are
expected when comparing variants: show every changed config path and declare
which differences the experiment intends, including intentional code or index
changes. Flag undeclared changes as confounders. Normally require the same
corpus snapshot; a corpus-change experiment must declare that treatment.
Run reuse and checkpoint resume use the stricter identity checks above.

Keep `paired.py` for a single pair of complete runs. Comparisons of repeat
groups follow the repeat protocol below; additional answers to one question
are not additional independent dataset samples.
Conversation runs additionally require matching prescribed user turns and
history policy, and use conversation-aware uncertainty as described below.

### Scoring and reports as separate commands

```bash
python -m rag.eval.score  <run> [--eval-set PATH]      # outputs -> new score revision
python -m rag.eval.judge  <run> --judge-model M        # answers -> judgments/
python -m rag.eval.report data/eval/experiments/header.yaml   # -> .md table
python -m rag.eval.report --baseline <run>:<score_revision> --runs <run>:<score_revision>…
```

- **Scoring reuses the existing code.** Retrieval scoring runs
  `rag/eval/relevance.py` and `rag/eval/metrics.py` over the stored chunk
  text, which is exactly what `retrieval_eval` does live today, minus
  retrieval. `unmatchable_spans` also depends on the labels and the full
  indexed corpus. Record its original value and label revision at run time;
  after a label fix, recompute it against the preserved index inventory or
  mark it unavailable. Never carry an old value onto new labels silently.
- **Reports pin run IDs and score revisions**, including the judgment attempts
  behind answer scores. A baseline that other analyses depend on (like
  Milestone 19 phase 4's) is pinned in its experiment file, so later rescoring
  cannot change a published comparison.
- **Reports reuse existing metrics and single-pair statistics.** Extend the
  report layer for revision selection and repeat summaries. The `.md` tables
  become generated reports; `measured-results.md` cites the precise run and
  score revision references next to its numbers.
- **Additional answer metrics are separate scoring revisions.** Correctness,
  faithfulness, answer-point coverage and citation support each identify their
  rubric, judge and applicability. Calibrate each scorer against human labels;
  do not assume a judge calibrated for pass/fail is calibrated for all metrics.
  Report denominators and unavailable/unparseable outcomes per metric. A
  legacy run missing required prompt text or citation mappings cannot receive
  the metric that needs those missing artifacts. Do not average the dimensions
  into one score.

### Answer quality beyond pass/fail

Connect [Milestone 22](backlog.md#milestone-22--reference-free-eval-metrics)
and Milestone 27's citation work to offline judging in Phase 4a:

- **Correctness** continues to compare the answer with the reference, preserving
  the existing pass/fail definition and separate refusal rubric.
- **Faithfulness** checks answer claims against the actual passages supplied
  to the generator, recording supported, unsupported and contradicted claims
  with evidence and reasons. `evidence_retrieved` only establishes that gold
  evidence was available; it cannot establish that the answer used it faithfully.
  Report faithfulness separately even when the answer contains the right figure.
- **Completeness** uses reviewed required answer points for questions asking
  for several facts. Store stable point IDs, the expected fact, supporting
  spans/documents and applicable company, period, quantity and units. Report
  point coverage and omissions alongside pass/fail; retain per-point judgments
  in the viewer. Reuse the multi-hop parts/conclusion approach. One-fact
  questions need no artificial decomposition, and a completeness label change
  requires a new judgment revision.
- **Citation support** checks whether the passage actually cited supports the
  attached claim. Preserve citation markers, claim-to-citation mappings and
  passage text; a claim supported elsewhere in the prompt may still have an
  incorrect citation. Track missing citations separately from incorrect support
  under an explicit citation requirement. Refusals and claim-free responses
  follow documented applicability rules rather than getting automatic perfect
  faithfulness or citation scores.
  Preserve invalid marker occurrences as deterministic validation results;
  an out-of-range marker discarded by `cited_chunk_ids` must not disappear
  from evaluation. An empty generation or its user-facing failure message is
  a generation failure, not a successful refusal. Runtime reporting is owned
  by [Milestone 19's output contract](milestone-19-plan.md#execution-and-output-contracts).

Answer relevance and context relevance remain Milestone 22's companion metrics
and can use the same offline scoring interface. Their definitions, calibration
and denominators must be explicit before they influence default selection.

### Conversational evaluation

Multi-hop asks for several facts in one question; conversational evaluation
tests dependence on earlier user and assistant turns. The current answer
runner calls `ask(sample.query)` without history even though the chat service
supports it. Phase 4c adds a separate conversation dataset and runner mode:

- Record conversation ID, ordered turn IDs, user messages, supplied history,
  per-turn references, evidence labels, expected answer/refusal behavior and
  optional required answer points. Validate a homogeneous sample type within
  each dataset and version this schema separately from single-turn samples.
- Cover pronoun/ellipsis follow-ups ("What about the following year?"), company
  and topic switches, user corrections, and unanswerable follow-ups at different
  positions. Review answerability and clarity against the supplied history.
- Run two explicitly named modes: **fixed history**, where each scored turn
  receives the same reviewed preceding conversation across variants, and
  **generated history**, where each variant's own previous answers are carried
  forward. Preserve the exact resulting history and rewritten retrieval query.
  Keep these modes in separate reports; their failure causes differ.
- Report retrieval and answer metrics per turn and by turn position, plus an
  explicitly defined conversation-level success measure. Keep related turns
  together when selecting subsets, creating confirmation splits and estimating
  uncertainty. Do not feed all dependent turns into the independent-sample CI
  as if they were unrelated questions. Input identity for generated-history
  comparisons means the same prescribed user turns and history policy; realized
  assistant histories may differ as part of the measured behavior.
- Retain the full ordered trace, prompts, rewrites and evidence in the run
  folder. Checkpoint generated-history runs with their preceding transcript so
  resumption cannot combine a later answer with a different earlier history.

### A viewer for comparing runs

Tables say *how much* two configs differ, not *why*. Answering why means
reading the questions that flipped, with each run's chunks and answers side
by side. Today that takes a rerun with `-v` and scrolling through logs.
Inspect's `inspect view` and HELM's `helm-server` each ship this view. Phase 4b
first tests MLflow against the requirements below. Use its UI for general run
tracking and comparison if the pilot succeeds; build only the specialized
views it cannot supply, such as span highlighting or conversation diagnostics.
If the pilot fails the acceptance checks, retain a standalone Streamlit viewer
over the run folders as the fallback:

```bash
streamlit run rag/ui/eval_viewer.py
```

- **Shared comparison logic, optional custom UI.** `rag/eval/compare.py`
  contains pure functions for pairing, flip classification, config differences
  and compatibility checks, reusable by CLI reports and tracker exports.
  If a custom view is needed, it is separate from the chat UI: `rag/ui/app.py`
  builds a `ChatService` at startup and fails without Ollama and Chroma. The viewer
  only reads files, so it has to work with neither running. The logic lives
  in those pure functions. The Streamlit wiring stays thin, the
  same `app.py`/`helpers.py` split Milestone 8 uses, so the logic is tested
  without a Streamlit runtime.
- **The run list** is filtered by eval set and experiment, and shows
  headline metrics per score revision. Pair selection follows comparison
  eligibility, including matching scoring-label revisions after a rescore;
  the original run's dataset hash alone does not decide eligibility.
- **Comparing two runs** shows:
  - the paired deltas with CI, W/L and McNemar's p, from `paired.py`;
  - a **config diff**: the dotted paths where the two manifests' effective
    configs differ, which confirms a variant changed only what it meant to;
  - a sample table filterable by outcome (win, loss, both hit, both miss)
    and by tier or `kind`, sorted with losses first.
- **One sample's detail view** shows:
  - the query and expected spans;
  - each run's ranked chunks side by side, with span matches highlighted,
    the rank of the first relevant chunk, and the filter or `routed_to`
    applied;
  - for answer runs: each answer, `evidence_retrieved`, and the reply from
    every selected judgment attempt, with its rubric and reference revision;
    multi-hop detail distinguishes retrieved evidence from prompt evidence.
  - where available: claim-level grounding, missing answer points and citation
    support; conversation details show turn order, history mode and retrieval
    rewrites. Exact rendered generator inputs are inspectable for every call.
- **Keep UI dependencies optional.** MLflow belongs in an optional tracking
  extra. Streamlit is already available for specialized or fallback views.
  `st.dataframe` takes plain lists of dicts, so custom views need not import
  pandas solely for tables.

### Declarative experiments

```yaml
# data/eval/experiments/chunk-header.yaml
description: Chunking plan Phase 2, deterministic header
corpus: [edgar]
eval_sets: [edgar_eval_set, edgar_period_set, edgar_underspecified_set]
stages: [retrieval]                # and/or answer, multihop, conversation
baseline: shipped                # initial run; resolve and pin before reporting
# Published comparisons pin baseline and candidate run:score_revision references.
variants:
  shipped: {}
  header=on:     {config: data/eval/config_header.yaml}
  rerank_header: {config: data/eval/config_header.yaml,
                  overrides: {reranker.include_header: true}}
```

- **One runner for everything:** `python -m rag.eval.run <experiment.yaml>`.
  It takes `--only`, `--variant`, `--sets`, `--limit` and `--repeats k`,
  covers every stage, and skips any (variant, set, stage) that already has a
  complete run with the same config, dataset, sample selection, code, corpus
  and index build identities. Generation reuse is independent of whether a
  requested judgment already exists; changing only the judge requests a new
  judgment, not new answers. Record the full config, but use its generation
  portion for this reuse check, excluding judge-only settings explicitly.
  That replaces `--merge`. With `--repeats k` it runs k more copies instead,
  which is the repeat mode Milestone 27 asks for.
- **Index resolution.** For each variant, the runner computes the
  `index_hash` and `corpus_hash` and validates them against the index manifest
  and actual inventory. Reuse the shipped index only when both match and its
  build identity is verified. Other builds live under
  `data/index_variants/<index_hash>/<corpus_hash>/<index_build_id>/`; variants
  can share a verified build without a hand-picked `index_dir`.
- **Missing indexes are never built by default.** A dry run
  (`--plan`, and also the default whenever an index is missing) lists what
  would run, which indexes are missing, and an estimated time taken from
  earlier runs' manifests. Building requires `--build-indexes`. The
  contextual index took about 1.9 hours and spends LLM calls, so it must
  never start as a side effect of a typo.
- `scripts/experiments/` stays as it is: those are one-off probes, not
  variant matrices.

### Measurement quality requirements

- **Review question quality explicitly.** Before accepting new samples, record
  answerability from the intended evidence, usefulness to the target user, and
  clarity of the question's intended meaning, with a decision and reason for
  each dimension. Verbatim span checks and an instruction to write realistic
  analyst questions do not establish usefulness. LLM critique may triage drafts;
  human review decides acceptance. Apply the rubric by tier: refusal questions
  must be unanswerable in the corpus, conversational questions may rely on their
  supplied history, and implicit/paraphrase questions must remain interpretable
  without accidentally requiring the hidden drafting passage. Do not apply a
  universal answerable-and-standalone filter. Retain draft rejections separately
  from the frozen benchmark and version the review rubric.
- **Quality-control human labels.** Write a short annotation guide with worked
  examples of partial evidence, wrong periods/units, contradictory passages,
  valid refusals and context-dependent questions. Independently double-review
  a predeclared subset spanning the tiers, record reviewer IDs and original
  decisions, and report agreement and disagreement counts. Adjudicate conflicts
  with a recorded reason before freezing labels. Apply this procedure to both
  dataset labels and judge-calibration answers, including claim, answer-point
  and citation annotations. If only one reviewer is available, record that
  limitation rather than claiming independent agreement. Sample across outcomes
  rather than checking only current-system misses.
- **Calibrate the judge in Phase 4.** Pull forward Milestone 27's fixture of
  about 60 human-labeled answers spanning single-hop passes and failures,
  refusals, and multi-hop parts, including wrong-period false passes. Retain
  the human labels and rationale. Report agreement, false-pass and false-fail
  rates with counts by category and uncertainty; repeat judging on the same
  stored answers to measure consistency separately. Run calibration whenever
  the judge model, rubric or parser changes. Define acceptable error bounds
  before evaluating a new judge; mark results exploratory if calibration
  cannot support the size of improvement being claimed. Matching a previous
  pass rate is not evidence that judgments are correct.
- **Analyze repeats in Phase 4.** Add repeat support to the answer matrix
  before the unified runner exists. Predeclare the number of repeats and
  pairing/order policy; retain repeat IDs, seeds when supported, and sampling
  settings. Initially run at least three repeats per compared variant and
  report per-run rates, mean, standard deviation and range, plus paired
  repeat deltas. This is an initial variance check, not a guarantee of power.
  Repeat judgments of fixed outputs measure judge noise; fresh generations
  under a fixed judge protocol measure generator variability plus residual
  judge noise. Temperature zero does not remove the need to check variability.
  Do not flatten question-by-repeat observations into `paired.py` as if n
  grew by the repeat count. Keep single-run paired CIs separate from repeat
  spread; any combined CI must account for repeated observations per question.
  Report inconclusive small effects when the repeats cannot resolve them.
- **Cover the intended questions.** Continue reporting generated, period,
  underspecified (including each `kind`), refusal and multi-hop tiers
  separately, with sample counts. Add the table tier under chunking-plan
  Phase 4 before evaluating the new chunker, and Milestone 27's harder
  period-shifted refusal cases before claiming reliable refusal behavior.
  Human-review and freeze the new sets before comparing variants. Record
  coverage gaps and audit label alternatives and reference-answer quality;
  a harness migration does not fix those gaps automatically.
- **Confirm selected defaults on untouched questions.** Reserve a separately
  reviewed and frozen confirmation set before further tuning, with the target
  tiers represented and related questions grouped by source fact/document
  family to avoid putting paraphrases of the same fact on both sides. Keep
  its outcomes out of variant selection. Declare the primary metrics, slices
  and practical improvement/regression criteria before running the selected
  candidate and baseline on it. Report confirmation results separately from
  exploratory matrices. After inspection it is no longer untouched for later
  tuning; record that use and replenish for subsequent selection cycles.
  Freezing labels alone does not prevent tuning to a repeatedly viewed set.

**Completion for measurement claims:** Phase 4 must deliver calibration and
repeat reports as well as storage. Claims about broader coverage or a new
default additionally need the applicable new tiers and a recorded confirmation
comparison. These are companion dataset tasks tracked in Milestone 27, not
evidence that follows automatically from the viewer or runner shipping.
Grounding/completeness claims additionally require Phase 4a's calibrated
metrics; conversational claims require Phase 4c's reviewed cases and both
history modes. Question review and annotation procedures precede new dataset
freezes and creation of the calibration fixture.

## Phases

Each phase ships on its own and leaves the harness usable. Estimates below
were for the original storage migration and must be revised for index build
provenance, immutable scoring revisions, calibration and repeat analysis.
Human dataset review and confirmation-set creation need separate estimates.
Phases 4a and 4c are additional scope with estimates to be set after their
rubrics and dataset sizes are agreed. Phase 4b's original one-day custom-viewer
estimate is superseded by the tracking pilot; estimate any specialized UI only
after its gaps are known.

### Phase 1 — Full-config fingerprint (half a day)

- Add `rag/eval/provenance.py` with `config_hash`, `index_hash`,
  `corpus_hash`, `dataset_hash`, and `selection_hash`; move `code_version`
  there from `checkpoint.py`. Extend index manifests to record and validate build
  identity and all index-affecting settings. Add the config classification
  tests and complete dataset snapshots described above.
- `run_matrix.py` and `run_answer_matrix.py` write `config_hash`,
  `dataset_hash`, `code_version` and a timestamp into every new row. They
  keep the old `fingerprint` beside it, so existing rows stay readable.
- **Done when** classification tests pass; changed source content, index build,
  chunk size/strategy, refusal tier, multi-hop parts/conclusion, or selected
  IDs invalidate the appropriate identity. A new baseline row's old
  `fingerprint` still matches the committed one (the old hash is unchanged).

### Phase 2 — Run folders for retrieval (1–1.5 days)

- `retrieval_eval` and `run_matrix.py` write run folders with
  `outputs.jsonl` (including per-query latency, which Milestone 24 needs and
  `retrieval_eval` doesn't record yet). They keep writing the legacy
  `results/*.json` and `.md` files in the same pass, so nothing downstream
  breaks yet.
- `outputs.jsonl` replaces `.partial/` checkpoints for retrieval.
- **Done when** the shipped baseline on `edgar_eval_set` produces a run
  whose initial score revision matches the current `baseline` row's metrics
  exactly, and an interrupted run resumes only against the same verified
  index build. Duplicate/missing samples prevent completion; a limited run
  cannot be reused as a full-set run.

### Phase 3 — Offline scoring and reports (1 day)

- `rag.eval.score` and `rag.eval.report`, reusing `relevance.py`,
  `metrics.py`, `paired.py` and `render_table`.
- Add immutable score revisions, explicit baseline/candidate revision pins,
  and comparison eligibility checks independent of config-hash equality.
- **Done when** a report built from Phase 2 runs is identical to the legacy
  `.md` table for the same variants. Also, rescoring a run against an eval
  set with one edited span label changes exactly that sample's scores and
  preserves the original revision. Changed inputs require fresh outputs;
  incompatible metric or label revisions cannot be paired. Config variants
  with declared differences can be paired. A pinned report stays unchanged
  after later rescoring.

### Phase 4 — Answer side: generate and judge as separate steps (1.5 days)

- Answer and multi-hop runs write answers to `outputs.jsonl` and verdicts to
  `judgments/<judge_hash>/<attempt_id>.jsonl`. Persist prompt passages and
  multi-hop search evidence so evidence scoring works without a live index.
  `rag.eval.judge` re-judges a finished run against explicit reference labels.
- Capture exact rendered generation inputs for every call, including history,
  system messages, tools, query rewrites and retries. Create the annotation
  guide and retain question-review and independent label-review records for
  the calibration fixture under Measurement quality requirements.
- Support [Milestone 19's per-step utility experiment](milestone-19-plan.md#per-step-retrieval-utility-and-stopping)
  through these same artifacts: record step prefixes and ledger snapshots,
  and link separately versioned diagnostic syntheses/judgments to their source
  run and step. Probes cannot mutate the original transcript or read future
  evidence; keep their cost apart from serving usage. Phase 4a supplies their
  calibrated evidence/answer metrics. A stopping policy must include its own
  inference overhead in serving cost and meet the plan's confirmation gate.
- `run_answer_matrix.py` writes run folders, as in Phase 2.
- If Milestone 19's [per-task evidence-state experiment](milestone-19-plan.md#per-task-evidence-state)
  is built, retain each state revision, candidate/support labels and exact
  source mappings beside the trajectory. Evaluate it under matched total
  budgets with Phase 4a's independent scoring; the agent's own support flags
  are not gold labels. Record updates/checks in serving cost. If resumable live
  tasks are introduced, preserve interruption/resume events and cumulative
  usage for analysis, while serving recovery remains Milestone 19's contract.
- Deliver the human calibration fixture and report, plus generation and judge
  repeat support and summaries under Measurement quality requirements.
- **Done when** evidence scores reproduce offline with the live index removed;
  changed references force a new judgment; repeated attempts and a second
  judge coexist without overwrites; calibration reports category error counts
  and rates; and at least three answer repeats per compared variant report
  spread without inflating the number of independent questions. Record any
  unresolved judge accuracy or variance limits on measurement claims.
  A saved trace must expose the exact assembled inputs without reconstructing
  them from current templates; calibration labels include review decisions and
  adjudication records or an explicit single-reviewer limitation.

### Phase 4a — Grounding, completeness and citation scoring (estimate pending)

- Implement the offline scorers under Answer quality beyond pass/fail, linking
  Milestones 22 and 27. Extend calibration examples to cover correct figures
  with unsupported explanations, omitted required facts, incorrect period or
  units, and claims citing the wrong passage.
- Include missing/invalid citation markers, correct claims citing irrelevant
  passages, and empty-generation failures whose fallback wording resembles a
  refusal. Deterministic validity and calibrated semantic support are separate
  results; neither can substitute for the other.
- Add optional reviewed answer points to compound questions while preserving
  existing simple questions and multi-hop scores. Each scorer produces its
  own immutable judgments and score revision.
- **Done when** reports distinguish correctness, faithfulness, completeness
  and citation support on the reviewed cases, including an answer that is
  correct but ungrounded and one that is grounded but incomplete. Each metric
  has a human-calibration report, per-item explanation and explicit denominator;
  reference-free scorers can run without an `expected_answer`. Re-judging needs
  neither regeneration nor a live index, and original pass/fail scores retain
  their meaning. Uncalibrated metrics remain exploratory.

### Phase 4b — Local MLflow pilot and comparison UI (estimate pending)

- Add shared comparison functions and an optional MLflow export adapter under
  [Dataset storage and experiment tracking](#dataset-storage-and-experiment-tracking).
  Run a local SQLite-backed tracking server with local artifacts. First import
  one baseline and one candidate over the same dataset, including their
  per-sample scores; include an answer pair once Phase 4 artifacts exist.
  The retrieval pilot can start after Phase 3.
- **Pilot acceptance checks:**
  - Preserve dataset/selection hashes, local run IDs, score/judge revisions and
    pinned baseline references. Re-exporting does not duplicate evaluations.
  - Inspect individual losses and side-by-side answers/evidence. Comparing
    `baseline` with `header=on` on `edgar_period_set` yields the same paired
    sample count and W/L as the CLI report; config differences are accessible.
  - Display or directly link our paired CIs, deltas and McNemar results with
    their pinned comparison inputs. Dataset/scorer mismatches remain refused
    by the harness even if the UI permits a looser comparison.
  - Rescore stored outputs into a new revision, export it, and reproduce the
    original pinned comparison unchanged. No model calls or live index are
    needed to browse stored runs or export existing scores.
  - With tracking unavailable, local execution/scoring/reporting still work.
    Demonstrate rebuilding the tracking view from retained canonical artifacts.
- **Done when** the pilot records these results, dependency/setup costs and
  remaining UI gaps, followed by an explicit implementation decision. If it
  passes, adopt MLflow for general tracking and build only missing specialized
  views. If it fails, document why and implement the file-backed Streamlit
  fallback against the same comparison requirements. Do not build both full UIs.
- Test shared pairing, flip classification, config diff and revision checks,
  plus adapter identity/re-export behavior. Inspect the chosen UI with Ollama
  stopped. Add Phase 4a's claim/point/citation detail and Phase 4c's conversation
  views when available; absent dimensions remain unavailable. Reconsider Weave
  using the same pilot contract if team review becomes the deciding requirement.

### Phase 4c — Conversational dataset and runner (estimate pending)

- Deliver the versioned conversation schema, reviewed/frozen scenarios and
  fixed-history/generated-history modes under Conversational evaluation.
- Extend storage, resume, reports and the viewer for ordered turns and their
  actual history, prompts and rewritten search queries. Calibrate conversational
  rubrics and keep conversations together in statistical comparisons.
- **Done when** both modes run the same reviewed scenarios with per-turn and
  conversation summaries; a changed supplied history invalidates reuse; and
  an interrupted generated-history run resumes its exact recorded transcript.
  Demonstrate ellipsis, company switch, correction and mid-conversation refusal
  cases, with separate mode reports and no inflated independent-sample count.

### Phase 5 — Experiment files and one runner (2 days)

- Add `rag.eval.run` and move the two `VARIANTS` lists into
  `data/eval/experiments/*.yaml`, one file per axis group, keeping the
  current names so reports line up.
- Add index resolution with `--plan` and `--build-indexes`.
- Register Phase 4a's scorers and Phase 4c's conversation modes without merging
  their metrics, dataset types or history policies into single-hop summaries.
- **Done when** the chunk-header experiment above runs end to end from its
  YAML, reusing the existing header index, and the dry run correctly reports
  a missing index for an unbuilt variant. A changed corpus/build or sample
  selection cannot reuse a stale completed run; repeated runs follow the
  Phase 4 protocol.

### Phase 6 — Retire the old paths and update the docs (half a day)

- Delete the variant lists, `--merge` and the legacy writers. Keep
  `run_matrix.py` and `run_answer_matrix.py` as thin wrappers, or delete
  them, depending on whether anything external calls them.
- Update the `measure-change` skill (reading a result now includes the
  viewer's losses), CLAUDE.md's "Running things" section (runner, score,
  judge, report, optional tracking/export and chosen viewer commands), the eval notes in
  `docs/milestone-notes.md`, and this plan's status.
- **Done when** `measure-change` describes only the new flow, and `pytest`,
  `ruff check .` and `mypy --ignore-missing-imports rag` pass.

## Migration

- **Legacy results stay where they are, read-only.** Links from
  `measured-results.md` keep working, and the numbers keep their meaning.
- No bulk conversion. When a legacy row is needed as a baseline, re-run it:
  retrieval takes minutes. Legacy answer rows cost tens of minutes each and
  are re-run only when something pairs against them (Milestone 19 phase 4's
  `crag=off`).

## Decisions and rejected alternatives

- **Borrow evaluation practices without requiring a framework migration.**
  The external references below inform dataset review, grounding and traces.
  Keep canonical local datasets and artifacts with optional tracking exports.
  Preserve corpus-specific challenge cases:
  repeated text across filings is intentional evidence for the period tier,
  so a blanket document-deduplication rule would remove what it tests.

- **Inspect (UK AI Security Institute): not adopted for these evals, and
  revisited at Milestone 19 phase 4.** Inspect's model (a dataset, a solver,
  a scorer, one log file per run) is the design this plan follows. Its
  `inspect view`, `inspect score` and epochs would replace Phases 2–4b.
  For retrieval and pipeline evals it loses on four counts:
  - it adds about 40 direct dependencies (0.3.272: `boto3`, `s3fs`,
    `fastapi`, `textual`, `tiktoken`…);
  - retrieval is not a model output, so ranked chunks would have to pass
    through sample metadata to a custom scorer;
  - it has no paired comparison between runs, only each run's own standard
    error, so `paired.py` would still decide every default;
  - moving to it means rewriting the three runners and both matrix scripts.

  Its strengths are agent transcripts, tool sandboxes and grids of models ×
  benchmarks, which is what Milestone 19's agentic evals need. So the
  question moves there, recorded in
  [Milestone 19 plan, phase 4](milestone-19-plan.md#4--measure-the-milestones-actual-deliverable).
  To keep that move cheap, keep this plan's manifest and per-sample fields
  close to Inspect's log schema: sample id, input, target, output, scores
  with explanation, model usage, and the full config.
- **MLflow: pilot first; W&B Weave: collaboration alternative.** This replaces
  the earlier blanket rejection. The expanded UI requirements justify testing
  MLflow locally before maintaining a full custom viewer. Adoption depends on
  Phase 4b's observed fit with our artifacts and comparison rules; current
  documentation alone does not establish that fit.
- **SQLite for tracking, files for canonical benchmarks.** MLflow's database
  provides indexing and navigation while Git retains reviewable labels and
  run artifacts retain execution evidence. Database storage does not prevent
  exporting immutable snapshots. Dataset authoring can move to a database
  later if collaboration requires it, with reviewed releases still frozen.
- **Nesting runs under experiment folders.** Rejected because one run serves
  several comparisons, such as the shipped baseline, which every experiment
  pairs against. Grouping lives in the experiment file and the manifest.
- **Keeping the allowlist and just adding the missing fields.** This is the
  quick fix, and it is roughly what Phase 1 would cost anyway. It fixes
  today's drift and leaves the mechanism that caused it. The denylist plus
  the classification test costs about the same and closes the gap for good.
- **Building indexes automatically without asking.** Rejected. See Index
  resolution.
- **A static HTML report instead of a viewer page.** A generated HTML report
  would need no running process, but it must be regenerated for each pair of
  runs. It also can't filter or drill into one sample without its own
  JavaScript. Streamlit is already a dependency and already the project's UI.
  The generated markdown tables stay as the committable summary.

## Open questions

- **What to commit.** Manifests, dataset snapshots and score revisions are
  committed for retained runs. Published comparisons pin exact artifacts,
  including referenced judgment attempts and calibration reports.
  `outputs.jsonl` with chunk text is estimated at around 1 MB per
  retrieval run (174 samples × 5 chunks × ~1.1k characters), which comes to
  roughly 60 MB for a 20-variant matrix over 3 sets. Too much for git history
  at every run. Recommendation: keep `outputs.jsonl` gitignored by default,
  gzip it, and commit it (`git add -f`) only for runs cited in
  `measured-results.md` or used as a CI baseline. Measure the real size in
  Phase 2 before settling this. The alternative is a store of chunk text
  keyed by hash and shared across runs (most variants retrieve the same
  chunks). It is much smaller but needs its own bookkeeping, and is worth it
  only if the gzipped files turn out too large.
- **Pairing across a label fix.** Bring both runs onto the same scoring-label
  revision, rescoring or re-judging as required. Pairing on the intersection of
  unchanged samples is also valid, but it quietly shrinks n, so it should be
  available only through an explicit flag.
- **The CI gate's index.** Milestone 23 must use the index recipe and corpus
  hashes for cache selection and validate the restored build identity. The
  cache transport and build-artifact retention policy remain implementation
  decisions for that milestone; config equality alone cannot validate a cache.

## Order and why

Phase 1 comes first because it fixes incorrect reuse and missing provenance.
Phases 2–4 come next because every later measurement depends on raw outputs,
versioned scoring, and calibrated judgments. Phase 4 also delivers repeat
analysis before small answer improvements are used to select defaults.
Prepare confirmation questions before further tuning, and schedule table and
hard-refusal coverage with their dependent measurements. Phase 4b is
where stored outputs pay off: pilot MLflow to inspect why variants lost before
building a custom viewer. Phase 5 is convenience, and it depends on index hashing
from Phase 1 and run folders from Phase 2. The chunking plan's Phases 4–5 would
benefit from Phases 1–2 landing first: Phase 5's new `chunking.strategy` is exactly the field the
current fingerprint misses.

Question-quality review and human annotation procedures begin before the next
dataset freeze. Phase 4 captures exact generation inputs; Phase 4a adds the
calibrated answer dimensions before making grounding or completeness claims.
Phase 4c adds conversational coverage after the core answer artifacts work.
The chosen tracking UI can expose single-turn comparisons first, with specialized
views added only for verified gaps as the new dimensions land. The unified
runner incorporates those capabilities while retaining optional tracking.

## External references and how they apply

- [MLflow: RAG Evaluation Datasets](https://mlflow.org/articles/rag-evaluation-datasets/)
  informs the annotation guide, independent review/adjudication, complete
  generator-input logging and conversational coverage. Adapt its practices
  to our corpus and scale rather than adopting fixed dataset percentages or
  removing intentionally repeated filing text.
- [Hugging Face: RAG Evaluation](https://huggingface.co/learn/cookbook/en/rag_evaluation)
  supplies the answerability, user relevance and standalone-clarity review
  dimensions. Apply them by tier and preserve human acceptance decisions;
  its correctness-only tutorial is not our complete metric set.
- [Ragas: Evaluation Dataset](https://docs.ragas.io/en/stable/concepts/components/eval_dataset/)
  and [Evaluation Sample](https://docs.ragas.io/en/stable/concepts/components/eval_sample/)
  inform the explicit input/context/response/reference contract and separation
  of single-turn and conversational samples. Our run artifacts provide this
  contract without requiring the Ragas classes as an internal dependency.
- [Ragas: Faithfulness](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/)
  motivates checking claims against supplied evidence separately from reference
  correctness. [Factual Correctness](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/factual_correctness/)
  informs claim-level coverage; our required answer points are reviewed labels
  and reuse the existing multi-hop approach.
- [MLflow backend stores](https://mlflow.org/docs/latest/self-hosting/architecture/backend-store/)
  and [evaluation datasets](https://mlflow.org/docs/latest/genai/datasets/)
  document local SQLite metadata storage and the SQL requirement for native
  dataset management. Our canonical dataset files remain separate.
- [MLflow custom scorers](https://mlflow.org/docs/latest/genai/eval-monitor/scorers/custom/)
  and [evaluation comparisons](https://mlflow.org/docs/latest/genai/eval-monitor/running-evaluation/prompts/)
  support trying our existing scoring logic and per-example comparisons in its
  UI; Phase 4b verifies the actual fit before adoption.
- [W&B Weave comparisons](https://docs.wandb.ai/weave/guides/evaluation/compare_evals)
  and [EvaluationLogger](https://docs.wandb.ai/weave/guides/evaluation/evaluation_logger)
  describe the alternative UI and integration from an existing runner. Weave
  can compare dataset intersections with warnings, so our stricter comparison
  rules must remain enforced outside the dashboard.
