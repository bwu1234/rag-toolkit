# Eval harness plan

How eval runs are identified, stored, scored and compared, where that falls
short today, and the phased work to fix it. It is groundwork for
[Milestone 23](backlog.md#milestone-23--eval-regression-gate-in-ci) (the CI
gate needs a committed baseline with per-sample scores) and for the parts of
[Milestone 27](backlog.md#milestone-27--eval-coverage-and-judge-reliability)
that re-judge or repeat runs (judge calibration, generator repeats). It
changes no metric and no eval set: the numbers in
[Measured results](measured-results.md) keep their meaning.

## What already works, and stays

- **Ground truth.** Verbatim `expected_spans` with grades, alternatives and
  `matching_mode`, `label_fixes` recorded on each sample, and the freeze rule
  ([Chunking plan, Phase 0](chunking-indexing-plan.md#phase-0--measurement-groundwork-35-days--done-2026-09-27)).
  The dataset format doesn't change.
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

### A run folder that is only ever added to

A **run** is one config against one eval set at one stage. Each run gets its
own folder, which is never edited after it completes:

```
data/eval/runs/20260928T140211Z-retrieval-edgar_eval_set-3f9c2a1b/
  manifest.json     # what produced it (below); status: running | complete
  outputs.jsonl     # first line: fingerprint header; then one line per sample
  scores.json       # derived from outputs.jsonl + the eval set; rewritable
  judgments/        # answer runs only: one file per judge
    <judge_hash>.jsonl
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
  - `index_hash` (see below), the eval set's path and `dataset_hash`;
  - `code_version` (the git commit plus a dirty-tree digest, from
    `checkpoint.code_version`);
  - the corpus selection and collection;
  - the generator, plus the judge where there is one;
  - start and end times, and the host.
- **`outputs.jsonl` is the run's raw record.** For retrieval, each line holds
  the ranked results (`chunk_id`, `document_id`, score, chunk text), the
  query filter used, `routed_to`, and the latency. For answers, it holds the
  retrieved context ids, the answer text, the latency and the token counts.
  It doubles as the checkpoint: a restarted run appends to it after the
  existing fingerprint-header check, which replaces `.partial/`.
- **Scores are derived, never primary.** `scores.json` is computed from
  `outputs.jsonl` and the eval set. It can be regenerated at any time and
  records the `dataset_hash` it was scored against.
- **Judging is its own step, keyed by judge.** An answer run generates once.
  Each judge writes `judgments/<judge_hash>.jsonl`, holding the verdict and
  the judge's full reply. A second judge, or a changed judge prompt, adds a
  file instead of regenerating. This retires the `answer_edgar__judge-*.json`
  naming workaround, and it is what Milestone 27's judge calibration needs.
- **A run marked `running` is never reported.** This keeps the checkpoint's
  rule that half a set is never presented as a whole one.

### Two hashes built from the full config, not an allowlist

- **`config_hash`** hashes the whole effective `RagConfig`, minus a
  **denylist** of fields that can't change results: `base_url`s,
  `concurrency`, `timeout_s`, cache paths, logging, `observability.*` and
  `api.*`. A new field then counts by default. The failure mode becomes a
  pair of runs wrongly refused as incomparable, which is visible, instead of
  wrongly paired, which is silent.
- **`index_hash`** covers what `rag/index_manifest.py` already checks before
  a query: the corpus selection, the chunker settings, the header,
  `carry_metadata`, the embedder and the contextual prompt. It says which
  index a run needs, which Phase 5 uses.
- **A test keeps the classification complete.** It walks every leaf of
  `RagConfig` and fails unless each one is hashed or on the denylist with a
  reason. Adding a config field then forces a decision about it, the same
  way `FINGERPRINTED` was meant to but couldn't enforce.
- **`dataset_hash`** is a SHA-256 of the canonical JSON of the samples (ids,
  queries, spans, doc ids, expected answers, matching mode). A label fix
  changes it, so a comparison across a label fix is refused, and the refusal
  names the changed samples. Rescoring the older run against the current set
  is then one command (below).

### Scoring and reports as separate commands

```bash
python -m rag.eval.score  <run> [--eval-set PATH]      # outputs -> scores.json
python -m rag.eval.judge  <run> --judge-model M        # answers -> judgments/
python -m rag.eval.report data/eval/experiments/header.yaml   # -> .md table
python -m rag.eval.report --baseline <run> --runs <run> <run>…
```

- **Scoring reuses the existing code.** Retrieval scoring runs
  `rag/eval/relevance.py` and `rag/eval/metrics.py` over the stored chunk
  text, which is exactly what `retrieval_eval` does live today, minus
  retrieval. `unmatchable_spans` depends on the chunker and the corpus, not
  on outputs, so it is computed at run time and stored in the manifest.
- **Reports name their baseline by run id**, instead of taking "the first
  variant in the file". A baseline that other analyses depend on (like
  Milestone 19 phase 4's) is pinned by id in its experiment file, so nothing
  can overwrite it.
- **Reports reuse `render_table` and `paired.py` unchanged.** The `.md`
  tables become generated reports from selected runs. `measured-results.md`
  cites run ids next to its numbers.

### A viewer for comparing runs

Tables say *how much* two configs differ, not *why*. Answering why means
reading the questions that flipped, with each run's chunks and answers side
by side. Today that takes a rerun with `-v` and scrolling through logs.
Inspect's `inspect view` and HELM's `helm-server` each ship this view. Here it
is a Streamlit page over the run folders:

```bash
streamlit run rag/ui/eval_viewer.py
```

- **A separate script from the chat UI.** `rag/ui/app.py` builds a
  `ChatService` at startup and fails without Ollama and Chroma. The viewer
  only reads files, so it has to work with neither running. The logic lives
  in `rag/eval/compare.py` as pure functions: loading runs, pairing samples,
  classifying flips, diffing configs. The Streamlit wiring stays thin, the
  same `app.py`/`helpers.py` split Milestone 8 uses, so the logic is tested
  without a Streamlit runtime.
- **The run list** is filtered by eval set and experiment, and shows
  headline metrics per run. Runs with different `dataset_hash`es are never
  offered as a pair.
- **Comparing two runs** shows four things:
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
    every judge that has judged the run.
- **No new dependency.** Streamlit is already one. `st.dataframe` takes
  plain lists of dicts, so the viewer doesn't import pandas.

### Declarative experiments

```yaml
# data/eval/experiments/chunk-header.yaml
description: Chunking plan Phase 2, deterministic header
corpus: [edgar]
eval_sets: [edgar_eval_set, edgar_period_set, edgar_underspecified_set]
stages: [retrieval]                # and/or answer, multihop
baseline: shipped
variants:
  shipped: {}
  header=on:     {config: data/eval/config_header.yaml}
  rerank_header: {config: data/eval/config_header.yaml,
                  overrides: {reranker.include_header: true}}
```

- **One runner for everything:** `python -m rag.eval.run <experiment.yaml>`.
  It takes `--only`, `--variant`, `--sets`, `--limit` and `--repeats k`,
  covers every stage, and skips any (variant, set, stage) that already has a
  complete run with the same config hash, dataset hash and code version.
  That replaces `--merge`. With `--repeats k` it runs k more copies instead,
  which is the repeat mode Milestone 27 asks for.
- **Index resolution.** For each variant, the runner computes the
  `index_hash` and checks it against the index manifest. A variant whose
  index settings match the shipped config uses the shipped index. Any other
  variant uses `data/index_variants/<index_hash>/`, so variants with the
  same index settings share one index without a hand-picked `index_dir`.
- **Missing indexes are never built by default.** A dry run
  (`--plan`, and also the default whenever an index is missing) lists what
  would run, which indexes are missing, and an estimated time taken from
  earlier runs' manifests. Building requires `--build-indexes`. The
  contextual index took about 1.9 hours and spends LLM calls, so it must
  never start as a side effect of a typo.
- `scripts/experiments/` stays as it is: those are one-off probes, not
  variant matrices.

## Phases

Each phase ships on its own and leaves the harness usable. Estimates are for
one person.

### Phase 1 — Full-config fingerprint (half a day)

- Add `rag/eval/provenance.py` with `config_hash`, `index_hash`,
  `dataset_hash`, and `code_version` moved from `checkpoint.py`. Add the
  denylist, and the test that every `RagConfig` leaf is classified.
- `run_matrix.py` and `run_answer_matrix.py` write `config_hash`,
  `dataset_hash`, `code_version` and a timestamp into every new row. They
  keep the old `fingerprint` beside it, so existing rows stay readable.
- **Done when** the classification test passes, and a new baseline row's old
  `fingerprint` still matches the committed one (the old hash is unchanged).

### Phase 2 — Run folders for retrieval (1–1.5 days)

- `retrieval_eval` and `run_matrix.py` write run folders with
  `outputs.jsonl` (including per-query latency, which Milestone 24 needs and
  `retrieval_eval` doesn't record yet). They keep writing the legacy
  `results/*.json` and `.md` files in the same pass, so nothing downstream
  breaks yet.
- `outputs.jsonl` replaces `.partial/` checkpoints for retrieval.
- **Done when** the shipped baseline on `edgar_eval_set` produces a run
  whose `scores.json` matches the current `baseline` row's metrics exactly,
  and an interrupted run resumes.

### Phase 3 — Offline scoring and reports (1 day)

- `rag.eval.score` and `rag.eval.report`, reusing `relevance.py`,
  `metrics.py`, `paired.py` and `render_table`.
- **Done when** a report built from Phase 2 runs is identical to the legacy
  `.md` table for the same variants. Also, rescoring a run against an eval
  set with one edited label changes exactly that sample's scores.

### Phase 4 — Answer side: generate and judge as separate steps (1.5 days)

- Answer and multi-hop runs write answers to `outputs.jsonl` and verdicts to
  `judgments/<judge_hash>.jsonl`. `rag.eval.judge` re-judges a finished run.
- `run_answer_matrix.py` writes run folders, as in Phase 2.
- **Done when** re-judging a finished run with the same judge reproduces its
  pass rate within the judge's own run-to-run variance (the judge's
  determinism is not measured yet, so record what it turns out to be), and a
  second judge's results sit beside the first's in the same run.

### Phase 4b — Run comparison viewer (1 day)

- `rag/eval/compare.py` (pure functions) and `rag/ui/eval_viewer.py`
  (Streamlit wiring), as described under
  [A viewer for comparing runs](#a-viewer-for-comparing-runs).
- It comes after Phase 4 so the answer panes have answers and judge
  replies to show. The retrieval half needs only Phase 3, and can ship first
  if Phase 4 slips.
- Tests cover `compare.py`: pairing, flip classification, config diff, and
  refusing mismatched dataset hashes. As with the chat UI, the Streamlit
  script is checked by running it.
- **Done when** comparing `baseline` against `header=on` on
  `edgar_period_set` lists the same wins and losses the report's W/L
  counts, every changed config path appears in the config diff, and the
  page loads with Ollama stopped.

### Phase 5 — Experiment files and one runner (2 days)

- Add `rag.eval.run` and move the two `VARIANTS` lists into
  `data/eval/experiments/*.yaml`, one file per axis group, keeping the
  current names so reports line up.
- Add index resolution with `--plan` and `--build-indexes`.
- **Done when** the chunk-header experiment above runs end to end from its
  YAML, reusing the existing header index, and the dry run correctly reports
  a missing index for an unbuilt variant.

### Phase 6 — Retire the old paths and update the docs (half a day)

- Delete the variant lists, `--merge` and the legacy writers. Keep
  `run_matrix.py` and `run_answer_matrix.py` as thin wrappers, or delete
  them, depending on whether anything external calls them.
- Update the `measure-change` skill (reading a result now includes the
  viewer's losses), CLAUDE.md's "Running things" section (runner, score,
  judge, report and viewer commands), the eval notes in
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
- **MLflow or Weights & Biases.** Either would give a UI and run tracking.
  Rejected: a large dependency plus a tracking server or hosted account, for
  a project that aims for minimal dependencies and runs locally. The part in
  use here is comparing runs you can reproduce, and a folder with a manifest
  does that. Revisit if runs outgrow a directory listing.
- **A database (SQLite) for runs.** It would make queries across runs easy,
  but outputs couldn't be diffed or committed and a new schema would need
  maintaining. Manifests are small JSON, and `report` can load all of them
  in well under a second at this scale.
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

- **What to commit.** `manifest.json` and `scores.json` are small and always
  committed. `outputs.jsonl` with chunk text is estimated at around 1 MB per
  retrieval run (174 samples × 5 chunks × ~1.1k characters), which comes to
  roughly 60 MB for a 20-variant matrix over 3 sets. Too much for git history
  at every run. Recommendation: keep `outputs.jsonl` gitignored by default,
  gzip it, and commit it (`git add -f`) only for runs cited in
  `measured-results.md` or used as a CI baseline. Measure the real size in
  Phase 2 before settling this. The alternative is a store of chunk text
  keyed by hash and shared across runs (most variants retrieve the same
  chunks). It is much smaller but needs its own bookkeeping, and is worth it
  only if the gzipped files turn out too large.
- **Pairing across a label fix.** The current plan refuses and asks for a
  rescore. Pairing on the intersection of unchanged samples is also valid,
  but it quietly shrinks n, so it should be available only through an
  explicit flag.
- **The CI gate's index.** Milestone 23 caches the index keyed on
  `FINGERPRINTED` chunking settings. It should key on `index_hash` instead.
  Decide that when Milestone 23 starts, not here.

## Order and why

Phase 1 comes first because it fixes a correctness problem (runs that
wrongly look comparable) and costs half a day. Phases 2–4 come next because
every later measurement depends on raw outputs being stored. Phase 4b is
where stored outputs pay off: reading why a variant lost, instead of
rerunning it with `-v`. Phase 5 is convenience, and it depends on index hashing from Phase 1 and run folders
from Phase 2. The chunking plan's Phases 4–5 would benefit from Phases 1–2
landing first: Phase 5's new `chunking.strategy` is exactly the field the
current fingerprint misses.
