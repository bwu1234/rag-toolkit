# Viewing evaluation runs

Evaluation commands save their results under `data/eval/runs/`. These run
folders are the canonical records: they contain the inputs, outputs and score
revisions used by reports. MLflow is an optional local browser for exploring
those records, traces and paired comparisons.

## Start MLflow

From the repository root, install the optional tracking dependency into the
environment you use for this project:

```bash
pip install -e '.[tracking]'
```

Export the saved runs, then start the UI:

```bash
python -m rag.eval.tracking export --all
python -m rag.eval.tracking serve
```

Keep the server command running in that terminal. Open
<http://127.0.0.1:5050> in your browser. The server is local to your machine;
press Ctrl-C in its terminal when you are finished. MLflow telemetry is
disabled by this command.

`export --all` imports complete runs and all their score revisions from
`data/eval/runs/` into `data/eval/mlflow/`. It does not run an evaluation or
call a model. Repeating the export is safe; already-exported revisions are
skipped. If you want to start over with the derived MLflow view, use
`python -m rag.eval.tracking export --all --replace`.

Replacing exports creates new MLflow run IDs and deletes the old runs and
their traces. Existing comparisons still link to those old IDs. Rebuild each
affected comparison with `compare --replace` to update its links, retaining
any `--declare` flags used for the original comparison:

```bash
python -m rag.eval.tracking compare \
  --baseline RUN_ID:REVISION \
  --candidate RUN_ID:REVISION \
  --replace
```

## Find a run and inspect its samples

In MLflow, open **Evaluation runs**. Choose the experiment named
`<stage>/<eval set>` (for example, `retrieval/edgar_eval_set`), then select a
run. Each score revision is represented by its own run; the run name and tags
include the source run and revision identifiers.

- **Traces** contains one trace per evaluated sample. Open a trace to see its
  query, ranked passages or answer, tool calls, scores and any recorded model
  calls.
- **Artifacts → transcripts/** contains a Markdown transcript for each sample
  when the run has answer-generation traces.
- **Metrics** and run parameters show aggregate scores and the recorded
  configuration. For paired statistical comparisons, use the harness report
  or the comparison command below; MLflow's aggregate run metrics alone are
  not the paired comparison.

The `data/eval/runs/` directory remains the source to consult if you need the
full saved record or want to verify which run/revision you exported. To find
IDs for a comparison, list the folders, read the `run_id` field in that
folder's `manifest.json`, and list the JSON files under its `scores/`
subdirectory. The run ID is the `run_id` value; each score revision is the
filename without `.json`. For example:

```bash
ls data/eval/runs/
cat data/eval/runs/<run-folder>/manifest.json
ls data/eval/runs/<run-folder>/scores/
```

The contents of `data/eval/mlflow/` are derived and can be rebuilt from those
run folders.

## Compare two saved revisions

Use the source run IDs and score revision labels from the run folders. The
comparison command checks that the pair is eligible under the same rules as
the CLI report, computes the paired comparison and exports it to MLflow:

```bash
python -m rag.eval.tracking compare \
  --baseline RUN_ID:REVISION \
  --candidate RUN_ID:REVISION
```

Open the resulting run in the `comparisons/<stage>/<eval set>` experiment and
follow the link in its description to MLflow's side-by-side evaluation view.
The comparison run also contains the paired report and per-sample flips.

For more on how runs, score revisions and comparisons are stored, see the
[eval harness documentation](eval-harness-plan.md).
