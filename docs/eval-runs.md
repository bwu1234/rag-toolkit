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

## Read an agent trace

Answer and multi-hop runs store each sample's whole turn: every model call
as it was sent and returned, and when each stage ran. Open a run's **Traces**
tab and click a sample. The span tree on the left follows the turn in time
order:

```text
answer
  agent_step
    chat 1            one model call: the conversation in, the reply out
      thinking        that call's reasoning
  rag_search          the tool call the model asked for
    embed, vector_search, sparse_search, fusion, rerank
  agent_step
    chat 2
  ...
  passages shown      every passage the model was shown, as a ranked list
  judge 1 · <item>    each judge call, one per graded item
```

Click a span to see its details:

- **`chat N`**: **Inputs / Outputs** shows the conversation as the model
  received it. That includes the system prompt, the question, the model's
  earlier tool calls, and each tool result as the text it read. The output is
  the reply: more tool calls, or the answer. **Attributes** lists the tools
  offered, token usage and the stop reason. It also holds
  `rendered_prompt (reconstructed)`: the whole prompt as ChatML text
  (`<|im_start|>system …`), explained in the last subsection below. In a
  pipeline run, the calls are named `generate N` and show a system prompt and
  a prompt instead of a conversation.
- **`thinking`**: the model's reasoning for that call. It has its own span
  because MLflow's chat view doesn't display reasoning.
- **`rag_search`** and other tool spans: the arguments the model sent
  (query, filters) and what came back (status, the passage numbers shown, and
  those passages). If the run recorded ranked hits, each retrieval stage
  nested inside lists them with their scores. A hit that was cut before the
  answer shows its text only while the run's index is unchanged. Otherwise a
  note says why there is no text.
- **`judge N · <item>`**: the judge's system prompt, the judge prompt, its
  full reply, and whether the item passed. The scores are in the trace's
  **Assessments** panel, with the judge's reply as the rationale.

Span times are offsets from the start of the turn. Each stage span has a
`timing` attribute:

- `recorded`: the run stored when the stage ended.
- `reconstructed`: an older run that didn't store that. The stage is laid out
  from the order of events and the model calls' times.
- `not recorded`: a zero-length marker.

Judge calls are graded after the turn without timings, so they appear as
markers at its end.

If a trace lacks spans described here, an older version of the export
produced it. Run `python -m rag.eval.tracking export --all --replace`, then
rebuild its comparisons as described above.

### The transcript

**Artifacts → transcripts/\<sample\>.md** holds the same turn as one document,
in the layout `scripts/trace_question.py` writes. It contains:

- the question, the answer and the expected answer;
- the timeline, the tool calls and retrieval;
- the passages shown and cited;
- each LLM call, with its messages, rendered prompt and response.

Each rendered prompt after the first notes how many characters it adds to
the previous one. The transcript is easier to read top to bottom than the
span tree, and you can download it.

### The rendered prompt is reconstructed

Eval runs call the model through Ollama's chat API, which applies the chat
template on the server. As a result, the prompt text the model actually read
isn't recorded. The export rebuilds it from the recorded messages with
`rag/llm/ollama_raw.py`, the same renderer that raw mode
(`trace_question.py --raw`) uses to build prompts. On `ad-airline-fuel`, the
rebuilt first prompt matched a raw-mode capture byte for byte. Limits:

- **qwen3.8 on Ollama only.** For other models, the span has no rendered
  prompt.
- **It assumes your Ollama renders prompts the way the port does.** A newer
  Ollama could render differently and nothing here would flag it. Raw mode's
  token-count check is what catches that.
- **It covers input only.** The model's unparsed output (reasoning and
  tool-call markup as generated) comes only from a raw-mode run
  (`llm.raw`). Its spans show `rendered_prompt`, without "(reconstructed)",
  and a `raw_output` attribute.

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
