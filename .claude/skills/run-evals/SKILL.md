---
name: run-evals
description: Exact commands and flags for running this repo's evals and inspecting turns — retrieval_eval (EDGAR and BEIR/trec_eval), answer_eval and multihop_eval (judge flags, --oracle), run_answer_matrix.py families (m19, m19-hosted, musique), trace_question.py (including --raw), cli chat --trace, and the turn log (cli turns). Use when running, re-running or debugging an eval or tracing a single question.
---

# Running evals and inspecting turns

To decide whether a change helps (which runs to compare, how to read and
record them), use the `measure-change` skill; this one is the command reference.

- Logged turns and feedback: `python -m rag.cli turns` (`--feedback down` for
  the thumbs-down ones). With the `jsonl` provider, the API, UI and `cli chat`
  append every turn to `observability.turn_log.path` (default
  `data/logs/turns.jsonl`); `provider: none` logs nothing. The eval runners
  never do.
- One turn end to end (system prompt, every LLM call and tool result, the
  answer): `python scripts/trace_question.py --id ad-airline-fuel` replays an
  eval question into `data/traces/<id>.md`; `--variant "agentic react / 27b,
  think=low"` runs it as that `run_answer_matrix.py` row. For a free-form
  question, `cli chat --trace out.md "..."`. Transcripts never go to the turn log.
  `--raw` adds each call's exact rendered prompt (chat template, special
  tokens) as a delta from the previous call, plus the model's unparsed output:
  it sets `llm.raw` on `agent.llm`, which renders client-side through
  `rag/llm/ollama_raw.py` (a port of Ollama 0.35.1's qwen3.8 renderer, so
  qwen3.8 models only) and calls `/api/generate` with `raw: true`. Each step
  is checked against `/api/chat`'s prompt token count; a mismatch after an
  Ollama upgrade means the port needs updating.
- Retrieval eval: `python -m rag.eval.retrieval_eval` (`-v` for per-sample;
  `--eval-set data/eval/edgar_eval_set.json --corpus edgar_md` for EDGAR).
  It and `scripts/run_matrix.py` record each run (one per matrix variant) as
  a folder under `data/eval/runs/` (manifest, dataset snapshot, ranked chunks
  and latency in `outputs.jsonl`, `scores/<revision>.json`); the matrix row
  names its `run` and `score_revision`. Rerunning the same command resumes an
  interrupted run on the same verified index build; `--fresh` starts over,
  `--limit N` evaluates N evenly spaced samples, `--no-run` writes nothing.
  Rescore a finished run offline after a label fix with `python -m
  rag.eval.score <run> --eval-set PATH` (a new score revision; refused if a
  query changed). Rebuild the matrix table from pinned runs with `python -m
  rag.eval.report --baseline <run>:<rev> --runs <run>:<rev> ... [--out F.md]`;
  it refuses runs on different samples, labels or scorer, and lists config
  differences the variant didn't declare (`--declare PATH|code|index|corpus`).
  To browse runs side by side, export them to the optional MLflow store
  (`pip install -e ".[tracking]"`): `python -m rag.eval.tracking export --all`
  (re-export is a no-op; `--replace` rebuilds), then `python -m
  rag.eval.tracking serve [--port 5050]`. `python -m rag.eval.tracking
  compare --baseline <run>:<rev> --candidate <run>:<rev> [--declare PATH]`
  prints and exports the paired comparison with a link to MLflow's
  side-by-side view. After `--replace` on a revision, re-export its
  comparisons with `--replace` too. A run's detail is under Evaluation runs
  -> the run -> Traces (each LLM call verbatim) and Artifacts ->
  `transcripts/` (each sample as Markdown); revisions exported before the
  transcripts existed (`rag.exporter` below `rag.eval.tracking/2`) need
  `export --replace` to get them.
  A BEIR qrels set (`scripts/beir_to_eval_set.py`) is scored `trec_eval`-style
  instead: `--config rag/config/beir.yaml --corpus beir-scifact --eval-set
  data/eval/beir_scifact_test.json`, with `--save-run DIR` to keep the
  rankings; `scripts/trec_eval_parity.py` checks them against `trec_eval`.
- Answer eval (LLM-as-judge): `python -m rag.eval.answer_eval`. The judge is
  `eval.judge`, else the generator grading itself (warned); `--judge-model`
  overrides, plus `--judge-provider` when the judge runs on a different
  provider than the one it inherits. Multi-hop: `python -m rag.eval.multihop_eval --corpus edgar_md`.
  Both take `--oracle` (gold chunks instead of retrieval, a diagnostic
  ceiling), and both write an answer run folder (answers, shown passages and
  every model call verbatim; `--no-run` to skip, `--fresh` to not resume),
  judged as a separate attempt under `judgments/`. Re-judge a finished run
  without regenerating: `python -m rag.eval.judge <run> --judge-model M
  [--eval-set PATH]` (a new attempt each time; another judge or changed
  reference answers get their own protocol directory). Rescoring an answer
  run needs no index or judge: `python -m rag.eval.score <run> [--eval-set
  PATH] [--judgment <attempt>]`; a changed reference refuses the old
  judgment. Judge calibration against reviewed human labels: `python -m
  rag.eval.calibrate draft --runs RUN ...`, then (after labeling per
  docs/judge-calibration-guide.md) `python -m rag.eval.calibrate run
  --judge-model M --attempts 3`. The Milestone 19 phase 4 matrix is `scripts/run_answer_matrix.py
  --family m19` (`--repeat N` for run-to-run noise, `--sets ...,adaptive` for
  the bridge/discovery set); `--family m19-hosted` is
  the Gemini Flash-Lite reference pair, on the free tier and bounded by
  `llm.requests_per_day`. `--family musique --config rag/config/musique.yaml
  --corpus musique-ans-train-tune` runs those rows on MuSiQue-Ans, scored by
  official EM/F1 (`scripts/fetch_musique.py`, then
  `scripts/musique_to_eval_set.py dev|train-tune`; public benchmarks plan).
