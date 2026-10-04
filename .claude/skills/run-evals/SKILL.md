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
  A BEIR qrels set (`scripts/beir_to_eval_set.py`) is scored `trec_eval`-style
  instead: `--config rag/config/beir.yaml --corpus beir-scifact --eval-set
  data/eval/beir_scifact_test.json`, with `--save-run DIR` to keep the
  rankings; `scripts/trec_eval_parity.py` checks them against `trec_eval`.
- Answer eval (LLM-as-judge): `python -m rag.eval.answer_eval`. The judge is
  `eval.judge`, else the generator grading itself (warned); `--judge-model`
  overrides, plus `--judge-provider` when the judge runs on a different
  provider than the one it inherits. Multi-hop: `python -m rag.eval.multihop_eval --corpus edgar_md`.
  Both take `--oracle` (gold chunks instead of retrieval, a diagnostic
  ceiling). The Milestone 19 phase 4 matrix is `scripts/run_answer_matrix.py
  --family m19` (`--repeat N` for run-to-run noise, `--sets ...,adaptive` for
  the bridge/discovery set); `--family m19-hosted` is
  the Gemini Flash-Lite reference pair, on the free tier and bounded by
  `llm.requests_per_day`. `--family musique --config rag/config/musique.yaml
  --corpus musique-ans-train-tune` runs those rows on MuSiQue-Ans, scored by
  official EM/F1 (`scripts/fetch_musique.py`, then
  `scripts/musique_to_eval_set.py dev|train-tune`; public benchmarks plan).
