# 0015 — Agentic search is the default; the pipeline stays for baselines and hosted models

- **Status:** Accepted (supersedes [0009](0009-pipeline-default-agentic-opt-in.md), 2026-10-03)
- **Recorded:** 2026-10-03, from
  [agentic results on `edgar_md`](../measured-results.md#agentic-retrieval-on-edgar_md-milestone-19-re-run-after-chunking-plan-phase-5)
  and the refusal re-check below

## Context

0009 kept the always-retrieve pipeline as the default and made the agent
opt-in. Its premise was a corpus Q&A tool, where most questions are one
lookup and a 1–2 minute turn buys little. The target use is research over an
organization's documents. There, a question's evidence is spread across
filings, companies and periods, and the question often names a fact or a
class rather than the document. That is the bridge and discovery shape.

The eval mix under-represents that use: 174 generated single-hop questions,
35 multi-hop questions that name every entity, and only 15 adaptive questions
that don't. But 0009 was not decided on a blended score. Every set was scored
on its own, and that per-set evidence already answers the quality question:

| `edgar_md`, mean per run | answerable /40 | refusals /15 | multi-hop /35 | adaptive /15 |
|---|---|---|---|---|
| `pipeline / 9b` (0009's default) | 37.3 | 15 | 24.7 | 6.7 |
| `agentic react / 9b` | 37.3 | 15 | 27.3 | 6.0 |
| `agentic react / 27b, think=low` | 38.5 | 14.5 | 32.5 | 13.0 |

Against `pipeline / 27b` (the loop alone), the think=low agent gains 6.2
multi-hop (7/0, p 0.016) and 6.0 adaptive (6/0, p 0.031), with answerable and
refusals within noise. It reaches the oracle on both hard sets. What 0009
weighed against that was latency, and that is a product judgment, not a
measurement. This record makes the opposite call for the research use.

## Decision

`chat.mode: agentic` is the shipped default, and `agent.llm` is the 27b at
`think: low`. The shipped config equals the measured
`agentic react / 27b, think=low` row exactly (a test holds them equal): react
strategy, `rag_search` only, 4096 tokens, 600 s budget, CRAG off. `llm` stays
the 9b for the utility calls.

Pinned to `pipeline`:

- **`vanilla.yaml`.** It is the plain-RAG floor, and an agent loop isn't plain.
- **The Flash-Lite configs.** Under the new default they would answer with
  the local 27b agent, not Flash-Lite. The hosted agent row hasn't been run,
  and each agent turn spends 2–7 requests of the free-tier day instead of one.

`run_answer_matrix.py` rows now start from `ROW_BASELINE` (`chat.mode:
pipeline`, `agent.llm: null`, i.e. what `config.yaml` was when every
recorded row ran). Without it, the shipped `think: low` merged into every
agent row's `agent.llm`. `agentic react / 27b` silently became a copy of the
think=low row, the 9b rows became a think=low 9b, and the CRAG rows, which
never pinned `chat.mode`, became agentic.

## Alternatives considered

- **Keep 0009.** It is the right call for a lookup tool and stays available as
  `chat.mode: pipeline`. It completes about half the bridge and discovery
  questions the agent does, and that is the target workload.
- **The 9b as the default agent.** It runs at pipeline speed but doesn't
  search a second time. On `edgar_md` it trails the pipeline on adaptive
  questions (evidence recall 0.58 vs 0.69). Switching `chat.mode` alone
  would have shipped this.
- **Pipeline first, escalating to the agent when retrieval looks incomplete.**
  This would keep 8–10 s single-hop turns. It is unbuilt and unmeasured, and
  it needs a signal that the pipeline's answer is incomplete. The one
  self-check measured so far, groundedness, caught 1 of the 8 failures the
  judge found on `edgar_md`. It belongs in the backlog item below, not in this flip.
- **Wait for a research-workload eval before switching.** This is the right
  mechanism. It is in the [backlog](../backlog.md#milestone-27--eval-coverage-and-judge-reliability).
  This record is a product decision made ahead of it, on the evidence above,
  and that eval is what can reverse it.

## Consequences

- **Latency.** Measured on `edgar_md`: about 34 s for a single-hop question
  (pipeline 8 s), 64 s for multi-hop, 91 s for bridge/discovery, and 104 s
  for a refusal. The agent keeps searching before it declines. Prompt tokens
  rise 4–13×.
- **Hardware.** Serving loads the 27b (18 GB) next to the 9b, the embedder
  and the reranker.
- **Eval runners follow the config.** `answer_eval` and `multihop_eval`
  without `--config` now measure the agent: a 40-question answerable run
  takes about 23 minutes of answering instead of 5. Never pair such a run against a
  recorded pipeline row. Use `run_answer_matrix.py` rows, which pin their
  mode.
- **Turn log hashes re-key.** The config hash leaves out `chat.mode` and
  `agent` only under the pipeline (0013), so turns logged from now on group
  apart from earlier ones. That is correct, because the turns differ.
- **What the agent doesn't have.** CRAG's grader and retries don't apply.
  Groundedness only reports its verdict. Metadata filters are pinned by the
  turn (`agent.model_filters` off, unmeasured). `agent.timeout_s` is still
  the 600 s batch-measurement budget. Whether the UI should get ~60 s is open
  and unmeasured, and a lower cap cuts off the searches that win.
- **Scope of the evidence.** One corpus, questions written from its own
  chunks, an LLM judge. The MuSiQue outside check hasn't run. This record
  should be reopened if MuSiQue or the research-workload tier shows the loop's
  gain doesn't transfer.

## Refusal re-check

0009's numbers left one open defect. Default-thinking 27b returned an empty
answer to `neg-entity-lilly-pipeline` in all four of its runs. A guard
(`stopped_reason: empty`) was added afterwards and never measured. think=low
didn't show the failure, but it is the row now shipped, so its refusal set
was re-run on the guarded code.

*Result: refusals held.* Run 2026-10-03 at `c595818` from a pinned
worktree: `agentic react / 27b, think=low` ×2 on the 15-question refusal set,
`edgar_md`, judge `gemma4:31b-mlx`. Raw records:
`data/eval/results_m19/adr0015_refusal_recheck/`.

| run | judged pass | empty answers | s/turn | LLM calls | prompt tokens |
|---|---|---|---|---|---|
| #1 | 15/15 | 0 | 93 | 4.8 | 15,788 |
| #2 | 15/15 | 0 | 104 | 4.7 | 14,833 |

- *The empty answer didn't recur.* `neg-entity-lilly-pipeline` declined
  correctly both times (6 and 5 calls, 119 s and 83 s).
- *Read by hand, one pass is lenient.* `neg-unanswerable-comparison` #2
  states the limitation, then undercuts it ("almost certainly the highest in
  the corpus") and uses NVIDIA's October 2025 quarter where #1 found April
  2026. I'd fail it, so it's 29/30 by hand. #1 passes as the rubric allows.
  This is within think=low's recorded 15 and 14 on `edgar_md`.
- *Cost matches the recorded row.* 93–104 s and ~15k prompt tokens per
  refusal, against 104 s and 18.5k recorded. A refusal remains the slowest
  turn: on that question, the agent spent 265–279 s searching for a ranking
  before qualifying it.
