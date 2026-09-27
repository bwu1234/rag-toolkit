# Milestone 19 plan — agentic retrieval

Implementation plan for the [backlog entry](backlog.md#milestone-19--agentic-retrieval).
The backlog says *what* and *why*; this says *how*, in what order, and what has
to be true before each step counts as done.

**Goal, unchanged from the backlog:** `chat.mode: pipeline | agentic`, compared
on the same eval sets. Comparing them is the deliverable. The agent becomes
the default only if it beats the pipeline by more than noise.

## What the pre-work measured

Full numbers in [measured results](measured-results.md#generator-model-qwen359b-vs-qwen3827b-pre-milestone-19).
The parts that shape this plan:

- **A bigger generator doesn't help the pipeline.** Three of the 9b's four
  answerable failures are retrieval misses that no generator can recover in a
  single pass. The gain from `qwen3.8:27b` comes from *searching again*, so it
  needs the agent loop.
- **The 9b can drive tools but won't iterate.** It never made a malformed call
  and decomposes multi-company questions perfectly, but it stops after one
  round. **The 27b iterates**, and on multi-hop questions that was the
  difference between 1/5 and 5/5 complete, correct answers (small n).
- **The prototype's failures tell us what the loop must guard against:** cap
  exhaustion with no answer, identical-query loops, parametric leakage, and
  prompt growth of ~6× over the pipeline.
- **The eval harness can't compare models today**, because the judge is the
  generator. That gets fixed first.

## Decisions

1. **Keep `generate()`; add tool calling as a capability subclass.**
   `generate(prompt, *, system)` has 14 call sites (contextualizer, expansion,
   CRAG, condenser, generation, both judges, the eval-set and tier-draft
   scripts), and none of them need tools.
   Add `ToolCallingLLM(LLMClient)` with
   `chat(messages, tools) -> AssistantTurn` (text + tool calls + token counts).
   `OllamaLLMClient` implements both. The agent's constructor takes
   `ToolCallingLLM`, so `chat.mode: agentic` with a provider that lacks tool
   support fails **at build time**, not mid-conversation.
   *Rejected:* widening the one-method ABC. That would force every test fake
   and future adapter to implement tool calling to answer a HyDE prompt.

   **Metering needs a matching wrapper.** Since Milestone 12, `build_chat_service`
   wraps every client in `MeteredLLMClient`, which subclasses only `LLMClient`.
   Wrapped that way, the agent's model fails the `ToolCallingLLM` check at
   build time; left unwrapped, agent turns record no tokens. So add
   `MeteredToolCallingLLM(ToolCallingLLM)`, whose `chat()` records each call's
   `AssistantTurn` counts into the same active `UsageMeter`, and have the builder
   pick the wrapper that matches the inner client's type.

2. **Give the agent its own model: `agent.llm`.** It is optional and falls back
   to `llm`. The measured split is 27b for the agent loop, 9b for everything
   else (condenser, contextualizer, CRAG). Tying them together would make every
   utility call pay 27b latency for no measured gain.

3. **One tool surface, shared with MCP.** `ToolSpec`, schema derivation and
   `RagTools` move from `rag/mcp/tools.py` to `rag/tools.py`, and
   `rag/mcp` imports them. The internal agent and external MCP agents then
   call the same `rag_search`, described by the same schema. A search-quality
   fix lands once, and the MCP path gets exercised by our own evals for free.
   The module is already dependency-free with respect to the `mcp` SDK, so the
   move is mechanical.

4. **`AgentService.ask()` returns a `ChatAnswer`, like `ChatService.ask()`.**
   `build_chat_service` picks one by `chat.mode`. The API, UI, CLI and both eval
   runners stay untouched. `ChatAnswer.search_queries` already exists and
   carries the agent's queries. Add `tool_calls: int` and
   `stopped_reason: "answered" | "cap" | "timeout"`. Like `ChatService.ask()`,
   `AgentService.ask()` opens its own `metered()` block and fills `llm_calls`,
   `llm_ms`, `prompt_tokens` and `completion_tokens` from it. The runners then
   read both modes' cost from the same fields.

5. **A passage ledger owns citation numbering.** Numbers are assigned in first-
   seen order across all calls and deduplicated by `chunk_id`, so a chunk found
   by two searches keeps one number. The ledger is the single source of truth
   for `[n] → Citation`, the job `build_rag_prompt` has in the pipeline. This
   is the backlog's "citations must survive accumulation" requirement.

6. **Loop guards are part of the design, not a safety net** (each one comes
   from a failure the prototype showed):

   | guard | config | why |
   |---|---|---|
   | search cap | `agent.max_tool_calls: 8` | the 27b searched 8× for an absent entity |
   | forced synthesis turn | always on | at the cap it returned `""` or "Let me try…" twice |
   | duplicate-query refusal | always on | same Tesla query 5× in a row; answer "already searched" without spending a search |
   | wall-clock budget | `agent.timeout_s` | hard questions took minutes on a shared GPU |
   | per-passage char cap | reuse MCP's `DEFAULT_MAX_CHARS` | prompt tokens grew ~6× over the pipeline |

7. **Grounding is enforced by the prompt, checked by the eval, and not yet
   gated at runtime.** The system prompt forbids outside knowledge *including
   "for reference"* (the 27b's FY2015 leak). Reusing CRAG's
   `GroundednessChecker` on the final answer stays switchable and measured
   separately, per the backlog. The CRAG table says not to assume it helps.

8. **History goes to the model; the condenser is bypassed in agentic mode.**
   Condensing exists because retrieval was stateless. The agent sees the real
   message list and writes its own standalone queries.

9. **Two loop strategies, `agent.strategy: react | planned`, sharing one
   ledger, tool surface and set of guards.** `react` is the loop the prototype
   ran: the model decides after every result whether to search again.
   `planned` asks the model once for sub-queries, runs each through `rag_search`
   with no model call between searches, then makes one synthesis call. It
   targets the 9b's measured profile: it decomposes multi-company questions
   perfectly but won't take a second round on its own, and `planned` doesn't
   need it to.
   An external result supports trying it. A 2026 ablation on a local 7B model
   ([arXiv 2606.21553](https://arxiv.org/abs/2606.21553), 5,000 HotpotQA
   questions) used exactly this plan-and-execute shape. It found that two
   retrieval rounds captured 95% of the gain from five, and that decomposition
   and reranking each helped significantly. If that transfers, most of the
   multi-hop gain may be available at 9b latency, which bears directly on the
   latency open question below.

## Phases

### 0 — Eval harness that can see the difference (before any agent code)

**Done.** The baseline is in
[measured results](measured-results.md#pipeline-baseline-under-a-fixed-judge-milestone-19-phase-0):
15/34 multi-hop questions complete, evidence recall 0.583. The runner is
`rag.eval.multihop_eval`, also a third set in `run_answer_matrix.py`; there is
no separate `run_agent_matrix.py`.

The token gap is closed in the runner. Each `MultihopSampleResult` copies the
turn's `llm_calls`, `llm_ms`, `prompt_tokens` and `completion_tokens` from
`ChatAnswer` (`llm_ms` was added to `ChatAnswer` for this; the meter already
had it). `MultihopReport` reports their means, and the answer matrix adds
them as columns. Token means cover only samples whose provider reported
counts, `None` otherwise, and `num_with_tokens` shows how many that was. The
judge's calls stay out, because `ask()` meters only its own calls and nested
meters are isolated (`rag/observability/usage.py`). The multi-hop baseline was re-run
with them: 1.0 LLM call, 9.3 s of LLM time and 1,680 / 184 prompt / generated
tokens per turn ([measured results](measured-results.md#pipeline-baseline-under-a-fixed-judge-milestone-19-phase-0)).
The answer matrix now checkpoints each finished sample and resumes a stopped
run (`rag/eval/checkpoint.py`), which phase 4's hours-long agent runs need.

Search count and cap-hit rate come with the agent in phase 3.

- **Separate judge:** `eval.judge: LLMConfig | None` in config, plus a
  `--judge-model` flag on `answer_eval` and `run_answer_matrix.py`. Default to
  the generator so old results reproduce, but log a warning whenever
  judge == generator.
- **Multi-hop eval set, `data/eval/edgar_multihop_set.json`, 30–40
  questions.** Build comparison questions by **pairing existing span-matched
  samples** (same metric, different company or period). Gold spans and answers
  are then inherited from verified samples rather than invented. Add a handful
  of aggregation questions ("which of X, Y, Z…") whose gold is the per-company
  spans.
- **New metrics** in a `run_agent_matrix.py`, or as an extension of the answer
  matrix:
  - *evidence recall*: the fraction of each question's gold spans present in
    the **union** of everything retrieved;
  - *per-entity completeness*: judge each company or period's sub-answer
    separately, which fixes the rubric blind spot the prototype exposed;
  - searches per turn, cap-hit rate, empty-answer rate, generated and prompt
    tokens, wall-clock.
- **Exit:** the pipeline baseline measured on all three sets (single-hop,
  refusal, multi-hop) under a fixed judge, recorded in
  `measured-results.md`.

### 1 — Interface and adapter

**Done.** What shipped, and where it departs from the original bullets:

- `ToolCallingLLM(LLMClient)` with `chat(messages, tools) -> AssistantTurn`
  in `rag/generation/llm.py`, plus provider-neutral `ToolDefinition`,
  `ToolCall` (with the provider's call `id`), `ChatMessage` and `ToolResult`.
  The returned `AssistantTurn` is also the assistant message sent back on the
  next call, so tool calls and any `thinking` trace round-trip unchanged.
  Tools are a neutral `ToolDefinition`, not an Ollama payload: the adapter owns
  the wire format, which makes phase 2's converter `ToolSpec → ToolDefinition`.
- `OllamaLLMClient` implements it on the same `/api/chat` call as
  `generate()`. Tool results go back with `tool_name`. `llm.think` takes
  `true | false | low | medium | high | xhigh`. A turn with no tools omits the
  `tools` key, which is how the forced-synthesis turn will work.
- `MeteredToolCallingLLM` in `rag/observability/usage.py`, and
  `metered_client()`, which picks the wrapper that matches the inner client.
  `build_chat_service` uses it. `build_agent_llm(config)` in
  `rag/generation/builder.py` builds the agent's metered client from
  `agent.llm or llm`, and raises at build time when the provider has no tool
  support (Gemini today).
- `agent:` in `settings.py` / `config.yaml`: `llm`, `strategy`,
  `max_tool_calls: 8`, `timeout_s: 600`, `max_passage_chars: 1200`,
  `num_ctx: 32768`. It is inert until phase 3, so `config_fingerprint`
  excludes it for now and logged turns keep their fingerprint. **Phase 3 must
  put it back in the hash when `chat.mode: agentic`.** `timeout_s: 600` is a
  measurement default. The interactive-latency question below is still open.
- **`num_ctx` alone doesn't prevent silent truncation, and the 90% warning
  doesn't detect it.** Measured on Ollama 0.34.4 with a ~6.9k-token prompt:

  | model (engine) | `num_ctx` | prompt tokens reported | answer |
  |---|---|---|---|
  | `qwen3.5:9b` (llama.cpp) | 4096 | 2,050 (start of prompt dropped) | wrong |
  | `qwen3.5:9b` (llama.cpp) | 4096, `truncate: false` | HTTP 400 `exceed_context_size_error` | — |
  | `qwen3.5:9b-mlx` (MLX) | 4096 | 6,921 (not truncated) | right |

  The llama.cpp engine truncates to about half the window and reports only
  the tokens it kept, so a truncated prompt looks like a small one, and a
  "prompt ≥ 90% of `num_ctx`" check never fires. The adapter therefore sends
  `truncate: false` alongside `num_ctx` and raises `ContextOverflowError` on
  the 400. The 90% warning stays as an early notice before the next,
  longer prompt overflows. Requests without `num_ctx`, which means the whole
  pipeline, are byte-identical to before.
  The MLX engine, which serves both shipped models (`-mlx`), **ignores
  per-request `num_ctx`**. `/api/ps` reports 32768 whatever the request asks
  for, and a 42.9k-token prompt was processed whole. So on MLX `num_ctx` is
  a no-op, and prompt size is bounded only by `max_tool_calls` ×
  `max_passage_chars`.
- **Tests:** a scripted fake `ToolCallingLLM` (`tests/fakes.py`), used for the
  metering and build-time type-check tests; adapter tests against a mocked
  transport, using the tool-call shape Ollama actually returned; and two
  `live`-marked tests (`tests/test_ollama_live.py`) that skip when Ollama or
  the model is absent. One is a real search-then-answer round trip through
  `build_agent_llm` that asserts 2 metered calls and prompt tokens below
  `num_ctx`. The other shows a GGUF model raising `ContextOverflowError`
  instead of truncating.

### 2 — Shared tools

- Move `ToolSpec`/`RagTools` to `rag/tools.py`, then re-export from
  `rag/mcp/tools.py` so the MCP tests pass unchanged.
- A `ToolSpec → ToolDefinition` converter (the adapter already turns a
  `ToolDefinition` into Ollama's format). The schema is already derived via
  pydantic, so this is a wrapper, not a second schema.
- **Exit:** `tests/test_mcp.py` green; a new test asserting the agent and MCP
  advertise the same `rag_search` schema.

### 3 — Agent loop

- `rag/generation/agent.py`: `AgentService` with the ledger, the guards from
  decision 6, `PipelineEvent`s per tool call (so the UI's progress view keeps
  working), and history handling.
- Both strategies from decision 9. `planned` is the smaller of the two: a
  plan call that returns sub-queries as structured output, the searches, then
  the forced-synthesis turn `react` already needs at the cap. The two share
  the ledger and guards: the duplicate-query refusal dedups the plan,
  `max_tool_calls` truncates it, and `timeout_s` applies unchanged.
- The builder switches on `chat.mode`, then `agent.strategy`. The agent's
  model comes from `build_agent_llm`. Add `agent` back into
  `config_fingerprint` for agentic turns.
- Catch `ContextOverflowError` like the search cap: stop searching and take
  the forced-synthesis turn with the passages already in the ledger.
- **Exit:** tests for each guard (cap leads to a synthesis turn with tools
  removed; a duplicate query spends no search; a timeout returns the best
  answer so far; ledger numbering and dedup); for `planned`, a test that the
  model is called exactly twice however many sub-queries there are, and that
  a plan longer than the cap is truncated. Plus the prototype's six multi-hop
  questions reproducing in the real code under `react`.

### 4 — Measure (the milestone's actual deliverable)

Matrix, one factor at a time, on all three sets:

| variant | purpose |
|---|---|
| `pipeline / 9b` | baseline |
| `oracle / 9b` | the ceiling for generation: gold evidence, no retrieval (diagnostic only) |
| `agentic react / 9b` | the loop's effect with the model we ship |
| `agentic planned / 9b` | whether plan-and-execute recovers the multi-hop gain without an iterating model (decision 9) |
| `agentic react / 27b` | the loop plus an iterating model |
| `agentic react / 27b, think=low` | whether reasoning improves search planning enough to justify its tokens |
| `agentic react / 27b + groundedness` | whether CRAG's checker catches the leakage cases |

**The oracle row** feeds the generator the indexed chunks that contain each
question's gold spans, found by the same `unmatched_spans` matching that
evidence recall uses, in place of retrieval. It separates two failures that
evidence recall alone can't: evidence never found (a retrieval problem, which
the agent can fix) and evidence found but mis-combined (a synthesis problem,
which it can't). In a 2026 multi-hop study
([arXiv 2601.19827](https://arxiv.org/abs/2601.19827)), 87% of errors were
composition failures on evidence that had been retrieved. If `oracle / 9b` is
not much above `pipeline / 9b` on multi-hop completeness, the headroom is in
the generator, not in searching. It needs an `--oracle` flag on
`multihop_eval`. It runs on the multi-hop and single-hop sets only, because
refusal questions have no gold evidence. It is never a candidate default.

**Default-flip criterion:** agentic beats pipeline on multi-hop
completeness by more than noise, while holding single-hop and refusal within
noise. Latency is reported next to it, not traded off silently. If the only
winning configuration is the 27b at minutes per hard question, the honest
outcome is **"agentic is an opt-in mode for hard questions"**, not a new
default. If `planned / 9b` wins by more than noise, prefer it over the 27b
for any default; it is the only agentic variant that runs at 9b speed.

### 5 — Follow-ups (only if phase 4 justifies the agent)

- **Structural tools:** `read_document(doc_id, section?)` and
  `read_span(doc_id, start, end)`. A single filing is roughly 15k tokens, well
  within the 27b's 262k context. This is the READ-paper direction, and it
  targets the table-split case the 27b needed 5 searches for.
- **Metadata filters on `rag_search`** (`company`, `period`, `form`), derived
  from the filename convention `TICKER_FORM_PERIOD.md`. This overlaps with
  Milestone 20's filter pushdown, so land it there and expose it here.
- **Streaming the agent's intermediate steps**, which is Milestone 21's SSE work.
- **A `planned_refine` strategy: a planned turn with one forced gap check.**
  After the planned searches, make one structured call ("which sub-question is
  still unanswered? return extra queries, or none"), run those searches, then
  synthesize. It is not "planned, then `react`": the 9b stops after one round,
  so an open-ended loop after the plan would mostly behave like `planned`. A
  single forced decision is something the 9b can make, and two rounds is where
  the plan-and-execute ablation cited in decision 9 found 95% of the gain.
  It reuses the ledger and guards, and the refine searches count toward
  `max_tool_calls`. Add it only if `planned / 9b` falls short on multi-hop
  and the misses are evidence that the first round's results pointed to but
  that was never searched for. Build it after the two pure strategies are
  measured, so that a win can be credited to either the plan or the extra
  round.

## Open questions

- **Interactive latency budget.** With the 27b, hard questions took minutes on
  this M2 Max. Is agentic mode meant for the API/UI, or batch and MCP-style
  use? The answer decides whether `timeout_s` is 60 or 600.
- **Hosted model as a reference point.** Running the phase 4 matrix once with a
  frontier model would separate "the method doesn't help" from "the local
  model can't drive it". That needs an Anthropic adapter (already a recognised
  `provider` value) and **spends real API budget**, so it needs explicit
  sign-off with a cost estimate first.
- **Does the superlative question belong in the refusal set?** Under the
  pipeline it's unanswerable. An agent with 14 searches could actually answer
  it. It probably moves to the multi-hop set with a real gold answer, which
  means computing operating margin for all 14 companies from their filings.
