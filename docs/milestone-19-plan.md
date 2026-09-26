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
   `generate(prompt, *, system)` has 11 call sites (contextualizer, expansion,
   CRAG, condenser, judge, eval-set generator), and none of them need tools.
   Add `ToolCallingLLM(LLMClient)` with
   `chat(messages, tools) -> AssistantTurn` (text + tool calls + token counts).
   `OllamaLLMClient` implements both. The agent's constructor takes
   `ToolCallingLLM`, so `chat.mode: agentic` with a provider that lacks tool
   support fails **at build time**, not mid-conversation.
   *Rejected:* widening the one-method ABC. That would force every test fake
   and future adapter to implement tool calling to answer a HyDE prompt.

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
   `stopped_reason: "answered" | "cap" | "timeout"`.

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

## Phases

### 0 — Eval harness that can see the difference (before any agent code)

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

- `ToolCallingLLM`, `AssistantTurn`, `ToolCall` in `rag/generation/llm.py`.
- `OllamaLLMClient.chat()`: `/api/chat` with `tools`; `tool_name` on tool
  messages; `think` passed through (bool or level string, since the 27b
  accepts `low|medium|xhigh`).
- `agent:` section in `settings.py` / `config.yaml`: `llm` (optional),
  `max_tool_calls`, `timeout_s`, `max_passage_chars`.
- **Exit:** unit tests with a scripted fake `ToolCallingLLM`; one live smoke
  test marked to skip when Ollama is absent.

### 2 — Shared tools

- Move `ToolSpec`/`RagTools` to `rag/tools.py`, then re-export from
  `rag/mcp/tools.py` so the MCP tests pass unchanged.
- A `ToolSpec → Ollama tool definition` converter. The schema is already
  derived via pydantic, so this is a wrapper, not a second schema.
- **Exit:** `tests/test_mcp.py` green; a new test asserting the agent and MCP
  advertise the same `rag_search` schema.

### 3 — Agent loop

- `rag/generation/agent.py`: `AgentService` with the ledger, the guards from
  decision 6, `PipelineEvent`s per tool call (so the UI's progress view keeps
  working), and history handling.
- The builder switches on `chat.mode`.
- **Exit:** tests for each guard (cap leads to a synthesis turn with tools
  removed; a duplicate query spends no search; a timeout returns the best
  answer so far; ledger numbering and dedup), plus the prototype's six
  multi-hop questions reproducing in the real code.

### 4 — Measure (the milestone's actual deliverable)

Matrix, one factor at a time, on all three sets:

| variant | purpose |
|---|---|
| `pipeline / 9b` | baseline |
| `agentic / 9b` | the loop's effect with the model we ship |
| `agentic / 27b` | the loop plus an iterating model |
| `agentic / 27b, think=low` | whether reasoning improves search planning enough to justify its tokens |
| `agentic / 27b + groundedness` | whether CRAG's checker catches the leakage cases |

**Default-flip criterion:** agentic beats pipeline on multi-hop
completeness by more than noise, while holding single-hop and refusal within
noise. Latency is reported next to it, not traded off silently. If the only
winning configuration is the 27b at minutes per hard question, the honest
outcome is **"agentic is an opt-in mode for hard questions"**, not a new
default.

### 5 — Follow-ups (only if phase 4 justifies the agent)

- **Structural tools:** `read_document(doc_id, section?)` and
  `read_span(doc_id, start, end)`. A single filing is roughly 15k tokens, well
  within the 27b's 262k context. This is the READ-paper direction, and it
  targets the table-split case the 27b needed 5 searches for.
- **Metadata filters on `rag_search`** (`company`, `period`, `form`), derived
  from the filename convention `TICKER_FORM_PERIOD.md`. This overlaps with
  Milestone 20's filter pushdown, so land it there and expose it here.
- **Streaming the agent's intermediate steps**, which is Milestone 21's SSE work.

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
