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
  support. (That was Gemini until the Gemini adapter gained `chat()`; see
  the note after this list.)
- `agent:` in `settings.py` / `config.yaml`: `llm`, `strategy`,
  `max_tool_calls: 8`, `timeout_s: 600`, `max_passage_chars: 1200`,
  `num_ctx: 32768`. It was inert until phase 3, so `config_fingerprint`
  excluded it; phase 3 put it back in the hash when `chat.mode: agentic` (see
  "Fingerprint" under phase 3). `timeout_s: 600` is a
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

**Gemini can drive the agent too** (added after phase 1). `GeminiLLMClient`
implements `chat()`, so `agent.llm` can name `gemini-3.5-flash-lite` or hosted
`gemma-4-31b-it`. Six probe requests settled what the docs left open:

- Both models return a signed `functionCall`: an `id` plus a
  `thoughtSignature`, even at `thinking_level: minimal`. It travels back on
  `ToolCall.signature`. With it stripped, the next request is a 400 ("Function
  call is missing a thought_signature"), so the round trip is load-bearing.
- A follow-up request with no tools declared is accepted after function
  calls. The forced-synthesis turn works as it does on Ollama, without
  `toolConfig: NONE`.
- `parametersJsonSchema` accepts the pydantic-derived schema (`anyOf`,
  `default`, `title`) as-is. The OpenAPI-subset `parameters` field would need
  a rewrite.
- Gemma adds a `thought: true` text part before its call. The adapter keeps it
  out of `content`.
- A step's results go back as one user turn of `functionResponse` parts,
  after all of its calls. Interleaving them is a 400.

`tests/test_gemini_live.py` repeats the round trip. It spends 3 requests, so
it runs only with `RAG_GEMINI_LIVE=1`.

### 2 — Shared tools

**Done.** What shipped, and where it departs from the original bullets:

- `ToolSpec`, `input_schema_for`, `RagTools`, `build_tool_specs`,
  `MAX_RESULTS` and `DEFAULT_MAX_CHARS` live in `rag/tools.py`.
  `rag/mcp/tools.py` keeps only what an MCP client alone sees
  (`SERVER_NAME`, `SERVER_VERSION`, `PROTOCOL_VERSION`, `INSTRUCTIONS`)
  and re-exports the rest, so every `rag.mcp` import is unchanged.
- **The MCP tests did not pass literally unchanged.** Two fixtures patched
  `rag.mcp.tools.build_retriever`, which now has to be `rag.tools.build_retriever`.
  Re-exporting `build_retriever` would have made the old patch target resolve,
  but the patch would then land on a name nothing calls, and the tests would
  quietly build a real retriever. Both fixtures were retargeted instead.
- The converter is `ToolSpec.definition -> ToolDefinition`, whose `parameters`
  are the same `input_schema` MCP's `tools/list` serves.
- **Added for phase 3:** `RagTools(config=...)`, as an alternative to a
  config path. The agent's builder already holds a `RagConfig`, including
  `--config` overlays and the eval matrices' in-memory overrides. Loading it
  a second time from disk would quietly search with a different config than
  the rest of the turn uses.
- **Tests (`tests/test_tools.py`):** the agent's `rag_search` definition
  equals the fallback's `tools/list` entry (name, description, schema) and
  the SDK transport's schema, and both adapters (Ollama, Gemini) send that
  schema to the model unmodified.

Left for phase 3, because it depends on the loop's design: which tools the
agent is offered (only `rag_search`, or `rag_list_corpora` too), and whether
the model may pick a `corpus` or the turn's corpus selection is pinned. Eval
runs select corpora with `--corpus`, and a model free to widen that
selection would be measuring a different index.

### 3 — Agent loop

**Done.** `chat.mode: agentic` builds an `AgentService`
(`rag/generation/agent.py`), with both strategies, the ledger and every guard
from decision 6. What shipped, and where it departs from the original bullets:

- **`ChatResponder`, a base class both responders share.** Decision 4 wanted
  the callers untouched. Their calls are, but their type annotations named
  `ChatService`. `ask()` (metering, timing, the turn record, recording a turn
  that raised) moved to a `ChatResponder` base class. `ChatService` and
  `AgentService` each implement only `_answer()`, and `build_chat_service`
  returns a `ChatResponder`. The callers changed one annotation each.
- **The model doesn't choose `corpus`, `top_k` or `max_chars`** (nor
  `filters`, added to `rag_search` afterwards by the chunking plan's Phase 3:
  it is pinned to the turn's own `/chat` filter; see phase 5). The agent
  offers `rag_search` as `ToolSpec.definition_without(...)`: the MCP schema
  with those three properties removed, the rest byte-identical. `corpus` is
  the turn's selection, because an eval's `--corpus` must be the index
  measured. `top_k` and `max_chars` are the prompt-size guards; a model
  asking for 20 full passages per search would undo them. It left the MCP
  tool description with a sentence pointing at `rag_list_corpora`, a tool
  the agent doesn't have. That pointer now lives only in the `corpus`
  argument's description, a small change to the text MCP clients see.
- **The `planned` strategy's plan is the tool calls of one turn,** not a JSON
  answer to parse. The 9b never made a malformed tool call in the probe, and
  tool calls are already structured output. Its planning prompt differs from
  `react`'s: it gets one round, so it must ask for every search at once.
- **The forced-synthesis instruction is its own user message, placed by
  measurement.** The first build appended it to the last tool result, in case
  Gemini rejected two user turns in a row. In the live run, two of three
  capped 27b turns still answered "Let me try one more search" or "". Both
  placements were replayed on the captured superlative conversation, 3
  samples each:

  | placement at the forced turn | lease | superlative |
  |---|---|---|
  | appended to the last tool result | 3/3 answered | **0/3: empty every time** |
  | separate user message (shipped) | 3/3 | 3/3 |
  | fresh single-shot grounded prompt | 3/3 | 3/3, but asserts a winner |

  The empty reply had no tool calls and no thinking, so it wasn't a tool call
  the loop dropped. The fresh prompt also worked, but it drops the
  conversation, and it named a single winner that the passages it had can't
  establish. Gemini documents the opposite placement (Gemini 3.x: "inline
  instructions should be appended directly to the response text"), so its
  adapter folds the message into the last function response. The agent stays
  provider-neutral. The Gemini fold follows the docs and is not live-verified.
- **`stopped_reason` gained `context`:** `answered | cap | timeout | context`.
  On `ContextOverflowError`, the latest step's results are replaced with a
  note, their passages leave the ledger (so they can't be cited), and the
  forced turn answers. If there's no step to roll back, the error propagates.
  `ContextOverflowError` moved to `rag/generation/llm.py` so the agent needn't
  import a concrete adapter.
- **`timeout_s` bounds the searching, not the whole turn.** It is checked
  before each model call and each search. An in-flight call isn't
  interrupted, and the forced answer is one more call after the budget.
- **`cap` means a guard cut something off.** `planned` with exactly
  `max_tool_calls` queries reports `answered`. `react` reports `cap` once it
  is out of searches or model steps: steps are bounded by `max_tool_calls`
  too, so a model that only repeats itself still runs out.
- **Citations carry the text the model saw,** capped at `max_passage_chars`,
  not the whole chunk, so the answer eval's judge and evidence recall see
  what the answer could have used. A chunk returned again is referenced as
  "already shown" rather than repeated.
- **History:** an earlier answer's `[n]` markers are stripped before it goes
  back to the model. They numbered that turn's passages, and a copied `[3]`
  would cite something this turn never showed.
- **Groundedness is check-only:** `crag.check_groundedness` reports a verdict
  on `ChatAnswer.grounded` without regenerating, so phase 4 can measure the
  checker by itself. CRAG's grader and retries are ignored in agentic mode,
  with a warning at build time.
- **Fingerprint:** `agent` counts only when `chat.mode: agentic`, and
  `chat.mode` itself is excluded while it's `pipeline`. Default and
  `vanilla.yaml` fingerprints match `main`'s exactly, so logged turns keep
  their keys.
- **Tests (`tests/test_agent.py`, 24):** the planned exit criteria (the cap
  leads to a synthesis turn with tools removed; a repeated query spends no
  search; a timeout answers with what was found; ledger numbering and dedup;
  `planned` calls the model exactly twice for 1, 3 or 6 sub-queries, and a
  plan over the cap is truncated), plus overflow rollback, history
  stripping, metering, and building each mode.
  Each guard was checked by breaking it in the source: all nine mutations
  failed a test.

**The prototype's six multi-hop questions, in the real code** (`react`,
`qwen3.8:27b-mlx`, EDGAR, M2 Max; searches / prompt tokens / seconds). Not
judged; grading is phase 4's job. Run twice: first with the instruction
appended to the tool result, then with the shipped separate message.

| question | prototype | first run | shipped |
|---|---|---|---|
| airlines revenue | 8 / 78k / 383 | 4, answered / 20k / 124 | 5, answered / 27k / 119 |
| lease obligations | 5 / 19k / 132 | 8, cap / 18k / 138, **"Let me try one more search."** | 8, cap / 18k / 131, "not in the passages" |
| retail comps | 4 / 12k / 132 | 3, answered / 11k / 69 | 4, answered / 12k / 98 |
| pharma IRA (3 cos) | 8 / 33k / 289 | 8, cap / 22k / 189 | 8, cap / 21k / 182 |
| Southwest fuel | 1 / 3k / 68 | 1, answered / 3k / 30 | 1, answered / 3k / 30 |
| superlative | 8 / 46k / 382 | 8, cap / 34k / 253, **""** | 8, cap / 38k / 265 |

- **Reproduces:** one search per entity, iteration when a passage points
  somewhere new (airlines, retail), and the cap on the hard ones. Prompt
  tokens are down, up to 3× on airlines, because the ledger references a
  repeated passage rather than resending it.
- **Every turn now ends in a real answer.** The first run returned the
  prototype's two cap failures verbatim; that led to the placement change
  above.
- **Doesn't reproduce: lease.** In both runs the 27b never found Apple's or
  Microsoft's lease table within 8 searches, and it said so rather than
  guessing. The prototype found them in 5. It also called COST "Costa Mesa"
  (it's Costco), a slip no passage supports.
- **The superlative answer names a winner (NVIDIA) after seeing margins for
  only some of the 14 companies.** That's a problem with the question, not
  a loop bug; see [the superlative question](#the-superlative-question-stays-a-refusal).

### 4 — Measure (the milestone's actual deliverable)

**Re-baseline the pipeline first.** The shipped pipeline changed after phase
0: the [chunking plan](chunking-indexing-plan.md)'s Phase 2 turned on the
chunk header and `reranker.include_header` (+7.5pp answer pass on the
generated set), and the 2026-09-26 label fixes touched one sample of the
40-sample `crag=off` row. Phase 0's multi-hop baseline (15/34) and that row
both predate it, so `pipeline / 9b` must be re-run at the current config
before any agentic row is paired against it. The agent searches the same
header-bearing index, so pairing it against the old baseline would credit it
with the header's gain.

Matrix, one factor at a time, on all three sets:

| variant | purpose |
|---|---|
| `pipeline / 9b` | baseline |
| `oracle / 9b` | the ceiling for generation: gold evidence, no retrieval (diagnostic only) |
| `agentic react / 9b` | the loop's effect with the model we ship |
| `agentic planned / 9b` | whether plan-and-execute recovers the multi-hop gain without an iterating model (decision 9) |
| `pipeline / 27b` | control for the row below: same model and settings, answering once, so the pair isolates the loop (added after stage 2's first runs) |
| `agentic react / 27b` | the loop plus an iterating model |
| `agentic react / 27b, think=low` | whether reasoning improves search planning enough to justify its tokens |
| `agentic react / 27b + groundedness` | whether CRAG's checker catches the leakage cases |

*Runner, shipped:* `scripts/run_answer_matrix.py --family m19` defines these
rows as written and writes to `data/eval/results_m19/`, apart from the CRAG
rows on the old labels. Every row pins its generator. The agent rows pin the
agent model's `max_tokens` (4096) and timeout (600 s), so 9b vs 27b is the
model alone. `--repeat N` runs each row N times as separate rows and reports
the spread. The groundedness row saves each verdict next to the judge's
outcome, because the agent's checker changes no answers. What that row
measures is whether the flags fall on the failures. Answerable and refusal
results now carry per-turn LLM calls and tokens, as multi-hop results already
did. Run order: `pipeline / 9b` first, since the table pairs every row against
the first one.

*The adaptive set, added after stage 1.* Stage 1 (the four 9b rows, three
repeats) found the multi-hop set can't separate the agent from the pipeline:
34 of its 35 questions name every company and period they ask about, so one
search with the whole question already reaches most of the evidence, and the
oracle completes about 33.7 of 35. `data/eval/edgar_adaptive_set.json`
(`scripts/build_adaptive_set.py`, 15 questions) asks what only a second,
different search can answer. The 10 **bridge** questions name an identifying
fact rather than the company ("the company that sells Enflonsia"), or ask a
follow-up about a comparison's winner. The 5 **discovery** questions name a
class ("the airlines in this corpus") or the latest period. It is a separate
file, so stage 1's multi-hop numbers stay valid. `--sets ...,adaptive` runs it,
and it reports completion per kind. It changes only the scripts and data, not
the answering code under `rag/`.

**The oracle row** feeds the generator the indexed chunks that contain each
question's gold spans, found by the same `unmatched_spans` matching that
evidence recall uses, in place of retrieval. It separates two failures that
evidence recall alone can't: evidence never found (a retrieval problem, which
the agent can fix) and evidence found but mis-combined (a synthesis problem,
which it can't). In a 2026 multi-hop study
([arXiv 2601.19827](https://arxiv.org/abs/2601.19827)), 87% of errors were
composition failures on evidence that had been retrieved. If `oracle / 9b` is
not much above `pipeline / 9b` on multi-hop completeness, the headroom is in
the generator, not in searching. It runs on the multi-hop and single-hop sets
only, because refusal questions have no gold evidence. It is never a candidate
default.

*Shipped:* `--oracle` on `multihop_eval` and `answer_eval`
(`rag/eval/oracle.py`). It swaps the retriever for one that returns the gold
chunks, so the prompt, generation and citations are the pipeline's own. It
refuses CRAG and the agent. On the current index every gold span is found:
1 chunk per single-hop question, 2–4 per multi-hop question. That is fewer
passages than the pipeline's 5, and none of them distractors, so the ceiling
is "gold evidence alone". If the oracle and the pipeline rows are close, a
padded variant (gold plus retrieved fill to `rerank_top_k`) would show whether
distractors are what costs the pipeline.

**Default-flip criterion:** agentic beats pipeline on multi-hop
completeness by more than noise, while holding single-hop and refusal within
noise. Latency is reported next to it, not traded off silently. If the only
winning configuration is the 27b at minutes per hard question, the honest
outcome is **"agentic is an opt-in mode for hard questions"**, not a new
default. If `planned / 9b` wins by more than noise, prefer it over the 27b
for any default; it is the only agentic variant that runs at 9b speed.

*Result (2026-10-01): opt-in, not the default.* Full numbers in
[measured results](measured-results.md#agentic-retrieval-milestone-19-phase-4).
`agentic react / 27b` against `pipeline / 27b` (the loop alone) gained 6.7 of
35 multi-hop and 5.7 of 15 adaptive questions, 7 wins and 0 losses on each,
and reached within a question of the oracle. Answerable and refusals held.
The 27b answering once gained 1.3 and 1.7, not shown at 95%. Both 9b
strategies stayed within noise of the pipeline on multi-hop and fell behind
it on the adaptive set, so `planned / 9b` didn't win. The win comes only from
the 27b, at 74–108 s per hard question against 10 s, which is the case this
criterion names: `chat.mode` stays `pipeline`, and agentic ships as an opt-in
mode for hard questions with the 27b at `think: low` (same quality, fewer
tokens). The groundedness checker flagged 9 of 104 answers, 6 of them ones
the judge passed, so it stays off.

*Re-run on `edgar_md` (2026-10-03): the same decision.* After chunking plan
Phase 5 moved EDGAR evals to `edgar_md`, every row was re-run there
([numbers](measured-results.md#agentic-retrieval-on-edgar_md-milestone-19-re-run-after-chunking-plan-phase-5)).

- **The 27b loop still gains.** +5.3 of 35 multi-hop (interval clear of
  zero, sign test p 0.12) and +6.0 of 15 adaptive (p 0.016), reaching the
  oracle on both sets.
- **`think: low` clears both sets at p ≤ 0.031**, with fewer tokens.
- **Default thinking returned an empty answer** to one refusal question in
  every run.
- **Hard questions take 60–84 s** against the pipeline's 8–10 s.

**Outside evidence (proposed).** Every set above was written from this
repo's own chunks. The public benchmarks plan drafts a
[MuSiQue follow-on](public-benchmarks-plan.md#follow-on-multi-hop-and-agentic-retrieval-on-musique)
that runs these rows on a pooled MuSiQue-Ans corpus with gold supporting
paragraphs and EM/F1 answers. It supports this criterion but does not replace
it: MuSiQue-Ans has no single-hop or refusal questions.

**Consider Inspect for the agentic rows.** The
[eval harness plan](eval-harness-plan.md#decisions-and-rejected-alternatives)
keeps pipeline evals on this repo's own run store, but defers this decision
to here. An agent failure is read in its transcript (which searches it ran,
what came back, why it stopped), and
[Inspect](https://inspect.aisi.org.uk/)'s `inspect view` shows every model
and tool call per sample. The run store only shows final outputs. Check
three things before adopting it:

- **How the agent's calls reach Inspect.** Inspect's agent bridge intercepts
  only the OpenAI, Anthropic and Google SDK APIs. `OllamaLLM` calls Ollama's
  native `/api/chat` over httpx, so the bridge would not see it. The fit here
  is an `inspect` `LLMClient` provider that delegates to Inspect's model API
  (use the `add-provider` skill). Then `rag/generation/agent.py` runs
  unchanged and every call lands in the transcript.
- **Pairing against the pipeline baseline.** Inspect reports each run's own
  standard error, not a paired difference. The default-flip criterion
  above still needs `paired.py` over per-sample scores exported from both
  logs, with sample ids matching the pipeline runs'.
- **The dependency.** About 40 direct dependencies. Put it in an optional
  `eval-agent` extra, never a core dependency, in the same way the `mcp`
  extra is optional.

If the transcript view doesn't change how a phase 4 failure gets diagnosed,
leave the agentic rows on the run store with the rest.

### 5 — Follow-ups (only if phase 4 justifies the agent)

**Injection surface.** Retrieved text reaches the agent's model undelimited,
as it does the pipeline's ([known limitations](known-limitations.md)).
Today that can corrupt an answer or steer the next search, but it can't
trigger a side effect: the agent's only tool, `rag_search`, is read-only, and
`corpus`, `top_k`, `max_chars` and `filters` are pinned. That is why
tool-hijack benchmarks (InjecAgent, ToolEmu, AgentHarm) don't apply yet.
Four follow-ups below widen the surface: model-set filters let injected text
narrow the search, the [navigation tools](#tool-surface-navigation-not-only-search)
put long windows of untrusted filings into the prompt, corpus routing
(enhancement 6) lets injected text pick which corpus is searched next, and web
routing (enhancement 3) brings in content nobody curates.
Each runs the injection tier from
[Milestone 28](backlog.md#milestone-28--production-hardening) in agentic mode
before it is adopted, next to its quality row.

- **Navigation tools:** listing, reading and finding text in documents, not
  only ranked search. The first follow-up; designed in
  [its own section](#tool-surface-navigation-not-only-search) below. It
  replaces the earlier `read_document(doc_id, section?)` and
  `read_span(doc_id, start, end)` pair.
- **Let the model set `filters` on `rag_search`.** The filter itself shipped
  with the chunking plan's Phase 3: `rag_search` takes a `QueryFilter` over
  the front-matter fields (`company`, `ticker`, `form`, `period_end`, …), and
  a company + period filter measured +10.9pp hit on the `period` tier and
  +7.6pp on `underspecified`. The agent currently pins `filters` to the
  turn's. Add an explicit tool argument and execution path for model-proposed
  filters; removing it from `PINNED_ARGUMENTS` alone does not apply it in
  `AgentService._search`. Intersect proposed filters with the caller's scope,
  never replace or widen it. Add a matrix row, `agentic react / 9b + filters`,
  to measure helpful narrowing versus filters that exclude the answer.
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

#### Tool surface: navigation, not only search

The agent has one tool, `rag_search`, which returns ranked chunks. Published
agentic retrieval systems also *navigate*. Claude Code
[replaced its vector index with glob, grep and read](https://officechai.com/ai/claude-researcher-explains-how-agentic-search-performed-better-than-rag-for-code-generation/).
Anthropic's [context-engineering guidance](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
describes agents that hold lightweight identifiers and load content "just in
time". OpenAI's [Deep Research](https://cdn.openai.com/deep-research-system-card.pdf)
was trained to search, open, scroll and read. Phase 4 points at the same gap
on this corpus:

- **Discovery is every row's weakest kind.** No 9b row completes a discovery
  question, and `react / 27b` completes 2.7 of 5. A discovery question asks
  for a set ("the airlines in this corpus"), and no ranked search can
  enumerate a set.
- **Split tables cost searches.** The 27b needed 5 searches to rebuild a
  table the chunker had split.
- **Refusals are the most expensive turn** (6.6 calls, 158 s). The agent
  can't establish that the corpus lacks something except by searching until it
  gives up. Seeing which companies and periods exist would let it decline
  sooner, and would make `neg-unanswerable-comparison` (missed in 3 of the
  27b agent's 6 runs) checkable.

**The tools, in build order.** Each one adds a matrix row, so a win is
credited to one tool:

1. **`rag_list_documents(filters?)`.** Lists the selected corpora's documents
   with their front-matter fields (`company`, `ticker`, `form`, `period_end`)
   and length. It reads documents and front matter only, with no embedder. It
   targets discovery questions and "which periods exist" questions.
   `rag_list_corpora` works at the corpus level; this works at the document
   level.
2. **`rag_read_document(document_id, start?, max_chars?)`.** Returns a window
   of the document's cleaned text: the text that chunk offsets index into
   (`chunk_selected_corpora`). It comes with `start`, `end`, `length` and
   `next_start`. One tool with an offset covers both earlier tools
   (`read_document` and `read_span`), and Anthropic's
   [tool-design guidance](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
   warns against tools whose uses overlap.
   - *Prerequisite.* `rag_search` results gain `char_start`/`char_end`, which
     chunk metadata already holds, so the model can read around a hit.
   - *Sections.* A `section` argument waits for the
     [chunking plan](chunking-indexing-plan.md)'s Phase 5 to supply heading
     paths. On `edgar_md` the Markdown headings already exist, so an outline
     in the response can come first there.
   - *Size.* This corrects the earlier "well within the 27b's 262k context".
     EDGAR's 61 filings average about 57k characters (about 14k tokens), and
     agent calls run at `agent.num_ctx` 32768. A whole filing doesn't fit
     beside the system prompt and earlier results. Reads are therefore
     windowed: a default of about 6k characters, under a per-turn budget
     `agent.max_read_chars` that is separate from `max_passage_chars`.
3. **`rag_find(phrase, document_id?)`.** A literal, case-insensitive phrase
   search over the selected documents' cleaned text. It returns matches with
   offsets and a short surrounding context. Search can miss an exact name or
   figure ("Enflonsia", a dollar amount) because hybrid ranking scores term
   overlap and the reranker then reorders, so it can fall out of the top 5.
   `rag_find` does no ranking. A linear scan of EDGAR's 3.5 MB takes
   milliseconds. At BEIR or MuSiQue scale, check first whether the contentless
   FTS5 table can answer phrase queries. Build this tool only if the read rows
   still miss questions that name an exact term.

**Constraints shared by all three:**

- **Citations through the ledger.** Every read window and find snippet becomes
  a ledger passage, with an id built from the document id and offsets, so
  `[n]` keeps one source of truth and the turn log records what was shown.
  Deduplicate by overlapping offsets, not only by chunk id.
- **Scope.** A `document_id` outside the turn's corpus selection or filters is
  a tool error. An id the model guesses gains it nothing.
- **Budgets.** Reads and finds count toward `max_tool_calls`. The existing
  context-overflow fallback covers a read that outgrows the window.
- **Shared with MCP.** Implement the tools in `RagTools` and register them in
  the MCP server, as `rag_search` is, so outside agents get them too. Update
  the contract in [MCP server](mcp-server.md).
- **Injection.** Run the Milestone 28 injection tier before adoption. A read
  window carries about five times more undelimited filing text than a search
  result.

**Measurement.**

- *Rows.* Cumulative rows on all four sets, each paired against the one
  before: `agentic react / 27b + list`, `+ list + read` and
  `+ list + read + find`. Also run `agentic react / 9b + list`: one listing
  call is the kind of single decision the 9b does make.
- *Evidence recall.* This needs one change. A 6k-character window contains
  gold spans more easily than a 1.2k-character passage, so report recall from
  searches and recall from reads separately, with characters read per turn.
- *Table tier.* Add the chunking plan's `table` tier when it exists, since
  split tables are where `rag_read_document` should show.
- *Adoption.* The same rule as every follow-up: paired quality and cost
  evidence on the same questions.

#### Enhancement follow-ups

These are proposed experiments, not shipped capabilities or prerequisites for
phase 4. Keep the current baseline and defaults unchanged. Implement one
intervention at a time after failure analysis identifies its target; adopt it
only with paired quality and cost evidence on the same corpus and questions.
The priorities below order this additional work. The
[navigation tools](#tool-surface-navigation-not-only-search) come before all
of it: they target the failures phase 4 actually showed (discovery questions,
split tables, refusal cost), and items 1 and 2 address grounding failures that
have not been measured yet. `planned_refine` stays conditional, as written
above.

1. **Measure grounding and citation failures before adding runtime gates.**
   Reuse [eval harness Phase 4a](eval-harness-plan.md#phase-4a--grounding-completeness-and-citation-scoring-estimate-pending)
   for claim-level faithfulness, answer-point coverage and citation support,
   and its Phase 4 judge calibration. This is the existing owner, not a second
   scoring implementation. Preserve the exact passages shown, citation mapping,
   draft and verdict so a wrong citation can be distinguished from missing
   evidence or incorrect synthesis. A whole-answer boolean is insufficient to
   target a repair. Include correct answers wrongly rejected by the checker
   and incomplete answers whose individual claims are all supported.

2. **Bounded repair after a failed draft check.** Today's optional agent
   checker only reports `grounded`; it neither blocks the answer nor restarts
   retrieval. Pipeline CRAG's regeneration uses the same passages, and its
   retrieval retry addresses rejected passages, not unsupported draft claims.
   Add an opt-in experiment that identifies the unsupported claim or missing
   answer point, searches for missing evidence when needed, and revises the
   draft. Repair synthesis or citation mappings from existing evidence when
   retrieval is already sufficient. Compare no check, report-only checking,
   and checking plus repair separately. Keep searches under the existing
   per-turn cap and deadline; also bound judge/revision calls and meter all of
   them. A repair must not reset budgets or widen caller corpus/filter scope.
   Specify outcomes for exhausted budgets and inconclusive/failed checks;
   evaluate explicit partial answers/refusals rather than silently deleting
   claims to improve faithfulness. Acceptance requires fewer unsupported or
   incorrectly cited claims without an unacceptable loss of completeness,
   correct answers or refusal accuracy. Report false rejection, repair success,
   latency and tokens alongside quality. Runtime gating remains off until the
   calibrated checker and these tradeoffs justify it.

3. **Dynamic retriever routing, only for demonstrated routing failures.**
   The agent currently gets one `rag_search` with configured dense/hybrid
   retrieval and optional web fusion; document routing selects documents, not
   tools. First compare the fixed hybrid baseline with agent-selected existing
   retrieval paths on exact-term, semantic and freshness cases. Reuse the shared
   `RagTools`/MCP tool surface and provider interfaces. Make every route and its
   results inspectable, and define fallbacks for an unavailable source or an
   invalid choice. Preserve caller scope; web must be explicitly configured and
   must not bypass metadata filters or an isolated-corpus evaluation. Evaluate
   web-enabled cases separately with captured source evidence. Measure route
   errors and end-answer quality/cost, not just tool-call success. SQL and graph
   tools are deferred until a real structured corpus, tool contract and labeled
   workload require them; SQLite FTS5 is not a structured-data query tool.

4. **Extend observability through its existing owner.** Local JSONL already
   records searches, passages, citations, events, timings, tokens and optional
   verdicts. [Milestone 26](backlog.md#milestone-26--opentelemetry-trace-export)
   owns OTel export; add agent steps, routing and repair/judge calls to that
   adapter when built. Reuse harness scoring rather than adding a vendor-specific
   evaluator stack. Report retrieval calls and token/cost usage per correct,
   grounded answer with explicit denominators, plus latency distributions;
   preserve separate correctness, grounding and completeness scores. On
   answerable questions, a refusal or vacuously faithful empty answer must not
   count as success; score appropriate refusals separately on unanswerable cases.

5. **Search budget and search-strategy prompt.** These are two rows on
   `react / 27b`, measured separately. Neither changes code under `rag/`.
   - *Wider budget.* Set `max_tool_calls` to 16 and `max_passage_chars` to
     2400. On Anthropic's
     [research eval](https://www.anthropic.com/engineering/multi-agent-research-system),
     token usage alone explained 80% of performance variance. This row tests
     whether the one question left between the agent and the oracle is a
     budget limit.
   - *Strategy prompt.* Tell the agent to start with short, broad queries and
     then narrow, to scale effort to the question (one or two searches for a
     single fact), and to decline when differently worded searches return
     nothing relevant. Anthropic names prompting as its main lever for agent
     behavior. The target is cost: refusals take 6.6 calls and 158 s, and
     single-hop questions take 36 s against the pipeline's 10 s. Adopt the
     prompt only if quality holds.

6. **Corpus routing within the caller's scope.** Today `corpus` is pinned, so
   the agent searches the turn's selection as one index. In this change, the
   turn's selection becomes an upper limit. The model may search any subset
   of it (one corpus, then another) but nothing outside it, and gets a
   `rag_list_corpora` restricted to the selection. That restriction is the
   same rule as model-set filters: narrow the scope, never widen it.
   - *Comparison.* Pair the agent routing between corpora against the agent
     searching the pooled index. Pooling already spans corpora in one search,
     so routing has to beat pooling, not beat searching a single corpus.
   - *Prerequisite: a workload.* The corpora are EDGAR plus public benchmarks
     that may only be evaluated in isolation, so no current set needs a
     second corpus. Routing needs a second real corpus that EDGAR questions
     plausibly draw on, with questions that span both. Registry entries also
     need a description field, because the model routes on what the listing
     says.

**Deferred: training the search policy, and multi-agent orchestration.**

- *Training the search policy.* The largest published gains come from models
  trained to search: OpenAI trained Deep Research with reinforcement learning
  on browsing tasks, and open recipes such as
  [Search-R1](https://arxiv.org/abs/2503.09516) do the same for 3–7b models
  against a retriever. Phase 4's result, where the 9b loop gains nothing and
  the 27b loop gains 6.7 questions, fits searching being a trained skill. The
  local lever is model choice: a tool-calling model that is a credible
  candidate gets a matrix row like the 27b's. Fine-tuning a search policy
  would be a research milestone with its own compute budget, not a follow-up
  here.
- *Multi-agent orchestration.* A lead agent with parallel subagents (Anthropic
  Research, about 15 times a chat's tokens) pays off by keeping each
  subagent's context separate and by covering broad questions in parallel.
  Here the costliest turns total about 35k prompt tokens across all their
  calls, over a few hops.
  Reconsider it only if the navigation tools push turns past the context
  window.

**Deferred: durable evidence memory and learned retrieval policies.** Recent
history and per-turn deduplication already exist. Require reviewed multi-turn
failures showing that cross-session evidence reuse would help before adding a
memory store; define source freshness, invalidation, corpus isolation and
provenance first. Similarly, agent-selected top-k, reranking depth or chunk size
needs a separate budget-controlled experiment, not an expansion of the initial
tool-routing scope.

Motivation: the [Hugging Face cookbook](https://huggingface.co/learn/cookbook/en/agent_rag)
and [retrieval-agents course](https://huggingface.co/learn/agents-course/en/unit2/smolagents/retrieval_agents)
describe query refinement and enhanced retrieval; most of those components
already exist here. The [FutureAGI article](https://futureagi.com/blog/agentic-rag-systems-2025/)
(updated May 2026) motivates the routing, draft-check/repair and tracing gaps.
These are design references, not evidence that the additions improve this repo.

## Open questions

- **Interactive latency budget.** With the 27b, hard questions took minutes on
  this M2 Max. Is agentic mode meant for the API/UI, or batch and MCP-style
  use? The answer decides whether `timeout_s` is 60 or 600.
- **Hosted model as a reference point.** Running the phase 4 matrix once with a
  frontier model would separate "the method doesn't help" from "the local
  model can't drive it". Gemini Flash-Lite can now drive the agent on the free
  tier, which costs quota rather than money. An agent turn makes up to 9 calls
  against the pipeline's 1, though, so a full matrix would take several days of
  quota; one reference row fits. A frontier model proper still needs an
  Anthropic adapter and **real API budget**, so explicit sign-off with a cost
  estimate first.
  *Built for Flash-Lite:* `run_answer_matrix.py --family m19-hosted` runs a
  pair, `pipeline / flash-lite` and `agentic react / flash-lite`. The same
  model with and without the loop separates the method from the model. The
  pair is its own family, so `--family m19` never spends quota. It refuses an
  unbounded answerable set. The client's daily budget stops the run at 450
  requests, and the same command resumes from the checkpoints after the reset.
  Estimated at about 400–450 requests for the agent row and 90 for the
  pipeline row, so 1–2 days of free quota per repeat. Measure calls per
  question on `--limit 5` first. One confound: Flash-Lite can't turn thinking
  off (`thinking_level: minimal`), while the 9b runs with it off.

### The superlative question stays a refusal

Resolved on 2026-09-29, before phase 4. The open question was whether an agent
with 14 searches could answer "which company had the highest operating margin
last quarter?", which would move it to the multi-hop set. It can't. Only MD&A
was extracted, and six of the 14 companies (AAPL, JNJ, MRK, PFE, XOM, CVX)
report no operating income there. "Last quarter" also ends on different dates,
March 31 to June 30, 2026, and Chevron's latest filing is a 10-K. So
`neg-unanswerable-comparison` stays in the refusal set with a corrected
rationale: the corpus can't support the ranking, however many searches are
made. Its `negative_type` changed from `requires_aggregation` to
`unsupported_by_corpus`. The answerable form is a new multi-hop question,
`mh-agg-airline-margin`: the airlines in the corpus, the quarter ended June 30,
2026. It still makes the agent find out which companies qualify. Its parts are
the set's first hand-authored gold, verified against the filings by
`scripts/build_multihop_set.py`. The set is now 35 questions, so totals from
before this change (15/34) are not directly comparable.
