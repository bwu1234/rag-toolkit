# 0016 — One repo, separate packages with a one-way dependency enforced in CI

- **Status:** Accepted
- **Recorded:** 2026-10-03

## Context

The agent (then `rag/generation/agent.py`) is built from retrieval: it searches
through the same tool surface the MCP server serves
([0014](0014-mcp-server-scope.md)). The question was whether agent and
retrieval should live in separate repos, and how to keep them decoupled
either way.

Moving code between repos does not decouple it. The interface between the
two does, and a repo boundary only changes who can change that interface and
how quickly. Here the two change together: agent quality depends on what a
search returns, how filters work and how scores are reported, and every eval
row ([0011](0011-measured-off-stays-off.md)) pins a retriever, an index and
an eval set that the agent must share. Nothing that would justify a split
holds today: one team, no unrelated product consuming retrieval (the MCP
server and the agent both already sit behind `rag.tools`), one runtime, one
licence, and an agent that is still about this corpus.

The seam already half-existed (`rag/tools.py`) but nothing kept it. A single
import from retrieval into the agent would have passed review and CI.

## Decision

Stay in one repo. Split the code into packages with a one-way dependency, and
make a violation a CI failure.

```text
rag.chat          picks pipeline or agent by chat.mode (build_chat_service)
  rag.agent       service, prompts, calculator, builder
    rag.generation   the pipeline, CRAG, the answer types the agent shares
      rag.llm        model clients and the tool-calling interface
```

`rag.llm` was extracted from `rag.generation` because retrieval needs a model
too (query expansion, contextual chunking). Before the move, the only way to
let retrieval import a model client was an exception in the contract. Now the
kernel is a package that imports none of the layers above it.
`build_chat_service` moved to `rag.chat` because it dispatches on `chat.mode`;
left inside either package, that package would import the other, and the
agent already imports the pipeline's answer types.

`import-linter` (`lint-imports`, contracts in `pyproject.toml`) runs in CI and
enforces four rules:

1. **Answering stack layers.** `chat > agent > generation > llm`: each
   imports only downward, so the pipeline never imports the agent and the
   model clients import neither.
2. **Retrieval never imports the answering stack.** Retrieval, chunking,
   vector store, embedding, ingestion, `rag.tools`, `rag.mcp` and `rag.llm`
   may not import `rag.generation`, `rag.agent` or `rag.chat`.
3. **The agent reaches retrieval only through `rag.tools`.** One exception is
   recorded in the contract: `rag.agent.service -> rag.vectorstore.base`, for
   the `ScoredChunk` type that `RagTools.retrieve` returns.
4. **Library code never imports an entrypoint** (`api`, `ui`, `cli`, `eval`).

Each contract was checked by adding a violating import and confirming the
check fails, including indirect ones.

The tool contract is `rag.tools` (`RagTools`, `ToolSpec`): a typed search
function whose JSON Schema is derived from its signature. It is the seam a
later repo split would cut along.

Evaluation already runs on each part and on the combination:
`retrieval_eval` scores retrieval alone, `answer_eval`/`multihop_eval` score
the agent end to end, and `run_answer_matrix.py` pairs agent and pipeline on
one pinned retriever and index.

## Alternatives considered

- **A separate agent repo now.** Rejected: no organizational signal, and the
  cost is a coordinated release for every search-quality change plus a version
  pairing for every eval result.
- **Directories and a CI rule only, no moves.** Where this started: contracts
  on the old layout with the LLM client allowed through by exception. Rejected
  once the move was cheap, since the exceptions encoded a layout the code did
  not have.
- **Separately installable distributions** (a `pyproject.toml` per package,
  each with its own dependency list). Not done. The packages share one
  dependency list, so `chromadb` and `sentence-transformers` still install
  with the agent. `rag.config` is imported by every package, so separate
  distributions would first need it extracted into a third. That buys nothing
  while one team ships one deployable, and the import rules already stop the
  agent from using what it need not install. It is the next step when a signal
  below appears.
- **Code review alone.** This is how boundaries erode. A rule that fails the
  build does not depend on a reviewer remembering it.

## Consequences

- A change that adds a forbidden import fails CI with the import chain. The
  fix is to route through `rag.tools`, or to move the shared piece to where
  both sides may import it. Adding an exception to a contract is a decision
  to record, not a way to make CI pass.
- Module paths changed: `rag.generation.{llm,factory,ollama_llm,gemini_llm,
  daily_budget}` are `rag.llm.{base,factory,ollama_llm,gemini_llm,
  daily_budget}`, `rag.generation.agent` is `rag.agent.service`,
  `rag.generation.calculator` is `rag.agent.calculator`, the agent's prompts
  moved to `rag.agent.prompts`, and `rag.generation.builder.build_chat_service`
  is `rag.chat.build_chat_service`. Branches open at the time need their
  imports updated on rebase. Eval rows and configs name no module paths, so
  recorded results are unaffected.
- `rag.tools` still imports `ToolDefinition` and `LLMClient` from `rag.llm`,
  so the contract is not yet free of LLM types. Moving `ToolSpec.definition`
  to the agent side would remove that.

## Signals to revisit

Split into a separate repo when one is true, not expected: different teams
own retrieval and agents, unrelated agents or products consume retrieval as a
service, the agent needs a sandbox or browser runtime a retrieval service
should not carry, one side needs different licensing or access control, or the
agent's main source stops being this corpus.
