# RAG_Project

Retrieval-Augmented Generation system, built up milestone by
milestone. Local-first by default: Ollama serves both embeddings and chat;
Chroma is the vector store. Model names live in `rag/config/config.yaml`.

## Status

What has shipped and what is next is tracked in [Backlog](docs/backlog.md);
the MCP server is also shipped.

- **Measured-off features stay off.** Contextual chunking, CRAG, query
  expansion, `retrieval.min_score` and `embedding.query_instruction` all
  measured as no better than noise on the EDGAR corpus — don't enable one
  without re-measuring
  ([Measured results](docs/measured-results.md)).
- **The MCP server is not Milestone 19.** It serves our retrieval to an outside
  agent; Milestone 19 is *this* system calling search as a tool.

## Architecture

Everything pipeline-relevant sits behind interfaces (`EmbeddingModel`,
`VectorStore`, `Reranker`, `LLMClient`, `QueryExpander`, `Chunker`) so
implementations can be swapped via config alone.

Pipeline code never references concrete classes — only the interfaces and the
config-driven factories. To add an implementation, use the `add-provider` skill.

## Config

All component selection and tunables live in `rag/config/config.yaml`,
validated by pydantic models in `rag/config/settings.py`. Load it via
`load_config()`. Don't hardcode model names, chunk sizes, or paths in pipeline
code — read them from `RagConfig`.

`chat.mode: agentic` swaps the retrieve-then-generate pipeline for an agent
that calls search as a tool (`rag/generation/agent.py`, settings under
`agent:`). It stays off by default until Milestone 19 phase 4 measures it.

A config file may start with `base: <path>` (relative to itself) to inherit
another and list only what it changes — mappings merge, lists replace.
`rag/config/vanilla.yaml` does this: plain dense RAG, no reranker,
`chat.prompt: plain`; pass it via `--config`.

### Corpora

`corpora.registry` names bodies of documents; `corpora.active` picks which ones
a command uses, overridden by `--corpus NAME` (repeatable; works on the CLI,
`scripts/demo.sh`, and both eval runners). Several names **pool** into one
index.

- Each selection gets its own index (`rag_corpus__edgar` vs.
  `rag_corpus__baseline+edgar`), so isolated and pooled indexes coexist.
  Order doesn't matter.
- An unknown corpus name raises. Pooling refuses duplicate `Document.id`s —
  don't "fix" that by namespacing ids; it would invalidate the eval sets'
  `expected_doc_ids`.
- Corpora live at `data/corpora/<name>/documents/`. `baseline` is committed;
  others (e.g. `edgar`) are gitignored and fetched from their `manifest.json`.

Why it's built this way: [milestone notes](docs/milestone-notes.md#named-corpora-notes-shipped-with-milestone-11).

## Running things

- Tests: `pytest` (`-m "not live"` skips the few that call a local Ollama;
  they also skip themselves when it isn't running; the Gemini one spends quota
  and runs only with `RAG_GEMINI_LIVE=1`). Before finishing, also run what CI runs: `ruff check .` and
  `mypy --ignore-missing-imports rag` (mypy is scoped to `rag/` on purpose —
  tests use structural fakes that nominal typing flags falsely).
- Chunk and index health: `python -m rag.cli index-report` (read-only, no
  embedder). Prints chunk sizes, chunks starting mid-table, duplicates, and
  whether the built index is in sync with the corpus and config. Run it before
  and after any chunking change.
- Whole pipeline end to end: `scripts/demo.sh` (takes `--corpus`, `--question "..."`)
- Build the index: `python -m rag.cli index`. Incremental: unchanged chunks
  are skipped, and chunks of deleted/shortened documents are removed. Add
  `--reset` to rebuild; it's **required** after changing the embedder or
  anything under `chunking.contextual`, and the index manifest
  (`rag/index_manifest.py`) refuses the run until you do. Generated
  contexts survive `--reset` on purpose (so rebuilds don't re-pay for them);
  `--clear-context-cache` forces regeneration.
- API: `uvicorn rag.api.main:app --reload` (`POST /chat` with `{"query": "..."}`, `/health`)
- MCP server: `python -m rag.mcp` (stdio), or `POST /mcp` on the running API.
  The `mcp` extra is required only for the streamable HTTP transport; stdio
  uses the fallback implementation without it. The MCP server serves exactly
  one protocol revision — see [MCP server](docs/mcp-server.md).
- Logged turns and feedback: `python -m rag.cli turns` (`--feedback down` for
  the thumbs-down ones). With the `jsonl` provider, the API, UI and `cli chat`
  append every turn to `observability.turn_log.path` (default
  `data/logs/turns.jsonl`); `provider: none` logs nothing. The eval runners
  never do.
- Retrieval eval: `python -m rag.eval.retrieval_eval` (`-v` for per-sample;
  `--eval-set data/eval/edgar_eval_set.json --corpus edgar` for EDGAR)
- Answer eval (LLM-as-judge): `python -m rag.eval.answer_eval`. The judge is
  `eval.judge`, else the generator grading itself (warned); `--judge-model`
  overrides, plus `--judge-provider` when the judge runs on a different
  provider than the one it inherits. Multi-hop: `python -m rag.eval.multihop_eval --corpus edgar`.

## Conventions

- Strong typing throughout — all public functions/classes are annotated;
  pydantic models for config and API schemas, plain typed dataclasses for
  internal data (`Document`, `Chunk`, etc.).
- `logging.getLogger(__name__)` per module; call `configure_logging()` once at
  each entrypoint (CLI, API, UI) — never at import time.
- Keep dependencies minimal — justify any new dependency against what's
  already available (e.g. don't add a second HTTP client, a second YAML
  parser, etc.).

## Documentation

Design rationale, history, and planning live in `docs/`, read on demand:

- **`docs/architecture.md`** — how index-time and query-time paths fit
  together and which entrypoint uses which layer. Read before a change that
  spans more than one stage.
- **`docs/milestone-notes.md`** — why each shipped component is built the way
  it is. Read before touching a component to see what tradeoff its current
  shape already encodes.
- **`docs/measured-results.md`** — eval numbers with the caveats
  needed to read them safely. Read before turning on anything that is off by
  default. To measure a new change, use the `measure-change` skill.
- **`docs/backlog.md`** — the planned milestones: the plan behind each, and
  why they are ordered as they are.
- **`docs/known-limitations.md`** — known gaps and failure modes in what's
  shipped, worth checking before recommending a feature that's off by default.
- **`docs/milestone-19-plan.md`** — phased plan for agentic retrieval, with the
  9b-vs-27b agent probe that shaped it. Read before starting Milestone 19.
- **`docs/chunking-indexing-plan.md`** — production chunking/indexing
  methodology, the measured gaps against it, and the phased plan (covers
  Milestones 14, 15 and 25 and part of 20). Read before changing the chunker,
  loaders, or index text.
- **`docs/mcp-server.md`** — the MCP tool contract, both transports, how to
  point an external agent at it, and why indexing is not exposed as a tool.
