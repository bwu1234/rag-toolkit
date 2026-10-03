---
name: add-provider
description: Add a new implementation of one of this repo's swappable interfaces (EmbeddingModel, VectorStore, Reranker, LLMClient, QueryExpander, Chunker) so it can be selected by a `provider:` string in config.yaml. Use whenever the task is "add a <X> backend", "support <model/service> as a reranker/embedder/vector store/LLM", "make <X> selectable via config", or when swapping in an alternative implementation of an existing component.
---

# Adding a provider

Every pipeline-relevant component in this repo sits behind an ABC and is
selected by a `provider:` string. Pipeline code never names a concrete class.
Adding a backend is therefore a fixed five-step edit, and **all five are
required** — a class with no factory branch is unreachable, and a factory
branch with no config field is unselectable.

## The five interfaces

| Interface | ABC | Factory | Config model |
|---|---|---|---|
| `EmbeddingModel` | `rag/embedding/base.py` | `rag/embedding/factory.py::get_embedder` | `EmbeddingConfig` |
| `VectorStore` | `rag/vectorstore/base.py` | `rag/vectorstore/factory.py::get_vector_store` | `VectorStoreConfig` |
| `Reranker` | `rag/retrieval/reranker.py` | `rag/retrieval/factory.py::get_reranker` | `RerankerConfig` |
| `QueryExpander` | `rag/retrieval/expansion.py` | `rag/retrieval/factory.py::get_query_expander` | `QueryExpansionConfig` |
| `LLMClient` | `rag/llm/base.py` | `rag/llm/factory.py::get_llm_client` | `LLMConfig` |
| `Chunker` | `rag/chunking/chunkers.py` | `rag/chunking/chunkers.py::get_chunker` | `ChunkingConfig` |

`Chunker` is the one exception to the `provider:` convention — it is selected by
`chunking.strategy` instead. Everything else below applies unchanged.

All config models live in `rag/config/settings.py`.

## Steps

**1. Read the ABC first.** Its docstring states the contract, and the contracts
carry non-obvious requirements that a naive implementation violates. Examples
already in the tree:

- `Reranker.rerank` takes `queries: list[str]`, not one query — `retrieval.expansion`
  can turn one question into several. `queries[0]` is always the user's real question.
- Rerankers must return scores in `[0, 1]` reflecting **their own** judgment, not
  the vector store's similarity. Use `normalize_rerank_score()` from
  `rag/retrieval/reranker.py` for unbounded logits, and `rescored()` to replace a
  score (`ScoredChunk` is frozen).
- Never mutate `chunk.text`. Citations quote the source verbatim; contextual
  chunking prepends context at index time precisely so the stored text stays clean.

**2. Write the adapter** in the interface's own package
(`rag/retrieval/my_reranker.py`, `rag/embedding/my_embedder.py`, …). Subclass the
ABC. Keep the constructor taking plain values (`model: str`, `base_url: str`),
not a config object — the factory does the unpacking, which keeps the adapter
testable without building a `RagConfig`.

Defer heavyweight setup (model download, client construction) until first use if
it is expensive; several tests instantiate these.

**3. Register a branch in the factory.** Follow the existing shape exactly — one
`if config.provider == "...":` per provider, and a final `raise ValueError` whose
message names the fix:

```python
raise ValueError(
    f"Unknown reranker provider: {config.provider!r}. "
    "Add an adapter and register it here to support a new provider."
)
```

Do not fall back to a default on an unknown provider. A typo must fail loudly —
silently returning a no-op produces a plausible-looking but meaningless eval run.

**4. Extend the config model** in `rag/config/settings.py`. The `provider` field
is a `Literal`, so the new name must be added to it or pydantic rejects the
config:

```python
provider: Literal["none", "cross_encoder", "my_backend"] = "none"
```

Add any new tunables as fields with defaults. Then add the stanza to
`rag/config/config.yaml` **with a comment saying what the setting does and what
it costs** — that file is the primary documentation for these knobs, and the
existing entries set the standard (see `RerankerConfig.query_prefix` for the
level of detail expected when a value is non-obvious).

**5. Mirror a test** in `tests/test_<component>.py`. The convention is to swap
the heavy dependency for a small fake rather than download a model — see
`_FakeCrossEncoder` in `tests/test_reranker.py`. Cover:

- the adapter's real logic against the fake (scoring, normalization, ordering);
- `get_<component>(config)` returns your class for the new provider string;
- an unknown provider still raises.

## Before finishing

- `ruff check .` and `mypy --ignore-missing-imports rag` — CI runs both, and mypy
  is scoped to `rag/` only (the suite's structural fakes are not ABC subclasses).
- `pytest`.
- If the new provider changes what retrieval returns, it is **unmeasured**. Do not
  make it the default in `config.yaml` on the strength of it working. Add it as a
  variant and use the `measure-change` skill — this repo has already shipped
  four features that turned out to be no better than noise.

## Two traps this repo has already hit

- **Loading a model is not using it correctly.** A BGE cross-encoder swapped in
  for ms-marco silently double-sigmoided every score (`predict()` applies a
  per-model default activation) into `[0.5, 0.73]`. Ranking order survived, so
  hit rate looked fine while `min_score` and `aggregate: mean` were broken.
  Check what activation/normalization your backend applies by default, and assert
  a known raw value maps to the expected normalized one in a test.
- **Inherited thresholds do not transfer.** `retrieval.min_score` is calibrated to
  a specific model's score distribution. A new reranker invalidates it. Leave it
  at `0.0` and re-derive rather than carrying the old value over.
