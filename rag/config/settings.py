"""Typed, config-driven settings for the RAG system.

All component choices (embedding model, vector store, reranker, LLM, chunking
parameters, etc.) live here as plain pydantic models loaded from a YAML file.
Swapping an implementation should mean: change a `provider` string in
config.yaml (and possibly add a small adapter class) — not editing pipeline
code.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal

import yaml
from pydantic import BaseModel, Field

# Repo root = two levels up from this file (rag/config/settings.py -> repo/)
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


class PathsConfig(BaseModel):
    """Filesystem locations used by the pipeline. Relative paths resolve against REPO_ROOT."""

    corpus_dir: Path = Path("data/corpora/baseline/documents")
    index_dir: Path = Path("data/index")

    def resolved(self) -> "PathsConfig":
        return PathsConfig(
            corpus_dir=(REPO_ROOT / self.corpus_dir).resolve(),
            index_dir=(REPO_ROOT / self.index_dir).resolve(),
        )


class CorpusConfig(BaseModel):
    """One named body of documents."""

    documents_dir: Path
    description: str = ""


class CorporaConfig(BaseModel):
    """Named corpora and which of them commands operate on.

    `active` is a **list** because that is what expresses the distinction the
    registry exists for:

    * one name -> that corpus indexed on its own (**isolated**), measuring
      retrieval quality within it;
    * several names -> those corpora indexed into one collection (**pooled**),
      measuring robustness to cross-corpus distractors.

    The gap between the two is the interference cost, and it is not observable
    with a single corpus. See `CorpusSelection` for how a selection maps onto
    on-disk index names.
    """

    active: list[str] = Field(default_factory=list)
    registry: dict[str, CorpusConfig] = Field(default_factory=dict)


@dataclass(frozen=True)
class CorpusSelection:
    """The resolved corpora a command operates on, and where their index lives.

    Index names are derived from the selection rather than fixed, so an isolated
    and a pooled index of the same corpora coexist instead of silently
    overwriting each other -- which matters because comparing them is the point.
    A run therefore cannot accidentally evaluate against an index built from a
    different set of documents.
    """

    names: tuple[str, ...]
    document_dirs: tuple[Path, ...]
    #: Base collection name from `vector_store.collection_name`.
    base_collection: str
    index_dir: Path

    @property
    def slug(self) -> str:
        """Stable identifier for this combination of corpora."""
        return "+".join(self.names)

    @property
    def collection_name(self) -> str:
        return f"{self.base_collection}__{self.slug}"

    @property
    def is_pooled(self) -> bool:
        return len(self.names) > 1

    def describe(self) -> str:
        kind = "pooled" if self.is_pooled else "isolated"
        return f"{', '.join(self.names)} ({kind})"


class EmbeddingConfig(BaseModel):
    """Which embedding model to use and how to reach it.

    Default: Ollama-served `qwen3-embedding:0.6b`. Because both embeddings and
    chat go through Ollama, the only network dependency for a fully local setup
    is the Ollama daemon — no extra ML libraries required for the default path.
    """

    provider: Literal["ollama", "sentence_transformers"] = "ollama"
    model: str = "qwen3-embedding:0.6b"
    base_url: str = "http://localhost:11434"
    # Embedding dimensionality is provider/model-specific; recorded here so the
    # vector store can validate it rather than discovering mismatches at query time.
    dimensions: int | None = None


class LLMConfig(BaseModel):
    """Which chat/generation model to use and how to reach it.

    Default: Ollama-served `qwen3.5:9b-mlx`.
    """

    provider: Literal["ollama", "anthropic", "openai"] = "ollama"
    model: str = "qwen3.5:9b-mlx"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.2
    max_tokens: int = 1024
    # qwen3.5 reasoning models spend `max_tokens` on a hidden "thinking" trace
    # before the real answer; on a long RAG prompt that can exhaust the
    # budget and leave `content` empty. Off by default for reliable answers.
    think: bool = False
    # Per-request HTTP timeout. The default suits a generation over 5 passages;
    # a judge grading a long multi-hop answer on a shared GPU needs far more
    # (the 9b-vs-27b probe needed 900s for an 11k-token grading prompt).
    timeout_s: float = Field(default=120.0, gt=0, description="Per-request timeout in seconds")


class ContextualChunkingConfig(BaseModel):
    """Index-time enrichment: prepend a generated document-context blurb to each chunk.

    Off by default because it's the most expensive operation in the pipeline:
    one LLM call *per chunk* during `python -m rag.cli index`, each carrying a
    slice of the parent document. The payoff is that the cost is paid per
    index rather than per query, and it addresses a failure that query-time
    tuning can't -- a chunk that never names its own subject (see
    `rag.chunking.contextualizer`).

    Changing any of these values changes what gets embedded, so re-index with
    `--reset` afterwards rather than relying on the incremental skip, which
    keys off chunk text alone.
    """

    enabled: bool = False
    max_document_chars: int = Field(
        default=8000,
        gt=0,
        description="Characters of the parent document included in each context prompt",
    )
    max_context_chars: int = Field(
        default=400,
        gt=0,
        description="Cap on a generated blurb, so context can't outweigh the chunk it describes",
    )
    concurrency: int = Field(
        default=4,
        ge=1,
        description=(
            "Context generations in flight at once. Per-chunk calls are independent, "
            "so this scales close to linearly -- up to the serving backend's own "
            "parallelism (Ollama's OLLAMA_NUM_PARALLEL), past which requests just queue."
        ),
    )
    cache: bool = Field(
        default=True,
        description=(
            "Checkpoint generated contexts to disk so an interrupted index run resumes. "
            "The cache key covers every input to the LLM call (prompts, truncation "
            "limits, model), so a config change misses rather than serving a stale blurb -- "
            "which is why the cache is kept across `index --reset`."
        ),
    )


class ChunkingConfig(BaseModel):
    """Parameters for splitting documents into retrievable chunks.

    `fixed` = simple character-based windows with overlap. Deliberately the
    simplest strategy that could work; structure-aware/semantic chunking is a
    later milestone once the end-to-end pipeline is proven out.
    """

    strategy: Literal["fixed"] = "fixed"
    chunk_size: int = Field(default=1000, gt=0, description="Target characters per chunk")
    chunk_overlap: int = Field(default=150, ge=0, description="Characters of overlap between consecutive chunks")
    contextual: ContextualChunkingConfig = ContextualChunkingConfig()


class VectorStoreConfig(BaseModel):
    """Vector store selection and connection details.

    Default: Chroma in persistent local mode (single directory on disk, no
    server process). Bundles vector search with metadata storage, which keeps
    the dependency count low for a local-first setup.
    """

    provider: Literal["chroma"] = "chroma"
    collection_name: str = "rag_corpus"


class RerankerConfig(BaseModel):
    """Reranker selection. `none` disables reranking (pure vector retrieval)."""

    provider: Literal["none", "cross_encoder"] = "none"
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # How to fold a candidate's per-query scores into one, when
    # `retrieval.expansion` produced several question phrasings. `max` keeps a
    # chunk that any phrasing found relevant -- the setting that lets multi-query
    # expansion actually change the final ranking rather than only stage 1.
    # `mean` requires broader agreement and suppresses chunks only one rewrite
    # liked. No effect with `retrieval.expansion.provider: none`.
    aggregate: Literal["max", "mean"] = "max"
    # Text wrapped around each side of a (query, passage) pair before scoring.
    #
    # Most cross-encoders (ms-marco, BGE) are trained on bare pairs and want
    # these empty. Instruction-tuned rerankers are not: Qwen3-Reranker was
    # trained behind an `<Instruct>/<Query>/<Document>` template, and scoring it
    # on bare pairs is off-distribution. It still ranks an easy pair correctly
    # but its margin collapses -- measured on one pair, a 6.8-logit separation
    # with the template versus 1.9 without -- and across 20 near-identical
    # passages that difference is the whole job. Unprefixed, it scored 0.276 hit
    # rate on this corpus, worse than using no reranker at all.
    #
    # `{query}` / `{document}` are substituted if present; otherwise the value is
    # treated as a plain prefix.
    query_prefix: str = ""
    document_prefix: str = ""


class QueryExpansionConfig(BaseModel):
    """Pre-retrieval query transformation. `none` keeps retrieval LLM-free.

    - ``hyde`` — generate a hypothetical answer passage and embed *that*,
      closing the shape mismatch between a short question and the long
      declarative passage that answers it.
    - ``multi_query`` — generate several rephrasings, retrieve for each, and
      fuse the ranked lists, so the corpus's choice of vocabulary matters less.

    Both add an LLM call to the retrieval path (and `hyde` with
    ``num_documents > 1`` adds one per document), plus one embedding round trip
    per generated query. That's a real latency cost on every search, which is
    why the default is `none`: turn one on and measure with
    `python -m rag.eval.retrieval_eval` rather than assuming it helps.
    """

    provider: Literal["none", "hyde", "multi_query"] = "none"
    num_queries: int = Field(
        default=3, ge=1, le=10, description="multi_query: rephrasings to generate (excluding the original)"
    )
    num_documents: int = Field(
        default=1, ge=1, le=5, description="hyde: hypothetical passages to generate and embed"
    )
    include_original: bool = Field(
        default=True,
        description="hyde: also embed the user's real question, so a bad generation can't sink the search",
    )


class WebSearchConfig(BaseModel):
    """Optional live web search source, fused into retrieval alongside dense/BM25.

    Disabled by default -- enabling it means every query leaves the local
    machine (a SearxNG instance, self-hosted or not) and costs one extra
    embedding round trip per result. `min_similarity`/`dedup_threshold`
    mirror the fixed thresholds Perplexica/Vane use for the same job (scoring
    raw search snippets by embedding cosine similarity against the query, then
    dropping near-duplicate snippets) -- see `rag.retrieval.websearch`.
    """

    enabled: bool = False
    searxng_url: str = "http://localhost:8080"
    top_k: int = Field(default=10, gt=0, description="Search results fetched from SearxNG per query")
    min_similarity: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Drop results below this query-similarity score"
    )
    dedup_threshold: float = Field(
        default=0.75, ge=0.0, le=1.0, description="Drop a result if another kept result exceeds this similarity to it"
    )
    timeout_s: float = Field(default=10.0, gt=0, description="HTTP timeout for the SearxNG request")


class RetrievalConfig(BaseModel):
    """Controls candidate retrieval, fusion, and post-rerank width.

    `mode`:
      - ``dense``  — vector similarity only (original behavior)
      - ``hybrid`` — dense + BM25 keyword search fused with Reciprocal Rank
        Fusion (RRF), then optionally reranked. Recovers exact keyword hits
        (acronyms, IDs, names) that pure embedding search often misses.
    """

    top_k: int = Field(default=20, gt=0, description="Candidates retrieved per retriever (dense and/or BM25)")
    rerank_top_k: int = Field(default=5, gt=0, description="Final number of chunks passed to the LLM after reranking")
    mode: Literal["dense", "hybrid"] = "dense"
    # RRF constant from Cormack et al.; 60 is the widely used default.
    rrf_k: int = Field(default=60, gt=0, description="RRF rank constant: score += 1 / (rrf_k + rank)")
    # Relevance floor applied to the *final* results, after reranking. Without
    # it a vector store always returns its nearest `top_k` neighbours no matter
    # how distant they are, so an off-corpus question still arrives at the LLM
    # with a full set of irrelevant passages to "ground" itself in. Dropping
    # everything below the floor is what lets `ChatService` distinguish "no
    # relevant context" from "empty index".
    #
    # The threshold is read against whatever the last stage's score means (see
    # the `Reranker` score convention): a cross-encoder's sigmoid-squashed
    # logit with `reranker.provider: cross_encoder`, raw cosine similarity with
    # `none`. Those scales differ, so retune this when switching providers.
    # 0.0 (the default) disables filtering.
    min_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Drop final results scoring below this")
    expansion: QueryExpansionConfig = QueryExpansionConfig()
    web_search: WebSearchConfig = WebSearchConfig()


class ChatConfig(BaseModel):
    """Turn-level behaviour of the chat pipeline (as opposed to retrieval tuning).

    `condense_history` controls conversational query rewriting: with it on, a
    follow-up like "what about part-time staff?" is rewritten into a
    standalone question using the preceding turns before it's embedded.
    Retrieval has no memory of its own -- embedding the raw follow-up searches
    the corpus for the literal words "what about part-time staff", which is
    almost never what the user meant.

    Costs one extra LLM round trip per turn that has history; set to `false`
    for a purely one-shot assistant (the CLI's `chat` command, the eval
    pipeline) where there's never any history to condense anyway.
    """

    condense_history: bool = True
    max_history_turns: int = Field(
        default=6,
        ge=0,
        description="Most recent turns fed to the condenser (keeps the rewrite prompt bounded)",
    )


class CragConfig(BaseModel):
    """Corrective RAG: grade what was retrieved, retry if it's bad, verify the answer.

    Plain RAG has exactly one shot at retrieval and no opinion about what came
    back. The relevance floor (`retrieval.min_score`) filters on the reranker's
    score, but a score is not a judgment about whether a passage *answers the
    question* -- so a confidently-scored, topically-adjacent passage still
    reaches the model as though it were an answer. CRAG adds three checks
    around the existing pipeline, each independently switchable:

    - `grade_documents` — ask the LLM, per retrieved passage, whether it
      actually helps answer the question, and drop the ones that don't.
    - `max_retries` — when grading leaves nothing, rewrite the query and search
      again rather than immediately giving up.
    - `check_groundedness` — after generating, ask whether the answer is
      actually supported by the passages, and regenerate if not.

    Every one of these costs LLM round trips on a path that previously had at
    most one, which is why the whole thing is off by default: `grade_documents`
    alone is one call per retrieved passage (`retrieval.rerank_top_k` of them).
    Measure with `python -m rag.eval.answer_eval` before leaving it on.
    """

    enabled: bool = False
    grade_documents: bool = True
    max_retries: int = Field(
        default=1,
        ge=0,
        le=3,
        description="Extra retrieve attempts with a rewritten query when grading leaves nothing",
    )
    check_groundedness: bool = True
    max_regenerations: int = Field(
        default=1,
        ge=0,
        le=3,
        description="Regeneration attempts when the groundedness check rejects an answer",
    )


class EvalConfig(BaseModel):
    """Settings read only by the eval runners -- nothing on the query path uses these.

    `judge` is the LLM that grades answers in `answer_eval`, `multihop_eval` and
    `scripts/run_answer_matrix.py`. `None` judges with `llm`, the generator,
    which is how every result before Milestone 19 was produced and so is kept
    as the default for reproducibility. It is also why those results cannot
    compare generators: swapping the generator swaps the judge with it. The
    runners warn whenever the judge and the generator are the same model.
    """

    judge: LLMConfig | None = None


class TurnLogConfig(BaseModel):
    """Where per-turn records and user feedback are persisted (Milestone 12).

    Only entrypoints a person talks to -- the API, the UI, `cli chat` -- pass a
    sink to `build_chat_service`; the eval runners don't, so evaluation traffic
    never lands in the log that is meant to become new eval samples.
    """

    provider: Literal["jsonl", "none"] = "jsonl"
    path: Path = Field(
        default=Path("data/logs/turns.jsonl"),
        description="JSONL file for the `jsonl` provider; relative paths resolve against the repo root",
    )


class ObservabilityConfig(BaseModel):
    turn_log: TurnLogConfig = TurnLogConfig()


class RagConfig(BaseModel):
    """Top-level config object — the single source of truth for component selection."""

    paths: PathsConfig = PathsConfig()
    corpora: CorporaConfig = CorporaConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    llm: LLMConfig = LLMConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    vector_store: VectorStoreConfig = VectorStoreConfig()
    reranker: RerankerConfig = RerankerConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    chat: ChatConfig = ChatConfig()
    crag: CragConfig = CragConfig()
    eval: EvalConfig = EvalConfig()
    observability: ObservabilityConfig = ObservabilityConfig()

    #: Name used when no registry is configured -- see `corpus_selection`.
    IMPLICIT_CORPUS_NAME: ClassVar[str] = "default"

    def corpus_selection(self, names: Sequence[str] | None = None) -> CorpusSelection:
        """Resolve which corpora to operate on, and where their index lives.

        `names` overrides `corpora.active` (this is what the CLI's `--corpus`
        flag passes). Falling back further, an empty `corpora.registry` yields a
        single implicit corpus at `paths.corpus_dir`, so a config with no
        `corpora:` section keeps working exactly as before the registry existed.

        Raises `ValueError` on an unknown name rather than silently indexing
        nothing -- a typo'd corpus name would otherwise produce an empty index
        and a plausible-looking eval run of all zeros.
        """

        paths = self.paths.resolved()
        registry = self.corpora.registry
        if not registry:
            registry = {self.IMPLICIT_CORPUS_NAME: CorpusConfig(documents_dir=paths.corpus_dir)}
            default_active = [self.IMPLICIT_CORPUS_NAME]
        else:
            default_active = self.corpora.active or sorted(registry)

        requested = list(names) if names else default_active
        unknown = [name for name in requested if name not in registry]
        if unknown:
            raise ValueError(
                f"Unknown corpus name(s): {', '.join(sorted(unknown))}. "
                f"Available: {', '.join(sorted(registry))}"
            )
        if not requested:
            raise ValueError("No corpora selected: set corpora.active or pass --corpus")

        # Sorted and deduped so `--corpus edgar --corpus baseline` and
        # `--corpus baseline --corpus edgar` resolve to the same index rather
        # than building two identical ones under different names.
        selected = tuple(sorted(set(requested)))
        return CorpusSelection(
            names=selected,
            document_dirs=tuple(
                (REPO_ROOT / registry[name].documents_dir).resolve() for name in selected
            ),
            base_collection=self.vector_store.collection_name,
            index_dir=paths.index_dir,
        )


def load_config(path: str | Path | None = None) -> RagConfig:
    """Load and validate config from a YAML file, falling back to defaults for missing keys.

    Passing no path loads `rag/config/config.yaml`. A missing file simply
    yields the default `RagConfig()` so the system runs out of the box.
    """

    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        return RagConfig()

    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    return RagConfig.model_validate(raw)
