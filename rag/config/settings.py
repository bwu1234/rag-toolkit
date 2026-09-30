"""Typed, config-driven settings for the RAG system.

All component choices (embedding model, vector store, reranker, LLM, chunking
parameters, etc.) live here as plain pydantic models loaded from a YAML file.
Swapping an implementation should mean: change a `provider` string in
config.yaml (and possibly add a small adapter class) — not editing pipeline
code.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

# Repo root = two levels up from this file (rag/config/settings.py -> repo/)
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"

# Top-level YAML key naming a config file to merge underneath this one (see
# `load_config`). Consumed at load time -- it never reaches `RagConfig`.
BASE_KEY = "base"

# How the generation prompt is phrased (`chat.prompt`); see
# `rag.generation.prompts`. Defined here so the config layer stays a leaf that
# the generation code imports from, not the other way round.
PromptStyle = Literal["grounded", "plain"]
#: Reasoning levels `llm.think` accepts besides on/off (Ollama passes them through).
ThinkLevel = Literal["low", "medium", "high", "xhigh"]
#: Values `llm.provider` accepts (the factory implements ollama and gemini).
LLMProvider = Literal["ollama", "gemini", "anthropic", "openai"]


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
    """One named body of documents.

    `clean: false` skips `rag.ingestion.cleaners.clean_text` for this corpus.
    For benchmark corpora whose text arrives already split and normalised
    (BEIR), where dehyphenation or whitespace rewriting would index a text the
    published scores were not computed on. It is a property of the corpus, not
    of a config file, so every config indexes that corpus the same way.
    Toggling it changes chunk text, which the incremental indexer's content
    hash already detects.
    """

    documents_dir: Path
    description: str = ""
    clean: bool = True


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
    # Task instruction for query vectors only, for instruction-tuned embedders;
    # documents are embedded without it. Each adapter uses its model family's
    # format: `ollama` sends `Instruct: {instruction}\nQuery:{query}`
    # (Qwen3-Embedding), `sentence_transformers` sends `{instruction} {query}`
    # (BGE). `null` embeds the bare query. Query-time only, so changing it
    # needs no reindex and it stays out of the index manifest.
    query_instruction: str | None = None
    # Hugging Face revision (commit) of a `sentence_transformers` model, so a
    # benchmark run names the exact weights. `null` takes the latest. Part of
    # the index manifest: different weights mean incomparable vectors.
    revision: str | None = None

    @model_validator(mode="after")
    def _revision_is_hugging_face_only(self) -> "EmbeddingConfig":
        # Ollama pins weights by tag, in `model`; a revision it silently
        # ignored would make a run look pinned when it isn't.
        if self.revision is not None and self.provider != "sentence_transformers":
            raise ValueError(
                f"embedding.revision pins a Hugging Face model; provider {self.provider!r} ignores it "
                "(pin an Ollama model by its tag in `model`)"
            )
        return self


#: Where `provider: gemini` points when `base_url` isn't set explicitly.
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com"


class LLMConfig(BaseModel):
    """Which chat/generation model to use and how to reach it.

    Default: Ollama-served `qwen3.5:9b-mlx`.
    """

    provider: LLMProvider = "ollama"
    model: str = "qwen3.5:9b-mlx"
    # Defaults to Ollama's; with `provider: gemini` and no explicit value it
    # becomes the Gemini API's (see `_provider_base_url`).
    base_url: str = "http://localhost:11434"
    temperature: float = 0.2
    max_tokens: int = 1024
    # qwen3.5 reasoning models spend `max_tokens` on a hidden "thinking" trace
    # before the real answer; on a long RAG prompt that can exhaust the
    # budget and leave `content` empty. Off by default for reliable answers.
    # A level string asks a model that supports one for that much reasoning
    # (the 27b takes low/medium/xhigh); Ollama only.
    think: bool | ThinkLevel = False
    # Per-request HTTP timeout. The default suits a generation over 5 passages;
    # a judge grading a long multi-hop answer on a shared GPU needs far more
    # (the 9b-vs-27b probe needed 900s for an 11k-token grading prompt).
    timeout_s: float = Field(default=120.0, gt=0, description="Per-request timeout in seconds")
    # Hosted providers only (ignored by ollama). The key is read from this
    # environment variable, never from a config file.
    api_key_env: str = "GEMINI_API_KEY"
    # The project's per-model free-tier limits, from AI Studio. The client paces
    # itself under both rather than firing and eating 429s. None = don't pace.
    requests_per_minute: int | None = Field(default=None, gt=0)
    tokens_per_minute: int | None = Field(default=None, gt=0)
    # The per-day request quota (Gemini only). With it set, the client counts
    # every request this machine sends to the model in `daily_request_log`
    # (relative to the repo) and refuses once the day's count reaches
    # `requests_per_day - requests_per_day_reserve`, so no one caller can spend
    # the whole day (see rag.generation.daily_budget). None = no daily count.
    requests_per_day: int | None = Field(default=None, gt=0)
    requests_per_day_reserve: int = Field(default=0, ge=0)
    daily_request_log: str = "data/logs/llm_daily_requests.json"
    # Gemini 3+ thinking models only (Flash-Lite 3.1/3.5). Thinking can't be
    # switched off there, and its tokens come out of `max_tokens`, so a deep
    # level on a long RAG prompt can leave no budget for the answer. None sends
    # no thinkingConfig, which non-thinking models (hosted Gemma) require: the
    # API rejects the field for them.
    thinking_level: Literal["minimal", "low", "medium", "high"] | None = None

    @model_validator(mode="after")
    def _provider_base_url(self) -> "LLMConfig":
        # The Ollama default URL is meaningless for a hosted provider, and making
        # every gemini stanza restate the endpoint invites a stale copy.
        if self.provider == "gemini" and "base_url" not in self.model_fields_set:
            self.base_url = GEMINI_BASE_URL
        return self

    @model_validator(mode="after")
    def _daily_budget_is_gemini_only(self) -> "LLMConfig":
        if self.requests_per_day is None:
            return self
        if self.provider != "gemini":
            raise ValueError(
                f"requests_per_day counts a hosted provider's daily quota; provider "
                f"{self.provider!r} has none and would ignore it"
            )
        if self.requests_per_day_reserve >= self.requests_per_day:
            raise ValueError(
                f"requests_per_day_reserve ({self.requests_per_day_reserve}) leaves nothing of "
                f"requests_per_day ({self.requests_per_day})"
            )
        return self

    @model_validator(mode="after")
    def _thinking_level_is_gemini_only(self) -> "LLMConfig":
        # Ollama would silently ignore it, and a run that believes it set a
        # reasoning level but didn't is a meaningless measurement.
        if self.thinking_level is not None and self.provider != "gemini":
            raise ValueError(
                f"thinking_level is a Gemini setting; provider {self.provider!r} ignores it "
                "(Ollama's reasoning switch is `think`)"
            )
        return self


class ContextualChunkingConfig(BaseModel):
    """Index-time enrichment: prepend a generated document-context blurb to each chunk.

    Off by default because it's the most expensive operation in the pipeline:
    one LLM call *per chunk* during `python -m rag.cli index`, each carrying a
    slice of the parent document. The payoff is that the cost is paid per
    index rather than per query, and it addresses a failure that query-time
    tuning can't -- a chunk that never names its own subject (see
    `rag.chunking.contextualizer`).

    Changing any of these values (other than `concurrency` and `cache`) changes
    what gets embedded, which the incremental skip -- keyed off chunk text
    alone -- can't see. The index manifest (`rag.index_manifest`) records them
    and refuses to extend an index built with different values until
    `index --reset`.
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


#: What `chunking.carry_metadata` defaults to, and what chunks carried before it existed.
DEFAULT_CARRY_METADATA = ("title", "page", "page_count")


class ChunkHeaderConfig(BaseModel):
    """A deterministic line naming each chunk's document, indexed ahead of its text.

    `template` is a `str.format` string over the document's metadata, e.g.
    `"{company} ({ticker}) {form}, period ended {period_end}"`. `null` (the
    default) adds no header. A document missing any field the template names
    gets no header rather than a half-filled one. The header is indexed
    (embedded and BM25) and shown to the answering model, but `Chunk.text`
    stays verbatim, as with contextual chunking. Changing it needs
    `index --reset`; the index manifest enforces that.
    """

    template: str | None = None


class ChunkingConfig(BaseModel):
    """Parameters for splitting documents into retrievable chunks.

    `fixed` = simple character-based windows with overlap. Deliberately the
    simplest strategy that could work; structure-aware/semantic chunking is a
    later milestone once the end-to-end pipeline is proven out.

    `none` = no splitting: each document becomes exactly one chunk with its
    text unchanged, and `chunk_size`/`chunk_overlap` are ignored. For corpora
    that arrive pre-split into retrieval units (BEIR), where splitting a
    passage would let one document fill several ranks.
    """

    strategy: Literal["fixed", "none"] = "fixed"
    chunk_size: int = Field(default=1000, gt=0, description="Target characters per chunk")
    chunk_overlap: int = Field(default=150, ge=0, description="Characters of overlap between consecutive chunks")
    contextual: ContextualChunkingConfig = ContextualChunkingConfig()
    # Document metadata keys copied onto each of its chunks, where the vector
    # store and BM25 index keep them. Keys a document lacks are skipped. Dates
    # are stored as YYYYMMDD integers: Chroma takes only primitives, and an
    # integer still supports range filters.
    carry_metadata: list[str] = Field(default_factory=lambda: list(DEFAULT_CARRY_METADATA))
    header: ChunkHeaderConfig = ChunkHeaderConfig()


#: Chroma's own default `ef_search`, which every collection used before
#: `vector_store.hnsw_ef_search` existed.
DEFAULT_HNSW_EF_SEARCH = 100


class VectorStoreConfig(BaseModel):
    """Vector store selection and connection details.

    Default: Chroma in persistent local mode (single directory on disk, no
    server process). Bundles vector search with metadata storage, which keeps
    the dependency count low for a local-first setup.
    """

    provider: Literal["chroma"] = "chroma"
    collection_name: str = "rag_corpus"
    # Width of the HNSW search beam: how many candidates the approximate
    # nearest-neighbour search explores per query (raised to `top_k` if lower).
    # Wider finds more of the exact top k and costs query time; it changes no
    # stored vector, so it needs no reindex and stays out of the index manifest.
    hnsw_ef_search: int = Field(default=DEFAULT_HNSW_EF_SEARCH, ge=1)


class SparseIndexConfig(BaseModel):
    """Keyword (BM25) index backend, maintained beside the vector store by `index`.

    - ``bm25``: rank_bm25 in memory, persisted as one JSON file. Rebuilds the
      whole model after any change, so it degrades beyond ~10^4 chunks.
    - ``sqlite_fts5``: an SQLite FTS5 table, updated in place and ranked by
      SQLite's `bm25()`. Scales to large corpora, but its BM25 parameters
      differ slightly from rank_bm25's, so rankings are not identical.
      Needs SQLite >= 3.43, which on Linux is the OS's library.

    Each backend keeps its own file, so switching needs no `--reset`: the next
    `index` run fills the new one (re-embedding every chunk, because the skip
    check requires a chunk in both stores).
    """

    provider: Literal["bm25", "sqlite_fts5"] = "bm25"


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
    # Score the passage with its chunk header (`chunking.header`) in front, so
    # the cross-encoder can see the company and period. Off by default: it
    # changes what every reranker measurement means. No effect on chunks
    # without a header.
    include_header: bool = False
    # Tokens per (query, passage) pair; longer pairs are truncated. null uses
    # the model's own limit (8,192 for bge-reranker-v2-m3), which the shipped
    # 1,000-char chunks never approach. Set it for corpora of long passages:
    # attention memory grows with the square of the length, and scoring 100
    # long BEIR FiQA passages at the full limit exhausted memory.
    max_length: int | None = Field(default=None, gt=0)


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


class DocumentRoutingConfig(BaseModel):
    """Pick the filing first, then rank chunks only inside it (chunking plan, Phase 3b).

    Each indexed document gets one short record, rendered from
    `record_template`, and a query is ranked against those records by BM25 and
    by dense similarity. When the two agree on the top document, chunk
    retrieval is filtered to the `top_m` best-fused documents; when they
    disagree, retrieval runs unfiltered. That agreement is the only gate: a
    router that is wrong filters the answer out entirely, so it acts only when
    two independent rankers pick the same document.

    `top_m: null` (the default) turns routing off. An explicit caller filter
    always wins over routing.

    `record_template` is a `str.format` string over the chunk header
    (`{header}`) and the document's carried metadata (`chunking.carry_metadata`).
    A `:date` spec spells a stored date the way questions say it, e.g.
    `{period_end:date}` -> "February 15, 2026". A document missing any named
    field has no record and can never be routed to. Query-time only: changing
    either setting needs no reindex.
    """

    top_m: int | None = Field(default=None, gt=0, description="Documents to route to; null disables routing")
    record_template: str = "{header}; period ended {period_end:date}"


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
    document_routing: DocumentRoutingConfig = DocumentRoutingConfig()


ChatMode = Literal["pipeline", "agentic"]
AgentStrategy = Literal["react", "planned"]


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

    # `pipeline`: retrieve once, then generate (everything above and in `crag`).
    # `agentic`: the model searches as a tool, as often as it needs, under the
    # `agent` section's guards. The condenser and CRAG's grader and retries
    # don't run in agentic mode: the model sees the conversation and writes
    # its own queries (Milestone 19).
    mode: ChatMode = "pipeline"
    condense_history: bool = True
    max_history_turns: int = Field(
        default=6,
        ge=0,
        description="Most recent turns fed to the condenser (keeps the rewrite prompt bounded)",
    )
    # `grounded` numbers the passages, asks for inline `[n]` citations and tells
    # the model to say so when the passages don't answer. `plain` is the
    # textbook "context + question" template with none of that -- a baseline for
    # measuring what the grounding instructions buy, not something to serve.
    # Under `plain` the answer carries no `[n]` markers, so a turn's
    # `cited_chunk_ids` is always empty.
    prompt: PromptStyle = "grounded"


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


class AgentConfig(BaseModel):
    """Agentic retrieval (Milestone 19): the model searches as a tool, as often as it needs.

    Read only when `chat.mode: agentic` builds the agent
    (`rag.generation.agent`); the pipeline never reads it. See
    `docs/milestone-19-plan.md` for why each guard exists -- every one answers
    a failure the prototype showed.
    """

    # The agent's own model; None uses `llm`. The measured split is a 27b for
    # the loop (it iterates; the 9b stops after one search round) and the 9b
    # for the condenser, contextualizer and CRAG calls, which gain nothing
    # from paying 27b latency.
    llm: LLMConfig | None = None
    # `react`: the model decides after every result whether to search again.
    # `planned`: one call plans sub-queries, all run with no model call in
    # between, then one synthesis call -- built for a model that decomposes
    # well but won't take a second round on its own.
    strategy: AgentStrategy = "react"
    max_tool_calls: int = Field(
        default=8,
        ge=1,
        description="Searches allowed per turn; at the cap the model gets one tool-free turn to answer",
    )
    # A default for measurement, not a decision about interactive use: hard
    # questions took minutes with the 27b, and a 60 s budget would cut off the
    # runs phase 4 exists to measure. Whether agentic mode serves the UI (and
    # so wants ~60 s) is an open question in the plan.
    # Checked before every model call and search, so it stops the searching;
    # an in-flight call isn't interrupted, and the forced-synthesis turn that
    # follows is one more call. It bounds when the agent stops looking, not
    # the turn's exact length.
    timeout_s: float = Field(default=600.0, gt=0, description="Wall-clock budget for one agent turn's searching")
    # Matches the MCP server's `DEFAULT_MAX_CHARS` (a test holds them equal):
    # agent prompts grew ~6x over the pipeline's in the prototype.
    max_passage_chars: int = Field(default=1200, ge=1, description="Per-passage character cap in search results")
    # Sent on every agent call. Ollama truncates a prompt longer than its
    # context window silently, and the shipped adapter otherwise sends no
    # window at all; 32768 is what the prototype ran with.
    num_ctx: int = Field(default=32768, ge=1024, description="Context window requested for agent calls (Ollama)")


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
    sparse_index: SparseIndexConfig = SparseIndexConfig()
    reranker: RerankerConfig = RerankerConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    chat: ChatConfig = ChatConfig()
    crag: CragConfig = CragConfig()
    eval: EvalConfig = EvalConfig()
    agent: AgentConfig = AgentConfig()
    observability: ObservabilityConfig = ObservabilityConfig()

    @model_validator(mode="after")
    def _plain_prompt_cannot_regenerate(self) -> "RagConfig":
        # CRAG regeneration answers under `REGROUND_SYSTEM_PROMPT`, a stricter
        # restatement of the *grounded* rules. With `chat.prompt: plain` a
        # rejected answer would silently switch prompt style mid-turn, so a run
        # meant to measure the plain prompt would partly measure the grounded
        # one. Grading and the groundedness verdict alone are fine.
        crag = self.crag
        if self.chat.prompt == "plain" and crag.enabled and crag.check_groundedness and crag.max_regenerations:
            raise ValueError(
                "chat.prompt: plain can't be combined with CRAG regeneration, which answers under "
                "the grounded prompt; set crag.max_regenerations: 0 (the groundedness check still runs)"
            )
        return self

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


ENV_OVERRIDE_PREFIX = "RAG__"


def _apply_env_overrides(raw: dict[str, Any], environ: Mapping[str, str]) -> None:
    """Overlay `RAG__SECTION__KEY=value` environment variables onto the parsed YAML.

    Exists for deployment-specific values the shared YAML can't know -- chiefly
    `llm.base_url` / `embedding.base_url`, which are `localhost` on a laptop but
    the `ollama` service name inside docker-compose. Keeping the override in the
    environment means there is no second config file to drift from this one.

    Values stay strings; pydantic coerces them (`"5"` -> int, `"false"` -> bool)
    during validation. Lists and mappings can't be expressed this way -- set
    those in YAML.
    """

    for name, value in environ.items():
        if not name.startswith(ENV_OVERRIDE_PREFIX):
            continue
        keys = [part.lower() for part in name[len(ENV_OVERRIDE_PREFIX):].split("__")]
        if not all(keys):
            raise ValueError(f"Malformed config override {name!r}: expected RAG__SECTION__KEY")
        node = raw
        for key in keys[:-1]:
            child = node.get(key)
            if child is None:
                child = node[key] = {}
            elif not isinstance(child, dict):
                raise ValueError(f"Config override {name!r} descends into non-mapping key {key!r}")
            node = child
        node[keys[-1]] = value


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Return `base` with `override` laid over it; neither input is mutated.

    Mappings merge key by key, recursively. Anything else -- scalars, lists,
    `null` -- replaces the base value outright. Lists replace rather than
    concatenate on purpose: `corpora.active: [edgar]` over a base of
    `[baseline]` means "edgar", not "baseline and edgar" (which would be a
    different, pooled index).
    """

    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, Mapping):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def _read_config_yaml(path: Path, chain: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Parse `path`, first merging in the file its `base:` key names, if any.

    `base` resolves relative to the file that names it (not the working
    directory), so `base: config.yaml` in `rag/config/vanilla.yaml` works from
    anywhere. Bases can chain. Unlike the top-level file, a named base that
    doesn't exist raises: silently falling back to defaults under a typo'd
    base would run a different pipeline than the one asked for.
    """

    resolved = path.resolve()
    if resolved in chain:
        cycle = " -> ".join(str(p) for p in (*chain, resolved))
        raise ValueError(f"Config `{BASE_KEY}` chain loops: {cycle}")

    with resolved.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config file {resolved} must contain a mapping at the top level")

    base = raw.pop(BASE_KEY, None)
    if base is None:
        return raw
    if not isinstance(base, str):
        raise ValueError(f"Config `{BASE_KEY}` in {resolved} must be a file path, got {base!r}")

    base_path = resolved.parent / base
    if not base_path.exists():
        raise FileNotFoundError(f"Config `{BASE_KEY}` in {resolved} names a missing file: {base_path}")
    return _deep_merge(_read_config_yaml(base_path, (*chain, resolved)), raw)


def load_config(path: str | Path | None = None) -> RagConfig:
    """Load and validate config from a YAML file, falling back to defaults for missing keys.

    Passing no path loads `rag/config/config.yaml`. A missing file simply
    yields the default `RagConfig()` so the system runs out of the box.

    A file may start with `base: <path>` to inherit another config and list
    only what it changes (see `_read_config_yaml` / `_deep_merge`) -- e.g.
    `rag/config/vanilla.yaml` is `config.yaml` with the extras switched off.

    `RAG__SECTION__KEY` environment variables override individual keys on top
    of the merged result (see `_apply_env_overrides`).
    """

    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    raw = _read_config_yaml(config_path) if config_path.exists() else {}

    _apply_env_overrides(raw, os.environ)
    return RagConfig.model_validate(raw)
