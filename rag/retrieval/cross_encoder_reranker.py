"""sentence-transformers `CrossEncoder`-backed `Reranker` adapter.

Unlike the embedder and chat model, this is the one component that runs a
local ML model directly in-process rather than going through Ollama --
sentence-transformers ships no Ollama-compatible serving path for cross-
encoders. That's the deliberate tradeoff behind `RerankerConfig.provider =
"cross_encoder"`: meaningfully better relevance ranking than pure vector
similarity, at the cost of pulling in `sentence-transformers` (and `torch`),
the heaviest dependency in the project. `provider: "none"` remains available
for anyone who'd rather not pay that cost.
"""

from __future__ import annotations

import logging
import os
from typing import Literal

from rag.retrieval.reranker import Reranker, normalize_rerank_score, rescored
from rag.vectorstore.base import ScoredChunk

logger = logging.getLogger(__name__)

RerankAggregation = Literal["max", "mean"]


class CrossEncoderReranker(Reranker):
    """Rescores `(query, chunk)` pairs with a local sentence-transformers `CrossEncoder`.

    The model is loaded lazily on first use (not at construction time) so
    constructing a `CrossEncoderReranker` -- e.g. while wiring up the
    pipeline, or in tests that monkeypatch `_model` -- never requires
    downloading or holding several hundred MB of model weights in memory.
    """

    def __init__(self, model: str, *, aggregate: RerankAggregation = "max") -> None:
        self.model_name = model
        self.aggregate: RerankAggregation = aggregate
        self._model: object | None = None

    def rerank(self, queries: list[str], candidates: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        if not candidates or not queries:
            return []

        # Every (query, candidate) pair goes through one predict() call: the
        # model batches internally, so N queries cost N× the pairs but not N×
        # the per-call overhead. That N× is the real price of reranking an
        # expanded query, and why `retrieval.expansion` isn't free after stage 1.
        count = len(candidates)
        pairs = [(query, candidate.text) for query in queries for candidate in candidates]
        raw_scores = [float(score) for score in self._cross_encoder.predict(pairs)]

        # `pairs` is query-major: candidate i under query j sits at j*count + i.
        scores_per_candidate = [
            [raw_scores[query_index * count + index] for query_index in range(len(queries))]
            for index in range(count)
        ]

        rescored_candidates = [
            rescored(candidate, normalize_rerank_score(self._combine(scores)))
            for candidate, scores in zip(candidates, scores_per_candidate)
        ]
        rescored_candidates.sort(key=lambda chunk: chunk.score, reverse=True)
        # Ranking only -- the relevance floor lives in `Retriever` (config:
        # `retrieval.min_score`) so it applies whichever reranker is selected,
        # including `NoOpReranker`. Filtering here would leave pure vector
        # retrieval with no floor at all.
        return rescored_candidates[:top_k]

    def _combine(self, scores: list[float]) -> float:
        """Fold one candidate's per-query logits into a single relevance score.

        Combining happens in *logit* space, before the sigmoid, so there's
        exactly one normalization point. `max` is monotonic through the sigmoid
        so the choice is immaterial there; averaging log-odds is the standard
        way to pool independent judgments and is what `mean` does.

        - `max` (default) — the candidate is relevant if *any* phrasing of the
          question judges it relevant. This is what makes multi-query expansion
          pay off: a chunk the user's original wording scores near zero can be
          rescued by a rephrasing that uses the corpus's vocabulary.
        - `mean` — the candidate must satisfy the phrasings *on average*. More
          conservative, and it punishes a chunk that only one rewrite liked --
          useful when a drifting rewrite keeps dragging in off-topic material.
        """

        if self.aggregate == "mean":
            return sum(scores) / len(scores)
        return max(scores)

    @property
    def _cross_encoder(self):  # type: ignore[no-untyped-def]
        if self._model is None:
            # Imported lazily, not at module load time -- `sentence-transformers`
            # (and the `torch` it pulls in) is the heaviest dependency in the
            # project, and importing it costs real time even when the
            # configured reranker provider is `"none"`.
            # `USE_TF=0` steers `transformers` (a sentence-transformers dep) away
            # from its TensorFlow integration -- this project only ever uses the
            # torch backend, and a TF install with a mismatched Keras version
            # would otherwise crash the import.
            os.environ.setdefault("USE_TF", "0")
            from sentence_transformers import CrossEncoder

            logger.info("Loading cross-encoder reranker model %r (first use)", self.model_name)
            self._model = CrossEncoder(self.model_name)
        return self._model
