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


def _load_with_raw_logits(cross_encoder_cls, model_name: str, identity):  # type: ignore[no-untyped-def]
    """Construct a `CrossEncoder` whose `predict()` returns unactivated logits.

    The keyword was renamed (`default_activation_function` -> `activation_fn`)
    in sentence-transformers 4, so try the current name and fall back rather
    than pinning a version for one argument.
    """

    for keyword in ("activation_fn", "default_activation_function"):
        try:
            return cross_encoder_cls(model_name, **{keyword: identity})
        except TypeError:
            continue
    # Neither keyword accepted: fall back to the default and set the attribute
    # directly, so an unexpected library version degrades to a warning rather
    # than a failed reranker.
    logger.warning(
        "sentence-transformers did not accept an activation override; reranker "
        "scores may be double-normalized for models defaulting to Sigmoid"
    )
    return cross_encoder_cls(model_name)


class CrossEncoderReranker(Reranker):
    """Rescores `(query, chunk)` pairs with a local sentence-transformers `CrossEncoder`.

    The model is loaded lazily on first use (not at construction time) so
    constructing a `CrossEncoderReranker` -- e.g. while wiring up the
    pipeline, or in tests that monkeypatch `_model` -- never requires
    downloading or holding several hundred MB of model weights in memory.
    """

    def __init__(
        self,
        model: str,
        *,
        aggregate: RerankAggregation = "max",
        query_prefix: str = "",
        document_prefix: str = "",
    ) -> None:
        self.model_name = model
        self.aggregate: RerankAggregation = aggregate
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self._model: object | None = None

    def _format(self, template: str, field: str, value: str) -> str:
        """Apply a prefix template, substituting `{query}`/`{document}` if present.

        Instruction-tuned rerankers expect their training template around each
        side of the pair; models trained on bare pairs want this to do nothing,
        which is what an empty template gives.
        """
        if not template:
            return value
        placeholder = "{" + field + "}"
        if placeholder in template:
            return template.replace(placeholder, value)
        return f"{template}{value}"

    def rerank(self, queries: list[str], candidates: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        if not candidates or not queries:
            return []

        # Every (query, candidate) pair goes through one predict() call: the
        # model batches internally, so N queries cost N× the pairs but not N×
        # the per-call overhead. That N× is the real price of reranking an
        # expanded query, and why `retrieval.expansion` isn't free after stage 1.
        count = len(candidates)
        pairs = [
            (
                self._format(self.query_prefix, "query", query),
                self._format(self.document_prefix, "document", candidate.text),
            )
            for query in queries
            for candidate in candidates
        ]
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
            from torch import nn

            logger.info("Loading cross-encoder reranker model %r (first use)", self.model_name)
            # Force raw logits out of `predict()`.
            #
            # sentence-transformers picks a per-model default activation, and it
            # is NOT the same across rerankers: `cross-encoder/ms-marco-*` uses
            # Identity (raw logits) while `BAAI/bge-reranker-*` uses Sigmoid.
            # Left alone, a BGE model returns values already in [0, 1] and
            # `normalize_rerank_score` sigmoids them a second time, crushing every
            # score into [0.5, 0.73]. Ranking order survives (sigmoid is
            # monotonic) so the damage is invisible in hit rate or NDCG -- but
            # `retrieval.min_score` becomes meaningless, and `aggregate: mean`
            # silently stops being log-odds pooling and starts averaging
            # probabilities.
            #
            # Normalizing in exactly one place is what makes the documented score
            # convention ("[0, 1], this reranker's own judgment") hold for every
            # model rather than for the one it was written against.
            self._model = _load_with_raw_logits(CrossEncoder, self.model_name, nn.Identity())
        return self._model
