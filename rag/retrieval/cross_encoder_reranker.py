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

from rag.retrieval.reranker import Reranker, normalize_rerank_score, rescored
from rag.vectorstore.base import ScoredChunk

logger = logging.getLogger(__name__)


class CrossEncoderReranker(Reranker):
    """Rescores `(query, chunk)` pairs with a local sentence-transformers `CrossEncoder`.

    The model is loaded lazily on first use (not at construction time) so
    constructing a `CrossEncoderReranker` -- e.g. while wiring up the
    pipeline, or in tests that monkeypatch `_model` -- never requires
    downloading or holding several hundred MB of model weights in memory.
    """

    def __init__(self, model: str) -> None:
        self.model_name = model
        self._model: object | None = None

    def rerank(self, query: str, candidates: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        if not candidates:
            return []

        pairs = [(query, candidate.text) for candidate in candidates]
        raw_scores = self._cross_encoder.predict(pairs)

        rescored_candidates = [
            rescored(candidate, normalize_rerank_score(float(raw_score)))
            for candidate, raw_score in zip(candidates, raw_scores)
        ]
        rescored_candidates.sort(key=lambda chunk: chunk.score, reverse=True)
        # Ranking only -- the relevance floor lives in `Retriever` (config:
        # `retrieval.min_score`) so it applies whichever reranker is selected,
        # including `NoOpReranker`. Filtering here would leave pure vector
        # retrieval with no floor at all.
        return rescored_candidates[:top_k]

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
