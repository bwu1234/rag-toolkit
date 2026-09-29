"""In-process `EmbeddingModel` adapter over the `sentence-transformers` library.

Runs the model in this process instead of behind a daemon. Added for the
public benchmarks plan, whose reference dense runs use `BAAI/bge-base-en-v1.5`,
a model Ollama doesn't serve with its published recipe (CLS pooling, 512
tokens, L2-normalised). A sentence-transformers checkpoint carries its own
pooling and truncation settings (`modules.json`, `sentence_bert_config.json`),
so loading it by name reproduces the model card's recipe without restating
it here.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from rag.embedding.base import EmbeddingModel

logger = logging.getLogger(__name__)

_DEFAULT_BATCH_SIZE = 32


class SentenceTransformersEmbedder(EmbeddingModel):
    """Embeds text with a local `SentenceTransformer` model.

    Vectors are always L2-normalised. The vector store ranks by cosine, which
    normalisation doesn't change, and it makes the stored vectors comparable
    with inner-product references (Pyserini's `--l2-norm` Faiss runs).
    """

    def __init__(
        self,
        model: str,
        dimensions: int | None = None,
        *,
        revision: str | None = None,
        query_instruction: str | None = None,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ) -> None:
        self.model_name = model
        self.revision = revision
        self.query_instruction = query_instruction
        self.batch_size = batch_size
        self._dimensions = dimensions
        self._model: Any = None

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._encode(texts)

    def embed_query(self, text: str) -> list[float]:
        # A plain prefix, joined by one space: the BGE model card's format and
        # Pyserini's `--query-prefix`. Not the Qwen `Instruct: ...\nQuery:`
        # template `OllamaEmbedder` builds -- each model family was trained on
        # its own.
        if self.query_instruction is not None:
            text = f"{self.query_instruction} {text}"
        return self._encode([text])[0]

    @property
    def dimensions(self) -> int:
        if self._dimensions is None:
            model = self._sentence_transformer
            # Renamed in sentence-transformers 5; pyproject still allows >= 3.
            get_dimension = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
            dims = get_dimension()
            if dims is None:
                raise RuntimeError(f"{self.model_name!r} does not report its embedding dimensionality")
            self._dimensions = int(dims)
        return self._dimensions

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self._sentence_transformer.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        result: list[list[float]] = vectors.tolist()
        return result

    @property
    def _sentence_transformer(self) -> Any:
        if self._model is None:
            # Imported lazily, as in `CrossEncoderReranker`: torch is the
            # heaviest import in the project, and building this adapter (in
            # tests, or to read `dimensions` from config) shouldn't pay for it.
            os.environ.setdefault("USE_TF", "0")
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, revision=self.revision)
            logger.info(
                "Loaded embedding model %r (revision %s) on %s, max_seq_length %s",
                self.model_name,
                self.revision or "latest",
                self._model.device,
                self._model.max_seq_length,
            )
        return self._model
