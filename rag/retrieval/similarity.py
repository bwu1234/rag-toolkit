"""Pure cosine similarity, shared by anything scoring raw (query, content) pairs
outside the `VectorStore` -- currently only `SearxNGWebSearch`, which scores
live search snippets that were never indexed and so never pass through
Chroma's own cosine-distance query path.
"""

from __future__ import annotations

import math


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two equal-length vectors, in `[-1, 1]`.

    Returns `0.0` for a zero vector rather than raising, since an all-zero
    embedding (an edge case, not something any real embedder should produce)
    has no defined direction to compare.
    """

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
