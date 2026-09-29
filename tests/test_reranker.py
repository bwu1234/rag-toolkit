"""Tests for the Reranker interface, NoOpReranker, CrossEncoderReranker, and factory.

`CrossEncoderReranker` is tested with its `_model` swapped for a tiny fake
that returns deterministic raw scores -- this exercises the real rescoring,
sigmoid-normalization, and re-sorting logic without downloading or running an
actual sentence-transformers model (the heaviest dependency in the project).
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import pytest

from rag.config.settings import RerankerConfig
from rag.retrieval.cross_encoder_reranker import CrossEncoderReranker
from rag.retrieval.factory import get_reranker
from rag.retrieval.reranker import NoOpReranker, Reranker, normalize_rerank_score, rescored
from rag.vectorstore.base import ScoredChunk


def _scored(chunk_id: str, text: str = "text", score: float = 0.5) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk_id,
        text=text,
        document_id="doc.md",
        source=Path("doc.md"),
        doc_type="markdown",
        score=score,
        metadata={},
    )


class _FakeCrossEncoder:
    """Returns one fixed raw logit per input text, looked up by text content."""

    def __init__(self, scores_by_text: dict[str, float]) -> None:
        self.scores_by_text = scores_by_text
        self.seen_pairs: list[tuple[str, str]] = []

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.seen_pairs.extend(pairs)
        return [self.scores_by_text[text] for _query, text in pairs]


# ---------------------------------------------------------------------------
# normalize_rerank_score / rescored helpers
# ---------------------------------------------------------------------------


def test_normalize_rerank_score_maps_logits_into_unit_interval() -> None:
    assert normalize_rerank_score(0.0) == pytest.approx(0.5)
    assert 0.0 < normalize_rerank_score(-10.0) < 0.5
    assert 0.5 < normalize_rerank_score(10.0) < 1.0
    # Symmetric around zero, as a sigmoid should be.
    assert normalize_rerank_score(2.0) + normalize_rerank_score(-2.0) == pytest.approx(1.0)


def test_rescored_returns_a_copy_with_new_score() -> None:
    original = _scored("a", score=0.1)

    updated = rescored(original, 0.9)

    assert updated.score == 0.9
    assert original.score == 0.1, "ScoredChunk is frozen -- rescoring must not mutate the original"
    assert updated.chunk_id == original.chunk_id
    assert updated.text == original.text


# ---------------------------------------------------------------------------
# NoOpReranker
# ---------------------------------------------------------------------------


def test_noop_reranker_preserves_order_and_scores() -> None:
    candidates = [_scored("a", score=0.9), _scored("b", score=0.5), _scored("c", score=0.1)]

    results = NoOpReranker().rerank(["query"], candidates, top_k=10)

    assert results == candidates


def test_noop_reranker_truncates_to_top_k() -> None:
    candidates = [_scored("a"), _scored("b"), _scored("c")]

    results = NoOpReranker().rerank(["query"], candidates, top_k=2)

    assert [r.chunk_id for r in results] == ["a", "b"]


# ---------------------------------------------------------------------------
# CrossEncoderReranker
# ---------------------------------------------------------------------------


def _cross_encoder_with(scores_by_text: dict[str, float]) -> tuple[CrossEncoderReranker, _FakeCrossEncoder]:
    reranker = CrossEncoderReranker(model="fake/cross-encoder")
    fake = _FakeCrossEncoder(scores_by_text)
    reranker._model = fake
    return reranker, fake


def test_cross_encoder_reranker_reorders_by_normalized_score() -> None:
    candidates = [_scored("low", "low-relevance text"), _scored("high", "high-relevance text")]
    reranker, fake = _cross_encoder_with({"low-relevance text": -5.0, "high-relevance text": 5.0})

    results = reranker.rerank(["query"], candidates, top_k=10)

    assert [r.chunk_id for r in results] == ["high", "low"]
    assert results[0].score > results[1].score
    assert all(0.0 < r.score < 1.0 for r in results)
    assert fake.seen_pairs == [("query", "low-relevance text"), ("query", "high-relevance text")]


def test_cross_encoder_reranker_scores_match_sigmoid_of_raw_logits() -> None:
    candidates = [_scored("a", "text-a")]
    reranker, _fake = _cross_encoder_with({"text-a": 2.0})

    [result] = reranker.rerank(["query"], candidates, top_k=10)

    assert result.score == pytest.approx(1.0 / (1.0 + math.exp(-2.0)))


def test_cross_encoder_reranker_truncates_to_top_k() -> None:
    candidates = [_scored("a", "ta"), _scored("b", "tb"), _scored("c", "tc")]
    reranker, _fake = _cross_encoder_with({"ta": 1.0, "tb": 3.0, "tc": 2.0})

    results = reranker.rerank(["query"], candidates, top_k=2)

    assert [r.chunk_id for r in results] == ["b", "c"]


def test_cross_encoder_reranker_on_empty_candidates_skips_model() -> None:
    reranker = CrossEncoderReranker(model="fake/cross-encoder")

    assert reranker.rerank(["query"], [], top_k=5) == []
    assert reranker._model is None, "the model should never be loaded for an empty candidate list"


def test_cross_encoder_reranker_loads_model_lazily(monkeypatch: pytest.MonkeyPatch) -> None:
    constructed: list[str] = []

    class _StubCrossEncoder:
        def __init__(self, model_name: str) -> None:
            constructed.append(model_name)

        def predict(self, pairs):
            return [0.0 for _ in pairs]

    import sys
    import types

    fake_module = types.ModuleType("sentence_transformers")
    fake_module.CrossEncoder = _StubCrossEncoder  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    reranker = CrossEncoderReranker(model="org/my-cross-encoder")
    assert constructed == [], "constructing the reranker must not load the model"

    reranker.rerank(["q"], [_scored("a", "ta")], top_k=1)

    assert constructed == ["org/my-cross-encoder"]
    # A second call must reuse the cached instance, not reconstruct it.
    reranker.rerank(["q"], [_scored("a", "ta")], top_k=1)
    assert constructed == ["org/my-cross-encoder"]


def test_cross_encoder_is_loaded_with_raw_logits(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: the model must not normalize its own scores before we do.

    sentence-transformers picks a per-model default activation, and it differs
    between rerankers: `cross-encoder/ms-marco-*` uses Identity (raw logits)
    while `BAAI/bge-reranker-*` uses Sigmoid. Left alone, a BGE model returns
    values already in [0, 1] and `normalize_rerank_score` sigmoids them a second
    time, crushing every score into [0.5, 0.73].

    Ranking order survives (sigmoid is monotonic), so this is invisible in hit
    rate or NDCG -- which is exactly why it needs a test. What breaks is
    `retrieval.min_score`, which is calibrated against the score *scale*, and
    `aggregate: mean`, which silently stops being log-odds pooling.
    """
    seen: dict[str, object] = {}

    class _StubCrossEncoder:
        def __init__(self, model_name: str, activation_fn=None) -> None:  # type: ignore[no-untyped-def]
            seen["model"] = model_name
            seen["activation"] = activation_fn

        def predict(self, pairs):  # type: ignore[no-untyped-def]
            # A raw logit of 0.0; one sigmoid puts it at exactly 0.5.
            return [0.0 for _ in pairs]

    import sys
    import types

    fake_module = types.ModuleType("sentence_transformers")
    fake_module.CrossEncoder = _StubCrossEncoder  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    reranker = CrossEncoderReranker(model="BAAI/bge-reranker-base")
    [result] = reranker.rerank(["q"], [_scored("a", "ta")], top_k=1)

    assert type(seen["activation"]).__name__ == "Identity", (
        "the model must be loaded with an identity activation so predict() "
        "returns raw logits"
    )
    assert result.score == pytest.approx(0.5), (
        "a raw logit of 0.0 must normalize to 0.5; 0.622 would mean the score "
        "was sigmoided twice"
    )


@pytest.mark.parametrize(("max_length", "expected"), [(None, {}), (512, {"max_length": 512})])
def test_max_length_reaches_the_model_only_when_set(
    monkeypatch: pytest.MonkeyPatch, max_length: int | None, expected: dict[str, int]
) -> None:
    seen: dict[str, object] = {}

    class _StubCrossEncoder:
        def __init__(self, model_name: str, activation_fn=None, **kwargs) -> None:  # type: ignore[no-untyped-def]
            seen.update(kwargs)

        def predict(self, pairs):  # type: ignore[no-untyped-def]
            return [0.0 for _ in pairs]

    import sys
    import types

    fake_module = types.ModuleType("sentence_transformers")
    fake_module.CrossEncoder = _StubCrossEncoder  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    get_reranker(RerankerConfig(provider="cross_encoder", model="m", max_length=max_length)).rerank(
        ["q"], [_scored("a", "ta")], top_k=1
    )
    assert seen == expected


def test_prefixes_default_to_leaving_pairs_untouched() -> None:
    """Models trained on bare pairs (ms-marco, BGE) must see exactly the pair."""
    reranker = CrossEncoderReranker(model="m")
    fake = _PairKeyedCrossEncoder({("what is x?", "text a"): 1.0})
    reranker._model = fake

    reranker.rerank(["what is x?"], [_scored("a", "text a")], top_k=1)

    assert fake.seen_pairs == [("what is x?", "text a")]


def test_prefix_templates_substitute_placeholders() -> None:
    """Instruction-tuned rerankers need their training template around each side."""
    reranker = CrossEncoderReranker(
        model="m",
        query_prefix="<Instruct>: rank\n<Query>: {query}",
        document_prefix="<Document>: {document}",
    )
    expected = ("<Instruct>: rank\n<Query>: what is x?", "<Document>: text a")
    fake = _PairKeyedCrossEncoder({expected: 1.0})
    reranker._model = fake

    reranker.rerank(["what is x?"], [_scored("a", "text a")], top_k=1)

    assert fake.seen_pairs == [expected]


def test_prefix_without_a_placeholder_is_prepended() -> None:
    reranker = CrossEncoderReranker(model="m", query_prefix="query: ", document_prefix="passage: ")
    expected = ("query: what is x?", "passage: text a")
    fake = _PairKeyedCrossEncoder({expected: 1.0})
    reranker._model = fake

    reranker.rerank(["what is x?"], [_scored("a", "text a")], top_k=1)

    assert fake.seen_pairs == [expected]


def test_cross_encoder_load_falls_back_when_activation_kwarg_is_unsupported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unexpected sentence-transformers version must degrade, not crash."""

    class _OldCrossEncoder:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        def predict(self, pairs):  # type: ignore[no-untyped-def]
            return [0.0 for _ in pairs]

    import sys
    import types

    fake_module = types.ModuleType("sentence_transformers")
    fake_module.CrossEncoder = _OldCrossEncoder  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    reranker = CrossEncoderReranker(model="org/legacy")
    [result] = reranker.rerank(["q"], [_scored("a", "ta")], top_k=1)
    assert result.score == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# get_reranker factory
# ---------------------------------------------------------------------------


def test_get_reranker_factory_selects_none() -> None:
    reranker = get_reranker(RerankerConfig(provider="none", model="unused"))

    assert isinstance(reranker, Reranker)
    assert isinstance(reranker, NoOpReranker)


def test_get_reranker_factory_selects_cross_encoder() -> None:
    reranker = get_reranker(RerankerConfig(provider="cross_encoder", model="org/model"))

    assert isinstance(reranker, Reranker)
    assert isinstance(reranker, CrossEncoderReranker)
    assert reranker.model_name == "org/model"
    assert reranker._model is None, "constructing via the factory must not load the model"


def test_get_reranker_factory_rejects_unknown_provider() -> None:
    config = RerankerConfig.model_construct(provider="bm25", model="m")

    with pytest.raises(ValueError, match="Unknown reranker provider"):
        get_reranker(config)


# ---------------------------------------------------------------------------
# Reranking against expanded queries
# ---------------------------------------------------------------------------


class _PairKeyedCrossEncoder:
    """Returns a raw logit per (query, text) pair -- needed to test aggregation."""

    def __init__(self, scores: dict[tuple[str, str], float]) -> None:
        self.scores = scores
        self.seen_pairs: list[tuple[str, str]] = []

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.seen_pairs.extend(pairs)
        return [self.scores[pair] for pair in pairs]


def _pair_keyed(scores: dict[tuple[str, str], float], *, aggregate: str = "max") -> CrossEncoderReranker:
    reranker = CrossEncoderReranker(model="fake/cross-encoder", aggregate=aggregate)  # type: ignore[arg-type]
    reranker._model = _PairKeyedCrossEncoder(scores)
    return reranker


def test_rerank_scores_every_candidate_against_every_query() -> None:
    reranker = _pair_keyed({
        ("q1", "ta"): 1.0, ("q1", "tb"): 2.0,
        ("q2", "ta"): 3.0, ("q2", "tb"): 4.0,
    })

    reranker.rerank(["q1", "q2"], [_scored("a", "ta"), _scored("b", "tb")], top_k=5)

    assert reranker._model.seen_pairs == [  # type: ignore[attr-defined]
        ("q1", "ta"), ("q1", "tb"), ("q2", "ta"), ("q2", "tb")
    ]


def test_max_aggregation_rescues_a_chunk_only_a_rephrasing_liked() -> None:
    # The whole point of reranking against expanded queries: the user's original
    # wording scores "b" near zero, a rephrasing using the corpus's vocabulary
    # scores it highly, and max lets that rescue it.
    reranker = _pair_keyed({
        ("original", "ta"): 1.0, ("original", "tb"): -5.0,
        ("rephrasing", "ta"): 0.5, ("rephrasing", "tb"): 6.0,
    })

    results = reranker.rerank(["original", "rephrasing"], [_scored("a", "ta"), _scored("b", "tb")], top_k=5)

    assert [r.chunk_id for r in results] == ["b", "a"]


def test_max_aggregation_uses_the_highest_logit_per_candidate() -> None:
    reranker = _pair_keyed({("q1", "ta"): -2.0, ("q2", "ta"): 3.0})

    [result] = reranker.rerank(["q1", "q2"], [_scored("a", "ta")], top_k=5)

    assert result.score == pytest.approx(1.0 / (1.0 + math.exp(-3.0)))


def test_mean_aggregation_averages_logits_before_the_sigmoid() -> None:
    reranker = _pair_keyed({("q1", "ta"): -2.0, ("q2", "ta"): 4.0}, aggregate="mean")

    [result] = reranker.rerank(["q1", "q2"], [_scored("a", "ta")], top_k=5)

    assert result.score == pytest.approx(1.0 / (1.0 + math.exp(-1.0)))


def test_mean_aggregation_suppresses_a_chunk_only_one_query_liked() -> None:
    # Same scores as the max test; mean reaches the opposite conclusion, which
    # is the tradeoff the knob exists to express.
    reranker = _pair_keyed({
        ("original", "ta"): 1.0, ("original", "tb"): -5.0,
        ("rephrasing", "ta"): 0.5, ("rephrasing", "tb"): 6.0,
    }, aggregate="mean")

    results = reranker.rerank(["original", "rephrasing"], [_scored("a", "ta"), _scored("b", "tb")], top_k=5)

    assert [r.chunk_id for r in results] == ["a", "b"]


def test_single_query_rerank_is_unaffected_by_aggregation() -> None:
    # Guards the unexpanded path: one query means max and mean are the same op.
    scores = {("q", "ta"): 2.0, ("q", "tb"): -1.0}
    by_max = _pair_keyed(scores).rerank(["q"], [_scored("a", "ta"), _scored("b", "tb")], top_k=5)
    by_mean = _pair_keyed(scores, aggregate="mean").rerank(
        ["q"], [_scored("a", "ta"), _scored("b", "tb")], top_k=5
    )

    assert [(r.chunk_id, r.score) for r in by_max] == [(r.chunk_id, r.score) for r in by_mean]


def test_rerank_on_empty_queries_returns_nothing_without_loading_the_model() -> None:
    reranker = CrossEncoderReranker(model="fake/cross-encoder")

    assert reranker.rerank([], [_scored("a", "ta")], top_k=5) == []
    assert reranker._model is None


def test_reranker_scores_the_header_only_when_configured() -> None:
    headed = dataclasses.replace(_scored("a", "text a"), header="Apple 10-K")

    plain = CrossEncoderReranker(model="m")
    plain_fake = _PairKeyedCrossEncoder({("q", "text a"): 1.0})
    plain._model = plain_fake
    plain.rerank(["q"], [headed], top_k=1)

    with_header = CrossEncoderReranker(model="m", include_header=True)
    header_fake = _PairKeyedCrossEncoder({("q", "Apple 10-K\n\ntext a"): 1.0, ("q", "text b"): 0.0})
    with_header._model = header_fake
    with_header.rerank(["q"], [headed, _scored("b", "text b")], top_k=2)

    assert plain_fake.seen_pairs == [("q", "text a")], "default: the pair is unchanged"
    assert header_fake.seen_pairs == [("q", "Apple 10-K\n\ntext a"), ("q", "text b")]


def test_get_reranker_passes_include_header() -> None:
    reranker = get_reranker(RerankerConfig(provider="cross_encoder", include_header=True))

    assert isinstance(reranker, CrossEncoderReranker)
    assert reranker.include_header is True
