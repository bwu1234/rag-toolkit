"""Retrieval quality metrics for the RAG evaluation pipeline.

All functions are pure — they operate on lists of ids and return scalars.
No I/O, no dependencies beyond the stdlib.

Retrieval is evaluated at the **document level**: a retrieved chunk is
considered relevant if its ``document_id`` appears in the sample's
``expected_doc_ids`` list, regardless of which specific chunk was returned.
This is deliberately lenient — for a corpus of page-granularity chunks it
ensures a hit anywhere in the right document counts, which is appropriate for
open-domain QA where any passage from the source could be helpful.

Metrics
-------
``hit_rate``
    1.0 if at least one relevant document appears in the top-k results,
    else 0.0.  Averaged across samples this equals Recall@k in the binary
    relevance sense (one relevant document per query).

``recall_at_k``
    Fraction of expected documents covered by the top-k results.  Equal to
    ``hit_rate`` when there is exactly one expected document per query, but
    more informative when a query has multiple ground-truth sources.

``precision_at_k``
    Fraction of the top-k results that are relevant.

``reciprocal_rank``
    1 / rank of the first relevant result (0.0 if none found within k).
    Average across samples = MRR (Mean Reciprocal Rank).
"""

from __future__ import annotations


def hit_rate(retrieved_doc_ids: list[str], expected_doc_ids: list[str]) -> float:
    """1.0 if any expected document appears in ``retrieved_doc_ids``, else 0.0."""
    if not expected_doc_ids:
        return 0.0
    expected = set(expected_doc_ids)
    return 1.0 if any(doc_id in expected for doc_id in retrieved_doc_ids) else 0.0


def recall_at_k(retrieved_doc_ids: list[str], expected_doc_ids: list[str]) -> float:
    """Fraction of ``expected_doc_ids`` covered by ``retrieved_doc_ids``."""
    if not expected_doc_ids:
        return 0.0
    expected = set(expected_doc_ids)
    hits = sum(1 for doc_id in expected if doc_id in set(retrieved_doc_ids))
    return hits / len(expected)


def precision_at_k(retrieved_doc_ids: list[str], expected_doc_ids: list[str]) -> float:
    """Fraction of ``retrieved_doc_ids`` that are relevant."""
    if not retrieved_doc_ids:
        return 0.0
    expected = set(expected_doc_ids)
    hits = sum(1 for doc_id in retrieved_doc_ids if doc_id in expected)
    return hits / len(retrieved_doc_ids)


def reciprocal_rank(retrieved_doc_ids: list[str], expected_doc_ids: list[str]) -> float:
    """1 / rank of the first relevant result; 0.0 if none found."""
    if not expected_doc_ids:
        return 0.0
    expected = set(expected_doc_ids)
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in expected:
            return 1.0 / rank
    return 0.0


def mean(values: list[float]) -> float:
    """Arithmetic mean of a non-empty list; returns 0.0 for an empty list."""
    return sum(values) / len(values) if values else 0.0
