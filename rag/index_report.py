"""Describe what the configured pipeline makes of a corpus, and whether the index matches.

The read side of `rag.cli index`, and the validation step of the indexing
pipeline in embryo (see docs/chunking-indexing-plan.md, stage 8): chunk counts
and sizes, the chunk-boundary defects this corpus is known to have, and whether
the built index is current with the corpus and the config. It reports; it does
not gate. Thresholds that fail the command come later, once a baseline says
where they belong.

Everything here is computed from the corpus with the configured chunker, the
same path `index` takes, so running it before and after a chunking change shows
what the change did without paying for embeddings.
"""

from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from rag.chunking.models import Chunk
from rag.ingestion.models import Document

#: Percentiles reported for the chunk-size distribution.
SIZE_PERCENTILES = (10, 50, 90, 99)


@dataclass
class IndexState:
    """How the built index compares with the corpus and the config."""

    #: The stored manifest, or None when the index predates manifests.
    manifest: dict[str, Any] | None
    #: `section.key: stored -> configured` lines; empty when they agree.
    manifest_differences: list[str]
    vector_chunks: int
    sparse_chunks: int
    #: Chunks the corpus produces that the index lacks, in either store.
    missing: int
    #: Chunks the index holds that the corpus no longer produces.
    stale: int
    #: Chunks whose stored content hash differs from the corpus's text.
    changed: int

    @property
    def in_sync(self) -> bool:
        return not (self.missing or self.stale or self.changed or self.manifest_differences)


@dataclass
class IndexReport:
    corpus: str
    chunking: dict[str, Any]
    documents: int
    chunks: int
    mean_chars: float
    #: Chunk size in characters at the min, each of SIZE_PERCENTILES, and max.
    size_percentiles: dict[str, int]
    min_chars: int
    under_floor: int
    max_chars: int
    over_cap: int
    #: Chunks holding at least one table row.
    table_chunks: int
    #: Chunks that begin inside a table, past its first row: they lost the header.
    mid_table_starts: int
    #: The subset of `mid_table_starts` that begin partway through a row.
    mid_row_starts: int
    #: Documents that produced no chunks (empty after cleaning).
    empty_documents: list[str] = field(default_factory=list)
    #: Distinct chunk texts that occur more than once.
    duplicate_groups: int = 0
    #: Chunks whose exact text occurs more than once, counting every copy.
    duplicate_chunks: int = 0
    index: IndexState | None = None


def is_table_row(line: str) -> bool:
    """Whether ``line`` is a table row as the loaders emit them (pipe-delimited)."""
    return line.lstrip().startswith("|")


def table_start(text: str, start: int) -> str | None:
    """Where a chunk starting at ``start`` in ``text`` falls relative to a table.

    Returns ``"mid_row"`` when it starts partway through a table row,
    ``"later_row"`` when it starts at the beginning of a row that isn't the
    table's first, and ``None`` otherwise (outside a table, or at its first
    row). Either non-None value means the table's header row is in an earlier
    chunk. Blank lines between rows don't end a table: the EDGAR fetcher puts
    one after every row.
    """
    while start < len(text) and text[start].isspace():
        start += 1
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", start)
    if not is_table_row(text[line_start : line_end if line_end != -1 else len(text)]):
        return None
    if text[line_start:start].strip():
        return "mid_row"
    previous = text[:line_start].rstrip().rsplit("\n", 1)[-1]
    return "later_row" if is_table_row(previous) else None


def build_report(
    *,
    corpus: str,
    chunking: dict[str, Any],
    documents: list[Document],
    chunks: list[Chunk],
    min_chars: int,
    max_chars: int,
    index: IndexState | None = None,
) -> IndexReport:
    """Compute the report for ``chunks``, which ``documents`` (cleaned) produced."""
    sizes = sorted(len(chunk.text) for chunk in chunks)
    text_by_doc = {document.id: document.text for document in documents}

    positions = Counter(
        table_start(text_by_doc[chunk.document_id], int(chunk.metadata.get("char_start", 0)))
        for chunk in chunks
        if chunk.document_id in text_by_doc
    )
    copies = Counter(chunk.text for chunk in chunks)
    chunked_docs = {chunk.document_id for chunk in chunks}

    return IndexReport(
        corpus=corpus,
        chunking=chunking,
        documents=len(documents),
        chunks=len(chunks),
        mean_chars=statistics.fmean(sizes) if sizes else 0.0,
        size_percentiles=_percentiles(sizes),
        min_chars=min_chars,
        under_floor=sum(1 for size in sizes if size < min_chars),
        max_chars=max_chars,
        over_cap=sum(1 for size in sizes if size > max_chars),
        table_chunks=sum(1 for chunk in chunks if any(is_table_row(line) for line in chunk.text.splitlines())),
        mid_table_starts=positions["mid_row"] + positions["later_row"],
        mid_row_starts=positions["mid_row"],
        empty_documents=sorted(document.id for document in documents if document.id not in chunked_docs),
        duplicate_groups=sum(1 for count in copies.values() if count > 1),
        duplicate_chunks=sum(count for count in copies.values() if count > 1),
        index=index,
    )


def _percentiles(sizes: list[int]) -> dict[str, int]:
    if not sizes:
        return {}
    result = {"min": sizes[0]}
    if len(sizes) > 1:
        # `inclusive` treats the data as the whole population rather than a
        # sample, so p99 of a small corpus never extrapolates past its max.
        cuts = statistics.quantiles(sizes, n=100, method="inclusive")
        result.update({f"p{p}": round(cuts[p - 1]) for p in SIZE_PERCENTILES})
    result["max"] = sizes[-1]
    return result


def format_report(report: IndexReport) -> str:
    """Human-readable rendering, one fact per line."""
    pct = lambda part: f"{part / report.chunks:.1%}" if report.chunks else "n/a"  # noqa: E731
    settings = "  ".join(f"{key}={value}" for key, value in report.chunking.items())
    sizes = "  ".join(f"{name}={value}" for name, value in report.size_percentiles.items())
    lines = [
        f"Corpus: {report.corpus}",
        f"Chunking: {settings}",
        "",
        f"Documents          {report.documents}",
        f"Chunks             {report.chunks}",
        f"Chunk chars        {sizes}  mean={report.mean_chars:.0f}",
        f"{f'Under {report.min_chars} chars':<19}{report.under_floor} ({pct(report.under_floor)})",
        f"{f'Over {report.max_chars} chars':<19}{report.over_cap} ({pct(report.over_cap)})",
        f"With table rows    {report.table_chunks} ({pct(report.table_chunks)})",
        f"Start mid-table    {report.mid_table_starts} ({pct(report.mid_table_starts)}): "
        f"{report.mid_row_starts} partway through a row, "
        f"{report.mid_table_starts - report.mid_row_starts} at a later row",
        f"Zero-chunk docs    {len(report.empty_documents)}"
        + (f": {', '.join(report.empty_documents)}" if report.empty_documents else ""),
        f"Duplicate chunks   {report.duplicate_chunks} ({report.duplicate_groups} distinct text(s))",
        "",
    ]
    state = report.index
    if state is None:
        lines.append("Index: not built")
        return "\n".join(lines)
    lines += [
        f"Index              {state.vector_chunks} vector / {state.sparse_chunks} BM25 chunk(s)",
        f"  vs. corpus       {state.missing} missing, {state.stale} stale, {state.changed} changed text",
    ]
    if state.manifest is None:
        lines.append("  manifest         none (built before manifests; settings unverified)")
    elif state.manifest_differences:
        lines.append("  manifest         DIFFERS from config:")
        lines += [f"    {line}" for line in state.manifest_differences]
    else:
        lines.append("  manifest         matches config")
    lines.append(f"  in sync          {'yes' if state.in_sync else 'NO -- re-run `rag.cli index`'}")
    return "\n".join(lines)
