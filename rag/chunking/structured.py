"""Structure-aware chunking over Markdown (`chunking.strategy: structured`).

Chunking plan Phase 5. `FixedSizeChunker` cuts every `chunk_size` characters
wherever that lands, which on EDGAR puts 690 chunk starts inside a table
with the table's header row in an earlier chunk. A row without its column
headers is a bare string of figures, and those questions pass half as often
(measured results, "`table` tier baseline"). This chunker cuts on the
structure Phase 4's renderer recovered instead:

1. **Blocks.** `parse_blocks` splits `Document.text` into ATX headings, pipe
   tables and paragraphs, with character offsets. A small hand-written pass,
   because that is all the Markdown the loaders emit; a Markdown library would
   be a dependency for three regular expressions.
2. **Units.** A block that fits in `chunk_size` is one unit. An oversized
   table is split by rows, and every piece after the first repeats the header
   rows (through the `|---|` line), so no piece is a headerless run of
   figures. An oversized paragraph is split at sentence ends, then at words,
   with `chunk_overlap` between the pieces.
3. **Packing.** Units are packed greedily up to `chunk_size`. A heading at or
   above `split_level` always starts a new chunk, unless what came before is
   under `min_chars` (a stub section joins its next sibling). A heading never
   ends a chunk: it moves forward with the unit it introduces.

`Chunk.text` stays a verbatim span of the document (`char_start`:`char_end`)
except for a repeated table header, which is prefixed to it: the answering
model has to see the column headers, so they can't live only in index text.
Span matching is by substring, so a quoted row still matches.

Sizes are characters, via an injectable `length` function, for the same
reason `FixedSizeChunker` uses characters: every chunk here is far below the
embedder's token limit. A hosted embedder with a hard token limit
(Milestone 18) would pass a token counter.
"""

from __future__ import annotations

import logging
import re
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Iterable, Literal

from rag.chunking.chunkers import FixedSizeChunker, carried_metadata, render_header, template_fields
from rag.chunking.models import Chunk, Chunker, make_chunk_id
from rag.config.settings import DEFAULT_CARRY_METADATA
from rag.ingestion.models import Document

logger = logging.getLogger(__name__)

BlockKind = Literal["heading", "table", "paragraph"]

_HEADING = re.compile(r"^(#{1,6})[ \t]+\S")
# A Markdown table's header/body separator: `|---|:--:|` and so on.
_TABLE_SEPARATOR = re.compile(r"^\s*\|(\s*:?-+:?\s*\|)+\s*$")
# End of a sentence: terminal punctuation, optional closing quotes or
# brackets, then whitespace. The match's end is where the next sentence starts.
_SENTENCE_END = re.compile(r"[.!?][\"')\]’”]*\s+")

#: The separator `section_path` joins heading texts with.
SECTION_SEPARATOR = " > "


@dataclass(frozen=True)
class Block:
    """One structural element of a document, as a span of its text.

    Attributes:
        kind: `heading`, `table` or `paragraph` (list items are paragraphs).
        start: Offset of the block's first character in `Document.text`.
        end: Offset just past its last non-whitespace character.
        level: A heading's level (number of `#`); 0 for other kinds.
        header_end: For a table, the offset just past its header rows (the
            `|---|` separator line when there is one, else the first row).
        rows: For a table, the (start, end) span of each row after the header.
    """

    kind: BlockKind
    start: int
    end: int
    level: int = 0
    header_end: int = 0
    rows: tuple[tuple[int, int], ...] = ()


def _lines(text: str) -> list[tuple[int, int]]:
    """(start, end) of every line in `text`, end excluding the newline."""

    spans = []
    start = 0
    for line in text.split("\n"):
        spans.append((start, start + len(line)))
        start += len(line) + 1
    return spans


def _is_row(line: str) -> bool:
    return line.lstrip().startswith("|")


def parse_blocks(text: str) -> list[Block]:
    """Split Markdown `text` into headings, tables and paragraphs, in order.

    Blank lines end a paragraph but not a table: the `edgar` fetcher writes a
    blank line after every pipe row, and those rows are still one table.
    """

    lines = _lines(text)
    blocks: list[Block] = []
    i = 0
    while i < len(lines):
        start, end = lines[i]
        line = text[start:end]
        if not line.strip():
            i += 1
            continue
        heading = _HEADING.match(line)
        if heading:
            blocks.append(Block("heading", start, end, level=len(heading.group(1))))
            i += 1
        elif _is_row(line):
            rows = [lines[i]]
            i += 1
            while i < len(lines):
                j = i
                while j < len(lines) and not text[lines[j][0] : lines[j][1]].strip():
                    j += 1
                if j < len(lines) and _is_row(text[lines[j][0] : lines[j][1]]):
                    rows.append(lines[j])
                    i = j + 1
                else:
                    break
            blocks.append(_table(text, rows))
        else:
            para_end = end
            i += 1
            while i < len(lines):
                s, e = lines[i]
                following = text[s:e]
                if not following.strip() or _HEADING.match(following) or _is_row(following):
                    break
                para_end = e
                i += 1
            blocks.append(Block("paragraph", start, _rstrip(text, start, para_end)))
    return blocks


def _rstrip(text: str, start: int, end: int) -> int:
    while end > start and text[end - 1].isspace():
        end -= 1
    return end


def _table(text: str, rows: list[tuple[int, int]]) -> Block:
    """A table block from its row lines, with the header rows split off."""

    header_rows = 1
    for index, (s, e) in enumerate(rows):
        if _TABLE_SEPARATOR.match(text[s:e]):
            header_rows = index + 1
            break
    if header_rows >= len(rows):  # nothing but header: treat the first row as it
        header_rows = min(1, len(rows))
    body = tuple((s, _rstrip(text, s, e)) for s, e in rows[header_rows:])
    return Block(
        "table",
        rows[0][0],
        _rstrip(text, *rows[-1]),
        header_end=_rstrip(text, *rows[header_rows - 1]),
        rows=body,
    )


@dataclass
class _Unit:
    """A span the packer places whole: a block, or one piece of a split block.

    `prefix` is text prepended to the span (a repeated table header); a unit
    with one always starts a chunk. `starts_chunk` is set on pieces that
    overlap their predecessor, which can't share a chunk with it.
    """

    kind: BlockKind
    start: int
    end: int
    path: tuple[str, ...]
    level: int = 0
    prefix: str | None = None
    starts_chunk: bool = False
    #: The whole block this unit is, or None for a piece of a split block.
    block: Block | None = None


@dataclass
class _Draft:
    """A chunk being packed: a contiguous span, plus an optional prefix."""

    units: list[_Unit] = field(default_factory=list)
    overlap_start: int | None = None

    @property
    def start(self) -> int:
        return self.overlap_start if self.overlap_start is not None else self.units[0].start

    @property
    def end(self) -> int:
        return self.units[-1].end

    @property
    def prefix(self) -> str | None:
        return self.units[0].prefix if self.units else None

    def text(self, document_text: str) -> str:
        body = document_text[self.start : self.end]
        return f"{self.prefix}\n{body}" if self.prefix else body

    def only_headings(self) -> bool:
        return all(unit.kind == "heading" for unit in self.units)


class StructuredChunker(Chunker):
    """Chunks Markdown on headings, table edges and paragraph breaks.

    Args:
        chunk_size: Most characters (per `length`) in a chunk. Exceeded only
            when a block that fits alone brings its lead-in with it (trailing
            headings, or a stub under `min_chars`), and then by at most that
            lead-in. An oversized block's first piece leaves room for it.
        chunk_overlap: Characters shared by consecutive pieces of an oversized
            paragraph, and, with `prose_overlap`, by chunks split between two
            paragraphs of one section. Never across a heading or table edge.
        split_level: Headings of this level or higher (fewer `#`) always start
            a chunk. Deeper headings are packed with their siblings.
        min_chars: A section shorter than this joins its next sibling instead
            of becoming a chunk of its own, and a final chunk this short joins
            its predecessor when both fit.
        prose_overlap: Whether `chunk_overlap` also applies where a chunk is
            split between paragraphs.
        length: Measures chunk size. Characters by default.
    """

    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        *,
        split_level: int = 2,
        min_chars: int = 200,
        prose_overlap: bool = False,
        carry_metadata: Iterable[str] = DEFAULT_CARRY_METADATA,
        header_template: str | None = None,
        length: Callable[[str], int] = len,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be smaller than chunk_size ({chunk_size}), "
                "otherwise an oversized paragraph never advances"
            )
        if min_chars >= chunk_size:
            raise ValueError(f"min_chars ({min_chars}) must be smaller than chunk_size ({chunk_size})")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.split_level = split_level
        self.min_chars = min_chars
        self.prose_overlap = prose_overlap
        self.carry_metadata = tuple(carry_metadata)
        if header_template is not None:
            template_fields(header_template)
        self.header_template = header_template
        self.length = length

    def chunk(self, documents: Iterable[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        document_count = 0
        headerless = 0
        for document in documents:
            document_count += 1
            header = render_header(self.header_template, document) if self.header_template else None
            if self.header_template and header is None:
                headerless += 1
            chunks.extend(self._chunk_one(document, header))
        logger.info("Chunked %d document(s) into %d structured chunk(s)", document_count, len(chunks))
        if headerless:
            logger.info(
                "%d of %d document(s) lack a field chunking.header.template names; "
                "their chunks have no header",
                headerless,
                document_count,
            )
        return chunks

    def _chunk_one(self, document: Document, header: str | None) -> list[Chunk]:
        text = document.text
        blocks = parse_blocks(text)
        if not blocks:
            return []
        inherited = carried_metadata(document, self.carry_metadata)
        drafts = self._pack(text, list(self._units(text, blocks)))
        result = []
        for index, draft in enumerate(drafts):
            result.append(
                Chunk(
                    id=make_chunk_id(document.id, index),
                    text=draft.text(text),
                    document_id=document.id,
                    source=document.source,
                    doc_type=document.doc_type,
                    metadata={
                        **inherited,
                        "chunk_index": index,
                        "char_start": draft.start,
                        "char_end": draft.end,
                        "section_path": _section_path(draft),
                        "block_types": ",".join(sorted({unit.kind for unit in draft.units})),
                    },
                    header=header,
                )
            )
        return result

    # -- units ---------------------------------------------------------------

    def _units(self, text: str, blocks: list[Block]) -> Iterable[_Unit]:
        """Blocks as whole units, with the heading path each sits under."""

        stack: list[tuple[int, str]] = []
        for position, block in enumerate(blocks):
            if block.kind == "heading":
                title = text[block.start : block.end].lstrip("#").strip()
                # A leading level-1 heading is the document's own title, which
                # the chunk header already names; keep it out of the path.
                is_document_title = position == 0 and block.level == 1
                while stack and stack[-1][0] >= block.level:
                    stack.pop()
                if not is_document_title:
                    stack.append((block.level, title))
            path = tuple(name for _, name in stack)
            yield _Unit(block.kind, block.start, block.end, path, level=block.level, block=block)

    def _split(self, text: str, unit: _Unit, room: int) -> list[_Unit]:
        """An oversized block as pieces, the first at most `room`, the rest `chunk_size`."""

        assert unit.block is not None
        if unit.kind == "table":
            return list(self._table_pieces(text, unit.block, unit.path, room))
        return list(self._prose_pieces(text, unit.start, unit.end, unit.path, room))

    def _table_pieces(self, text: str, block: Block, path: tuple[str, ...], room: int) -> Iterable[_Unit]:
        """An oversized table split by rows; pieces after the first repeat the header rows."""

        header = text[block.start : block.header_end]
        piece_start = block.start  # the first piece holds the header in place
        prefix: str | None = None
        last_end: int | None = None
        budget = room
        for row_start, row_end in block.rows:
            if self.length(f"{header}\n{text[row_start:row_end]}") > self.chunk_size:
                # One row too big for a chunk even alone (a rendering artifact:
                # a whole list flattened into a cell). Close what's open and
                # split the row by words, each piece under the header.
                header_in_place = last_end is None and prefix is None
                if last_end is not None:
                    yield _Unit("table", piece_start, last_end, path, prefix=prefix, starts_chunk=prefix is not None)
                pieces = self._prose_pieces(
                    text, row_start, row_end, path, self.chunk_size, reserve=self.length(header) + 1
                )
                for index, piece in enumerate(pieces):
                    if index == 0 and header_in_place:  # the header precedes this row in the text
                        yield _Unit("table", block.start, piece.end, path)
                    else:
                        yield _Unit("table", piece.start, piece.end, path, prefix=header, starts_chunk=True)
                piece_start, prefix, last_end, budget = row_end, header, None, self.chunk_size
                continue
            if last_end is None:
                piece_start = row_start if prefix is not None else piece_start
            else:
                candidate = text[piece_start:row_end]
                if self.length(f"{prefix}\n{candidate}" if prefix else candidate) > budget:
                    yield _Unit("table", piece_start, last_end, path, prefix=prefix, starts_chunk=prefix is not None)
                    piece_start, prefix, budget = row_start, header, self.chunk_size
            last_end = row_end
        if last_end is not None:
            yield _Unit("table", piece_start, last_end, path, prefix=prefix, starts_chunk=prefix is not None)
        elif not block.rows:  # header rows only; nothing to split on
            yield _Unit("table", block.start, block.end, path)

    def _prose_pieces(
        self, text: str, start: int, stop: int, path: tuple[str, ...], room: int, *, reserve: int = 0
    ) -> Iterable[_Unit]:
        """`text[start:stop]` split at sentence ends, else words, with `chunk_overlap` between pieces.

        The first piece is at most `room`, the rest `chunk_size`, each less
        `reserve` (room kept for a prefix the caller adds).
        """

        boundaries = [m.end() for m in _SENTENCE_END.finditer(text, start, stop)]
        boundaries.append(stop)
        budget = room - reserve
        first = True
        previous_end = start  # each piece must end past the last, or overlap could stall it
        while start < stop:
            fitting = [
                b
                for b in boundaries
                if _rstrip(text, start, b) > previous_end and self.length(text[start:b].rstrip()) <= budget
            ]
            if fitting:
                end = _rstrip(text, start, fitting[-1]) if fitting[-1] < stop else stop
            else:  # no sentence end fits: fall back to words
                end = min(stop, start + budget)
                if end < stop:
                    end = _rstrip(text, start, FixedSizeChunker._snap_backward_to_boundary(text, end))
                if end <= previous_end:  # no break anywhere in the window: cut mid-word
                    end = min(stop, start + budget)
            yield _Unit("paragraph", start, end, path, starts_chunk=not first)
            if end >= stop:
                return
            first = False
            previous_end = end
            budget = self.chunk_size - reserve
            next_start = self._overlap_start(text, start, end) if self.chunk_overlap else end
            while next_start < stop and text[next_start].isspace():
                next_start += 1
            start = next_start if next_start > start else end

    def _overlap_start(self, text: str, floor: int, end: int, overlap: int | None = None) -> int:
        """Where a span sharing `overlap` (default `chunk_overlap`) characters with `[.., end)` starts.

        Snapped forward to a word, so the overlap never begins mid-word, and
        never before `floor`. Returns `end` (no overlap) when there's no room.
        """

        overlap = self.chunk_overlap if overlap is None else overlap
        if overlap <= 0:
            return end
        target = max(floor + 1, end - overlap)
        return FixedSizeChunker._snap_forward_to_boundary(text, target, limit=end)

    # -- packing -------------------------------------------------------------

    def _size(self, text: str, draft: _Draft, unit: _Unit | None = None) -> int:
        end = unit.end if unit is not None else draft.end
        body = text[draft.start : end]
        return self.length(f"{draft.prefix}\n{body}" if draft.prefix else body)

    def _oversized(self, text: str, unit: _Unit) -> bool:
        return unit.block is not None and self.length(text[unit.start : unit.end]) > self.chunk_size

    def _pack(self, text: str, units: list[_Unit]) -> list[_Draft]:
        drafts: list[_Draft] = []
        current = _Draft()
        queue = deque(units)
        while queue:
            unit = queue.popleft()
            if not current.units:
                if self._oversized(text, unit):
                    pieces = self._split(text, unit, self.chunk_size)
                    queue.extendleft(reversed(pieces[1:]))
                    unit = pieces[0]
                current.units.append(unit)
                continue
            if unit.kind == "heading" and unit.level <= self.split_level:
                if current.only_headings() or self._size(text, current) < self.min_chars:
                    current.units.append(unit)  # a stub section joins its next sibling
                else:
                    drafts.append(current)
                    current = _Draft([unit])
                continue
            fits = unit.prefix is None and not unit.starts_chunk and self._size(text, current, unit) <= self.chunk_size
            if fits:
                current.units.append(unit)
                continue

            # Close the draft. What leads into the unit moves forward with it:
            # trailing headings always, and the whole draft if it's a stub
            # (a lead-in line such as "(amounts in millions)").
            lead = self._lead(text, current, unit)
            rest = current.units[: len(current.units) - len(lead)]
            if rest:
                drafts.append(_Draft(rest, current.overlap_start))
            previous = drafts[-1] if drafts and rest else None
            if lead and self._oversized(text, unit):
                lead_size = self.length(text[lead[0].start : unit.start])
                pieces = self._split(text, unit, max(self.chunk_size - lead_size, self.chunk_size // 2))
                queue.extendleft(reversed(pieces[1:]))
                unit = pieces[0]
            elif not lead and self._oversized(text, unit):
                pieces = self._split(text, unit, self.chunk_size)
                queue.extendleft(reversed(pieces[1:]))
                unit = pieces[0]
            current = _Draft(lead + [unit])
            if not lead and previous is not None and self._shares_prose(previous, unit):
                # Only as much overlap as the chunk has room for: overlap is
                # context, not a reason to exceed the cap.
                last = previous.units[-1]
                room = self.chunk_size - self.length(text[last.end : unit.end])
                start = self._overlap_start(text, last.start, last.end, min(self.chunk_overlap, room))
                if start < last.end:
                    current.overlap_start = start
        if current.units:
            drafts.append(current)
        return self._merge_tail(text, drafts)

    def _lead(self, text: str, current: _Draft, unit: _Unit) -> list[_Unit]:
        """The units at the end of `current` that belong with `unit` in the next chunk."""

        if unit.prefix is not None or unit.starts_chunk:
            lead: list[_Unit] = []  # a continuation piece has its own context
        elif current.prefix is None and current.overlap_start is None and not current.units[0].starts_chunk and (
            self._size(text, current) < self.min_chars
        ):
            lead = list(current.units)
        else:
            lead = []
        if not lead:
            count = 0
            while count < len(current.units) and current.units[-1 - count].kind == "heading":
                count += 1
            lead = current.units[len(current.units) - count :] if count else []
            if lead and lead[0] is current.units[0] and current.prefix is not None:
                lead = []  # never strand a prefix
        return lead

    def _shares_prose(self, previous: _Draft, unit: _Unit) -> bool:
        """Whether `prose_overlap` applies between `previous` and a chunk starting at `unit`."""

        last = previous.units[-1]
        return (
            self.prose_overlap
            and self.chunk_overlap > 0
            and unit.kind == "paragraph"
            and unit.prefix is None
            and not unit.starts_chunk
            and last.kind == "paragraph"
            and last.path == unit.path
        )

    def _merge_tail(self, text: str, drafts: list[_Draft]) -> list[_Draft]:
        """Fold a final stub (or heading-only chunk) into its predecessor when both fit."""

        if len(drafts) < 2:
            return drafts
        last, previous = drafts[-1], drafts[-2]
        stub = last.only_headings() or self._size(text, last) < self.min_chars
        if not stub or last.prefix is not None or last.overlap_start is not None or last.units[0].starts_chunk:
            return drafts
        merged = _Draft(previous.units + last.units, previous.overlap_start)
        if self._size(text, merged) > self.chunk_size and not last.only_headings():
            return drafts
        return drafts[:-2] + [merged]


def _section_path(draft: _Draft) -> str:
    """The heading path the chunk's content shares, else its first content's.

    Content, not headings: a chunk opening with the document title would
    otherwise share only the empty path.
    """

    content = [unit for unit in draft.units if unit.kind != "heading"] or draft.units
    paths = [unit.path for unit in content]
    common: list[str] = []
    for names in zip(*paths):
        if len(set(names)) != 1:
            break
        common.append(names[0])
    return SECTION_SEPARATOR.join(common or paths[0])
