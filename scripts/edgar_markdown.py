"""Render a span of an EDGAR HTML filing as Markdown (chunking plan, Phase 4).

``fetch_edgar.html_to_text`` flattens a filing to text: headings become plain
lines, and table cells become `` | ``-joined lines with no header row. So a
chunker can't find a section boundary or a table's header. This module
re-reads the same HTML and keeps that structure:

* **Headings** become ``##``-``####`` lines. Filings almost never use ``<h*>``
  tags. A heading is a short block set entirely in a style more prominent
  than the filing's body text (bold, larger, underlined, italic, or
  coloured), so detection is relative to each filing's own body style.
* **Tables** become Markdown tables: one line per ``<tr>``, cell contents
  flattened inline (some filers wrap every cell in ``<p>``, which the text
  flattener turns into one line per cell), layout-only columns dropped, and a
  ``|---|`` line after the header rows.
* **Page furniture** is dropped: page numbers, "Table of Contents" links, and
  running page headers that repeat on every page.

What it renders is fixed by the corpus on disk, not re-selected:
``fetch_edgar.pinned_span`` locates each document's body in the
``html_to_text`` output, and :func:`render_span` renders exactly the HTML
that produced that text. The mapping goes through :func:`parse`, which
records where every character of the flattened text came from.

Heuristic, like MD&A extraction: tuned against this corpus's 14 filers, and
checked by the heading-precision sample and the span-presence check before
it counts as done (see ``docs/chunking-indexing-plan.md``, Phase 4).
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from html.parser import HTMLParser

# Tags the flattened text puts line breaks around, and the ones it skips.
# Changing either changes `fetch_edgar.html_to_text`, and so the text every
# eval set was labeled against: `fetch_edgar.py --cache-raw` checks it still
# reproduces the corpus.
BLOCK_TAGS = frozenset(
    {"p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table", "section", "article", "header", "footer"}
)
SKIP_TAGS = frozenset({"script", "style", "head", "title"})
_VOID_TAGS = frozenset({"br", "hr", "img", "meta", "link", "input", "col", "area", "base", "wbr"})

# Typographic characters `html_to_text` rewrites, so BM25 sees one spelling.
PUNCTUATION = (
    ("\xa0", " "), ("’", "'"), ("‘", "'"),
    ("“", '"'), ("”", '"'),
    ("—", "--"), ("–", "-"), ("…", "..."),
)

# A heading is a label, not a sentence. Longer than this, or ending in a
# sentence's punctuation, and a bold block is emphasized prose instead.
MAX_HEADING_CHARS = 120
# Heading levels below the document title (`#`): `##` to `####`. A filing
# with more heading styles than this puts the least prominent ones at `####`
# rather than dropping them.
MAX_HEADING_LEVELS = 3

_PAGE_NUMBER = re.compile(r"^(?:page\s+)?[-–—]?\s*[ivxlc\d]{1,4}\s*[-–—]?$", re.IGNORECASE)
_TABLE_OF_CONTENTS = re.compile(r"^(?:return\s+to\s+|back\s+to\s+)?table\s+of\s+contents$", re.IGNORECASE)


# --------------------------------------------------------------------------
# Parse: a minimal element tree that also records html_to_text's offsets
# --------------------------------------------------------------------------


@dataclass
class Text:
    """A text node, and where it starts in the raw (pre-normalization) stream `parse` builds."""

    data: str
    raw_start: int

    @property
    def raw_end(self) -> int:
        return self.raw_start + len(self.data)


@dataclass
class Element:
    tag: str
    attrs: dict[str, str]
    children: list[Element | Text] = field(default_factory=list)

    @property
    def style(self) -> dict[str, str]:
        return parse_style(self.attrs.get("style", ""))


def parse_style(style: str) -> dict[str, str]:
    declarations = (part.split(":", 1) for part in style.split(";") if ":" in part)
    return {name.strip().lower(): value.strip().lower() for name, value in declarations}


class _TreeBuilder(HTMLParser):
    """Builds an `Element` tree while emitting the raw stream `fetch_edgar.html_to_text` normalizes.

    Each `Text` node records its offset in that stream, which is what lets a
    span of `html_to_text` output be traced back to the elements behind it.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = Element("#root", {})
        self._stack: list[Element] = [self.root]
        self._raw: list[str] = []
        self._raw_len = 0
        self._skip_depth = 0

    def _emit(self, part: str) -> None:
        self._raw.append(part)
        self._raw_len += len(part)

    def raw(self) -> str:
        return "".join(self._raw)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in SKIP_TAGS:
            self._skip_depth += 1
        elif tag in BLOCK_TAGS:
            self._emit("\n")
        elif tag in ("td", "th"):
            self._emit(" | ")
        element = Element(tag, {name: value or "" for name, value in attrs})
        self._stack[-1].children.append(element)
        if tag not in _VOID_TAGS:
            self._stack.append(element)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        # `<br/>`: HTMLParser calls handle_starttag then handle_endtag, which
        # is also what fetch_edgar's extractor sees.
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag in SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag in BLOCK_TAGS:
            self._emit("\n")
        if tag in _VOID_TAGS:
            return
        # Close up to the matching open tag; ignore a stray end tag.
        for depth in range(len(self._stack) - 1, 0, -1):
            if self._stack[depth].tag == tag:
                del self._stack[depth:]
                return

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        self._stack[-1].children.append(Text(data, self._raw_len))
        self._emit(data)


# --------------------------------------------------------------------------
# Flatten: the plain text, with an origin for every output character
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Flattened:
    """``html_to_text``'s output, plus the raw-stream offset each character came from."""

    text: str
    origin: list[int]

    def raw_range(self, start: int, end: int) -> tuple[int, int]:
        """The raw-stream range behind ``text[start:end]`` (non-empty)."""
        return self.origin[start], self.origin[end - 1] + 1


def _mapped_sub(pattern: re.Pattern[str], replacement: str, text: str, origin: list[int]) -> tuple[str, list[int]]:
    """``pattern.sub(replacement, text)`` that carries ``origin`` along.

    ``replacement`` is literal. Its characters take the origin of the match's
    first character, which is exact for the one-character-to-many rewrites
    here and close enough for collapsed whitespace, which carries no content.
    """
    parts: list[str] = []
    origins: list[int] = []
    position = 0
    for match in pattern.finditer(text):
        parts.append(text[position : match.start()])
        origins.extend(origin[position : match.start()])
        if replacement:
            parts.append(replacement)
            anchor = origin[match.start()] if match.start() < len(origin) else (origin[-1] + 1 if origin else 0)
            origins.extend([anchor] * len(replacement))
        position = match.end()
    parts.append(text[position:])
    origins.extend(origin[position:])
    return "".join(parts), origins


def _mapped_strip(text: str, origin: list[int]) -> tuple[str, list[int]]:
    start = len(text) - len(text.lstrip())
    end = len(text.rstrip())
    return text[start:end], origin[start:end]


def normalize_raw(raw: str) -> Flattened:
    """The normalization ``html_to_text`` applies to its raw stream, tracking origins."""
    text, origin = raw, list(range(len(raw)))
    for fancy, plain in PUNCTUATION:
        text, origin = _mapped_sub(re.compile(re.escape(fancy)), plain, text, origin)
    text, origin = _mapped_sub(re.compile(r"[ \t]+"), " ", text, origin)
    # ` *\n *` -> `\n`: done as two passes so the newline keeps its own origin.
    text, origin = _mapped_sub(re.compile(r" +(?=\n)"), "", text, origin)
    text, origin = _mapped_sub(re.compile(r"(?<=\n) +"), "", text, origin)
    text, origin = _mapped_sub(re.compile(r"\n{3,}"), "\n\n", text, origin)
    text, origin = _mapped_sub(re.compile(r"\n(?:\s*\|\s*)+\n"), "\n", text, origin)
    text, origin = _mapped_strip(text, origin)
    return Flattened(text, origin)


def parse(html: str) -> tuple[Element, Flattened]:
    """The filing's element tree, and its ``html_to_text`` output with origins."""
    builder = _TreeBuilder()
    builder.feed(html)
    builder.close()
    return builder.root, normalize_raw(builder.raw())


# --------------------------------------------------------------------------
# Blocks
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Run:
    """Inline text with the style it's set in."""

    text: str
    bold: bool
    italic: bool
    underline: bool
    size: float | None
    color: str | None


@dataclass
class Paragraph:
    runs: list[Run]

    @property
    def text(self) -> str:
        return clean_inline("".join(run.text for run in self.runs))


@dataclass
class Table:
    rows: list[list[str]]
    #: ``(row, first column, width)`` of every cell spanning more than one column.
    spans: list[tuple[int, int, int]] = field(default_factory=list)


@dataclass
class PageBreak:
    pass


Block = Paragraph | Table | PageBreak


def clean_inline(text: str) -> str:
    """Inline text the way ``html_to_text`` writes it, minus line breaks."""
    for fancy, plain in PUNCTUATION:
        text = text.replace(fancy, plain)
    return re.sub(r"\s+", " ", text).strip()


def _font_size(value: str | None) -> float | None:
    if not value:
        return None
    match = re.match(r"([\d.]+)\s*pt", value)
    return float(match.group(1)) if match else None


def _is_bold(style: dict[str, str], tag: str) -> bool:
    weight = style.get("font-weight", "")
    return tag in ("b", "strong") or weight == "bold" or (weight.isdigit() and int(weight) >= 600)


def _is_hidden(element: Element) -> bool:
    return element.style.get("display") == "none" or element.tag == "ix:header"


def _is_page_break(element: Element) -> bool:
    # EDGAR filers mark page breaks with an `<hr>`, styled or not (TGT's are
    # bare), and some with a page-break style on a `<div>`.
    if element.tag == "hr":
        return True
    style = element.style
    return "always" in (style.get("page-break-after", ""), style.get("page-break-before", ""), style.get("break-before", ""), style.get("break-after", ""))


class _BlockCollector:
    """Walks the tree, keeping only text inside the raw range being rendered."""

    def __init__(self, raw_start: int, raw_end: int) -> None:
        self.raw_start = raw_start
        self.raw_end = raw_end
        self.blocks: list[Block] = []
        self._runs: list[Run] = []

    def _clip(self, node: Text) -> str:
        if node.raw_end <= self.raw_start or node.raw_start >= self.raw_end:
            return ""
        return node.data[max(0, self.raw_start - node.raw_start) : self.raw_end - node.raw_start]

    def _flush(self) -> None:
        if self._runs and "".join(run.text for run in self._runs).strip():
            self.blocks.append(Paragraph(self._runs))
        self._runs = []

    def walk(self, element: Element, inherited: Run) -> None:
        for child in element.children:
            if isinstance(child, Text):
                data = self._clip(child)
                if data:
                    self._runs.append(Run(data, inherited.bold, inherited.italic, inherited.underline, inherited.size, inherited.color))
                continue
            if child.tag in SKIP_TAGS or _is_hidden(child):
                continue
            if _is_page_break(child):
                self._flush()
                self.blocks.append(PageBreak())
            if child.tag == "table":
                self._flush()
                table = self._table(child)
                if table is not None:
                    self.blocks.append(table)
                continue
            style = child.style
            run = Run(
                "",
                bold=inherited.bold or _is_bold(style, child.tag),
                italic=inherited.italic or child.tag in ("i", "em") or style.get("font-style") == "italic",
                underline=inherited.underline or child.tag == "u" or "underline" in style.get("text-decoration", ""),
                size=_font_size(style.get("font-size")) or inherited.size,
                color=style.get("color") or inherited.color,
            )
            block = child.tag in BLOCK_TAGS or child.tag == "hr"
            if block:
                self._flush()
            self.walk(child, run)
            if block:
                self._flush()

    def _table(self, table: Element) -> Table | None:
        rows: list[list[str]] = []
        spans: list[tuple[int, int, int]] = []
        for row in _descendants(table, "tr"):
            cells: list[str] = []
            row_spans: list[tuple[int, int, int]] = []
            for cell in row.children:
                if isinstance(cell, Element) and cell.tag in ("td", "th"):
                    text = clean_inline(self._cell_text(cell))
                    colspan = cell.attrs.get("colspan", "1")
                    width = int(colspan) if colspan.isdigit() and int(colspan) > 1 else 1
                    if width > 1 and text:
                        row_spans.append((len(rows), len(cells), width))
                    # A spanning cell keeps its columns, so rows stay aligned.
                    cells.extend([text] + [""] * (width - 1))
            if any(cells):
                rows.append(cells)
                spans.extend(row_spans)
        return Table(rows, spans) if rows else None


    def _cell_text(self, element: Element) -> str:
        """A cell's text on one line, with a space where a block inside it ended.

        MSFT wraps "Three Months Ended" and "March 31," in separate ``<p>``s
        within one cell; joined bare they'd read "EndedMarch".
        """
        parts: list[str] = []
        for child in element.children:
            if isinstance(child, Text):
                parts.append(self._clip(child))
            elif child.tag not in SKIP_TAGS and not _is_hidden(child):
                inner = self._cell_text(child)
                parts.append(f" {inner} " if child.tag in BLOCK_TAGS else inner)
        return "".join(parts)


def _descendants(element: Element, tag: str) -> list[Element]:
    found: list[Element] = []
    for child in element.children:
        if isinstance(child, Element):
            if child.tag == tag:
                found.append(child)
            elif child.tag != "table":  # a nested table's rows are its own
                found.extend(_descendants(child, tag))
    return found


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------

# Cells EDGAR puts in their own column beside a figure: a currency symbol
# before it, and a closing parenthesis or percent sign after it.
_PREFIX_CELLS = frozenset({"$", "€", "£", "¥"})
_SUFFIX_CELLS = frozenset({")", "%", ")%", "%)", "pts", "bps"})
_NUMERIC = re.compile(r"^[($€£¥]*-?[\d,]*\.?\d+[)%]*$|^[-—–]+$")


def glue_figures(rows: list[list[str]]) -> list[list[str]]:
    """Glue split figures back together, keeping every column.

    EDGAR sets ``$``, ``)`` and ``%`` in their own cells beside the figure, so
    ``$ | 94,930`` becomes ``$94,930`` and ``(1.2 | )`` becomes ``(1.2)``. The
    vacated cell stays empty, so column indices (and ``Table.spans``) still hold.
    """
    width = max(len(row) for row in rows)
    glued: list[list[str]] = []
    for row in rows:
        cells: list[str] = []
        prefix = ""
        for cell in row + [""] * (width - len(row)):
            if cell in _PREFIX_CELLS:
                prefix = cell
                cells.append("")
            elif cell in _SUFFIX_CELLS and any(cells):
                target = max(index for index, value in enumerate(cells) if value)
                cells[target] += cell
                cells.append("")
            elif prefix and cell:
                cells.append(prefix + cell)
                prefix = ""
            else:
                cells.append(cell)
        glued.append(cells)
    return glued


def merge_spanned_columns(rows: list[list[str]], spans: list[tuple[int, int, int]]) -> list[list[str]]:
    """Collapse the columns a spanning cell covers into one, where every row allows it.

    A header like "August 2, 2025" spans the ``$``, figure and ``)`` columns
    below it, and one row's figure can itself span the columns another row
    splits into ``$ | 5,404`` (DAL). After gluing, the figure sits in any of
    them from row to row. Where no row fills more than one of those columns,
    they are one logical column. A super-header ("Three Months Ended") spans
    several filled columns, so its columns stay apart. Then columns empty in
    every row go.
    """
    width = len(rows[0])
    group = list(range(width))  # column -> first column of its group
    for _row, start, span in sorted(spans, key=lambda s: s[2]):
        end = min(start + span, width)
        if end - start < 2:
            continue
        if all(sum(1 for column in range(start, end) if cells[column]) <= 1 for cells in rows):
            for column in range(start, end):
                group[column] = group[start]

    merged: list[list[str]] = []
    for cells in rows:
        out: dict[int, list[str]] = {}
        for column, value in enumerate(cells):
            if value:
                out.setdefault(group[column], []).append(value)
        merged.append([" ".join(out.get(column, [])) for column in sorted(set(group))])
    keep = [index for index in range(len(merged[0])) if any(row[index] for row in merged)]
    return [[row[index] for index in keep] for row in merged]


# A layout table's first column: bullets, and footnote or list markers.
_MARKER = re.compile(r"^(?:[•●◦▪■·\-–—*]|\(?\d{1,2}\)|\(?[a-z]\)|\d{1,2}\.)$")


def layout_lines(rows: list[list[str]]) -> list[str] | None:
    """A table that is really a list or text box, as its lines; ``None`` for a data table.

    Filers lay out bullet lists and footnotes as two-column tables
    (``• | text``, ``(1) | text``) and boxed text as one-cell tables.
    """
    if all(len(row) == 1 for row in rows):
        return [row[0] for row in rows]
    if all(len(row) == 2 and _MARKER.match(row[0]) for row in rows):
        return [f"{marker} {text}" for marker, text in rows]
    return None


_YEAR = re.compile(r"^(?:19|20)\d\d$")


def header_row_count(rows: list[list[str]]) -> int:
    """How many leading rows are column headers: those before the first row holding a figure.

    A row counts as data when any cell after its first (the row label) is
    numeric. A bare year ("2026") is a column label, not a figure. At least
    one row is a header, since Markdown requires one.
    """
    for index, row in enumerate(rows):
        if any(_NUMERIC.match(cell) and not _YEAR.match(cell) for cell in row[1:] if cell):
            return max(1, index)
    return 1


def render_table(table: Table) -> str:
    rows = glue_figures(table.rows)
    lines = layout_lines([[cell for cell in row if cell] for row in rows])
    if lines is not None:
        return "\n\n".join(lines)
    headers = header_row_count([[cell for cell in row if cell] for row in rows])
    rows = merge_spanned_columns(rows, table.spans)
    if not rows or not rows[0]:
        return ""
    lines = [_table_line(row) for row in rows[:headers]]
    lines.append("|" + "|".join(["---"] * len(rows[0])) + "|")
    lines.extend(_table_line(row) for row in rows[headers:])
    return "\n".join(lines)


def _table_line(row: list[str]) -> str:
    return "| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |"


# --------------------------------------------------------------------------
# Headings and furniture
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Style:
    bold: bool
    italic: bool
    underline: bool
    size: float | None
    color: str | None
    # Set in capitals. Not a font property, but filers use it as one: COST's
    # "RESULTS OF OPERATIONS" and "Net Sales" share a font and differ only here.
    caps: bool = False


def _block_style(paragraph: Paragraph) -> _Style | None:
    """The style shared by every non-blank run, or ``None`` if the runs differ."""
    styles = {
        _Style(run.bold, run.italic, run.underline, run.size, _plain_color(run.color))
        for run in paragraph.runs
        if run.text.strip()
    }
    if len(styles) != 1:
        return None
    text = paragraph.text
    return _Style(**{**styles.pop().__dict__, "caps": text.upper() == text and any(c.isalpha() for c in text)})


def _plain_color(color: str | None) -> str | None:
    return None if color in (None, "", "#000000", "#000", "black", "windowtext") else color


def body_style(blocks: list[Block]) -> _Style:
    """The style most of the filing's paragraph text is set in, by character count."""
    counts: Counter[_Style] = Counter()
    for block in blocks:
        if isinstance(block, Paragraph):
            for run in block.runs:
                counts[_Style(run.bold, run.italic, run.underline, run.size, _plain_color(run.color))] += len(run.text.strip())
    return counts.most_common(1)[0][0] if counts else _Style(False, False, False, None, None)


def _prominence(style: _Style, body: _Style) -> tuple[float, int, int, int, int, int] | None:
    """How much a style stands out from the body text, or ``None`` if it doesn't.

    Ordered so the more prominent style gets the higher heading level:
    larger type first, then capitals, bold, colour, underline, italic.
    Capitals alone don't make a heading (a short all-caps line in body type
    is usually a label or table caption), but they rank one.
    """
    size_gain = (style.size or 0) - (body.size or 0) if style.size and body.size else 0.0
    marks = (
        int(style.bold and not body.bold),
        int(style.color is not None and style.color != body.color),
        int(style.underline and not body.underline),
        int(style.italic and not body.italic),
    )
    if size_gain <= 0 and not any(marks):
        return None
    if size_gain < 0 and not marks[0]:
        return None  # smaller type: a footnote or caption, not a heading
    return (size_gain, int(style.caps), *marks)


def looks_like_heading(text: str) -> bool:
    return (
        0 < len(text) <= MAX_HEADING_CHARS
        and not text.endswith((".", ",", ";", ":"))
        and any(char.isalpha() for char in text)
        and not text.startswith(("•", "(", "*"))
    )


def is_furniture(text: str, running_headers: set[str]) -> bool:
    return bool(_PAGE_NUMBER.match(text) or _TABLE_OF_CONTENTS.match(text) or _page_key(text) in running_headers)


def _page_key(text: str) -> str:
    """``text`` with its numbers masked, so "Form 10-Q | 14" and "Form 10-Q | 15" match."""
    return re.sub(r"\d+", "#", text)


# A running header or footer is short. Tables up to this many rows are
# considered, since some filers set theirs as a small navigation table.
_MAX_FURNITURE_ROWS = 3


def running_headers(blocks: list[Block]) -> set[str]:
    """Lines that open or close three or more pages: the filing's running headers and footers.

    Keys are :func:`_page_key` forms. Only blocks next to a page break count,
    which keeps a sub-heading that recurs through the text ("Quarter-to-Date")
    from being mistaken for one. A sentence never counts: headers and footers
    are labels, and a page that happens to open with similar prose three
    times ("... segment 1 ...", "... segment 2 ...") must keep it.
    """
    edges: Counter[str] = Counter()
    for index, block in enumerate(blocks):
        if not isinstance(block, PageBreak):
            continue
        for neighbour in (*blocks[max(0, index - 2) : index], *blocks[index + 1 : index + 3]):
            text = block_text(neighbour)
            if text and not text.endswith((".", "?", "!")):
                edges[_page_key(text)] += 1
    return {key for key, count in edges.items() if count >= 3 and len(key) <= MAX_HEADING_CHARS}


def block_text(block: Block) -> str | None:
    """A paragraph's text, or a small table's cells joined, for furniture matching."""
    if isinstance(block, Paragraph):
        return block.text
    if isinstance(block, Table) and len(block.rows) <= _MAX_FURNITURE_ROWS:
        return clean_inline(" ".join(cell for row in block.rows for cell in row if cell))
    return None


# --------------------------------------------------------------------------
# Render
# --------------------------------------------------------------------------


@dataclass
class Rendered:
    markdown: str
    headings: list[str]
    tables: int
    dropped_furniture: int


def render_blocks(blocks: list[Block], furniture_blocks: list[Block] | None = None) -> Rendered:
    """Markdown for ``blocks``. Heading styles and running headers come from ``furniture_blocks`` (default: ``blocks``).

    Pass the whole filing as ``furniture_blocks`` when rendering a span of
    it, so page edges and the body style are measured over every page.
    """
    context = furniture_blocks if furniture_blocks is not None else blocks
    body = body_style(context)
    repeated = running_headers(context)

    candidates: dict[_Style, tuple[float, int, int, int, int, int]] = {}
    for block in blocks:
        if isinstance(block, Paragraph) and looks_like_heading(block.text) and not is_furniture(block.text, repeated):
            style = _block_style(block)
            if style is not None and (prominence := _prominence(style, body)) is not None:
                candidates[style] = prominence
    ranked = sorted(set(candidates.values()), reverse=True)
    level_of = {style: 2 + min(ranked.index(p), MAX_HEADING_LEVELS - 1) for style, p in candidates.items()}

    parts: list[str] = []
    headings: list[str] = []
    tables = dropped = 0
    for block in blocks:
        if isinstance(block, PageBreak):
            continue
        if isinstance(block, Table):
            small = block_text(block)
            if small is not None and is_furniture(small, repeated):
                dropped += 1
                continue
            rendered = render_table(block)
            if rendered:
                parts.append(rendered)
                tables += rendered.startswith("|")
            continue
        text = block.text
        if is_furniture(text, repeated):
            dropped += 1
            continue
        style = _block_style(block)
        level = level_of.get(style) if style is not None and looks_like_heading(text) else None
        if level is not None:
            parts.append(f"{'#' * level} {text}")
            headings.append(text)
        else:
            parts.append(text)
    return Rendered("\n\n".join(parts), headings, tables, dropped)


def blocks_in(root: Element, raw_start: int, raw_end: int) -> list[Block]:
    collector = _BlockCollector(raw_start, raw_end)
    collector.walk(root, Run("", False, False, False, None, None))
    collector._flush()
    return collector.blocks


def render_span(html: str, start: int, end: int) -> Rendered:
    """Markdown for the HTML behind ``html_to_text(html)[start:end]``.

    Blocks are clipped to the span at the character, so a section that starts
    mid-line (``Item 2.Management's...``) starts where the span does.
    """
    root, flattened = parse(html)
    raw_start, raw_end = flattened.raw_range(start, end)
    whole = blocks_in(root, 0, flattened.origin[-1] + 1 if flattened.origin else 0)
    return render_blocks(blocks_in(root, raw_start, raw_end), furniture_blocks=whole)
