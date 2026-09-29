"""Tests for the EDGAR HTML -> Markdown renderer (chunking plan, Phase 4).

Each case is a small, synthetic version of a pattern found in the real
filings; the filer it came from is named where it matters.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from edgar_markdown import Table, parse, render_span, render_table  # noqa: E402
from fetch_edgar import html_to_text  # noqa: E402

_BODY = "font-size:10pt;font-weight:400"
_BOLD = "font-size:10pt;font-weight:700"


def _div(text: str, style: str = _BODY) -> str:
    return f'<div><span style="{style}">{text}</span></div>'


def _render_all(html: str) -> str:
    text = html_to_text(html)
    return render_span(html, 0, len(text)).markdown


# ---------------------------------------------------------------- flattening


def test_flattened_text_keeps_the_legacy_format() -> None:
    html = (
        "<html><head><title>skip me</title></head><body>"
        "<p>Apple’s revenue — up&#160;8%…</p>"
        "<table><tr><td>Net sales</td><td>$</td><td>94,930</td></tr><tr><td> </td><td></td></tr></table>"
        "</body></html>"
    )

    assert html_to_text(html) == "Apple's revenue -- up 8%...\n\n| Net sales | $ | 94,930"


def test_every_flattened_character_traces_back_to_its_text_node() -> None:
    html = "<div>First   paragraph.</div><div>Second “quoted” one.</div>"
    root, flattened = parse(html)
    start = flattened.text.index("Second")

    raw_start, raw_end = flattened.raw_range(start, len(flattened.text))

    second = root.children[1].children[0]
    assert (raw_start, raw_end) == (second.raw_start, second.raw_end)


# ------------------------------------------------------------------ headings


def test_a_short_block_set_bolder_than_the_body_is_a_heading() -> None:
    html = _div("Liquidity and Capital Resources", _BOLD) + _div("Cash was $12.2 billion at year end.") * 3

    assert _render_all(html).startswith("## Liquidity and Capital Resources\n\nCash was")


def test_emphasized_prose_and_run_in_headings_are_not_headings() -> None:
    html = (
        _div("This whole sentence is bold for emphasis.", _BOLD)
        # UAL's run-in heading: a bold label opening a prose paragraph.
        + f'<div><span style="{_BOLD}">Income Taxes. </span><span style="{_BODY}">See Note 5.</span></div>'
        + _div("Body text sets the baseline style for this filing.") * 3
    )

    assert "#" not in _render_all(html)


def test_larger_type_outranks_bold_and_extra_styles_are_clamped_not_dropped() -> None:
    styles = [
        "font-size:16pt;font-weight:700",
        "font-size:12pt;font-weight:700",
        "font-size:10pt;font-weight:700;text-decoration:underline",
        _BOLD,
        "font-size:10pt;font-style:italic",
    ]
    html = "".join(_div(f"Heading {n}", style) for n, style in enumerate(styles)) + _div("Body text here.") * 10

    lines = _render_all(html).split("\n\n")

    assert lines[:5] == ["## Heading 0", "### Heading 1", "#### Heading 2", "#### Heading 3", "#### Heading 4"]


def test_capitals_rank_above_the_same_font_in_mixed_case() -> None:
    # COST: "RESULTS OF OPERATIONS" and "Net Sales" differ only in case.
    html = _div("RESULTS OF OPERATIONS", _BOLD) + _div("Net Sales", _BOLD) + _div("Body text here.") * 5

    assert _render_all(html).startswith("## RESULTS OF OPERATIONS\n\n### Net Sales")


# ----------------------------------------------------------------- furniture


def test_page_numbers_contents_links_and_running_headers_are_dropped() -> None:
    page = "<hr/>"
    html = ""
    for number in range(1, 5):
        html += _div(f"Revenue grew in segment {number} this period.") + _div(str(number)) + _div("Table of Contents")
        # TGT's running header: a one-row table whose page number changes.
        html += f"{page}<table><tr><td>TARGET CORPORATION</td><td>Q2 2025 Form 10-Q</td><td>{number}</td></tr></table>"

    rendered = _render_all(html)

    assert rendered.split("\n\n") == [f"Revenue grew in segment {number} this period." for number in range(1, 5)]


# -------------------------------------------------------------------- tables


def test_split_figures_are_glued_and_spanned_headers_sit_over_their_figures() -> None:
    # TGT: each date header spans the `$`, figure and `)%` columns below it,
    # and a row's dash sits in the span's first column rather than its figure's.
    table = Table(
        rows=[
            ["", "August 2, 2025", "", "", "Change", ""],
            ["GAAP EPS", "$", "2.05", "", "(20.2", ")%"],
            ["Adjustments", "--", "", "", "", ""],
        ],
        spans=[(0, 1, 3), (0, 4, 2)],
    )

    assert render_table(table).split("\n") == [
        "|  | August 2, 2025 | Change |",
        "|---|---|---|",
        "| GAAP EPS | $2.05 | (20.2)% |",
        "| Adjustments | -- |  |",
    ]


def test_a_data_cell_spanning_the_columns_another_row_splits_is_one_column() -> None:
    # DAL: `$ | 5,404` in one row, `5,363` with colspan=2 in the next.
    table = Table(rows=[["", "2026"], ["Main cabin", "$", "5,404"], ["Premium", "5,363", ""]], spans=[(2, 1, 2)])

    assert render_table(table).split("\n")[2:] == ["| Main cabin | $5,404 |", "| Premium | 5,363 |"]


def test_year_rows_are_headers_not_data() -> None:
    table = Table(rows=[["(In millions)", "2026", "2025"], ["Revenue", "82,886", "70,066"]])

    assert render_table(table).split("\n")[1] == "|---|---|---|"


def test_bullet_and_footnote_layout_tables_render_as_lines() -> None:
    table = Table(rows=[["•", "MacBook Air 13-in.; and"], ["(1)", "Excludes one-time items."]])

    assert render_table(table) == "• MacBook Air 13-in.; and\n\n(1) Excludes one-time items."


def test_block_boundaries_inside_a_cell_become_spaces() -> None:
    # MSFT wraps each line of a header cell in its own <p>.
    html = (
        "<table><tr><td></td><td><p>Three Months Ended</p><p>March 31,</p></td></tr>"
        "<tr><td>Revenue</td><td>82,886</td></tr></table>"
    )

    assert "| Three Months Ended March 31, |" in _render_all(html)


# --------------------------------------------------------------------- spans


def test_a_span_starting_mid_line_is_clipped_at_the_character() -> None:
    # CVX: the section heading shares a line with its item number.
    html = _div("Item 2.Management's Discussion and Analysis", _BOLD) + _div("Revenue rose in every segment.") * 3
    text = html_to_text(html)
    start = text.index("Management's")

    rendered = render_span(html, start, len(text))

    assert rendered.markdown.startswith("## Management's Discussion and Analysis\n\nRevenue rose in every segment.")
    assert "Item 2." not in rendered.markdown
