#!/usr/bin/env python
"""Fetch SEC EDGAR filings and write their MD&A sections as a local corpus.

Why EDGAR, and why MD&A specifically
------------------------------------
The stock corpus (``data/corpora/baseline/documents``, ~26k characters over 8 documents) is too
small to evaluate against: with ``retrieval.top_k: 20`` every query already
retrieves roughly two thirds of it, so hit rate and recall@k are saturated
before any config knob is touched.  Nothing in Milestone 11 can produce signal
against a corpus that small.

EDGAR fixes that, and it is the *right* kind of harder rather than merely
bigger:

* **Long documents.**  A single 10-K MD&A section runs 40k-80k characters --
  more than this project's entire current corpus -- which is what exposes the
  fact that document-level eval matching stops being a proxy for "found the
  right passage".
* **Near-duplicate documents.**  Twelve companies x five filings produces
  documents that are structurally near-identical and differ only in entity,
  period, and figures.  That is a brutal distractor environment, and it is
  exactly the case contextual chunking exists for: a chunk reading "revenue
  increased 8% driven by higher volumes" is unusable without knowing whose
  revenue and which quarter.
* **Free and redistributable.**  Commercial transcript providers (Motley Fool,
  Seeking Alpha) carry restrictive terms and cannot be committed to a public
  repo.  EDGAR is a public filing system with a documented API.

We extract **MD&A prose only**, not the financial statement tables, on purpose.
``pypdf``/HTML table extraction collapses tables into whitespace soup (a known
limitation of this repo, see Milestone 14), and indexing mangled tables would
confound a retrieval measurement with a parsing failure -- you would read
"contextual chunking didn't help" when the truth is the number never survived
extraction.  Tables come later, as Milestone 14's motivating evidence.

Corpus documents are **not committed**.  This script plus the committed
manifest reproduce the corpus, which keeps the repo small and sidesteps
redistribution questions entirely.

Dependencies: ``httpx`` (already a project dependency -- no second HTTP client)
and the standard library's ``html.parser`` (no BeautifulSoup/lxml), per the
project's "keep dependencies minimal" rule.

Usage
-----
    python scripts/fetch_edgar.py --manifest data/corpora/edgar/manifest.json

    # Preview what would be fetched without writing anything:
    python scripts/fetch_edgar.py --manifest ... --dry-run

The SEC requires a User-Agent identifying the requester.  Override the default
with ``--user-agent`` or the ``SEC_USER_AGENT`` environment variable.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.logging_config import configure_logging  # noqa: E402

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# SEC endpoints and politeness
# --------------------------------------------------------------------------

TICKER_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"

# The SEC asks for a User-Agent identifying the requester and rate-limits to
# 10 requests/second. We stay at 5 -- this is a batch job with no deadline, and
# getting an IP blocked costs far more than the time saved.
DEFAULT_USER_AGENT = "rag-toolkit research bwu1991@gmail.com"
REQUESTS_PER_SECOND = 5.0

# MD&A is a substantial section by regulation, so a short span is never the real
# thing -- it is a table-of-contents entry or a cross-reference to another
# filing's subsection ("...see MD&A - Financial and Derivative Instrument Market
# Risk in Chevron's 2025 Annual Report..."). Enforced per form because a 10-K
# covers a full year and is correspondingly longer than a quarterly 10-Q.
MIN_SECTION_CHARS = {"10-Q": 12_000, "10-K": 15_000}
DEFAULT_MIN_SECTION_CHARS = 12_000

# Loose backstop for a span that ran to end-of-filing because the end marker
# was missed. The digit-ratio band below does the real discrimination.
MAX_SECTION_CHARS = 250_000

# A table-of-contents entry matches the heading pattern but is immediately
# followed by more TOC entries. Real prose contains almost no "Item N" markers.
MAX_ITEM_MARKERS = 4
_TOC_PROBE_CHARS = 2_500

# MD&A is *numeric prose* -- "revenue increased 8% to $94.0 billion" -- which
# gives it a characteristic digit density, measured across a sample of filers:
#
#   0.001-0.020  forward-looking-statement boilerplate, risk factors, and the
#                business section. Prose, but the WRONG prose.
#   0.030-0.085  genuine MD&A (AAPL 0.066/0.072, MSFT 0.035/0.044,
#                UAL 0.041/0.065, COST 0.044/0.055, GS 0.039).
#   0.105+       collapsed financial-statement tables -- the banks
#                (BAC 0.107-0.112) fold theirs directly into MD&A.
#
# The bounds are deliberately loose at the top, because the legitimate range and
# the table-dump range nearly touch: Delta's real MD&A measures 0.091 (airlines
# report operating statistics -- ASMs, load factor, CASM -- as prose) against
# BAC's 0.107. A tight threshold between those two is a coin flip, so the ceiling
# only catches the clear cases (BAC's acronym glossary, 0.112) and cell-separator
# density below does the rest of the table discrimination.
MIN_DIGIT_RATIO = 0.030
MAX_DIGIT_RATIO = 0.100

# Fraction of characters that are table-cell separators emitted by the HTML
# extractor. This separates MD&A from table dumps far more cleanly than digit
# density does: real MD&A measures 0.000-0.045 (MSFT 0.000, COST 0.017,
# GS 0.024, AAPL 0.025, UAL 0.042) against BAC's 0.070-0.073.
MAX_PIPE_RATIO = 0.055

# Some filers open MD&A with an acronym glossary (`| MDL | Multi-District
# Litigation`), which the whole-span ratio above absorbs without noticing --
# PFE measures 0.039 overall while its first page is pure table. A real MD&A
# opens with prose, so the leading window is checked separately and tightly.
MAX_LEADING_PIPE_RATIO = 0.020
_LEADING_PROBE_CHARS = 1_500

# Filings cross-reference their own sections in quotes -- `see "Management's
# Discussion and Analysis of Financial Condition and Results of Operations" in
# our Form 10-K`. The closing quote lands ~85 characters into the match, while a
# real section heading is followed by prose. Checking a short window is what
# keeps this from firing on the parenthetical quote in a genuine MD&A opening
# ("...this Quarterly Report on Form 10-Q ("Form 10-Q") contains...", ~161 in).
_CROSS_REFERENCE_PROBE_CHARS = 100


@dataclass(frozen=True)
class FilingRef:
    """A single filing selected for download."""

    ticker: str
    company: str
    cik: int
    form: str
    filing_date: str
    report_date: str
    accession: str
    document: str

    @property
    def url(self) -> str:
        return ARCHIVE_URL.format(
            cik=self.cik,
            accession=self.accession.replace("-", ""),
            document=self.document,
        )

    @property
    def filename(self) -> str:
        """Descriptive filename -- becomes ``Document.id`` downstream.

        Entity, form, and period all live in the id so a citation or a
        retrieved chunk is traceable to a specific filing without a lookup.
        """
        period = self.report_date or self.filing_date
        return f"{self.ticker}_{self.form.replace('/', '-')}_{period}.md"


# --------------------------------------------------------------------------
# HTML -> text
# --------------------------------------------------------------------------

_BLOCK_TAGS = {
    "p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6",
    "table", "section", "article", "header", "footer",
}
_SKIP_TAGS = {"script", "style", "head", "title"}


class _TextExtractor(HTMLParser):
    """Collapse an EDGAR HTML filing into plain text.

    Inline-XBRL filings are mostly presentational markup wrapping short text
    runs, so a tag-stripper gets very close to clean prose.  Table cells are
    joined with ``|`` rather than dropped so that a row label stays attached to
    its figure -- strictly better than the whitespace soup ``pypdf`` produces,
    though still not structured table extraction (that is Milestone 14).
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: object) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        elif tag in _BLOCK_TAGS:
            self._parts.append("\n")
        elif tag in ("td", "th"):
            self._parts.append(" | ")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._parts.append(data)

    def text(self) -> str:
        return "".join(self._parts)


def html_to_text(html: str) -> str:
    """Extract normalized plain text from an EDGAR HTML filing."""
    parser = _TextExtractor()
    parser.feed(html)
    text = parser.text()

    # Non-breaking spaces are pervasive in EDGAR markup and would otherwise
    # survive into chunk text and defeat whitespace-boundary chunk snapping.
    # Normalize the typographic punctuation EDGAR filings are full of. Left as
    # smart quotes it reaches BM25's tokenizer, where "Company's" and
    # "Company’s" are different terms and only one of them matches a query.
    for fancy, plain in (
        ("\xa0", " "), ("’", "'"), ("‘", "'"),
        ("“", '"'), ("”", '"'),
        ("—", "--"), ("–", "-"), ("…", "..."),
    ):
        text = text.replace(fancy, plain)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Empty table rows left behind by layout-only cells.
    text = re.sub(r"\n(?:\s*\|\s*)+\n", "\n", text)
    return text.strip()


# --------------------------------------------------------------------------
# Section extraction
# --------------------------------------------------------------------------

# 10-Q: Part I Item 2.  10-K: Item 7 (Item 7A is the following section).
#
# Tried in priority order. Most filers repeat the "Item 2./Item 7." heading at
# the section itself, which is the precise signal. Some -- banks especially --
# print the heading only in the table of contents and open the section with
# prose, so the second pattern matches that opening sentence instead.
_MDA_START_PATTERNS = (
    re.compile(r"item\s*[27][.\s)]{0,8}management'?s\s+discussion", re.IGNORECASE),
    re.compile(
        r"(?:the\s+following\s+is\s+)?management'?s\s+discussion\s+and\s+analysis"
        r"\s+of\s+(?:the\s+)?financial\s+condition",
        re.IGNORECASE,
    ),
)
_MDA_END = re.compile(
    r"item\s*(?:3|7a|8)[.\s)]{0,8}"
    r"(?:quantitative|financial\s+statements|consolidated\s+financial)",
    re.IGNORECASE,
)
_ITEM_MARKER = re.compile(r"Item\s*\d", re.IGNORECASE)


def digit_ratio(text: str) -> float:
    """Fraction of non-whitespace characters that are digits.

    The discriminator between MD&A, the prose sections around it, and collapsed
    financial tables -- see ``MIN_DIGIT_RATIO`` for measured values.
    """
    non_space = [char for char in text if not char.isspace()]
    if not non_space:
        return 0.0
    return sum(char.isdigit() for char in non_space) / len(non_space)


def pipe_ratio(text: str) -> float:
    """Density of table-cell separators -- the table-dump discriminator.

    ``_TextExtractor`` joins table cells with ``|``, so this measures how much
    of a span is tabular. See ``MAX_PIPE_RATIO`` for measured values.
    """
    if not text:
        return 0.0
    return text.count("|") / len(text)


def extract_mda(text: str, form: str = "10-Q") -> str | None:
    """Slice the MD&A section out of a filing's full text.

    A filing names its own sections several times -- in the table of contents,
    at the section itself, and in cross-references scattered through the prose
    ("...set forth under Item 7. Management's Discussion and Analysis... and
    elsewhere") -- so matching the heading is nowhere near enough.  Filers also
    vary wildly: some repeat the heading in ALL CAPS at the real section, some
    never print "Item" at all, and some fold their entire financial-statement
    tables into MD&A.

    Every *positional* rule fails on some real filer.  Taking the longest span
    picks the table-of-contents entry, whose start pairs with the real section's
    end marker.  Taking the last picks a trailing cross-reference.  So selection
    is **content-based** instead: keep candidates that look like MD&A and take
    the tightest one.

    1. Reject spans shorter than this form's minimum, or absurdly long ones.
    2. Reject table-of-contents entries by their density of "Item N" markers.
    3. Reject quoted cross-references by the closing quote that follows them.
    4. Reject low-digit spans -- risk factors, the business section, and
       forward-looking boilerplate are prose, but the wrong prose.
    5. Reject table-dominated spans by cell-separator density, overall and
       across the leading window (which catches an opening acronym glossary
       that the whole-span ratio would average away).
    6. Among survivors take the **shortest**, which is the span that has not
       swallowed a neighbouring section.

    This is a pile of content heuristics, not a filing parser, and it is tuned
    against a sample of filers rather than derived from the spec.  It therefore
    fails *closed*: anything that does not clearly look like MD&A is skipped and
    logged, because a corpus that is silently wrong would invalidate every metric
    computed against it.

    Returns ``None`` when nothing qualifies, so the caller can skip the filing
    rather than index the wrong section of it.
    """
    # Both patterns frequently match a few characters apart on the same heading;
    # keying by start offset keeps them all as distinct candidates and lets the
    # shortest-wins rule pick between them harmlessly.
    candidates: dict[int, str] = {}
    for pattern in _MDA_START_PATTERNS:
        for start_match in pattern.finditer(text):
            start = start_match.start()
            # Small skip so an end marker cannot match the start heading itself,
            # but short enough that a TOC entry still finds its neighbouring TOC
            # entry and is correctly measured as tiny.
            end_match = _MDA_END.search(text, start + 50)
            end = end_match.start() if end_match else len(text)
            candidates[start] = text[start:end].strip()

    min_chars = MIN_SECTION_CHARS.get(form, DEFAULT_MIN_SECTION_CHARS)
    qualified = [
        candidate
        for candidate in candidates.values()
        if min_chars <= len(candidate) <= MAX_SECTION_CHARS
        and len(_ITEM_MARKER.findall(candidate[:_TOC_PROBE_CHARS])) < MAX_ITEM_MARKERS
        and '"' not in candidate[:_CROSS_REFERENCE_PROBE_CHARS]
        and MIN_DIGIT_RATIO <= digit_ratio(candidate) <= MAX_DIGIT_RATIO
        and pipe_ratio(candidate) <= MAX_PIPE_RATIO
        and pipe_ratio(candidate[:_LEADING_PROBE_CHARS]) <= MAX_LEADING_PIPE_RATIO
    ]
    if not qualified:
        return None
    return min(qualified, key=len)


# --------------------------------------------------------------------------
# EDGAR client
# --------------------------------------------------------------------------


class EdgarClient:
    """Minimal, rate-limited SEC EDGAR client."""

    def __init__(self, user_agent: str, timeout: float = 30.0) -> None:
        # trust_env=False mirrors OllamaEmbedder/OllamaLLMClient: keep a batch
        # job's network behaviour predictable regardless of ambient proxy vars.
        self._client = httpx.Client(
            headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"},
            timeout=timeout,
            follow_redirects=True,
            trust_env=False,
        )
        self._min_interval = 1.0 / REQUESTS_PER_SECOND
        self._last_request = 0.0

    def __enter__(self) -> "EdgarClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self._client.close()

    def _get(self, url: str) -> httpx.Response:
        elapsed = time.monotonic() - self._last_request
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request = time.monotonic()
        response = self._client.get(url)
        response.raise_for_status()
        return response

    def ticker_to_cik(self) -> dict[str, tuple[int, str]]:
        """Map upper-case ticker -> (CIK, company name)."""
        data = self._get(TICKER_URL).json()
        return {
            row["ticker"].upper(): (int(row["cik_str"]), row["title"])
            for row in data.values()
        }

    def recent_filings(
        self, ticker: str, cik: int, company: str, forms: dict[str, int]
    ) -> list[FilingRef]:
        """Most recent ``count`` filings per form type, newest first."""
        recent = self._get(SUBMISSIONS_URL.format(cik=cik)).json()["filings"]["recent"]
        remaining = dict(forms)
        selected: list[FilingRef] = []

        for i, form in enumerate(recent["form"]):
            if remaining.get(form, 0) <= 0:
                continue
            document = recent["primaryDocument"][i]
            if not document.lower().endswith((".htm", ".html")):
                continue
            remaining[form] -= 1
            selected.append(
                FilingRef(
                    ticker=ticker,
                    company=company,
                    cik=cik,
                    form=form,
                    filing_date=recent["filingDate"][i],
                    report_date=recent["reportDate"][i],
                    accession=recent["accessionNumber"][i],
                    document=document,
                )
            )
            if all(v <= 0 for v in remaining.values()):
                break
        return selected

    def fetch_document(self, filing: FilingRef) -> str:
        return self._get(filing.url).text


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def render_document(filing: FilingRef, body: str) -> str:
    """Wrap an extracted section in a Markdown document with a title heading.

    The heading is what ``MarkdownLoader`` picks up as ``metadata['title']``,
    and it names the entity and period -- which the body text frequently does
    not.  Without it, a chunk from the middle of an MD&A is anonymous.
    """
    header = f"# {filing.company} ({filing.ticker}) {filing.form} -- period ended {filing.report_date}"
    provenance = (
        f"_Source: SEC EDGAR, accession {filing.accession}, filed {filing.filing_date}._"
    )
    return f"{header}\n\n{provenance}\n\n{body}\n"


def fetch_corpus(
    manifest: dict, out_dir: Path, user_agent: str, *, dry_run: bool = False
) -> int:
    """Fetch every filing named by ``manifest`` into ``out_dir``. Returns count written."""
    forms: dict[str, int] = manifest["forms"]
    tickers: list[str] = [t.upper() for t in manifest["tickers"]]

    with EdgarClient(user_agent) as client:
        logger.info("Resolving %d tickers to CIKs", len(tickers))
        lookup = client.ticker_to_cik()

        unknown = [t for t in tickers if t not in lookup]
        if unknown:
            logger.warning("Unknown tickers (skipped): %s", ", ".join(unknown))

        plan: list[FilingRef] = []
        for ticker in tickers:
            if ticker not in lookup:
                continue
            cik, company = lookup[ticker]
            filings = client.recent_filings(ticker, cik, company, forms)
            logger.info("%-6s %-28s %d filings", ticker, company[:28], len(filings))
            plan.extend(filings)

        if dry_run:
            logger.info("--- dry run: %d filings would be fetched ---", len(plan))
            for f in plan:
                logger.info("  %s  %s", f.filename, f.url)
            return 0

        out_dir.mkdir(parents=True, exist_ok=True)
        written = 0
        skipped: list[str] = []

        for filing in plan:
            destination = out_dir / filing.filename
            if destination.exists():
                logger.debug("exists, skipping: %s", filing.filename)
                written += 1
                continue

            try:
                html = client.fetch_document(filing)
            except httpx.HTTPError as exc:
                logger.warning("fetch failed for %s: %s", filing.filename, exc)
                skipped.append(filing.filename)
                continue

            text = html_to_text(html)
            body = extract_mda(text, filing.form)
            if body is None:
                # Deliberately skip rather than fall back to the whole filing:
                # a full 10-K is mostly risk factors and statement tables, which
                # is the content we are specifically trying to keep out.
                logger.warning(
                    "no MD&A section found in %s (%d chars extracted) -- skipping",
                    filing.filename,
                    len(text),
                )
                skipped.append(filing.filename)
                continue

            destination.write_text(render_document(filing, body), encoding="utf-8")
            # Log the discriminator alongside the size: a corpus built by
            # heuristic extraction should be inspectable, not trusted.
            logger.info(
                "wrote %-34s %7d chars  digit=%.3f pipe=%.3f",
                filing.filename,
                len(body),
                digit_ratio(body),
                pipe_ratio(body),
            )
            written += 1

        if skipped:
            logger.warning("%d filing(s) skipped: %s", len(skipped), ", ".join(skipped))
        return written


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch SEC EDGAR filings and write their MD&A sections as a local corpus."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/corpora/edgar/manifest.json"),
        help="Corpus manifest declaring tickers and form counts.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: the manifest's directory + /documents).",
    )
    parser.add_argument(
        "--user-agent",
        default=os.environ.get("SEC_USER_AGENT", DEFAULT_USER_AGENT),
        help="User-Agent header. The SEC requires one identifying the requester.",
    )
    parser.add_argument("--dry-run", action="store_true", help="List filings, write nothing.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    configure_logging(logging.DEBUG if args.verbose else logging.INFO)

    if not args.manifest.exists():
        logger.error("Manifest not found: %s", args.manifest)
        return 1

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    out_dir = args.out_dir or args.manifest.parent / "documents"

    count = fetch_corpus(manifest, out_dir, args.user_agent, dry_run=args.dry_run)
    if not args.dry_run:
        total = sum(p.stat().st_size for p in out_dir.glob("*.md"))
        logger.info("Corpus ready: %d documents, %.1f MB in %s", count, total / 1e6, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
