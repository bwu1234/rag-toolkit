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

    # Add YAML front matter to files fetched before the fetcher wrote it
    # (offline; reads the title and source lines this script wrote):
    python scripts/fetch_edgar.py --manifest ... --backfill-front-matter

    # Cache the raw HTML of exactly the filings on disk (pinned by accession),
    # and check each document's body is found in it:
    python scripts/fetch_edgar.py --manifest ... --cache-raw

    # Re-render those documents as Markdown from the cache (offline):
    python scripts/fetch_edgar.py --manifest ... --render-markdown data/corpora/edgar_md/documents

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
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import httpx
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from edgar_markdown import Rendered, parse, render_span  # noqa: E402

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

def html_to_text(html: str) -> str:
    """Extract normalized plain text from an EDGAR HTML filing.

    Block elements become line breaks, table cells are joined with `` | `` so
    a row label stays attached to its figure, typographic punctuation is
    folded to ASCII (left as smart quotes it reaches BM25's tokenizer, where
    "Company's" and "Company’s" are different terms), and whitespace is
    collapsed. ``edgar_markdown.parse`` does the work: it also maps every
    output character back to the HTML, which is what lets
    ``edgar_markdown.render_span`` re-render exactly this text as Markdown.
    """
    return parse(html)[1].text


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

    ``html_to_text`` joins table cells with ``|``, so this measures how much
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

    def recent_submissions(self, cik: int) -> dict[str, list]:
        """The company's recent filings as parallel column arrays (``form``, ``accessionNumber``, ...)."""
        recent: dict[str, list] = self._get(SUBMISSIONS_URL.format(cik=cik)).json()["filings"]["recent"]
        return recent

    def recent_filings(
        self, ticker: str, cik: int, company: str, forms: dict[str, int]
    ) -> list[FilingRef]:
        """Most recent ``count`` filings per form type, newest first."""
        recent = self.recent_submissions(cik)
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


def front_matter(
    *, company: str, ticker: str, form: str, period_end: str, filed: str, accession: str
) -> str:
    """The YAML front matter block ``MarkdownLoader`` parses into ``Document.metadata``.

    Typed fields, so metadata filters and chunk headers read them directly
    rather than parsing the title line. Dates are written as YAML dates, which
    load as ``datetime.date``. The loader strips the block from the text, so
    adding it moves no character offset.
    """
    fields = {
        "company": company,
        "ticker": ticker,
        "form": form,
        "period_end": date.fromisoformat(period_end),
        "filed": date.fromisoformat(filed),
        "accession": accession,
    }
    return f"---\n{yaml.safe_dump(fields, sort_keys=False)}---\n"


def render_document(filing: FilingRef, body: str) -> str:
    """Wrap an extracted section in a Markdown document: front matter, then a title heading.

    The heading is what ``MarkdownLoader`` picks up as ``metadata['title']``,
    and it names the entity and period -- which the body text frequently does
    not.  Without it, a chunk from the middle of an MD&A is anonymous.
    """
    header = f"# {filing.company} ({filing.ticker}) {filing.form} -- period ended {filing.report_date}"
    provenance = (
        f"_Source: SEC EDGAR, accession {filing.accession}, filed {filing.filing_date}._"
    )
    meta = front_matter(
        company=filing.company,
        ticker=filing.ticker,
        form=filing.form,
        period_end=filing.report_date,
        filed=filing.filing_date,
        accession=filing.accession,
    )
    return f"{meta}{header}\n\n{provenance}\n\n{body}\n"


# The two lines ``render_document`` writes under the front matter. Parsed only
# by the one-off backfill below, never at load time: that would tie every
# loader to this title format.
_TITLE_LINE = re.compile(
    r"^# (?P<company>.+) \((?P<ticker>[^)]+)\) (?P<form>\S+) -- period ended (?P<period_end>\d{4}-\d{2}-\d{2})$"
)
_SOURCE_LINE = re.compile(
    r"^_Source: SEC EDGAR, accession (?P<accession>[\d-]+), filed (?P<filed>\d{4}-\d{2}-\d{2})\._$"
)


def backfill_front_matter(out_dir: Path) -> int:
    """Prepend front matter to fetched files that predate it. Returns the count changed.

    Offline, and it doesn't re-fetch: the manifest selects each ticker's
    *most recent* filings, so a re-fetch can return a different corpus and
    invalidate every eval set's ``expected_doc_ids``. The fields come from the
    title and source lines this script already wrote. Files that already
    have front matter are left alone, and a file whose lines don't parse
    raises instead of being skipped.
    """
    changed = 0
    for path in sorted(out_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        if text.startswith("---\n"):
            continue
        lines = text.split("\n", 3)
        title = _TITLE_LINE.match(lines[0])
        source = _SOURCE_LINE.match(lines[2]) if len(lines) > 2 else None
        if title is None or source is None:
            raise ValueError(f"{path.name}: title/source lines not in the format render_document writes")
        path.write_text(front_matter(**title.groupdict(), **source.groupdict()) + text, encoding="utf-8")
        changed += 1
    return changed


# --------------------------------------------------------------------------
# Raw HTML cache, pinned to the corpus on disk
# --------------------------------------------------------------------------

# Written next to the cached HTML: one FilingRef per accession, so rendering
# from the cache needs no network call.
RAW_INDEX_NAME = "index.json"


def read_front_matter(text: str) -> dict | None:
    """The YAML front matter ``render_document`` writes, or ``None`` if the file has none."""
    if not text.startswith("---\n"):
        return None
    end = text.index("\n---\n", 3)
    fields: dict = yaml.safe_load(text[4:end])
    return fields


def pinned_accessions(out_dir: Path) -> dict[str, dict[str, str]]:
    """``ticker -> {accession: filename}`` for every fetched document in ``out_dir``.

    The corpus is defined by what is on disk, not by the manifest: the manifest
    selects each ticker's *most recent* filings, so re-running it later returns
    a different set and invalidates every eval set's ``expected_doc_ids``.

    Raises:
        ValueError: a document has no front matter (run
            ``--backfill-front-matter`` first).
    """
    pinned: dict[str, dict[str, str]] = {}
    for path in sorted(out_dir.glob("*.md")):
        fields = read_front_matter(path.read_text(encoding="utf-8"))
        if fields is None:
            raise ValueError(f"{path.name}: no front matter; run --backfill-front-matter first")
        pinned.setdefault(str(fields["ticker"]), {})[str(fields["accession"])] = path.name
    return pinned


def filing_ref_for(recent: dict[str, list], *, ticker: str, company: str, cik: int, accession: str) -> FilingRef | None:
    """The ``FilingRef`` for one accession in a submissions listing, or ``None`` if it isn't listed."""
    for i, listed in enumerate(recent["accessionNumber"]):
        if listed == accession:
            return FilingRef(
                ticker=ticker,
                company=company,
                cik=cik,
                form=recent["form"][i],
                filing_date=recent["filingDate"][i],
                report_date=recent["reportDate"][i],
                accession=accession,
                document=recent["primaryDocument"][i],
            )
    return None


def load_raw_index(raw_dir: Path) -> dict[str, FilingRef]:
    """The cached ``accession -> FilingRef`` index, empty if nothing is cached yet."""
    path = raw_dir / RAW_INDEX_NAME
    if not path.exists():
        return {}
    return {accession: FilingRef(**fields) for accession, fields in json.loads(path.read_text(encoding="utf-8")).items()}


def raw_html_path(raw_dir: Path, accession: str) -> Path:
    return raw_dir / f"{accession}.htm"


def cache_raw_filings(out_dir: Path, raw_dir: Path, user_agent: str) -> int:
    """Download the HTML of exactly the filings already in ``out_dir``. Returns the count downloaded.

    Idempotent and resumable: a filing already cached is not requested again,
    and when every filing is cached no request is made at all. Each filing is
    matched by accession, and its ``FilingRef.filename`` must equal the file on
    disk, which pins form and period as well.

    Raises:
        ValueError: an accession isn't in its company's recent submissions, or
            resolves to a different filename than the one on disk.
    """
    pinned = pinned_accessions(out_dir)
    index = load_raw_index(raw_dir)
    missing = {
        ticker: {acc: name for acc, name in filings.items() if acc not in index or not raw_html_path(raw_dir, acc).exists()}
        for ticker, filings in pinned.items()
    }
    missing = {ticker: filings for ticker, filings in missing.items() if filings}
    if not missing:
        logger.info("All %d filing(s) already cached in %s", len(index), raw_dir)
        return 0

    raw_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    with EdgarClient(user_agent) as client:
        lookup = client.ticker_to_cik()
        for ticker, filings in missing.items():
            cik, company = lookup[ticker]
            recent = client.recent_submissions(cik)
            for accession, filename in filings.items():
                ref = filing_ref_for(recent, ticker=ticker, company=company, cik=cik, accession=accession)
                if ref is None:
                    raise ValueError(f"{filename}: accession {accession} not in {ticker}'s recent submissions")
                if ref.filename != filename:
                    raise ValueError(f"{filename}: accession {accession} resolves to {ref.filename}")
                destination = raw_html_path(raw_dir, accession)
                # Write-then-rename, so an interrupted download never leaves a
                # truncated file that the next run would take as cached.
                partial = destination.with_suffix(".partial")
                partial.write_text(client.fetch_document(ref), encoding="utf-8")
                partial.replace(destination)
                index[accession] = ref
                downloaded += 1
                logger.info("cached %-34s %s", filename, ref.document)
                # The index is rewritten after each filing for the same reason.
                (raw_dir / RAW_INDEX_NAME).write_text(
                    json.dumps({acc: asdict(r) for acc, r in sorted(index.items())}, indent=2) + "\n",
                    encoding="utf-8",
                )
    return downloaded


def pinned_body(ref: FilingRef, document: str) -> str:
    """The extracted body inside a document ``render_document`` wrote for ``ref``.

    Raises:
        ValueError: the front matter, title or source line differ from what
            ``render_document`` writes for ``ref``.
    """
    head = render_document(ref, "")[:-1]  # everything up to the body, which ends "\n"
    if not (document.startswith(head) and document.endswith("\n")):
        raise ValueError(f"{ref.filename}: header lines differ from what render_document writes")
    return document[len(head) : -1]


def pinned_span(ref: FilingRef, document: str, text: str) -> tuple[int, int]:
    """Where the on-disk body sits in ``text`` (``html_to_text`` of the filing), as ``(start, end)``.

    This, not ``extract_mda``, is what fixes the section a re-render covers.
    The corpus was fetched by an earlier version of the selection rules, and
    today's rules pick a different span, or none, for 10 of the 61 filings
    (see ``selection_drift``). Every eval set was labeled against the corpus
    as fetched, so a re-render has to cover exactly that text.

    Raises:
        ValueError: the body doesn't occur in ``text`` exactly once, so the
            cache doesn't hold the filing the document came from.
    """
    body = pinned_body(ref, document)
    count = text.count(body)
    if count != 1:
        raise ValueError(f"{ref.filename}: extracted body occurs {count} times in the cached filing, expected 1")
    start = text.index(body)
    return start, start + len(body)


def verify_raw_cache(out_dir: Path, raw_dir: Path) -> list[str]:
    """Filenames the cache can't reproduce: not cached, or ``pinned_span`` fails.

    An empty result proves the cache holds the same filings, extracted by the
    same ``html_to_text``, as the corpus every eval set was labeled against.
    """
    index = load_raw_index(raw_dir)
    failed: list[str] = []
    for filings in pinned_accessions(out_dir).values():
        for accession, filename in filings.items():
            ref = index.get(accession)
            path = raw_html_path(raw_dir, accession)
            if ref is None or not path.exists():
                failed.append(filename)
                continue
            try:
                pinned_span(ref, (out_dir / filename).read_text(encoding="utf-8"), html_to_text(path.read_text(encoding="utf-8")))
            except ValueError as exc:
                logger.warning("%s", exc)
                failed.append(filename)
    return sorted(failed)


def selection_drift(out_dir: Path, raw_dir: Path) -> list[str]:
    """Filenames where today's ``extract_mda`` would select a different section, or none.

    A fresh fetch would drop or change these documents. Rendering is
    unaffected, since it goes through ``pinned_span``. Assumes a verified cache.
    """
    index = load_raw_index(raw_dir)
    drifted: list[str] = []
    for filings in pinned_accessions(out_dir).values():
        for accession, filename in filings.items():
            ref = index[accession]
            body = extract_mda(html_to_text(raw_html_path(raw_dir, accession).read_text(encoding="utf-8")), ref.form)
            if body != pinned_body(ref, (out_dir / filename).read_text(encoding="utf-8")):
                drifted.append(filename)
    return sorted(drifted)


def render_markdown_corpus(out_dir: Path, raw_dir: Path, dest_dir: Path) -> dict[str, Rendered]:
    """Re-render every document in ``out_dir`` as Markdown into ``dest_dir``, from the cache. Offline.

    Each document keeps its filename (so its ``Document.id`` and every eval
    set's ``expected_doc_ids``), its front matter and title, and covers exactly
    the text it covers today (``pinned_span``). Only the body's form changes:
    headings, Markdown tables, no page furniture. See ``edgar_markdown``.

    Raises:
        ValueError: a document isn't cached, or its body isn't found in the
            cached filing (run ``--cache-raw`` first).
    """
    index = load_raw_index(raw_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Rendered] = {}
    for filings in pinned_accessions(out_dir).values():
        for accession, filename in filings.items():
            ref = index.get(accession)
            if ref is None or not raw_html_path(raw_dir, accession).exists():
                raise ValueError(f"{filename}: not in the raw cache; run --cache-raw first")
            html = raw_html_path(raw_dir, accession).read_text(encoding="utf-8")
            start, end = pinned_span(ref, (out_dir / filename).read_text(encoding="utf-8"), html_to_text(html))
            rendered = render_span(html, start, end)
            (dest_dir / filename).write_text(render_document(ref, rendered.markdown), encoding="utf-8")
            results[filename] = rendered
            logger.info(
                "rendered %-34s %3d headings  %3d tables  %3d furniture dropped",
                filename,
                len(rendered.headings),
                rendered.tables,
                rendered.dropped_furniture,
            )
    return results


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
    parser.add_argument(
        "--backfill-front-matter",
        action="store_true",
        help="Add YAML front matter to already-fetched files that lack it. Offline.",
    )
    parser.add_argument(
        "--cache-raw",
        action="store_true",
        help=(
            "Download the HTML of exactly the filings already in the output directory "
            "(pinned by accession) into --raw-dir, then check each document's body is "
            "found in it exactly once."
        ),
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="Raw HTML cache (default: the manifest's directory + /raw).",
    )
    parser.add_argument(
        "--render-markdown",
        type=Path,
        metavar="DEST_DIR",
        default=None,
        help=(
            "Re-render every document as Markdown (headings, tables, no page furniture) "
            "from the raw cache into DEST_DIR. Offline; needs --cache-raw first."
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    configure_logging(logging.DEBUG if args.verbose else logging.INFO)

    if not args.manifest.exists():
        logger.error("Manifest not found: %s", args.manifest)
        return 1

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    out_dir = args.out_dir or args.manifest.parent / "documents"

    raw_dir = args.raw_dir or args.manifest.parent / "raw"
    if args.render_markdown:
        if args.render_markdown.resolve() == out_dir.resolve():
            logger.error("--render-markdown would overwrite the documents it renders from; pick another directory")
            return 1
        results = render_markdown_corpus(out_dir, raw_dir, args.render_markdown)
        logger.info("Rendered %d document(s) into %s", len(results), args.render_markdown)
        return 0

    if args.cache_raw:
        downloaded = cache_raw_filings(out_dir, raw_dir, args.user_agent)
        logger.info("Downloaded %d filing(s) into %s", downloaded, raw_dir)
        failed = verify_raw_cache(out_dir, raw_dir)
        if failed:
            logger.error("%d document(s) not reproduced from the cache: %s", len(failed), ", ".join(failed))
            return 1
        logger.info("Every document's extracted body found exactly once in its cached filing")
        drifted = selection_drift(out_dir, raw_dir)
        if drifted:
            logger.warning(
                "Today's MD&A selection differs for %d document(s); a fresh fetch would drop or change them: %s",
                len(drifted),
                ", ".join(drifted),
            )
        return 0

    if args.backfill_front_matter:
        changed = backfill_front_matter(out_dir)
        logger.info("Added front matter to %d file(s) in %s", changed, out_dir)
        return 0

    count = fetch_corpus(manifest, out_dir, args.user_agent, dry_run=args.dry_run)
    if not args.dry_run:
        total = sum(p.stat().st_size for p in out_dir.glob("*.md"))
        logger.info("Corpus ready: %d documents, %.1f MB in %s", count, total / 1e6, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
