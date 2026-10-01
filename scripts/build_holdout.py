#!/usr/bin/env python
"""Assemble, audit, review and freeze the fresh grouped holdout (`fresh_v1`).

Step 2 of the [evaluation rigor plan](../docs/evaluation-rigor-plan.md) and the
"untouched confirmation set" of Milestone 27. The protocol -- workload, task
families, grouping, split, review rubric and freeze rules -- is
[docs/holdout-protocol.md](../docs/holdout-protocol.md); this script enforces
the mechanical parts of it.

Why company-level groups
------------------------
Every existing EDGAR set is development data: the generated, period,
underspecified, table, multi-hop, adaptive and refusal sets have all informed
configuration choices. A fresh question is only independent of them if no
underlying fact crosses the boundary, and an EDGAR fact travels: a 10-K
restates last year's figures as comparatives, a 10-Q repeats the 10-K's
liquidity paragraph, a multi-hop question reuses two single-hop facts. All of
that stays inside one company, so the company is the grouping key, and every
sample whose companies straddle the split is refused. The split itself is a
seeded, sector-stratified draw (:func:`assign_splits`), recorded before any
question was written; the five companies of ``edgar_md_holdout`` are holdout
by construction (entity transfer).

The group is coarse on purpose: a finer key (filing, or fact) would need the
restatement links between filings to be found by hand before the split could
be trusted, and the company key needs none. Its cost is fewer, larger groups,
which the protocol's grouped bootstrap accounts for.

Lifecycle
---------
``assemble``
    Read the authored sources (compact JSON, one file per company) and write
    the draft file with every derived field filled in: tier, kind, split,
    group, transfer slice, matching mode and a pending review record.
    Refuses to overwrite a draft that carries any review decision.
``audit``
    Every mechanical check, on the whole draft: schema per expected behavior,
    verbatim evidence in an expected filing, copies in filings not listed,
    split integrity, fact overlap with the development sets, absence checks
    for unanswerable questions, and spans the configured chunker splits
    (reported, not refused: the plan keeps those tasks and tracks the loss).
``review``
    Print drafts with the context a reviewer needs: the evidence's section
    and paragraph, other filings with the same text, leakage flags.
``finalize``
    Refuse while any verdict is pending, re-run the audit on the accepted
    samples, then write one file per split and runner plus a freeze record
    pinning the corpus (sha256 per document), the question files and the
    split assignment.

Usage
-----
    python scripts/build_holdout.py split
    python scripts/build_holdout.py assemble SRC_DIR
    python scripts/build_holdout.py audit [--config PATH]
    python scripts/build_holdout.py review [--show pending|accept|reject|all] [--id ID]
    python scripts/build_holdout.py finalize [--config PATH]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.chunking.chunkers import get_chunker  # noqa: E402
from rag.config.settings import RagConfig, load_config  # noqa: E402
from rag.eval.dataset import MODE_SPAN_AND_DOCUMENT, EvalDataset, EvalSample, ExpectedSpan  # noqa: E402
from rag.eval.relevance import normalize  # noqa: E402
from rag.ingestion.corpora import load_selected_corpora  # noqa: E402
from rag.ingestion.models import Document  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "data" / "eval"
DRAFT = EVAL_DIR / "drafts" / "fresh_v1_draft.json"
HOLDOUT_DIR = EVAL_DIR / "holdout"
FREEZE = HOLDOUT_DIR / "fresh_v1_freeze.json"

#: Bumped when the protocol's rules change; recorded in the freeze file.
PROTOCOL_VERSION = "fresh_v1 / holdout-protocol 2026-10-01"

#: Every set that has been run against a configuration choice. All of it is
#: development data: no holdout sample may share a fact with any of it.
DEV_SETS = (
    "edgar_eval_set.json",
    "edgar_period_set.json",
    "edgar_underspecified_set.json",
    "edgar_table_set.json",
    "edgar_multihop_set.json",
    "edgar_adaptive_set.json",
    "edgar_refusal_set.json",
)

CORPORA = ("edgar_md", "edgar_md_holdout")

#: Companies already in `edgar`, by sector -- the strata of the split draw.
EXISTING_SECTORS: dict[str, tuple[str, ...]] = {
    "tech": ("AAPL", "MSFT", "NVDA"),
    "retail": ("COST", "TGT", "WMT"),
    "airlines": ("DAL", "LUV", "UAL"),
    "pharma": ("JNJ", "MRK", "PFE"),
    "energy": ("CVX", "XOM"),
}
#: Companies only in `edgar_md_holdout`: holdout by construction.
NEW_COMPANIES = ("AAL", "ALK", "AMD", "COP", "HD")
#: Fixed before any question was written (2026-10-01); see the protocol.
SPLIT_SEED = 20261001

DEV, HOLDOUT = "dev", "holdout"
FRESH_FACT, NEW_ENTITY = "fresh_fact", "new_entity"

#: Primary task families, in the order the rigor plan lists them.
FAMILIES = (
    "direct_lookup",
    "entity_period",
    "table_calc",
    "distributed",
    "synthesis",
    "revision",
    "ambiguity",
    "unanswerable",
)
#: What a correct response does.
ANSWER, CLARIFY, REFUSE, PARTIAL = "answer", "clarify", "refuse", "partial"
BEHAVIORS = (ANSWER, CLARIFY, REFUSE, PARTIAL)

VERDICTS = ("pending", "accept", "reject")
#: The rubric dimensions the harness plan requires a decision on.
REVIEW_DIMENSIONS = ("answerability", "usefulness", "clarity")

_FIGURE = re.compile(r"\$?\d[\d,]*(?:\.\d+)?\s*(?:%|percent|billion|million|thousand)?", re.IGNORECASE)
_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


# --------------------------------------------------------------------------
# Split
# --------------------------------------------------------------------------


def assign_splits(seed: int = SPLIT_SEED) -> dict[str, str]:
    """Company -> split: one existing company per sector to holdout, every new company holdout.

    One per sector keeps every sector's same-company-different-period
    distractors on both sides while leaving most existing companies for
    development, where the new task families need worked examples too.
    """
    rng = random.Random(seed)
    splits = {ticker: HOLDOUT for ticker in NEW_COMPANIES}
    for tickers in EXISTING_SECTORS.values():
        chosen = rng.choice(sorted(tickers))
        for ticker in tickers:
            splits[ticker] = HOLDOUT if ticker == chosen else DEV
    return splits


def ticker_of(doc_id: str) -> str:
    """`AAL_10-K_2025-12-31.md` -> `AAL`."""
    return doc_id.split("_", 1)[0]


# --------------------------------------------------------------------------
# Assemble
# --------------------------------------------------------------------------


def _spans_json(raw: list[Any]) -> list[str | dict[str, Any]]:
    return [ExpectedSpan.from_json(s).to_json() for s in raw]


def assemble_sample(src: dict[str, Any], splits: dict[str, str], *, drafted_with: str, drafted_on: str) -> dict[str, Any]:
    """Expand one authored sample into the draft schema.

    Authored samples carry only what a person writes: id, family, behavior,
    query, evidence (``spans``/``docs`` or ``parts``), the reference answer,
    and optional ``tags``, ``calc``, ``absent_terms``, ``companies`` and
    ``note``. Everything else is derived here so it can't drift.
    """
    behavior = src["behavior"]
    parts = src.get("parts") or []
    docs: list[str] = list(src.get("docs") or [])
    for part in parts:
        for doc_id in part.get("docs", []):
            if doc_id not in docs:
                docs.append(doc_id)
    companies = sorted(set(src.get("companies") or []) | {ticker_of(d) for d in docs})
    sample_splits = {splits.get(c, "unknown") for c in companies}
    split = sample_splits.pop() if len(sample_splits) == 1 else "mixed"

    out: dict[str, Any] = {
        "id": src["id"],
        "query": src["query"],
        "expected_doc_ids": docs,
    }
    spans = list(src.get("spans") or [])
    if parts:
        # The runners read the union of the parts' spans as the sample's evidence.
        spans = [s for part in parts for s in part.get("spans", [])]
    if spans:
        out["expected_spans"] = _spans_json(spans)
        out["matching_mode"] = MODE_SPAN_AND_DOCUMENT
    out["expected_answer"] = src["answer"]
    out["tier"] = "refusal" if behavior == REFUSE else "fresh"
    out["kind"] = src["family"]
    out["family"] = src["family"]
    out["tags"] = list(src.get("tags") or [])
    out["behavior"] = behavior
    out["companies"] = companies
    out["group"] = "+".join(companies)
    out["split"] = split
    out["transfer"] = NEW_ENTITY if any(c in NEW_COMPANIES for c in companies) else FRESH_FACT
    if parts:
        out["parts"] = [
            {"label": p["label"], "answer": p["answer"], "spans": _spans_json(p.get("spans", []))}
            | ({"available": False} if p.get("available") is False else {})
            for p in parts
        ]
        if src.get("conclusion"):
            out["conclusion"] = src["conclusion"]
    for key in ("calc", "absent_terms", "note"):
        if src.get(key):
            out[key] = src[key]
    out["origin"] = "authored_from_whole_filing"
    out["drafted_with"] = drafted_with
    out["drafted_on"] = drafted_on
    out["review"] = {
        "verdict": "pending",
        "reviewer": None,
        **{dim: {"decision": None, "reason": None} for dim in REVIEW_DIMENSIONS},
        "edits": None,
    }
    return out


def has_review_decision(sample: dict[str, Any]) -> bool:
    review = sample.get("review") or {}
    return review.get("verdict") != "pending" or any(
        (review.get(dim) or {}).get("decision") is not None for dim in REVIEW_DIMENSIONS
    )


# --------------------------------------------------------------------------
# Corpus views used by the checks
# --------------------------------------------------------------------------


@dataclass
class Corpus:
    """The pooled holdout corpus as the checks need it: normalized text and structure."""

    documents: dict[str, Document]
    normalized: dict[str, str] = field(default_factory=dict)
    paragraphs: dict[str, list[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for doc_id, doc in self.documents.items():
            self.normalized[doc_id] = normalize(doc.text)
            self.paragraphs[doc_id] = [normalize(p) for p in split_units(doc.text)]

    def containing(self, quote: str) -> list[str]:
        needle = normalize(quote)
        return sorted(d for d, text in self.normalized.items() if needle in text)

    def units_containing(self, doc_id: str, quote: str) -> set[int]:
        """Indices of the paragraphs (or table rows) that hold part of ``quote``."""
        needle = normalize(quote)
        units = self.paragraphs.get(doc_id, [])
        hits = {i for i, u in enumerate(units) if needle in u or (u and u in needle)}
        if hits:
            return hits
        # A quote spanning a paragraph break: credit every paragraph it touches.
        return {i for i, u in enumerate(units) if u and _overlaps(u, needle)}

    def section_of(self, doc_id: str, quote: str) -> str | None:
        """The heading path above the first occurrence of ``quote`` -- its stable location."""
        doc = self.documents.get(doc_id)
        if doc is None:
            return None
        needle = normalize(quote)
        stack: list[tuple[int, str]] = []
        for line in doc.text.splitlines():
            match = _HEADING.match(line)
            if match:
                level = len(match.group(1))
                stack = [(lvl, txt) for lvl, txt in stack if lvl < level] + [(level, match.group(2))]
                continue
            if needle and needle[:80] in normalize(line):
                # Skip the document title (level 1) -- it names the filing, not a section.
                return " > ".join(txt for lvl, txt in stack if lvl > 1) or None
        return None


def split_units(text: str) -> list[str]:
    """Paragraphs, with each Markdown table row its own unit.

    A table is one block of text, but each row is a separate fact: treating the
    whole table as one paragraph would flag a fresh row as overlapping a
    development row of the same table.
    """
    units: list[str] = []
    for block in re.split(r"\n\s*\n", text):
        lines = block.strip().splitlines()
        if lines and all(line.lstrip().startswith("|") for line in lines):
            units.extend(line.strip() for line in lines)
        elif block.strip():
            units.append(block.strip())
    return units


def _overlaps(unit: str, needle: str, min_chars: int = 40) -> bool:
    """Whether a quote crossing a paragraph break shares a long piece with ``unit``."""
    head, tail = needle[:min_chars], needle[-min_chars:]
    return len(needle) >= min_chars and (head in unit or tail in unit)


def figures(text: str) -> set[str]:
    """The figures in a passage specific enough to identify a fact, normalized.

    A currency amount or a magnitude (`$1.2 billion`), a number with four or
    more digits (`45,781`, `19.90`), or a percentage with a decimal (`22.5%`).
    Bare small percentages (`16%`), years and day numbers recur everywhere, and
    counting them would flag unrelated facts as leaks.
    """
    out = set()
    for match in _FIGURE.finditer(text):
        token = normalize(match.group(0)).strip().rstrip(",.")
        digits = re.sub(r"\D", "", token)
        if re.fullmatch(r"(19|20)\d\d", token):
            continue
        has_unit = "$" in token or any(u in token for u in ("billion", "million", "thousand"))
        is_percent = "%" in token or "percent" in token
        if has_unit or len(digits) >= 4 or (is_percent and len(digits) >= 3):
            out.add(token)
    return out


@dataclass(frozen=True)
class DevSpan:
    """One quote from a development set, where it came from and what it states."""

    set_name: str
    sample_id: str
    quote: str


def load_dev_spans(eval_dir: Path = EVAL_DIR) -> list[DevSpan]:
    """Every quote any development set grades on, alternatives and part spans included."""
    out: list[DevSpan] = []
    for name in DEV_SETS:
        path = eval_dir / name
        if not path.exists():
            continue
        for sample in EvalDataset.load(path):
            quotes = [q for span in sample.expected_spans for q in span.quotes]
            for part in sample.extra.get("parts") or []:
                quotes += [q for s in part.get("spans", []) for q in ExpectedSpan.from_json(s).quotes]
            out.extend(DevSpan(name, sample.id, q) for q in dict.fromkeys(quotes))
    return out


@dataclass
class DevIndex:
    """Where each development quote sits in the corpus, for the leakage check."""

    units: dict[tuple[str, int], list[DevSpan]]
    figures_by_company: dict[str, dict[str, list[DevSpan]]]

    @classmethod
    def build(cls, dev_spans: list[DevSpan], corpus: Corpus) -> DevIndex:
        units: dict[tuple[str, int], list[DevSpan]] = defaultdict(list)
        by_company: dict[str, dict[str, list[DevSpan]]] = defaultdict(lambda: defaultdict(list))
        for span in dev_spans:
            for doc_id in corpus.containing(span.quote):
                for i in corpus.units_containing(doc_id, span.quote):
                    units[(doc_id, i)].append(span)
                for fig in figures(span.quote):
                    by_company[ticker_of(doc_id)][fig].append(span)
        return cls(dict(units), {c: dict(f) for c, f in by_company.items()})

    def overlaps(self, corpus: Corpus, quote: str) -> list[str]:
        """Development samples that share this quote's paragraph, or a figure from the same company."""
        found: list[str] = []
        for doc_id in corpus.containing(quote):
            for i in corpus.units_containing(doc_id, quote):
                for span in self.units.get((doc_id, i), []):
                    found.append(f"same paragraph as {span.set_name}:{span.sample_id}")
            for fig in figures(quote):
                for span in self.figures_by_company.get(ticker_of(doc_id), {}).get(fig, []):
                    found.append(f"figure {fig!r} also in {span.set_name}:{span.sample_id}")
        return sorted(set(found))


# --------------------------------------------------------------------------
# Audit
# --------------------------------------------------------------------------


@dataclass
class Finding:
    sample_id: str
    level: str  # "error" refuses finalize; "warn" is for the reviewer
    message: str


def all_quotes(sample: EvalSample) -> list[tuple[str, list[str]]]:
    """(primary quote, every accepted quote) for each span on the sample."""
    return [(span.text, list(span.quotes)) for span in sample.expected_spans]


def check_schema(sample: EvalSample) -> list[str]:
    """Structural problems by expected behavior. Pure: no corpus needed."""
    x = sample.extra
    problems: list[str] = []
    if x.get("family") not in FAMILIES:
        problems.append(f"unknown family {x.get('family')!r}")
    behavior = x.get("behavior")
    if behavior not in BEHAVIORS:
        problems.append(f"unknown behavior {behavior!r}")
    if not sample.expected_answer:
        problems.append("no expected_answer (reference answer or criteria)")
    if x.get("split") not in (DEV, HOLDOUT):
        problems.append(f"split is {x.get('split')!r}: its companies straddle the split or are unknown")
    parts = x.get("parts") or []
    if behavior == ANSWER and not sample.expected_spans:
        problems.append("an answerable sample needs evidence spans")
    if behavior == REFUSE:
        if sample.expected_spans or parts:
            problems.append("a refusal sample must carry no evidence")
        if not x.get("absent_terms"):
            problems.append("a refusal sample needs absent_terms the audit can check")
        if x.get("tier") != "refusal":
            problems.append("a refusal sample must have tier 'refusal' (the answer judge keys on it)")
    if behavior == PARTIAL:
        if not any(p.get("available") is False for p in parts):
            problems.append("a partial sample needs at least one part marked available: false")
        if not any(p.get("available", True) and p.get("spans") for p in parts):
            problems.append("a partial sample needs at least one supported part with spans")
    for part in parts:
        if part.get("available", True) and not part.get("spans"):
            problems.append(f"part {part.get('label')!r} is answerable but has no spans")
    calc = x.get("calc")
    if calc is not None:
        for key in ("inputs", "formula", "units", "tolerance"):
            if key not in calc:
                problems.append(f"calc is missing {key!r}")
    if x.get("family") == "table_calc" and "calculation" in x.get("tags", []) and calc is None:
        problems.append("a calculation sample needs calc {inputs, formula, units, tolerance}")
    return problems


def audit(
    samples: list[EvalSample],
    corpus: Corpus,
    dev_index: DevIndex,
    splits: dict[str, str],
    chunk_texts: dict[str, list[str]] | None = None,
) -> list[Finding]:
    """Every mechanical check, per sample. See the module docstring."""
    findings: list[Finding] = []
    ids = Counter(s.id for s in samples)
    queries = Counter(normalize(s.query) for s in samples)

    for sample in samples:
        sid = sample.id

        def add(level: str, message: str, _sid: str = sid) -> None:
            findings.append(Finding(_sid, level, message))

        if ids[sid] > 1:
            add("error", "duplicate id")
        if queries[normalize(sample.query)] > 1:
            add("error", "duplicate query text")
        for problem in check_schema(sample):
            add("error", problem)

        companies = sample.extra.get("companies") or []
        for company in companies:
            if company not in splits:
                add("error", f"company {company} is in neither corpus")
        if {splits.get(c) for c in companies} != {sample.extra.get("split")}:
            add("error", f"split {sample.extra.get('split')!r} disagrees with its companies {companies}")

        for doc_id in sample.expected_doc_ids:
            if doc_id not in corpus.documents:
                add("error", f"expected document {doc_id} is not in the corpus")

        for primary, quotes in all_quotes(sample):
            found_in = sorted({d for q in quotes for d in corpus.containing(q)})
            if not set(found_in) & set(sample.expected_doc_ids):
                add("error", f"span not verbatim in any expected document: {primary[:90]!r}")
            for q in quotes:
                if not corpus.containing(q):
                    add("error", f"quote not found anywhere: {q[:90]!r}")
            elsewhere = sorted(set(found_in) - set(sample.expected_doc_ids))
            if elsewhere:
                add("warn", f"span also verbatim in unlisted {', '.join(elsewhere)} "
                            f"(add it if it states the same period's fact): {primary[:60]!r}")
            if len(normalize(primary)) < 25:
                add("warn", f"span is very short; it may not identify the fact: {primary!r}")
            for q in quotes:
                overlaps = dev_index.overlaps(corpus, q)
                if overlaps:
                    level = "error" if sample.extra.get("split") == HOLDOUT else "warn"
                    for o in overlaps:
                        add(level, f"leakage: {o} -- {q[:60]!r}")
            if chunk_texts is not None:
                needles = [normalize(q) for q in quotes]
                haystack = [t for d in sample.expected_doc_ids for t in chunk_texts.get(d, [])]
                if not any(n in t for n in needles for t in haystack):
                    add("warn", f"no chunk of the configured chunker contains this span "
                                f"(chunk-boundary loss, kept by protocol): {primary[:60]!r}")

        # Absence is checked in the companies' own filings: "guidance" missing
        # from Home Depot's MD&A is the claim, and ConocoPhillips' use of the
        # word says nothing about it.
        own = {d for d in corpus.documents if ticker_of(d) in companies}
        for term in sample.extra.get("absent_terms") or []:
            present = [d for d in corpus.containing(term) if d in own]
            if present:
                add("error", f"absent term {term!r} appears in {', '.join(present[:4])}"
                             f"{' ...' if len(present) > 4 else ''}")

    return findings


# --------------------------------------------------------------------------
# I/O helpers
# --------------------------------------------------------------------------


def load_corpus_view(config: RagConfig) -> tuple[Corpus, dict[str, list[str]]]:
    _, documents = load_selected_corpora(config, list(CORPORA), clean=True)
    chunks = get_chunker(config.chunking).chunk(documents)
    chunk_texts: dict[str, list[str]] = defaultdict(list)
    for chunk in chunks:
        chunk_texts[chunk.document_id].append(normalize(chunk.text))
    return Corpus({d.id: d for d in documents}), dict(chunk_texts)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def corpus_snapshot(config: RagConfig) -> dict[str, dict[str, str]]:
    """sha256 and accession of every raw document file in the pooled corpora."""
    snapshot: dict[str, dict[str, str]] = {}
    for name in CORPORA:
        entry = config.corpora.registry[name]
        for path in sorted(Path(entry.documents_dir).glob("*.md")):
            text = path.read_text(encoding="utf-8")
            accession = re.search(r"^accession:\s*(\S+)", text, re.MULTILINE)
            snapshot[path.name] = {
                "corpus": name,
                "sha256": sha256_text(text),
                "accession": accession.group(1) if accession else "",
            }
    return snapshot


def print_findings(findings: list[Finding]) -> None:
    by_level = Counter(f.level for f in findings)
    for f in sorted(findings, key=lambda f: (f.level != "error", f.sample_id)):
        print(f"  {f.level:<5} {f.sample_id}: {f.message}")
    print(f"\n{by_level['error']} error(s), {by_level['warn']} warning(s)")


def coverage_table(samples: list[EvalSample]) -> str:
    counts = Counter((s.extra.get("family"), s.extra.get("split")) for s in samples)
    transfer = Counter((s.extra.get("split"), s.extra.get("transfer")) for s in samples)
    behaviors = Counter((s.extra.get("behavior"), s.extra.get("split")) for s in samples)
    lines = [f"{'family':<16}{'dev':>6}{'holdout':>9}"]
    for fam in FAMILIES:
        lines.append(f"{fam:<16}{counts[(fam, DEV)]:>6}{counts[(fam, HOLDOUT)]:>9}")
    lines.append(f"{'total':<16}{sum(v for (_, s), v in counts.items() if s == DEV):>6}"
                 f"{sum(v for (_, s), v in counts.items() if s == HOLDOUT):>9}")
    lines.append("")
    for b in BEHAVIORS:
        lines.append(f"behavior {b:<8}{behaviors[(b, DEV)]:>6}{behaviors[(b, HOLDOUT)]:>9}")
    lines.append("")
    for (split, slice_), n in sorted(transfer.items()):
        lines.append(f"{split} {slice_}: {n}")
    groups = Counter((s.extra.get("split"), s.extra.get("group")) for s in samples)
    lines.append(f"groups: dev {sum(1 for (s, _) in groups if s == DEV)}, "
                 f"holdout {sum(1 for (s, _) in groups if s == HOLDOUT)}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------


def cmd_split() -> int:
    splits = assign_splits()
    for split in (DEV, HOLDOUT):
        print(f"{split:<8} {' '.join(sorted(t for t, s in splits.items() if s == split))}")
    return 0


def cmd_assemble(src_dir: Path, out: Path, drafted_with: str) -> int:
    if out.exists():
        existing = json.loads(out.read_text(encoding="utf-8"))
        if any(has_review_decision(s) for s in existing):
            print(f"Refusing to overwrite {out}: it carries review decisions. Edit it in place.")
            return 1
    splits = assign_splits()
    today = date.today().isoformat()
    samples: list[dict[str, Any]] = []
    for path in sorted(src_dir.glob("*.json")):
        for src in json.loads(path.read_text(encoding="utf-8")):
            samples.append(assemble_sample(src, splits, drafted_with=drafted_with, drafted_on=today))
    samples.sort(key=lambda s: (s["split"], s["group"], s["id"]))
    EvalDataset(samples=[EvalSample.from_dict(s) for s in samples]).save(out)
    print(f"Wrote {len(samples)} draft(s) to {out}")
    return 0


def _corpus_and_index(config_path: str | None) -> tuple[RagConfig, Corpus, dict[str, list[str]], DevIndex]:
    config = load_config(config_path)
    corpus, chunk_texts = load_corpus_view(config)
    return config, corpus, chunk_texts, DevIndex.build(load_dev_spans(), corpus)


def check_snapshot(config: RagConfig) -> list[str]:
    """Documents that differ from the frozen snapshot, if one exists."""
    if not FREEZE.exists():
        return []
    frozen = json.loads(FREEZE.read_text(encoding="utf-8"))["corpus"]
    now = corpus_snapshot(config)
    problems = [f"{d}: changed since the freeze" for d in frozen if d in now and now[d]["sha256"] != frozen[d]["sha256"]]
    problems += [f"{d}: missing" for d in frozen if d not in now]
    problems += [f"{d}: not in the frozen snapshot" for d in now if d not in frozen]
    return problems


def cmd_audit(path: Path, config_path: str | None) -> int:
    samples = EvalDataset.load(path).samples
    config, corpus, chunk_texts, dev_index = _corpus_and_index(config_path)
    findings = audit(samples, corpus, dev_index, assign_splits(), chunk_texts)
    findings += [Finding("(corpus)", "error", p) for p in check_snapshot(config)]
    print_findings(findings)
    print("\n" + coverage_table(samples))
    return 1 if any(f.level == "error" for f in findings) else 0


def cmd_review(path: Path, show: str, only: str | None, config_path: str | None) -> int:
    samples = EvalDataset.load(path).samples
    _, corpus, _, dev_index = _corpus_and_index(config_path)
    counts = Counter(s.extra["review"]["verdict"] for s in samples)
    print(f"{path}: {len(samples)} draft(s), " + ", ".join(f"{v} {counts[v]}" for v in VERDICTS))
    for sample in samples:
        x = sample.extra
        if only and sample.id != only:
            continue
        if not only and show != "all" and x["review"]["verdict"] != show:
            continue
        print(f"\n{'-' * 78}\n[{sample.id}]  {x['split']} / {x['family']} / {x['behavior']}"
              f"  ({x['transfer']}, group {x['group']})  verdict: {x['review']['verdict']}")
        if x.get("tags"):
            print(f"  tags:     {', '.join(x['tags'])}")
        print(f"  Q:        {sample.query}")
        print(f"  answer:   {sample.expected_answer}")
        for part in x.get("parts") or []:
            flag = "" if part.get("available", True) else "  [NOT IN CORPUS]"
            print(f"  part:     {part['label']}: {part['answer']}{flag}")
        if x.get("conclusion"):
            print(f"  conclude: {x['conclusion']}")
        if x.get("calc"):
            print(f"  calc:     {json.dumps(x['calc'], ensure_ascii=False)}")
        for primary, quotes in all_quotes(sample):
            print(f"  span:     {primary!r}")
            for alt in quotes[1:]:
                print(f"     or:    {alt!r}")
            for doc_id in sample.expected_doc_ids:
                if normalize(primary) in corpus.normalized.get(doc_id, ""):
                    print(f"     in:    {doc_id} § {corpus.section_of(doc_id, primary)}")
            others = sorted(set(corpus.containing(primary)) - set(sample.expected_doc_ids))
            if others:
                print(f"     also:  {', '.join(others)}")
            for o in dev_index.overlaps(corpus, primary):
                print(f"     LEAK?: {o}")
        if x.get("absent_terms"):
            print(f"  absent:   {', '.join(x['absent_terms'])}  (checked in the company's own filings)")
        if x.get("note"):
            print(f"  note:     {x['note']}")
    return 0


def runner_files(samples: list[EvalSample]) -> dict[str, list[EvalSample]]:
    """Split accepted samples by the runner that scores them.

    ``retrieval`` holds every sample with evidence spans (the retrieval runner
    scores a gold-less sample as a miss). ``answer`` holds single-answer
    samples, refusals included (the answer judge keys on ``tier``).
    ``multipart`` holds samples with ``parts`` (the multi-hop runner judges each
    part and the conclusion).
    """
    return {
        "retrieval": [s for s in samples if s.expected_spans],
        "answer": [s for s in samples if not s.extra.get("parts")],
        "multipart": [s for s in samples if s.extra.get("parts")],
    }


def cmd_finalize(path: Path, config_path: str | None) -> int:
    samples = EvalDataset.load(path).samples
    pending = [s.id for s in samples if s.extra["review"]["verdict"] == "pending"]
    unknown = [s.id for s in samples if s.extra["review"]["verdict"] not in VERDICTS]
    undecided = [
        s.id for s in samples
        if s.extra["review"]["verdict"] in ("accept", "reject")
        and not all((s.extra["review"].get(d) or {}).get("decision") for d in REVIEW_DIMENSIONS)
    ]
    if pending or unknown or undecided:
        print(f"Refusing: {len(pending)} pending, {len(unknown)} unknown verdict(s), "
              f"{len(undecided)} without a decision on every rubric dimension.")
        return 1
    accepted = [s for s in samples if s.extra["review"]["verdict"] == "accept"]
    config, corpus, chunk_texts, dev_index = _corpus_and_index(config_path)
    errors = [f for f in audit(accepted, corpus, dev_index, assign_splits(), chunk_texts) if f.level == "error"]
    if errors:
        print_findings(errors)
        print("Refusing: accepted samples fail a mechanical check.")
        return 1

    HOLDOUT_DIR.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    for split in (DEV, HOLDOUT):
        in_split = [s for s in accepted if s.extra["split"] == split]
        for runner, chosen in runner_files(in_split).items():
            out = HOLDOUT_DIR / f"fresh_v1_{split}_{runner}.json"
            for s in chosen:
                # Anchor each span to a section of the frozen snapshot, so the
                # evidence stays locatable if quotes are ever re-matched.
                if s.expected_spans:
                    s.extra["evidence_locations"] = [
                        {"span": span.text, "doc_id": doc_id, "section": corpus.section_of(doc_id, span.text)}
                        for span in s.expected_spans
                        for doc_id in s.expected_doc_ids
                        if normalize(span.text) in corpus.normalized.get(doc_id, "")
                    ]
            EvalDataset(samples=chosen).save(out)
            files[out.name] = sha256_text(out.read_text(encoding="utf-8"))
    reviewers = sorted({s.extra["review"].get("reviewer") or "unrecorded" for s in samples})
    freeze = {
        "protocol": PROTOCOL_VERSION,
        "frozen_on": date.today().isoformat(),
        "split_seed": SPLIT_SEED,
        "splits": assign_splits(),
        "draft": {"path": str(path.relative_to(ROOT)), "sha256": sha256_text(path.read_text(encoding="utf-8"))},
        "files": files,
        "corpus": corpus_snapshot(config),
        "counts": {
            "drafted": len(samples),
            "accepted": len(accepted),
            "by_family_split": {f"{f}/{s}": n for (f, s), n in sorted(
                Counter((x.extra["family"], x.extra["split"]) for x in accepted).items())},
        },
        "reviewers": reviewers,
        "single_reviewer_limitation": len([r for r in reviewers if r != "unrecorded"]) < 2,
        "holdout_uses": [],
    }
    FREEZE.write_text(json.dumps(freeze, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Froze {len(accepted)} of {len(samples)} draft(s); wrote {len(files)} file(s) and {FREEZE}")
    print("\n" + coverage_table(accepted))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("split", help="Print the seeded company split")
    asm = sub.add_parser("assemble", help="Expand authored sources into the draft file")
    asm.add_argument("src_dir", type=Path)
    asm.add_argument("--out", type=Path, default=DRAFT)
    asm.add_argument("--drafted-with", default="claude-opus-5-5")
    for name, help_ in (("audit", "Run every mechanical check"), ("finalize", "Freeze the accepted samples")):
        p = sub.add_parser(name, help=help_)
        p.add_argument("--draft", type=Path, default=DRAFT)
        p.add_argument("--config", default=None)
    rev = sub.add_parser("review", help="Print drafts with the context to judge them")
    rev.add_argument("--draft", type=Path, default=DRAFT)
    rev.add_argument("--config", default=None)
    rev.add_argument("--show", choices=("all", *VERDICTS), default="pending")
    rev.add_argument("--id", default=None)
    args = parser.parse_args()

    if args.command == "split":
        return cmd_split()
    if args.command == "assemble":
        return cmd_assemble(args.src_dir, args.out, args.drafted_with)
    if args.command == "audit":
        return cmd_audit(args.draft, args.config)
    if args.command == "review":
        return cmd_review(args.draft, args.show, args.id, args.config)
    return cmd_finalize(args.draft, args.config)


if __name__ == "__main__":
    raise SystemExit(main())
