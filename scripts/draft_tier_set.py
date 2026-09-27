#!/usr/bin/env python
"""Draft, review and finalize the `period` and `underspecified` eval tiers.

Phase 0 step 2 of the chunking and indexing plan. The generated set
(`edgar_eval_set.json`) has one shape of question: a single span in a single
filing, naming its company and period. Two defects the plan targets are
invisible to it, so each gets its own tier, in its own file, reported
separately and never averaged into the generated set's numbers.

`period`
    Drafted from the paragraphs repeated verbatim across filings of the same
    company: 440 of at least 200 characters, every one within a single
    company, less 6 table-shaped ones.
    The question names one filing's company and period-end date; the span is
    in that filing *and* in at least one other period's. Samples use
    ``matching_mode: span_and_document``, so the other period's identical copy
    counts as a miss. That tie is what metadata filtering (Phase 3) exists to
    break, and BM25 can't: the text is identical.

`underspecified`
    Rewrites of a random sample of generated questions, in two kinds, each
    reported on its own. Both keep the source sample's reviewed labels, which
    stay valid only because each rewrite keeps the question single-answer:

    * ``paraphrase`` keeps company and period but rewords the rest, so at most
      a third of its content words appear in the span (the generated set's
      median is 0.625 by :func:`content_overlap`).
    * ``implicit`` names the company only through a product, brand or
      description, never its name or ticker, and keeps the period.

    Dropping company or period outright was rejected: the question would then
    have many correct answers, and a label naming one of them would score the
    others as misses.

The LLM drafts, a person labels
-------------------------------
Nothing drafted here is auto-accepted. ``draft`` writes a file under
``data/eval/drafts/`` in which every sample carries
``review: {"verdict": "pending", ...}``. A person reads each one
(``review`` prints them with the context needed to judge), edits the query,
spans or answer in place if needed, and sets the verdict to ``accept`` or
``reject``. ``finalize`` re-runs every mechanical check on the accepted
samples, refuses while any verdict is still pending, and writes the tier
file. Commit it before measuring anything on it (the freeze rule).

What the reviewer checks, from the label check of the generated set
(docs/measured-results.md):

* **Other statements of the asked fact.** Search the expected filing for
  another chunk stating the same thing in other words and add it as a span
  ``alternative``, or reject. ``review`` lists chunks sharing the span's
  figures as a starting point, which catches restated numbers but not
  reworded prose.
* **The span states the fact whole**, not a fragment that changes its meaning.
* **The question pins the entity and period granularity** (segment, not only
  company; three months vs. year to date) and doesn't quote source jargon.
* For ``implicit``: the description identifies exactly one of the corpus's
  companies. "The airline" doesn't: there are three.
* For ``paraphrase``: the rewrite asks for the same fact as the source query.

Usage
-----
    python scripts/draft_tier_set.py draft period --attempts 150
    python scripts/draft_tier_set.py draft underspecified --per-kind 40
    python scripts/draft_tier_set.py review data/eval/drafts/edgar_period_draft.json
    python scripts/draft_tier_set.py finalize data/eval/drafts/edgar_period_draft.json
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_eval_set import MIN_SPAN_CHARS, parse_reply, verify_label  # noqa: E402

from rag.chunking.chunkers import get_chunker  # noqa: E402
from rag.chunking.models import Chunk  # noqa: E402
from rag.config.settings import load_config  # noqa: E402
from rag.eval.dataset import (  # noqa: E402
    MODE_SPAN_AND_DOCUMENT,
    EvalDataset,
    EvalSample,
    ExpectedSpan,
)
from rag.eval.relevance import normalize  # noqa: E402
from rag.generation.factory import get_llm_client  # noqa: E402
from rag.ingestion.cleaners import clean_documents  # noqa: E402
from rag.ingestion.loaders import load_corpus  # noqa: E402
from rag.ingestion.models import Document  # noqa: E402
from rag.logging_config import configure_logging  # noqa: E402

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent
CORPUS = REPO / "data/corpora/edgar/documents"
GENERATED_SET = REPO / "data/eval/edgar_eval_set.json"
DRAFTS = REPO / "data/eval/drafts"
TIER_FILES = {
    "period": REPO / "data/eval/edgar_period_set.json",
    "underspecified": REPO / "data/eval/edgar_underspecified_set.json",
}

# Spans up to this long. Not `chunking.chunk_overlap`, the generator's cap:
# that bounded a heuristic Phase 0 step 1 replaced. Every draft is checked to
# fit inside one chunk of the configured chunker, and every eval run reports
# spans a different chunker splits as `unmatchable_spans`. Short still matters
# for that reason: a 250-character span in 1,000-character chunks survives
# most boundaries.
MAX_SPAN_CHARS = 250

# Paragraphs this long or longer are what the plan's 437 count was taken over.
MIN_PARAGRAPH_CHARS = 200

# `paraphrase` rewrites may share at most this fraction of their content words
# with the span, by :func:`content_overlap`. The generated set's median is
# 0.625, and only 35 of its 174 questions are at or under a third.
MAX_PARAPHRASE_OVERLAP = 1 / 3

VERDICTS = ("pending", "accept", "reject")

# Question words that carry no topic: function words, and the words every
# question uses to name a period, which both kinds keep on purpose.
_STOPWORDS = frozenset(
    """a an the of in on at for to from by with and or as is was were be been are
    what which who whom how much many did does do its it their this that these
    those during over per than s company inc corp co fiscal year years quarter
    quarters month months ended ending three six nine twelve period periods
    january february march april may june july august september october november
    december""".split()
)
_PERIOD_TOKEN = re.compile(r"(19|20)\d\d|\d{1,2}|fy\d*|q[1-4]")
_WORD = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
_YEAR = re.compile(r"\b(?:19|20)\d\d\b")

# Figures a restatement would repeat: "$15.6 billion", "19%", "1,234".
_FIGURE = re.compile(r"\$?\d[\d,]*(?:\.\d+)?(?:\s*(?:%|percent|billion|million))?")

# Aliases the title's first word misses. ExxonMobil's title spells it as one
# word, but a question can write "Exxon" or "Exxon Mobil".
_EXTRA_ALIASES = {"XOM": ("exxon", "mobil")}


# ---------------------------------------------------------------------------
# Filing identity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Filing:
    """What a question needs to name to pin one document."""

    doc_id: str
    company: str
    ticker: str
    form: str
    period_end: date

    @classmethod
    def from_document(cls, document: Document) -> "Filing":
        title = str(document.metadata.get("title", ""))
        match = re.match(r"(.+?) \((\w+)\) (10-[KQ]) -- period ended (\d{4}-\d{2}-\d{2})", title)
        if match is None:
            raise ValueError(f"{document.id}: title {title!r} doesn't name company, form and period")
        company, ticker, form, period = match.groups()
        return cls(document.id, company, ticker, form, date.fromisoformat(period))

    @property
    def display_name(self) -> str:
        """The title's company name as a person writes it: "Costco Wholesale Corp", not
        "COSTCO WHOLESALE CORP /NEW"."""
        name = re.sub(r"\s*/\w+$", "", self.company)
        return name.title() if name.isupper() else name

    @property
    def period_phrase(self) -> str:
        """The exact wording a period question must use, e.g. "March 28, 2026"."""
        return f"{self.period_end:%B} {self.period_end.day}, {self.period_end.year}"

    @property
    def descriptor(self) -> str:
        kind = "fiscal year" if self.form == "10-K" else "quarter"
        return f"Form {self.form} for the {kind} ended {self.period_phrase}"

    @property
    def name_tokens(self) -> frozenset[str]:
        """Lowercase words that name the company outright. `implicit` questions use none.

        Tickers aren't here: COST is "cost" once lowercased. :func:`names_company`
        matches them case-sensitively instead.
        """
        first = re.sub(r"[^a-z]", "", self.company.split()[0].lower())
        return frozenset({first, *_EXTRA_ALIASES.get(self.ticker, ())})


def load_filings(documents: list[Document]) -> dict[str, Filing]:
    return {d.id: Filing.from_document(d) for d in documents}


def question_words(text: str) -> list[str]:
    return _WORD.findall(normalize(text))


def names_company(question: str, filing: Filing) -> bool:
    """True if the question names the company or its ticker.

    Some names are also ordinary words (Target, Delta, United), so this can
    flag a question that only uses the word. That errs toward rejecting an
    `implicit` draft, which is the safe direction.
    """
    if re.search(rf"\b{filing.ticker}\b", question):
        return True
    return any(w in filing.name_tokens for w in question_words(question))


def content_overlap(question: str, spans: list[str], exclude: frozenset[str] = frozenset()) -> float:
    """Fraction of the question's topic words that also appear in its spans.

    Function words, period words and ``exclude`` (the company's names) don't
    count: both tiers keep company and period on purpose, so counting them
    would score every rewrite as overlapping. Returns 0.0 for a question with
    no topic words at all.
    """
    span_words = set(question_words(" ".join(spans)))
    words = [
        w
        for w in question_words(question)
        if w not in _STOPWORDS and w not in exclude and not _PERIOD_TOKEN.fullmatch(w)
    ]
    if not words:
        return 0.0
    return sum(w in span_words for w in words) / len(words)


def figures(text: str) -> list[str]:
    """The figures in a span worth searching for elsewhere.

    Amounts, percentages, decimals and thousands only: a day or a year would
    match half the filing.
    """
    found = [f.strip().rstrip(",.") for f in _FIGURE.findall(text)]
    return [f for f in found if re.search(r"[$%]|\d\.\d|\d,\d{3}|billion|million|percent", f)]


# ---------------------------------------------------------------------------
# Corpus lookups
# ---------------------------------------------------------------------------


class Corpus:
    """Normalized text per document and per chunk, for span and figure lookups."""

    def __init__(self, documents: list[Document], chunks: list[Chunk]) -> None:
        self.documents = documents
        self.filings = load_filings(documents)
        self.text = {d.id: normalize(d.text) for d in documents}
        self.chunks: dict[str, list[Chunk]] = defaultdict(list)
        for chunk in chunks:
            self.chunks[chunk.document_id].append(chunk)
        self._chunk_text = {c.id: normalize(c.text) for c in chunks}

    def documents_containing(self, quote: str) -> list[str]:
        needle = normalize(quote)
        return sorted(doc_id for doc_id, text in self.text.items() if needle in text)

    def chunks_containing(self, doc_id: str, quote: str) -> list[str]:
        needle = normalize(quote)
        return [c.id for c in self.chunks[doc_id] if needle in self._chunk_text[c.id]]

    def restatement_candidates(self, doc_ids: list[str], span: str) -> list[str]:
        """Chunks of ``doc_ids`` stating every figure in ``span`` without quoting it.

        A reviewer's starting point for alternatives, not a verdict: it finds a
        restated number, never a reworded sentence.
        """
        wanted = [normalize(f) for f in figures(span)]
        if not wanted:
            return []
        needle = normalize(span)
        out = []
        for doc_id in doc_ids:
            for chunk in self.chunks[doc_id]:
                text = self._chunk_text[chunk.id]
                if needle not in text and all(f in text for f in wanted):
                    out.append(chunk.id)
        return out


def repeated_paragraphs(documents: list[Document]) -> list[tuple[str, list[str]]]:
    """Paragraphs of at least MIN_PARAGRAPH_CHARS appearing verbatim in two or more filings.

    Table-shaped paragraphs are dropped: tables get their own tier after
    Phase 4 re-renders them, and spans written against today's pipe rows
    would stop matching.
    """
    where: dict[str, set[str]] = defaultdict(set)
    for document in documents:
        for paragraph in re.split(r"\n\s*\n", document.text):
            paragraph = paragraph.strip()
            if len(paragraph) >= MIN_PARAGRAPH_CHARS:
                where[paragraph].add(document.id)
    return sorted(
        (p, sorted(ids))
        for p, ids in where.items()
        if len(ids) > 1 and p.count("|") <= 4
    )


# ---------------------------------------------------------------------------
# Drafting: period
# ---------------------------------------------------------------------------

PERIOD_SYSTEM_PROMPT = (
    "You write evaluation questions for a search system over SEC filings. "
    "You are given one paragraph from a filing and the filing it is from. The "
    "same paragraph also appears word for word in the company's filings for "
    "other periods, so the question must identify THIS filing: it must name the "
    "company and state the period end date exactly as given. "
    "Ask about a substantive fact, policy, risk or explanation that the "
    "paragraph states, phrased the way an analyst would, not by restating the "
    "paragraph. Do not ask about forward-looking-statement boilerplate. "
    "Also return the shortest verbatim quote from the paragraph that answers "
    "the question, copied EXACTLY, character for character, no ellipsis, and a "
    "one-sentence answer. The quote must be a complete clause, not a bare "
    "figure. "
    'Reply with only a JSON object: {"question": "...", "answer_span": "...", '
    '"answer": "..."} and nothing else.'
)


def build_period_prompt(paragraph: str, filing: Filing) -> str:
    return (
        f"<company>{filing.display_name}</company>\n"
        f"<filing>{filing.descriptor}</filing>\n"
        f"<period_end>{filing.period_phrase}</period_end>\n\n"
        f"<paragraph>\n{paragraph}\n</paragraph>\n\n"
        f"The question must contain the exact text \"{filing.period_phrase}\". "
        f"The answer_span must be one sentence or clause of at most {MAX_SPAN_CHARS} characters and must "
        "appear word-for-word inside the paragraph.\nJSON:"
    )


@dataclass
class PeriodTarget:
    # Position in the sorted repeated-paragraph pool: stable across runs, so
    # it names the sample.
    index: int
    paragraph: str
    doc_ids: list[str]
    target: str


def sample_period_targets(
    pool: list[tuple[str, list[str]]], count: int, seed: int
) -> list[PeriodTarget]:
    """Round-robin across companies, one paragraph each turn, target filing at random.

    By company rather than uniformly because COST, MRK and MSFT hold 40% of
    the repeated paragraphs, and a tier that is mostly three companies says
    little about the other ten.
    """
    rng = random.Random(seed)
    by_company: dict[str, list[tuple[int, str, list[str]]]] = defaultdict(list)
    for index, (paragraph, doc_ids) in enumerate(pool):
        by_company[doc_ids[0].split("_")[0]].append((index, paragraph, doc_ids))
    for paragraphs in by_company.values():
        rng.shuffle(paragraphs)
    companies = sorted(by_company)
    rng.shuffle(companies)

    targets: list[PeriodTarget] = []
    while len(targets) < count and any(by_company.values()):
        for company in companies:
            if by_company[company] and len(targets) < count:
                index, paragraph, doc_ids = by_company[company].pop()
                targets.append(PeriodTarget(index, paragraph, doc_ids, rng.choice(doc_ids)))
    return targets


class Tally:
    """Counts rejections by reason and keeps each rejected draft.

    The counts say how a run's yield broke down; the drafts say whether a
    check is wrong. A 70% verification failure rate looked like a bad drafter
    until the rejected drafts showed a verifier failing sound labels.
    """

    def __init__(self) -> None:
        self.reasons: Counter[str] = Counter()
        self.rejected: list[dict[str, Any]] = []
        self._context: dict[str, Any] = {}

    def _reject(self, reason: str) -> None:
        self.reasons[reason] += 1
        self.rejected.append({"reason": reason, **self._context})
        return None


class PeriodValidator(Tally):
    """Mechanical checks on a period draft."""

    def __init__(self, corpus: Corpus) -> None:
        super().__init__()
        self.corpus = corpus

    def validate(self, target: PeriodTarget, reply: dict[str, Any] | None) -> EvalSample | None:
        self._context = {"target": target.target, "paragraph_index": target.index}
        if reply is None:
            return self._reject("generation_failed")
        question = str(reply.get("question", "")).strip()
        span = str(reply.get("answer_span", "")).strip()
        answer = str(reply.get("answer", "")).strip()
        self._context.update(question=question, span=span, answer=answer)
        filing = self.corpus.filings[target.target]
        if not question or not span:
            return self._reject("generation_failed")
        if not MIN_SPAN_CHARS <= len(span) <= MAX_SPAN_CHARS:
            return self._reject("span_length")
        if normalize(span) not in normalize(target.paragraph):
            return self._reject("span_not_verbatim")
        if not names_company(question, filing):
            return self._reject("question_missing_company")
        if normalize(filing.period_phrase) not in normalize(question):
            return self._reject("question_missing_period")
        docs = self.corpus.documents_containing(span)
        if target.target not in docs or len(docs) < 2:
            # Can't happen for a span inside a repeated paragraph unless
            # cleaning and paragraph splitting disagree; worth a count if so.
            return self._reject("span_not_repeated")
        if not self.corpus.chunks_containing(target.target, span):
            return self._reject("span_split_across_chunks")
        if content_overlap(question, [span], filing.name_tokens) >= 0.9:
            return self._reject("question_copies_span")

        competing = [d for d in docs if d != target.target]
        return EvalSample(
            id=f"period::{target.target}::para{target.index}",
            query=question,
            expected_spans=[ExpectedSpan(text=span)],
            expected_doc_ids=[target.target],
            expected_answer=answer or None,
            explicit_mode=MODE_SPAN_AND_DOCUMENT,
            extra={
                "tier": "period",
                "competing_doc_ids": competing,
                "review": {
                    "verdict": "pending",
                    "note": "",
                    "paragraph": target.paragraph,
                    "restatement_candidates": self.corpus.restatement_candidates(
                        [target.target], span
                    ),
                },
            },
        )


# ---------------------------------------------------------------------------
# Drafting: underspecified
# ---------------------------------------------------------------------------

PARAPHRASE_SYSTEM_PROMPT = (
    "You rewrite search questions about SEC filings the way a person who has "
    "not read the filing would ask them. Keep the company name and the exact "
    "period, and ask for exactly the same fact. Reword everything else in "
    "plain everyday language: avoid the accounting terms and phrases used in "
    "the source passage, and use synonyms or a description of what is being "
    "asked instead. Reply with only the rewritten question."
)

# The first draft of this prompt leaked the name in 24 of 60 rewrites, always
# through a brand that contains it ("the operator of Costco warehouses", "the
# airline that operates Delta SkyMiles"). The forbidden words are now spelled
# out per company, and a leak gets one retry naming the word.
IMPLICIT_SYSTEM_PROMPT = (
    "You rewrite search questions about SEC filings so they do not name the "
    "company. Replace the company's name with a description, product or brand "
    "that identifies it uniquely among these companies: {companies}. For "
    "example 'the iPhone maker', 'the Atlanta-based airline' or 'the operator "
    "of Sam's Club'. The rewrite must not contain any of these words, on their "
    "own or inside a brand name: {forbidden}. Keep the period exactly and ask "
    "for exactly the same fact. Change nothing else. "
    "Reply with only the rewritten question."
)

# The generator's verifier fails every period draft: its "BAD if the passage
# is about a different period" fires because the question names a period and
# the span, identical across periods by construction, can't. This one says
# where the passage comes from, and is otherwise as strict.
PERIOD_VERIFY_SYSTEM_PROMPT = (
    "You check evaluation labels for a document retrieval benchmark. "
    "You are given a question, a passage quoted from the filing the question "
    "names, and a proposed answer. The passage itself need not restate the "
    "company or the period: it is known to come from that filing. "
    "Reply GOOD only if the passage genuinely answers what the question asks "
    "AND the proposed answer says what the passage states. "
    "Reply BAD if the passage is about a different topic or quantity, answers "
    "only part of the question, or the proposed answer adds claims the passage "
    "does not make. "
    "Be strict: a label you are unsure about is BAD. "
    "Reply with one word, GOOD or BAD, and nothing else."
)

# For `implicit`, "same company" can't be judged from the two questions: the
# check rejected "the Atlanta-based airline" for Delta and passed "the maker
# of Tylenol" for J&J (Tylenol is Kenvue's since 2023). So the meaning check
# is told the company reference changed, and identification is a separate
# blind call: which corpus company does the rewrite refer to?
IMPLICIT_SAME_SYSTEM_PROMPT = (
    "You check rewrites of search questions. The rewrite deliberately refers to "
    "the company by a description, product or brand instead of its name: do "
    "NOT judge that part. Reply SAME only if, apart from how the company is "
    "referred to, both questions ask for the same fact about the same period, "
    "segment and quantity. Reply DIFFERENT otherwise. Be strict. Reply with one "
    "word, SAME or DIFFERENT."
)

IDENTIFY_SYSTEM_PROMPT = (
    "Which one of these companies does the question refer to: {companies}? "
    "Use what is true today, including spin-offs and divestitures. Reply with "
    "the company's name exactly as listed, or UNSURE if the question could "
    "refer to more than one of them or to none. Reply with nothing else."
)

SAME_QUESTION_SYSTEM_PROMPT = (
    "You check rewrites of search questions. Reply SAME only if both questions "
    "ask for the same fact about the same company and the same period, so one "
    "correct answer answers both. Reply DIFFERENT otherwise, including when the "
    "rewrite is vaguer about the period, the segment or the quantity. Be "
    "strict. Reply with one word, SAME or DIFFERENT."
)


def spans_of(sample: EvalSample) -> list[str]:
    return [s.text for s in sample.expected_spans]


class UnderspecifiedValidator(Tally):
    def __init__(self, corpus: Corpus) -> None:
        super().__init__()
        self.corpus = corpus

    def validate(
        self, kind: str, source: EvalSample, rewrite: str | None, same: bool
    ) -> EvalSample | None:
        self._context = {"source_sample": source.id, "source_query": source.query,
                         "rewrite": rewrite}
        if not rewrite:
            return self._reject(f"{kind}:generation_failed")
        filing = self.corpus.filings[source.expected_doc_ids[0]]
        if normalize(rewrite) == normalize(source.query):
            return self._reject(f"{kind}:unchanged")
        if set(_YEAR.findall(source.query)) - set(_YEAR.findall(rewrite)):
            return self._reject(f"{kind}:dropped_period")
        overlap = content_overlap(rewrite, spans_of(source), filing.name_tokens)
        if kind == "paraphrase":
            if not names_company(rewrite, filing):
                return self._reject(f"{kind}:dropped_company")
            if overlap > MAX_PARAPHRASE_OVERLAP:
                return self._reject(f"{kind}:overlap_too_high")
        elif names_company(rewrite, filing):
            return self._reject(f"{kind}:names_company")
        if not same:
            return self._reject(f"{kind}:changed_meaning")

        extra = {
            k: v for k, v in source.extra.items() if k not in ("tier", "source_chunk", "label_fixes")
        }
        return EvalSample(
            id=f"{kind}::{source.id}",
            query=rewrite,
            expected_spans=source.expected_spans,
            expected_doc_ids=source.expected_doc_ids,
            expected_answer=source.expected_answer,
            explicit_mode=source.explicit_mode,
            extra={
                **extra,
                "tier": "underspecified",
                "kind": kind,
                "source_sample": source.id,
                "source_query": source.query,
                "content_overlap": round(overlap, 2),
                "review": {"verdict": "pending", "note": ""},
            },
        )


def sample_sources(samples: list[EvalSample], per_kind: int, seed: int) -> dict[str, list[EvalSample]]:
    """Disjoint random samples of the generated set, one per kind.

    Random, never chosen from the current config's misses (the freeze rule):
    a tier drawn from failures makes any change aimed at them look good.
    """
    rng = random.Random(seed)
    pool = sorted(samples, key=lambda s: s.id)
    rng.shuffle(pool)
    return {"paraphrase": pool[:per_kind], "implicit": pool[per_kind : 2 * per_kind]}


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def _load(
    config_path: str | None, model: str | None, check_model: str | None
) -> tuple[Corpus, Any, Any]:
    """The corpus, the drafting client, and the client for every check call.

    Checks default to the drafting model. A different one is better: a model
    grading its own drafts shares their blind spots.
    """
    config = load_config(config_path)
    draft_llm = config.llm.model_copy(update={"model": model or config.llm.model})
    check_llm = draft_llm.model_copy(update={"model": check_model or draft_llm.model})
    documents = clean_documents(load_corpus(CORPUS))
    chunks = get_chunker(config.chunking).chunk(documents)
    logger.info("Corpus: %d document(s) -> %d chunk(s); drafting with %s, checking with %s",
                len(documents), len(chunks), draft_llm.model, check_llm.model)
    return Corpus(documents, chunks), get_llm_client(draft_llm), get_llm_client(check_llm)


def _parallel(fn: Callable[[Any], Any], items: list[Any], concurrency: int) -> list[Any]:
    """Map ``fn`` over ``items`` concurrently, results in input order."""
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        results = []
        for i, result in enumerate(pool.map(fn, items), 1):
            results.append(result)
            if i % 20 == 0:
                logger.info("  ... %d/%d", i, len(items))
        return results


def _generate(llm: Any, prompt: str, system: str) -> str | None:
    try:
        return str(llm.generate(prompt, system=system)).strip()
    except Exception:  # noqa: BLE001 -- one bad call must not kill the run
        logger.exception("Generation failed")
        return None


def draft_period(args: argparse.Namespace) -> tuple[list[EvalSample], list[dict[str, Any]]]:
    corpus, llm, checker = _load(args.config, args.model, args.check_model)
    pool = repeated_paragraphs(corpus.documents)
    logger.info("%d repeated paragraph(s) to draw from", len(pool))
    targets = sample_period_targets(pool, args.attempts, args.seed)

    def draft(target: PeriodTarget) -> dict[str, Any] | None:
        reply = _generate(
            llm,
            build_period_prompt(target.paragraph, corpus.filings[target.target]),
            PERIOD_SYSTEM_PROMPT,
        )
        return parse_reply(reply) if reply else None

    validator = PeriodValidator(corpus)
    drafted = [
        s for t, r in zip(targets, _parallel(draft, targets, args.concurrency))
        if (s := validator.validate(t, r)) is not None
    ]

    def verify(sample: EvalSample) -> bool:
        return verify_label(checker, sample.query, spans_of(sample)[0],
                            sample.expected_answer or "", label=sample.id,
                            system=PERIOD_VERIFY_SYSTEM_PROMPT)

    verdicts = _parallel(verify, drafted, args.concurrency)
    for sample, ok in zip(drafted, verdicts):
        if not ok:
            validator.reasons["failed_verification"] += 1
            validator.rejected.append({
                "reason": "failed_verification", "target": sample.expected_doc_ids[0],
                "question": sample.query, "span": spans_of(sample)[0],
                "answer": sample.expected_answer,
            })
    _report(len(targets), validator.reasons)
    return [s for s, ok in zip(drafted, verdicts) if ok], validator.rejected


def draft_underspecified(
    args: argparse.Namespace,
) -> tuple[list[EvalSample], list[dict[str, Any]]]:
    corpus, llm, checker = _load(args.config, args.model, args.check_model)
    sources = sample_sources(EvalDataset.load(GENERATED_SET).samples, args.per_kind, args.seed)
    companies = ", ".join(sorted({f.display_name for f in corpus.filings.values()}))
    jobs = [(kind, s) for kind, samples in sources.items() for s in samples]
    # What the blind identification call answered, per source sample, so a
    # rejection says whether the description or the fact was the problem.
    identified: dict[str, str] = {}

    def system_for(kind: str, filing: Filing) -> str:
        if kind == "paraphrase":
            return PARAPHRASE_SYSTEM_PROMPT
        forbidden = ", ".join(sorted(w.title() for w in filing.name_tokens) + [filing.ticker])
        return IMPLICIT_SYSTEM_PROMPT.format(companies=companies, forbidden=forbidden)

    def first_line(text: str | None) -> str | None:
        return text.strip().strip('"').splitlines()[0].strip() if text else None

    def rewrite(job: tuple[str, EvalSample]) -> tuple[str | None, bool]:
        kind, source = job
        filing = corpus.filings[source.expected_doc_ids[0]]
        prompt = f"<question>{source.query}</question>\nRewritten question:"
        text = first_line(_generate(llm, prompt, system_for(kind, filing)))
        if text and kind == "implicit" and names_company(text, filing):
            retry = (f"{prompt} {text}\n\nThat still names the company. Rewrite it "
                     "again without any of the forbidden words.\nRewritten question:")
            text = first_line(_generate(llm, retry, system_for(kind, filing)))
        if not text:
            return None, False
        check = _generate(
            checker,
            f"<original>{source.query}</original>\n<rewrite>{text}</rewrite>\n\nSAME or DIFFERENT:",
            IMPLICIT_SAME_SYSTEM_PROMPT if kind == "implicit" else SAME_QUESTION_SYSTEM_PROMPT,
        )
        # Fails closed, like the generator's verifier: an unreadable check
        # drops the draft rather than passing it to review as if checked.
        same = bool(check) and check.upper().startswith("SAME")
        if same and kind == "implicit":
            named = _generate(checker, f"<question>{text}</question>",
                              IDENTIFY_SYSTEM_PROMPT.format(companies=companies))
            identified[source.id] = named or ""
            same = bool(named) and names_company(named, filing)
        return text, same

    validator = UnderspecifiedValidator(corpus)
    drafted = [
        s for (kind, source), (text, same) in zip(jobs, _parallel(rewrite, jobs, args.concurrency))
        if (s := validator.validate(kind, source, text, same)) is not None
    ]
    for record in validator.rejected:
        if record["source_sample"] in identified:
            record["identified_as"] = identified[record["source_sample"]]
    _report(len(jobs), validator.reasons)
    return drafted, validator.rejected


def _report(attempted: int, reasons: Counter[str]) -> None:
    print(f"\n{'=' * 72}\n  {attempted} attempt(s)\n{'=' * 72}")
    for reason, n in sorted(reasons.items()):
        print(f"  rejected: {reason:<36} {n:>4}")


def review(path: Path, show: str) -> None:
    samples = EvalDataset.load(path).samples
    counts = Counter(s.extra["review"]["verdict"] for s in samples)
    print(f"{path}: {len(samples)} draft(s), " + ", ".join(f"{v} {counts[v]}" for v in VERDICTS))
    for sample in samples:
        verdict = sample.extra["review"]["verdict"]
        if show != "all" and verdict != show:
            continue
        print(f"\n{'-' * 72}\n[{sample.id}]  verdict: {verdict}")
        if sample.extra.get("tier") == "underspecified":
            print(f"  kind:     {sample.extra['kind']}  (overlap {sample.extra['content_overlap']})")
            print(f"  source Q: {sample.extra['source_query']}")
        print(f"  Q:        {sample.query}")
        for span in sample.expected_spans:
            print(f"  span:     {span.text!r}")
            for alt in span.alternatives:
                print(f"     or:    {alt!r}")
        print(f"  answer:   {sample.expected_answer}")
        print(f"  expected: {', '.join(sample.expected_doc_ids)}")
        info = sample.extra["review"]
        if "competing_doc_ids" in sample.extra:
            print(f"  also in:  {', '.join(sample.extra['competing_doc_ids'])}")
        if info.get("restatement_candidates"):
            print(f"  CHECK restated figures in: {', '.join(info['restatement_candidates'])}")
        if "paragraph" in info:
            print(f"  paragraph: {info['paragraph']}")
        if info.get("suggestion"):
            reason = f" -- {info['suggestion_reason']}" if info.get("suggestion_reason") else ""
            print(f"  suggest:  {info['suggestion']}{reason}")
        if info.get("note"):
            print(f"  note:     {info['note']}")


def check_accepted(sample: EvalSample, corpus: Corpus) -> list[str]:
    """Mechanical checks re-run at finalize, after a reviewer's edits."""
    problems = []
    for span in sample.expected_spans:
        if not any(set(corpus.documents_containing(q)) & set(sample.expected_doc_ids)
                   for q in span.quotes):
            problems.append(f"span not found in any expected document: {span.text!r}")
    filing = corpus.filings[sample.expected_doc_ids[0]]
    if sample.extra.get("tier") == "period":
        if sample.matching_mode != MODE_SPAN_AND_DOCUMENT:
            problems.append("period samples must use span_and_document matching")
        others = set(corpus.documents_containing(sample.expected_spans[0].text)) - set(
            sample.expected_doc_ids
        )
        if not others:
            problems.append("span no longer appears in another period's filing")
        if normalize(filing.period_phrase) not in normalize(sample.query):
            problems.append("question doesn't state the period end date")
    elif sample.extra.get("kind") == "implicit" and names_company(sample.query, filing):
        problems.append("implicit question names the company")
    elif sample.extra.get("kind") == "paraphrase" and not names_company(sample.query, filing):
        problems.append("paraphrase question dropped the company")
    return problems


def finalize(path: Path, out: Path | None, config_path: str | None) -> int:
    samples = EvalDataset.load(path).samples
    pending = [s.id for s in samples if s.extra["review"]["verdict"] == "pending"]
    unknown = [s.id for s in samples if s.extra["review"]["verdict"] not in VERDICTS]
    if pending or unknown:
        print(f"Refusing: {len(pending)} pending, {len(unknown)} with an unknown verdict. "
              "Every draft needs accept or reject before the tier is frozen.")
        return 1

    config = load_config(config_path)
    documents = clean_documents(load_corpus(CORPUS))
    corpus = Corpus(documents, get_chunker(config.chunking).chunk(documents))
    accepted = [s for s in samples if s.extra["review"]["verdict"] == "accept"]
    failed = {s.id: p for s in accepted if (p := check_accepted(s, corpus))}
    for sample_id, problems in failed.items():
        for problem in problems:
            print(f"  {sample_id}: {problem}")
    if failed:
        print(f"Refusing: {len(failed)} accepted sample(s) fail a mechanical check.")
        return 1

    tier = accepted[0].extra["tier"] if accepted else ""
    out = out or TIER_FILES[tier]
    for sample in accepted:
        # The review block is the drafting record; it stays in the draft file.
        sample.extra = {k: v for k, v in sample.extra.items() if k != "review"}
    EvalDataset(samples=accepted).save(out)
    by_kind = Counter(s.extra.get("kind", tier) for s in accepted)
    print(f"Wrote {len(accepted)} of {len(samples)} draft(s) to {out}: {dict(by_kind)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    draft = sub.add_parser("draft", help="Draft a tier for review")
    draft.add_argument("tier", choices=sorted(TIER_FILES))
    draft.add_argument("--attempts", type=int, default=150, help="period: paragraphs to try")
    draft.add_argument("--per-kind", type=int, default=40,
                       help="underspecified: source questions per kind")
    draft.add_argument("--seed", type=int, default=0)
    draft.add_argument("--concurrency", type=int, default=4)
    draft.add_argument("--config", default=None)
    draft.add_argument("--model", default=None, help="Override llm.model for drafting")
    draft.add_argument("--check-model", default=None,
                       help="Model for verification and rewrite checks (default: --model)")
    draft.add_argument("--out", type=Path, default=None)

    show = sub.add_parser("review", help="Print drafts with the context to judge them")
    show.add_argument("path", type=Path)
    show.add_argument("--show", choices=("all", *VERDICTS), default="pending")

    fin = sub.add_parser("finalize", help="Write the accepted drafts as the tier file")
    fin.add_argument("path", type=Path)
    fin.add_argument("--out", type=Path, default=None)
    fin.add_argument("--config", default=None)

    args = parser.parse_args()
    configure_logging()

    if args.command == "review":
        review(args.path, args.show)
        return 0
    if args.command == "finalize":
        return finalize(args.path, args.out, args.config)

    out = args.out or DRAFTS / f"edgar_{args.tier}_draft.json"
    if out.exists():
        print(f"Refusing to overwrite {out}: it may hold review verdicts. Move it or pass --out.")
        return 1
    samples, rejected = draft_period(args) if args.tier == "period" else draft_underspecified(args)
    model = args.model or load_config(args.config).llm.model
    for sample in samples:
        sample.extra["drafted_with"] = model
        sample.extra["checked_with"] = args.check_model or model
    samples.sort(key=lambda s: s.id)
    EvalDataset(samples=samples).save(out)
    rejects = out.with_name(out.name.replace("_draft.json", "_rejects.jsonl"))
    rejects.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rejected))
    print(f"\nWrote {len(samples)} draft(s) to {out}, {len(rejected)} reject(s) to {rejects}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
