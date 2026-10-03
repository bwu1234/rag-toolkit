#!/usr/bin/env python
"""Generate a span-labelled eval set from the EDGAR corpus.

Why this is generated rather than hand-written
----------------------------------------------
``data/eval/eval_set.json`` is 43 hand-authored samples against 26k characters.
The EDGAR corpus is 3.6M characters over 61 filings; hand-authoring proportional
coverage is not realistic, and an eval set that touches 1% of the corpus mostly
measures which 1% you picked.

How it works: sample a chunk, show it to the LLM, and ask for a question it
answers plus the **verbatim span** that answers it.  Recording the span at
generation time is what makes span-level ground truth affordable -- the labour
that usually makes chunk-level labels expensive is finding the passage, and here
the passage is the input.

Known bias, stated plainly
--------------------------
Questions generated *from* a passage are lexically closer to it than real user
questions are, which inflates retrieval scores across the board.  Two things
limit the damage, neither of which eliminates it:

* the prompt asks for an analyst's phrasing rather than a restatement of the
  excerpt, and rejects questions that copy a long literal run from it;
* absolute numbers from this set are **not** comparable to numbers from a
  hand-authored set.  It is built to compare *configurations against each other*
  on identical questions, which is what Milestone 11 needs.

Audit a sample by hand before trusting a result.  ``--review`` prints samples
for eyeballing.

Entity and period disambiguation
--------------------------------
The corpus is deliberately full of near-duplicate documents: 14 companies
discussing the same topics across multiple periods.  A question like "what drove
revenue growth?" is answered correctly by dozens of chunks, so single-span
ground truth would mark correct retrievals wrong.  The prompt therefore requires
each question to name its company and period, and generated spans are checked
for uniqueness across the whole corpus -- a span appearing in more than one
filing is rejected rather than labelled.

Validation is the load-bearing part
-----------------------------------
A local 9b model paraphrases when asked to quote.  Every generated span is
checked to appear **verbatim** in its source chunk (under the same normalization
:mod:`rag.eval.relevance` uses at eval time) and to be short enough to survive
chunking.  Anything that fails is dropped and counted, never repaired -- a
silently wrong label is worse than a smaller eval set.

Usage
-----
    python scripts/generate_eval_set.py --samples 150
    python scripts/generate_eval_set.py --review 5      # print samples, write nothing
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.chunking.chunkers import get_chunker  # noqa: E402
from rag.chunking.models import Chunk  # noqa: E402
from rag.config.settings import RagConfig, load_config  # noqa: E402
from rag.eval.dataset import EvalDataset, EvalSample, ExpectedSpan  # noqa: E402
from rag.eval.relevance import normalize  # noqa: E402
from rag.llm.factory import get_llm_client  # noqa: E402
from rag.ingestion.cleaners import clean_documents  # noqa: E402
from rag.ingestion.loaders import load_corpus  # noqa: E402
from rag.logging_config import configure_logging  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_CORPUS = Path("data/corpora/edgar/documents")
DEFAULT_OUT = Path("data/eval/edgar_eval_set.json")

# A chunk this short is usually a heading or a stray table row -- not enough
# material for a question with a defensible single answer.
MIN_CHUNK_CHARS = 400

# A span shorter than this is a bare figure ("$359 million") rather than a
# clause. Those are poor ground truth twice over: they pass the corpus
# uniqueness check by luck rather than by being distinctive, and they anchor a
# passage so weakly that a near-miss chunk containing the same number scores as
# a hit. Requiring a clause makes uniqueness meaningful.
MIN_SPAN_CHARS = 40

SYSTEM_PROMPT = (
    "You write evaluation questions for a financial-document search system. "
    "Given one excerpt from an SEC filing, you write a question that a financial "
    "analyst would actually ask, which the excerpt answers. "
    "The question MUST name the company and the reporting period explicitly, "
    "because the corpus contains many companies and periods and an ambiguous "
    "question has many correct answers. "
    "Phrase the question the way an analyst would, NOT by restating the excerpt: "
    "do not copy long phrases out of it. "
    "You also return the shortest verbatim quote from the excerpt that answers "
    "the question -- copied EXACTLY, character for character, no paraphrasing, "
    "no ellipsis, no added words. "
    'Reply with only a JSON object: {"question": "...", "answer_span": "...", '
    '"answer": "..."} and nothing else. '
    "The answer_span must be a complete clause or sentence that on its own "
    "states the answer -- not a bare figure -- and the answer must be exactly "
    "what that span says, not a different number from elsewhere in the excerpt."
)

VERIFY_SYSTEM_PROMPT = (
    "You check evaluation labels for a document retrieval benchmark. "
    "You are given a question, a quoted passage, and a proposed answer. "
    "Reply GOOD only if the passage genuinely answers the question AND the "
    "proposed answer is exactly what the passage states. "
    "Reply BAD if the passage is about a different quantity, a different period, "
    "or if the proposed answer is a number the passage does not actually give. "
    "Be strict: a label you are unsure about is BAD. "
    "Reply with one word, GOOD or BAD, and nothing else."
)


def build_verify_prompt(question: str, span: str, answer: str) -> str:
    return (
        f"<question>{question}</question>\n"
        f"<passage>{span}</passage>\n"
        f"<proposed_answer>{answer}</proposed_answer>\n\n"
        "GOOD or BAD:"
    )


@dataclass
class Candidate:
    """One generated sample before validation."""

    chunk: Chunk
    question: str
    answer_span: str
    answer: str


def build_prompt(chunk: Chunk, max_span_chars: int) -> str:
    title = chunk.metadata.get("title", chunk.document_id)
    return (
        f"<filing>{title}</filing>\n"
        f"<document_id>{chunk.document_id}</document_id>\n\n"
        f"<excerpt>\n{chunk.text}\n</excerpt>\n\n"
        f"The answer_span must be at most {max_span_chars} characters and must "
        "appear word-for-word inside the excerpt above.\n"
        "JSON:"
    )


def parse_reply(reply: str) -> dict | None:
    """Pull the JSON object out of a model reply, tolerating fences and preamble."""
    text = reply.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def generate_candidate(llm_client, chunk: Chunk, max_span_chars: int) -> Candidate | None:
    """Ask the model for one question/span pair. Returns None on any failure."""
    try:
        reply = llm_client.generate(
            build_prompt(chunk, max_span_chars), system=SYSTEM_PROMPT
        )
    except Exception:  # noqa: BLE001 -- one bad chunk must not kill the run
        logger.exception("Generation failed for chunk %r", chunk.id)
        return None

    parsed = parse_reply(reply)
    if parsed is None:
        return None

    question = str(parsed.get("question", "")).strip()
    span = str(parsed.get("answer_span", "")).strip()
    answer = str(parsed.get("answer", "")).strip()
    if not question or not span:
        return None
    return Candidate(chunk=chunk, question=question, answer_span=span, answer=answer)


def verify_candidate(llm_client, candidate: Candidate) -> bool:
    """Verify a generated candidate -- see :func:`verify_label`."""
    return verify_label(
        llm_client,
        candidate.question,
        candidate.answer_span,
        candidate.answer,
        label=candidate.chunk.id,
    )


def verify_label(
    llm_client,
    question: str,
    span: str,
    answer: str,
    *,
    label: str,
    system: str = VERIFY_SYSTEM_PROMPT,
) -> bool:
    """Second-opinion check that the span answers the question and supports the answer.

    Deliberately **fails closed**, which is the opposite of every runtime
    component in this repo (the contextualizer, CRAG's three checks, the query
    condenser all fail *open*). The asymmetry is the point: at runtime a broken
    judgment must not degrade a live turn, so proceeding is the safe default.
    Here a bad label silently corrupts every measurement taken against this set
    afterwards, so an unverifiable sample is dropped rather than kept.

    This exists because the generating model reliably produces spans that
    contradict their own stated answer -- asking for operating income and
    answering with the year-over-year *increase*, or quoting "increased 22% and
    21%" beside an answer of "9%".
    """
    try:
        reply = llm_client.generate(
            build_verify_prompt(question, span, answer), system=system
        )
    except Exception:  # noqa: BLE001
        logger.exception("Verification failed for %r", label)
        return False
    return reply.strip().upper().startswith("GOOD")


class Validator:
    """Rejects candidates that would produce a silently wrong label.

    Every rejection reason is counted so the run reports *why* it dropped what it
    dropped -- a generator whose yield quietly collapses is a generator whose
    output you should not trust.
    """

    def __init__(self, chunks: list[Chunk], *, max_span_chars: int) -> None:
        self.max_span_chars = max_span_chars
        self.reasons: Counter[str] = Counter()
        # Normalized full text per document, for the corpus-uniqueness check.
        self._documents: dict[str, str] = {}
        for chunk in chunks:
            self._documents.setdefault(chunk.document_id, "")
            self._documents[chunk.document_id] += " " + normalize(chunk.text)

    def validate(self, candidate: Candidate) -> EvalSample | None:
        span = candidate.answer_span
        needle = normalize(span)

        if len(span) > self.max_span_chars:
            # Consecutive chunk windows overlap by chunk_overlap characters, so a
            # longer span can straddle every boundary and match nothing.
            self.reasons["span_too_long"] += 1
            return None

        if len(span) < MIN_SPAN_CHARS:
            self.reasons["span_too_short"] += 1
            return None

        if not needle:
            self.reasons["span_empty"] += 1
            return None

        if needle not in normalize(candidate.chunk.text):
            # The model paraphrased instead of quoting. This is the single most
            # common failure and the one that would silently poison the set.
            self.reasons["span_not_verbatim"] += 1
            return None

        matches = [doc_id for doc_id, text in self._documents.items() if needle in text]
        if len(matches) > 1:
            # Boilerplate that appears in several filings cannot identify one
            # passage, so a retriever surfacing a different filing would be
            # marked wrong for being right.
            self.reasons["span_not_unique"] += 1
            return None

        if self._question_copies_the_excerpt(candidate):
            self.reasons["question_copies_excerpt"] += 1
            return None

        return EvalSample(
            id=f"{candidate.chunk.id}",
            query=candidate.question,
            expected_spans=[ExpectedSpan(text=span)],
            expected_doc_ids=[candidate.chunk.document_id],
            expected_answer=candidate.answer or None,
            extra={"tier": "generated", "source_chunk": candidate.chunk.id},
        )

    def _question_copies_the_excerpt(self, candidate: Candidate, run_words: int = 8) -> bool:
        """True if the question lifts a long literal run out of the excerpt.

        A question that quotes its own source is trivially retrievable and
        measures nothing. This only catches the blatant cases; it does not undo
        the general lexical-overlap bias described in the module docstring.
        """
        question_words = normalize(candidate.question).split()
        chunk_text = normalize(candidate.chunk.text)
        for i in range(len(question_words) - run_words + 1):
            if " ".join(question_words[i : i + run_words]) in chunk_text:
                return True
        return False


def sample_chunks(chunks: list[Chunk], count: int, seed: int) -> list[Chunk]:
    """Pick chunks spread evenly across documents rather than clustered.

    Uniform random sampling over 4,236 chunks would over-weight the longest
    filings (COST's 10-K alone is ~160 chunks) and could miss short ones
    entirely, so the eval set would silently describe a few big documents.
    """
    rng = random.Random(seed)
    by_document: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        if len(chunk.text) >= MIN_CHUNK_CHARS:
            by_document.setdefault(chunk.document_id, []).append(chunk)

    for pool in by_document.values():
        rng.shuffle(pool)

    selected: list[Chunk] = []
    doc_ids = sorted(by_document)
    rng.shuffle(doc_ids)
    # Round-robin across documents until the target is met or every pool is dry.
    while len(selected) < count:
        progressed = False
        for doc_id in doc_ids:
            if by_document[doc_id]:
                selected.append(by_document[doc_id].pop())
                progressed = True
                if len(selected) >= count:
                    break
        if not progressed:
            break
    return selected


def generate(
    chunks: list[Chunk],
    config: RagConfig,
    *,
    count: int,
    concurrency: int,
    seed: int,
) -> tuple[list[EvalSample], Counter[str]]:
    """Generate and validate ``count`` samples. Returns (samples, rejection reasons)."""
    max_span_chars = config.chunking.chunk_overlap
    llm_client = get_llm_client(config.llm)
    validator = Validator(chunks, max_span_chars=max_span_chars)
    targets = sample_chunks(chunks, count, seed)

    logger.info(
        "Generating from %d chunk(s) with %d concurrent request(s); spans capped at "
        "%d chars (chunking.chunk_overlap)",
        len(targets),
        concurrency,
        max_span_chars,
    )

    samples: list[EvalSample] = []
    completed = 0
    survivors: list[tuple[Candidate, EvalSample]] = []

    # Workers make LLM calls; validation and all counting happen on the main
    # thread as results arrive. Counter increments are not atomic, and the
    # rejection tally is the only evidence of *why* a run's yield collapsed --
    # it is not worth making it racy to save a substring scan.
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = [
            pool.submit(generate_candidate, llm_client, chunk, max_span_chars)
            for chunk in targets
        ]
        for future in as_completed(futures):
            completed += 1
            candidate = future.result()
            if candidate is None:
                validator.reasons["generation_failed"] += 1
            else:
                sample = validator.validate(candidate)
                if sample is not None:
                    survivors.append((candidate, sample))
            if completed % 25 == 0:
                logger.info(
                    "  ... %d/%d generated, %d passed local checks",
                    completed, len(targets), len(survivors),
                )

        # Verification runs only on survivors: it costs a second LLM call, and
        # there is no sense spending one on a span that was paraphrased or
        # appears in three other filings.
        logger.info("Verifying %d candidate(s)", len(survivors))
        verify_futures = {
            pool.submit(verify_candidate, llm_client, candidate): sample
            for candidate, sample in survivors
        }
        for future in as_completed(verify_futures):
            if future.result():
                samples.append(verify_futures[future])
            else:
                validator.reasons["failed_verification"] += 1

    # Stable order so re-running with the same seed produces a comparable file.
    samples.sort(key=lambda s: s.id)
    return samples, validator.reasons


def print_review(samples: list[EvalSample], limit: int) -> None:
    print(f"\n{'=' * 72}\n  Sample review ({min(limit, len(samples))} of {len(samples)})\n{'=' * 72}")
    for sample in samples[:limit]:
        print(f"\n[{sample.id}]")
        print(f"  Q:      {sample.query}")
        print(f"  span:   {sample.expected_spans[0].text!r}")
        print(f"  answer: {sample.expected_answer}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a span-labelled eval set from a corpus."
    )
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--config", default=None, help="Path to config YAML")
    parser.add_argument("--samples", type=int, default=150, help="Chunks to attempt")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed (reproducibility)")
    parser.add_argument(
        "--review", type=int, default=0, metavar="N",
        help="Print N generated samples for eyeballing (still writes the file)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Generate but write nothing")
    parser.add_argument(
        "--merge", action="store_true",
        help=(
            "Merge into the existing --out file instead of replacing it. Yield is ~17%% "
            "of attempts, so reaching a useful sample count means several runs at "
            "different --seed values; samples are keyed by source chunk id, so a chunk "
            "drawn twice across seeds is kept once."
        ),
    )
    args = parser.parse_args()

    configure_logging()
    if not args.corpus_dir.exists():
        logger.error("Corpus not found: %s -- run scripts/fetch_edgar.py first", args.corpus_dir)
        return 1

    config = load_config(args.config)
    documents = clean_documents(load_corpus(args.corpus_dir))
    chunks = get_chunker(config.chunking).chunk(documents)
    logger.info("Corpus: %d document(s) -> %d chunk(s)", len(documents), len(chunks))

    samples, reasons = generate(
        chunks, config, count=args.samples, concurrency=args.concurrency, seed=args.seed
    )

    attempted = len(samples) + sum(reasons.values())
    print(f"\n{'=' * 72}")
    print(f"  Generated {len(samples)} usable sample(s) from {attempted} attempt(s)")
    print(f"{'=' * 72}")
    for reason, n in reasons.most_common():
        print(f"  rejected: {reason:<28} {n:>4}")
    if not samples:
        logger.error("No usable samples generated")
        return 1
    covered = len({s.expected_doc_ids[0] for s in samples})
    print(f"  documents covered:           {covered:>4}")

    if args.review:
        print_review(samples, args.review)

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0

    if args.merge and args.out.exists():
        existing = EvalDataset.load(args.out).samples
        merged = {s.id: s for s in existing}
        added = sum(1 for s in samples if s.id not in merged)
        merged.update({s.id: s for s in samples})
        samples = sorted(merged.values(), key=lambda s: s.id)
        print(f"\nMerged into {len(existing)} existing sample(s): +{added} new")

    EvalDataset(samples=samples).save(args.out)
    print(f"\nWrote {len(samples)} sample(s) to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
