"""Tests for the tier drafter's checks (`scripts/draft_tier_set.py`).

Like the generator's tests, only the pure parts: the LLM's drafts are
throwaway, but a check that lets a bad draft through reaches a person's review
looking already vetted, and a finalize that skips a check freezes the error
into the tier.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from draft_tier_set import (  # noqa: E402
    Corpus,
    Filing,
    PeriodTarget,
    PeriodValidator,
    TableTarget,
    TableValidator,
    UnderspecifiedValidator,
    check_accepted,
    content_overlap,
    figures,
    finalize,
    markdown_tables,
    names_company,
    repeated_paragraphs,
    sample_period_targets,
    sample_sources,
    sample_table_targets,
    table_rows_by_company,
)

from rag.chunking.models import Chunk  # noqa: E402
from rag.eval.dataset import EvalDataset, EvalSample, ExpectedSpan  # noqa: E402
from rag.ingestion.models import Document  # noqa: E402

POLICY = (
    "Higher tariffs are more likely to adversely impact rather than improve our "
    "results. The degree of our exposure depends on the type of goods and timing. "
    "Government actions in various countries relating to tariffs affect our costs."
)
assert len(POLICY) >= 200, "fixture must clear the repeated-paragraph floor"
SPAN = "Higher tariffs are more likely to adversely impact rather than improve our results."


def _doc(doc_id: str, text: str) -> Document:
    ticker, form, period = doc_id.removesuffix(".md").split("_")
    title = f"COSTCO WHOLESALE CORP /NEW ({ticker}) {form} -- period ended {period}"
    return Document(id=doc_id, text=text, source=Path(doc_id), doc_type="markdown",
                    metadata={"title": title})


def _corpus(*docs: Document) -> Corpus:
    # One chunk per document is enough for span-in-chunk checks.
    chunks = [
        Chunk(id=f"{d.id}::chunk0", text=d.text, document_id=d.id, source=d.source,
              doc_type="markdown", metadata={"chunk_index": 0})
        for d in docs
    ]
    return Corpus(list(docs), chunks)


FY25 = _doc("COST_10-K_2025-08-31.md", f"Intro.\n\n{POLICY}\n\nOther text $2.1 billion.")
Q1 = _doc("COST_10-Q_2025-11-23.md", f"Quarter intro.\n\n{POLICY}")
UNIQUE = _doc("COST_10-Q_2026-02-15.md", "Nothing repeated here.")


def _period_reply(question: str, span: str = SPAN) -> dict:
    return {"question": question, "answer_span": span, "answer": "Adversely."}


GOOD_Q = "How does Costco view tariffs in its 10-K for the fiscal year ended August 31, 2025?"


# ---------------------------------------------------------------------------
# Filing identity and word checks
# ---------------------------------------------------------------------------


def test_filing_parses_title_and_formats_the_period() -> None:
    filing = Filing.from_document(FY25)
    assert (filing.ticker, filing.form) == ("COST", "10-K")
    assert filing.period_phrase == "August 31, 2025"
    assert filing.display_name == "Costco Wholesale Corp"


def test_ticker_that_is_a_word_matches_only_in_capitals() -> None:
    filing = Filing.from_document(FY25)
    assert not names_company("What was the cost of sales in fiscal 2025?", filing)
    assert names_company("What was COST's cost of sales?", filing)
    assert names_company("What was Costco's cost of sales?", filing)


def test_overlap_ignores_company_and_period_words() -> None:
    filing = Filing.from_document(FY25)
    spans = ["Operating income increased 17% during 2025."]
    # Only "operating" and "income" are topic words; both appear in the span.
    assert content_overlap("What was Costco's operating income in fiscal 2025?", spans,
                           filing.name_tokens) == 1.0
    assert content_overlap("How much did Costco earn from running its stores in 2025?", spans,
                           filing.name_tokens) == 0.0


def test_figures_skip_dates_and_keep_amounts() -> None:
    text = "As of September 28, 2024, obligations were $15.6 billion, or 19%, on 1,234 leases."
    assert figures(text) == ["$15.6 billion", "19%", "1,234"]


# ---------------------------------------------------------------------------
# Period tier
# ---------------------------------------------------------------------------


def test_repeated_paragraphs_need_two_filings_and_no_table() -> None:
    table = "| a | b |\n| 1 | 2 |\n| 3 | 4 |" + " x" * 100
    docs = [FY25, Q1, UNIQUE, _doc("COST_10-Q_2026-05-10.md", table),
            _doc("COST_10-Q_2025-05-11.md", table)]
    assert repeated_paragraphs(docs) == [(POLICY, [FY25.id, Q1.id])]


def test_period_targets_spread_across_companies_and_are_seeded() -> None:
    pool = [(f"{t} paragraph {i}", [f"{t}_10-K_2025-01-01.md", f"{t}_10-Q_2025-04-01.md"])
            for t in ("AAA", "BBB") for i in range(5)]
    targets = sample_period_targets(pool, 4, seed=1)
    assert sorted(t.doc_ids[0][:3] for t in targets) == ["AAA", "AAA", "BBB", "BBB"]
    assert [t.index for t in targets] == [t.index for t in sample_period_targets(pool, 4, seed=1)]


def _target(doc: Document = FY25) -> PeriodTarget:
    return PeriodTarget(index=7, paragraph=POLICY, doc_ids=[FY25.id, Q1.id], target=doc.id)


def test_good_period_draft_uses_span_and_document_matching() -> None:
    sample = PeriodValidator(_corpus(FY25, Q1)).validate(_target(), _period_reply(GOOD_Q))
    assert sample is not None
    assert sample.id == "period::COST_10-K_2025-08-31.md::para7"
    assert sample.matching_mode == "span_and_document"
    assert sample.expected_doc_ids == [FY25.id]
    assert sample.extra["competing_doc_ids"] == [Q1.id]
    assert sample.extra["review"]["verdict"] == "pending"


@pytest.mark.parametrize(
    ("reply", "reason"),
    [
        (_period_reply("How does Costco view tariffs in fiscal 2025?"), "question_missing_period"),
        (_period_reply("How does the retailer view tariffs as of August 31, 2025?"),
         "question_missing_company"),
        (_period_reply(GOOD_Q, span="Higher tariffs may hurt results, the company believes now."),
         "span_not_verbatim"),
        (_period_reply(GOOD_Q, span="Higher tariffs"), "span_length"),
        (None, "generation_failed"),
    ],
)
def test_period_draft_rejections_are_counted(reply: dict | None, reason: str) -> None:
    validator = PeriodValidator(_corpus(FY25, Q1))
    assert validator.validate(_target(), reply) is None
    assert validator.reasons == {reason: 1}
    # The rejected draft is kept, so a miscalibrated check can be read, not just counted.
    assert validator.rejected[0]["reason"] == reason
    assert validator.rejected[0]["target"] == FY25.id


def test_period_draft_needs_the_span_in_another_filing() -> None:
    # Q1 missing from the corpus: the span is in one filing only, so it tests nothing.
    validator = PeriodValidator(_corpus(FY25))
    assert validator.validate(_target(), _period_reply(GOOD_Q)) is None
    assert validator.reasons == {"span_not_repeated": 1}


def test_restated_figures_are_flagged_for_review() -> None:
    doc = _doc("COST_10-K_2025-08-31.md", "Revenue was $2.1 billion.\n\nIt rose to $2.1 billion.")
    corpus = _corpus(doc)
    assert corpus.restatement_candidates([doc.id], "It rose to $2.1 billion.") == []
    other = _doc("COST_10-Q_2025-11-23.md", "Prior year revenue of $2.1 billion.")
    corpus = _corpus(doc, other)
    assert corpus.restatement_candidates([other.id], "It rose to $2.1 billion.") == [
        f"{other.id}::chunk0"
    ]


# ---------------------------------------------------------------------------
# Underspecified tier
# ---------------------------------------------------------------------------

SOURCE = EvalSample(
    id="COST_10-K_2025-08-31.md::chunk3",
    query="What was Costco's operating income for fiscal year 2025?",
    expected_spans=[ExpectedSpan(text="Operating income increased to $9.3 billion in 2025.")],
    expected_doc_ids=[FY25.id],
    expected_answer="$9.3 billion",
    extra={"tier": "generated", "source_chunk": "x", "label_fixes": [{"date": "d"}]},
)


def _under(kind: str, rewrite: str | None, same: bool = True) -> tuple[EvalSample | None, dict]:
    validator = UnderspecifiedValidator(_corpus(FY25))
    return validator.validate(kind, SOURCE, rewrite, same), dict(validator.reasons)


def test_paraphrase_inherits_labels_and_drops_generation_provenance() -> None:
    sample, _ = _under("paraphrase", "How much did Costco earn from running its stores in 2025?")
    assert sample is not None
    assert sample.id == "paraphrase::COST_10-K_2025-08-31.md::chunk3"
    assert sample.expected_spans == SOURCE.expected_spans
    assert sample.extra["source_query"] == SOURCE.query
    assert "label_fixes" not in sample.extra and "source_chunk" not in sample.extra


@pytest.mark.parametrize(
    ("kind", "rewrite", "same", "reason"),
    [
        ("paraphrase", "What was Costco's operating income in 2025?", True,
         "paraphrase:overlap_too_high"),
        ("paraphrase", "How much did the warehouse club earn from its stores in 2025?", True,
         "paraphrase:dropped_company"),
        ("paraphrase", "How much did Costco earn from running its stores?", True,
         "paraphrase:dropped_period"),
        ("paraphrase", "How much did Costco earn from running its stores in 2025?", False,
         "paraphrase:changed_meaning"),
        ("implicit", "What was COST's operating income for fiscal year 2025?", True,
         "implicit:names_company"),
        ("implicit", None, True, "implicit:generation_failed"),
    ],
)
def test_underspecified_rejections_are_counted(
    kind: str, rewrite: str | None, same: bool, reason: str
) -> None:
    sample, reasons = _under(kind, rewrite, same)
    assert sample is None
    assert reasons == {reason: 1}


def test_implicit_draft_accepted_without_the_name() -> None:
    sample, _ = _under("implicit",
                       "What was the largest warehouse club's operating income for fiscal year 2025?")
    assert sample is not None and sample.extra["kind"] == "implicit"


def test_source_samples_are_disjoint_between_kinds() -> None:
    samples = [EvalSample(id=f"s{i}", query="q") for i in range(10)]
    picked = sample_sources(samples, 4, seed=0)
    ids = [s.id for kind in ("paraphrase", "implicit") for s in picked[kind]]
    assert len(ids) == len(set(ids)) == 8


# ---------------------------------------------------------------------------
# Finalize
# ---------------------------------------------------------------------------


def test_finalize_rechecks_reviewer_edits() -> None:
    corpus = _corpus(FY25, Q1)
    sample = PeriodValidator(corpus).validate(_target(), _period_reply(GOOD_Q))
    assert sample is not None and check_accepted(sample, corpus) == []
    sample.query = "How does Costco view tariffs?"
    assert check_accepted(sample, corpus) == ["question doesn't state the period end date"]


def test_finalize_refuses_while_any_draft_is_pending(tmp_path: Path) -> None:
    sample = PeriodValidator(_corpus(FY25, Q1)).validate(_target(), _period_reply(GOOD_Q))
    assert sample is not None
    path = tmp_path / "draft.json"
    EvalDataset(samples=[sample]).save(path)
    out = tmp_path / "tier.json"
    assert finalize(path, out, None) == 1
    assert not out.exists()
    # The draft file is plain JSON a reviewer edits by hand.
    assert json.loads(path.read_text())[0]["review"]["verdict"] == "pending"


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------

TABLE_TEXT = """# Costco Wholesale Corp (COST) 10-K -- period ended 2025-08-31

## RESULTS OF OPERATIONS

### Net Sales

The following table summarizes net sales (dollars in millions).

| | 52 Weeks Ended | | |
| | 2025 | 2024 | Change |
|---|---|---|---|
| Net sales | $269,912 | $249,625 | 8.1% |
| Membership fees | 5,323 | 4,828 | 10.3% |

## LIQUIDITY

| Not a data table | text |
|---|---|
| Only | words |"""

TABLE_DOC = _doc("COST_10-K_2025-08-31.md", TABLE_TEXT)
# The prior year's 10-K has the figure without its "$" (it's no longer the first row).
PRIOR = _doc("COST_10-K_2024-09-01.md", "| | 2024 | 2023 |\n|---|---|---|\n| Net sales | 249,625 | 237,710 |")


def test_markdown_tables_keep_headings_caption_and_data_tables_only() -> None:
    [table] = markdown_tables(TABLE_DOC)

    assert table.headings == ["RESULTS OF OPERATIONS", "Net Sales"]
    assert table.lead_in.startswith("The following table summarizes net sales")
    assert table.rows[0] == "| Net sales | $269,912 | $249,625 | 8.1% |"


def test_a_spanning_header_labels_the_columns_to_its_right() -> None:
    [table] = markdown_tables(TABLE_DOC)

    assert table.column_path(2) == ["52 Weeks Ended", "2024"]
    assert table.column_path(3) == ["52 Weeks Ended", "Change"]


def test_table_targets_are_seeded_figures_under_labeled_columns() -> None:
    tables = markdown_tables(TABLE_DOC)

    first = sample_table_targets(tables, 5, seed=3)
    assert [t.sample_id for t in first] == [t.sample_id for t in sample_table_targets(tables, 5, seed=3)]
    assert all(t.column > 0 and t.table.column_path(t.column) for t in first)


def _table_sample(question: str, answer: str = "Net sales were $249,625 million.", column: int = 2):
    corpus = _corpus(TABLE_DOC, PRIOR)
    [table] = markdown_tables(TABLE_DOC)
    validator = TableValidator(corpus, table_rows_by_company([TABLE_DOC, PRIOR]))
    return validator.validate(TableTarget(table, 0, column), {"question": question, "answer": answer}), validator


def test_good_table_draft_quotes_the_whole_row_and_lists_restatements() -> None:
    sample, _ = _table_sample("What were Costco's net sales for fiscal 2024?")

    assert sample is not None
    assert sample.expected_spans[0].text == "| Net sales | $269,912 | $249,625 | 8.1% |"
    assert sample.expected_doc_ids == ["COST_10-K_2025-08-31.md"]
    review = sample.extra["review"]
    assert review["column_path"] == "52 Weeks Ended > 2024"
    assert review["value"] == "$249,625"
    # The prior year's 10-K reports the same row and figure.
    assert review["restatement_candidates"] == ["COST_10-K_2024-09-01.md"]
    assert review["restatement_rows"] == [
        {"doc_id": "COST_10-K_2024-09-01.md", "row": "| Net sales | 249,625 | 237,710 |"}
    ]
    assert sample.matching_mode == "span_and_document"
    assert check_accepted(sample, _corpus(TABLE_DOC, PRIOR)) == []


@pytest.mark.parametrize(
    ("question", "answer", "reason"),
    [
        ("What were net sales in fiscal 2024?", "Net sales were $249,625 million.", "question_missing_company"),
        ("Were Costco's fiscal 2024 net sales $249,625 million?", "Yes, $249,625 million.", "question_leaks_value"),
        ("What were Costco's net sales for fiscal 2024?", "Net sales rose.", "answer_missing_value"),
        ("What were Costco's net sales last year?", "Net sales were $249,625 million.", "question_missing_column_year"),
    ],
)
def test_table_draft_rejections_are_counted(question: str, answer: str, reason: str) -> None:
    sample, validator = _table_sample(question, answer)

    assert sample is None
    assert validator.reasons == {reason: 1}
