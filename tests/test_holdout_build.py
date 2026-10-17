"""Tests for the fresh holdout builder (`scripts/build_holdout.py`).

Only the mechanical rules. The questions themselves are reviewed by a person;
what these tests protect is the boundary that review relies on: a split that
can't drift, a leakage check that fires on a shared fact, and a finalize that
won't freeze undecided drafts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_holdout  # noqa: E402
from build_holdout import (  # noqa: E402
    DEV,
    EXISTING_SECTORS,
    HOLDOUT,
    NEW_COMPANIES,
    Corpus,
    DevIndex,
    DevSpan,
    assemble_sample,
    assign_splits,
    audit,
    check_schema,
    figures,
    has_review_decision,
    runner_files,
    split_units,
)

from rag.eval.dataset import EvalSample  # noqa: E402
from rag.ingestion.models import Document  # noqa: E402

FUEL = "Aircraft fuel expense was $4.9 billion in the second quarter of 2026, up 83.3%."
LIQUIDITY = "As of June 30, 2026, we had $11.3 billion in total available liquidity."
ROW = "| Passenger load factor (percent) | 83.2 | 84.7 | (1.5)pts |"
OTHER_ROW = "| Yield (cents) | 22.33 | 19.96 | 11.9% |"


def _doc(doc_id: str, text: str) -> Document:
    return Document(id=doc_id, text=text, source=Path(doc_id), doc_type="markdown", metadata={})


AAL_Q2 = _doc(
    "AAL_10-Q_2026-06-30.md",
    f"# AAL 10-Q\n\n## Results\n\n{FUEL}\n\n#### Liquidity\n\n{LIQUIDITY}\n\n{ROW}\n{OTHER_ROW}\n",
)
AAPL_Q3 = _doc(
    "AAPL_10-Q_2026-06-27.md",
    "# AAPL 10-Q\n\n## Services\n\nServices net sales increased due to advertising.\n\n"
    "The Company repurchased $25.8 billion of its common stock.\n\n"
    "The Company paid the remaining $8.8 billion balance of the deemed repatriation tax.\n",
)
AAPL_Q2 = _doc(
    "AAPL_10-Q_2026-03-28.md",
    "# AAPL 10-Q\n\nComparative: the Company repurchased $25.8 billion in the prior quarter.\n",
)


def _corpus() -> Corpus:
    return Corpus({d.id: d for d in (AAL_Q2, AAPL_Q3, AAPL_Q2)})


def _sample(**src: object) -> EvalSample:
    base: dict = {"id": "s", "family": "direct_lookup", "behavior": "answer", "query": "q?", "answer": "a"}
    base.update(src)
    return EvalSample.from_dict(assemble_sample(base, assign_splits(), drafted_with="t", drafted_on="2026-10-01"))


# --------------------------------------------------------------------------
# Split
# --------------------------------------------------------------------------


def test_split_is_seeded_and_one_existing_company_per_sector() -> None:
    splits = assign_splits()
    assert splits == assign_splits(), "the draw must be reproducible"
    for tickers in EXISTING_SECTORS.values():
        assert sum(splits[t] == HOLDOUT for t in tickers) == 1
    assert all(splits[t] == HOLDOUT for t in NEW_COMPANIES)


def test_split_recorded_in_protocol_matches_the_draw() -> None:
    # docs/holdout-protocol.md names the drawn companies; a seed or code change
    # that moves them would silently move questions across the boundary.
    splits = assign_splits()
    assert {t for t, s in splits.items() if s == HOLDOUT} - set(NEW_COMPANIES) == {
        "AAPL", "TGT", "LUV", "PFE", "XOM"
    }


# --------------------------------------------------------------------------
# Assemble
# --------------------------------------------------------------------------


def test_assemble_derives_group_split_and_matching_mode() -> None:
    s = _sample(id="a", docs=["AAL_10-Q_2026-06-30.md"], spans=[FUEL])
    assert s.extra["companies"] == ["AAL"]
    assert s.extra["split"] == HOLDOUT
    assert s.extra["transfer"] == "new_entity"
    assert s.matching_mode == "span_and_document"
    assert s.extra["review"]["verdict"] == "pending"


def test_assemble_marks_a_cross_split_question_mixed() -> None:
    # AAL is holdout, DAL development: the question can't sit on either side.
    s = _sample(id="x", docs=["AAL_10-Q_2026-06-30.md", "DAL_10-Q_2026-06-30.md"], spans=[FUEL])
    assert s.extra["split"] == "mixed"
    assert any("straddle" in p for p in check_schema(s))


def test_assemble_unions_part_spans_and_docs() -> None:
    s = _sample(
        id="p", family="distributed",
        parts=[
            {"label": "fuel", "answer": "$4.9B", "docs": ["AAL_10-Q_2026-06-30.md"], "spans": [FUEL]},
            {"label": "liq", "answer": "$11.3B", "docs": ["AAL_10-Q_2026-06-30.md"], "spans": [LIQUIDITY]},
        ],
    )
    assert [sp.text for sp in s.expected_spans] == [FUEL, LIQUIDITY]
    assert s.expected_doc_ids == ["AAL_10-Q_2026-06-30.md"]


def test_has_review_decision_protects_a_started_review() -> None:
    raw = assemble_sample({"id": "r", "family": "direct_lookup", "behavior": "answer", "query": "q",
                           "answer": "a", "docs": [], "spans": []}, assign_splits(),
                          drafted_with="t", drafted_on="d")
    assert not has_review_decision(raw)
    raw["review"]["clarity"]["decision"] = "pass"
    assert has_review_decision(raw)


# --------------------------------------------------------------------------
# Schema by behavior
# --------------------------------------------------------------------------


def test_refusal_needs_tier_absent_terms_and_no_evidence() -> None:
    ok = _sample(id="r", family="unanswerable", behavior="refuse", companies=["AAL"], absent_terms=["Q3 2026"])
    assert check_schema(ok) == []
    assert ok.extra["tier"] == "refusal"
    no_terms = _sample(id="r2", family="unanswerable", behavior="refuse", companies=["AAL"])
    assert any("absent_terms" in p for p in check_schema(no_terms))


def test_partial_needs_one_supported_and_one_unavailable_part() -> None:
    supported = {"label": "Q2", "answer": "x", "docs": ["AAL_10-Q_2026-06-30.md"], "spans": [FUEL]}
    missing = {"label": "Q3", "answer": "not available", "available": False}
    assert check_schema(_sample(id="p", family="unanswerable", behavior="partial",
                                parts=[supported, missing])) == []
    assert any("available: false" in p for p in check_schema(
        _sample(id="p2", family="unanswerable", behavior="partial", parts=[supported])))


def test_calculation_tag_requires_calc_block() -> None:
    s = _sample(id="c", family="table_calc", tags=["calculation"],
                docs=["AAL_10-Q_2026-06-30.md"], spans=[FUEL])
    assert any("calc" in p for p in check_schema(s))


# --------------------------------------------------------------------------
# Corpus checks
# --------------------------------------------------------------------------


def test_split_units_treats_each_table_row_as_its_own_fact() -> None:
    units = split_units(AAL_Q2.text)
    assert ROW in units and OTHER_ROW in units


def test_figures_ignore_years_and_bare_small_percentages() -> None:
    assert figures("In 2025, sales rose 16% to $4.9 billion and 83.3%.") == {"$4.9 billion", "83.3%"}


def test_section_of_returns_heading_path_without_title() -> None:
    assert _corpus().section_of(AAL_Q2.id, LIQUIDITY) == "Results > Liquidity"


def test_audit_flags_quote_missing_from_expected_document() -> None:
    s = _sample(id="m", docs=["AAL_10-Q_2026-06-30.md"], spans=["This sentence is not in the filing at all."])
    findings = audit([s], _corpus(), DevIndex({}, {}), assign_splits())
    assert any(f.level == "error" and "not verbatim" in f.message for f in findings)


def test_audit_errors_on_holdout_leak_by_paragraph_and_by_figure() -> None:
    corpus = _corpus()
    dev = DevIndex.build([DevSpan("edgar_eval_set.json", "dev-1",
                                  "the Company repurchased $25.8 billion in the prior quarter")], corpus)
    # Different paragraph and filing, same company, same figure: a restatement.
    s = _sample(id="leak", docs=["AAPL_10-Q_2026-06-27.md"],
                spans=["The Company repurchased $25.8 billion of its common stock."])
    findings = audit([s], corpus, dev, assign_splits())
    assert any(f.level == "error" and "figure '$25.8 billion'" in f.message for f in findings)
    fresh = _sample(id="fresh", docs=["AAPL_10-Q_2026-06-27.md"],
                    spans=["The Company paid the remaining $8.8 billion balance of the deemed repatriation tax."])
    assert not [f for f in audit([fresh], corpus, dev, assign_splits()) if "leakage" in f.message]


def test_audit_treats_a_leak_on_development_samples_as_a_warning() -> None:
    corpus = Corpus({"DAL_10-Q_2026-06-30.md": _doc("DAL_10-Q_2026-06-30.md", f"{FUEL}\n")})
    dev = DevIndex.build([DevSpan("edgar_eval_set.json", "dev-1", FUEL)], corpus)
    s = _sample(id="d", docs=["DAL_10-Q_2026-06-30.md"], spans=[FUEL])
    assert s.extra["split"] == DEV
    leaks = [f for f in audit([s], corpus, dev, assign_splits()) if "leakage" in f.message]
    assert leaks and all(f.level == "warn" for f in leaks)


def test_absent_terms_are_checked_in_the_companies_own_filings_only() -> None:
    # "repatriation" appears in Apple's filing, not American's: absence is a
    # claim about AAL, and Apple's wording is irrelevant to it.
    aal = _sample(id="ra", family="unanswerable", behavior="refuse", companies=["AAL"],
                  absent_terms=["repatriation"])
    assert not [f for f in audit([aal], _corpus(), DevIndex({}, {}), assign_splits()) if "absent term" in f.message]
    aapl = _sample(id="rb", family="unanswerable", behavior="refuse", companies=["AAPL"],
                   absent_terms=["repatriation"])
    assert [f for f in audit([aapl], _corpus(), DevIndex({}, {}), assign_splits()) if "absent term" in f.message]


def test_audit_warns_when_the_span_is_verbatim_in_an_unlisted_filing() -> None:
    corpus = _corpus()
    s = _sample(id="w", docs=["AAPL_10-Q_2026-06-27.md"], spans=["repurchased $25.8 billion"])
    assert any(f.level == "warn" and "AAPL_10-Q_2026-03-28.md" in f.message
               for f in audit([s], corpus, DevIndex({}, {}), assign_splits()))


# --------------------------------------------------------------------------
# Finalize
# --------------------------------------------------------------------------


def test_runner_files_keep_goldless_samples_out_of_retrieval() -> None:
    answer = _sample(id="a", docs=["AAL_10-Q_2026-06-30.md"], spans=[FUEL])
    refuse = _sample(id="r", family="unanswerable", behavior="refuse", companies=["AAL"], absent_terms=["x"])
    parts = _sample(id="p", family="distributed", parts=[
        {"label": "l", "answer": "a", "docs": ["AAL_10-Q_2026-06-30.md"], "spans": [FUEL]}])
    files = runner_files([answer, refuse, parts])
    assert [s.id for s in files["retrieval"]] == ["a", "p"]
    assert [s.id for s in files["answer"]] == ["a", "r"]
    assert [s.id for s in files["multipart"]] == ["p"]


def test_finalize_refuses_pending_or_undecided_reviews(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    raw = assemble_sample({"id": "f", "family": "direct_lookup", "behavior": "answer", "query": "q?",
                           "answer": "a", "docs": ["AAL_10-Q_2026-06-30.md"], "spans": [FUEL]},
                          assign_splits(), drafted_with="t", drafted_on="d")
    draft = tmp_path / "draft.json"
    draft.write_text(json.dumps([raw]))
    assert build_holdout.cmd_finalize(draft, None) == 1
    assert "1 pending" in capsys.readouterr().out

    raw["review"]["verdict"] = "accept"  # a verdict without rubric decisions
    draft.write_text(json.dumps([raw]))
    assert build_holdout.cmd_finalize(draft, None) == 1
    assert "without a complete review record" in capsys.readouterr().out
