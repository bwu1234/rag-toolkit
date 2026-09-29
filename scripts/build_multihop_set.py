#!/usr/bin/env python
"""Build data/eval/edgar_multihop_set.json from pairs of verified single-hop samples.

Phase 0 of the Milestone 19 plan: the agent loop can only be compared with the
pipeline on questions that need more than one search, and there were six such
questions, hand-written for the probe. This builds 30-40 more, without new
ground truth to verify: each question names two or more samples of
`edgar_eval_set.json` (same metric, different company or period, or several
companies for an aggregation), and each *part* inherits that sample's gold
spans and answer. The only hand-authored gold is the optional `conclusion`
(which is larger, which way it moved), computed from the inherited figures and
kept to what the question explicitly asks for.

`answer` may override a part's inherited answer where the source is ambiguous
out of context -- Costco reports in millions without saying so in the span, and
some generated answers dropped half of what the span states. The spans are
never overridden.

Every span is checked against the corpus documents when they are present, so a
typo in a source id or a stale sample fails the build rather than producing a
question no retrieval could satisfy.

A part may instead be hand-authored (`G`), with spans quoted from the filing,
when no single-hop sample covers the fact. That is new ground truth, so it is
kept rare and passes the same span-in-corpus check. The first one is the airline
operating-margin question, the answerable form of the probe's superlative
("highest operating margin last quarter" across all 14 companies). The
original stays in the refusal set: operating income is not in the MD&A of six
of the 14 companies and their latest quarters end on different dates, so even
an agent that searches everything should decline or qualify it.

Usage:
    python scripts/build_multihop_set.py            # writes the set
    python scripts/build_multihop_set.py --check    # validate only
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.eval.dataset import EvalDataset, EvalSample, ExpectedSpan  # noqa: E402
from rag.eval.relevance import normalize  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
SOURCE = REPO / "data/eval/edgar_eval_set.json"
OUTPUT = REPO / "data/eval/edgar_multihop_set.json"
CORPUS = REPO / "data/corpora/edgar/documents"


def P(source: str, label: str, answer: str | None = None) -> dict[str, Any]:
    """A part: `source` is the single-hop sample id, `FILE#chunkN` shorthand."""
    return {"source": source.replace("#", ".md::chunk"), "label": label, "answer": answer}


def G(source: str, label: str, answer: str, spans: list[str]) -> dict[str, Any]:
    """A hand-authored part: `spans` are quoted from the filing, `source` names the chunk holding the first."""
    return {"source": source.replace("#", ".md::chunk"), "label": label, "answer": answer, "spans": spans}


# (id, kind, question, parts, conclusion or None)
SPEC: list[tuple[str, str, str, list[dict[str, Any]], str | None]] = [
    # --- cross_period: one company, one metric, two or more periods -------------
    ("mh-aapl-lease-yoy", "cross_period",
     "How did Apple's fixed lease payment obligations change between September 28, 2024 and September 27, 2025?",
     [P("AAPL_10-K_2024-09-28#12", "Apple, as of September 28, 2024", "$15.6 billion"),
      P("AAPL_10-K_2025-09-27#14", "Apple, as of September 27, 2025", "$16.8 billion")],
     "They increased, from $15.6 billion to $16.8 billion."),
    ("mh-aapl-buyback-q2q3", "cross_period",
     "How much of its common stock did Apple repurchase in the second quarter and in the third quarter of fiscal 2026, and in which quarter was it larger?",
     [P("AAPL_10-Q_2026-03-28#23", "Apple, second quarter of fiscal 2026"),
      P("AAPL_10-Q_2026-06-27#22", "Apple, third quarter of fiscal 2026")],
     "The third quarter ($25.8 billion vs $11.0 billion)."),
    ("mh-dal-oplease-yoy", "cross_period",
     "How did Delta Air Lines' minimum operating lease obligations change from December 31, 2024 to December 31, 2025?",
     [P("DAL_10-K_2024-12-31#35", "Delta, as of December 31, 2024", "$8.4 billion"),
      P("DAL_10-K_2025-12-31#33", "Delta, as of December 31, 2025", "$7.8 billion")],
     "They decreased, from $8.4 billion to $7.8 billion."),
    ("mh-dal-debt-yoy", "cross_period",
     "How did the principal amount of Delta Air Lines' debt and finance leases change from December 31, 2024 to December 31, 2025?",
     [P("DAL_10-K_2024-12-31#42", "Delta, at December 31, 2024"),
      P("DAL_10-K_2025-12-31#40", "Delta, at December 31, 2025")],
     "It decreased, from $16.2 billion to $14.1 billion."),
    ("mh-dal-debt-repay", "cross_period",
     "What cash outflows did Delta Air Lines report for repayments of debt and finance lease obligations in the nine months ended September 30, 2025, and in the three months ended March 31, 2026?",
     [P("DAL_10-Q_2025-09-30#39", "Delta, nine months ended September 30, 2025", "$3.9 billion"),
      P("DAL_10-Q_2026-03-31#28", "Delta, three months ended March 31, 2026")],
     None),
    ("mh-dal-buyback-2026", "cross_period",
     "Had Delta Air Lines repurchased any shares under its June 2025 repurchase program as of March 31, 2026, and as of June 30, 2026?",
     [P("DAL_10-Q_2026-03-31#30", "Delta, through March 31, 2026", "No shares had been repurchased."),
      P("DAL_10-Q_2026-06-30#41", "Delta, through June 30, 2026", "No shares had been repurchased.")],
     None),
    ("mh-cost-credit", "cross_period",
     "How did Costco's borrowing capacity under its bank credit facilities change between September 1, 2024 and February 15, 2026?",
     [P("COST_10-K_2024-09-01#29", "Costco, at September 1, 2024", "$1,198 million"),
      P("COST_10-Q_2026-02-15#34", "Costco, at February 15, 2026", "$1,447 million")],
     "It increased, by $249 million."),
    ("mh-cost-ocf-q2", "cross_period",
     "Costco reports net cash provided by operating activities for the first quarter and the first half of fiscal 2026. What were those figures, and how much did it generate in the second quarter alone?",
     [P("COST_10-Q_2025-11-23#24", "Costco, first quarter of fiscal 2026", "$4,688 million"),
      P("COST_10-Q_2026-02-15#30", "Costco, first half of fiscal 2026", "$7,684 million")],
     "About $2,996 million in the second quarter ($7,684 million minus $4,688 million)."),
    ("mh-nvda-cash", "cross_period",
     "How much did NVIDIA's cash, cash equivalents, and marketable securities change between January 26, 2025 and October 26, 2025?",
     [P("NVDA_10-K_2025-01-26#39", "NVIDIA, as of January 26, 2025", "$43.2 billion"),
      P("NVDA_10-Q_2025-10-26#33", "NVIDIA, as of October 26, 2025", "$60.6 billion")],
     "They increased by $17.4 billion."),
    ("mh-nvda-gm", "cross_period",
     "What was NVIDIA's gross margin for the third quarter of fiscal 2026, and for the full fiscal year 2026?",
     [P("NVDA_10-Q_2025-10-26#23", "NVIDIA, third quarter of fiscal 2026"),
      P("NVDA_10-K_2026-01-25#27", "NVIDIA, fiscal year 2026")],
     None),
    ("mh-msft-ocf", "cross_period",
     "What was Microsoft's cash from operations for the three months ended September 30, 2025, and for the nine months ended March 31, 2026? What does that imply for the six months ended March 31, 2026?",
     [P("MSFT_10-Q_2025-09-30#38", "Microsoft, three months ended September 30, 2025", "$45.1 billion"),
      P("MSFT_10-Q_2026-03-31#51", "Microsoft, nine months ended March 31, 2026", "$127.5 billion")],
     "About $82.4 billion for the six months ended March 31, 2026."),
    ("mh-msft-div", "cross_period",
     "What dividends did Microsoft's Board of Directors declare in fiscal year 2025, and in the nine months ended March 31, 2026?",
     [P("MSFT_10-K_2025-06-30#41", "Microsoft, fiscal year 2025"),
      P("MSFT_10-Q_2026-03-31#55", "Microsoft, nine months ended March 31, 2026")],
     None),
    ("mh-luv-salaries", "cross_period",
     "How did the year-over-year growth in Southwest Airlines' salaries, wages, and benefits expense evolve across full-year 2025, the first quarter of 2026, and the second quarter of 2026?",
     [P("LUV_10-K_2025-12-31#22", "Southwest, full-year 2025"),
      P("LUV_10-Q_2026-03-31#17", "Southwest, first quarter of 2026"),
      P("LUV_10-Q_2026-06-30#21", "Southwest, second quarter of 2026")],
     "The growth rate accelerated in each period (5.9%, then 6.3%, then 7.3%)."),
    ("mh-luv-casm", "cross_period",
     "How did Southwest Airlines' operating expenses per available seat mile change in the third quarter of 2025 and in the first six months of 2026, each compared with the prior-year period?",
     [P("LUV_10-Q_2025-09-30#30", "Southwest, third quarter of 2025"),
      P("LUV_10-Q_2026-06-30#33", "Southwest, first six months of 2026", "Increased 9.3 percent")],
     None),
    ("mh-ual-fuel", "cross_period",
     "How did United Airlines' aircraft fuel expense change year over year in the first quarter of 2026 and in the second quarter of 2026?",
     [P("UAL_10-Q_2026-03-31#11", "United, first quarter of 2026", "Increased $339 million, or 12.6%"),
      P("UAL_10-Q_2026-06-30#11", "United, second quarter of 2026", "Increased $2.3 billion, or 84.1%")],
     None),
    ("mh-ual-unrealized", "cross_period",
     "What were United Airlines' net unrealized losses on investments for 2024, and for the first nine months of 2025?",
     [P("UAL_10-K_2024-12-31#24", "United, 2024"),
      P("UAL_10-Q_2025-09-30#30", "United, first nine months of 2025")],
     None),
    ("mh-wmt-etr", "cross_period",
     "Compare Walmart's effective income tax rate for fiscal 2025 with its rate for the three months ended April 30, 2026.",
     [P("WMT_10-K_2025-01-31#33", "Walmart, fiscal 2025"),
      P("WMT_10-Q_2026-04-30#25", "Walmart, three months ended April 30, 2026")],
     "Slightly lower in the 2026 quarter (23.2% vs 23.4%)."),
    ("mh-wmt-comps", "cross_period",
     "What was Walmart U.S. comparable sales growth for the three months ended July 31, 2025, and for full fiscal 2026?",
     [P("WMT_10-Q_2025-07-31#6", "Walmart U.S., three months ended July 31, 2025"),
      P("WMT_10-K_2026-01-31#9", "Walmart U.S., fiscal 2026")],
     None),
    ("mh-tgt-gm", "cross_period",
     "Compare Target's gross margin rate for the quarter ended August 2, 2025 with its rate for the quarter ended May 2, 2026.",
     [P("TGT_10-Q_2025-08-02#12", "Target, quarter ended August 2, 2025"),
      P("TGT_10-Q_2026-05-02#14", "Target, quarter ended May 2, 2026")],
     "They were the same, 29.0 percent in both quarters."),
    ("mh-tgt-moodys", "cross_period",
     "What was Target's Moody's long-term debt credit rating as of February 1, 2025, and as of May 2, 2026?",
     [P("TGT_10-K_2025-02-01#30", "Target, as of February 1, 2025", "A2"),
      P("TGT_10-Q_2026-05-02#28", "Target, as of May 2, 2026")],
     None),
    ("mh-cvx-fields", "cross_period",
     "Which has the longer estimated remaining production life: Chevron's Ballymore field, or its Jack and St. Malo fields?",
     [P("CVX_10-K_2024-12-31#25", "Chevron, Ballymore field", "30 years"),
      P("CVX_10-K_2024-12-31#26", "Chevron, Jack and St. Malo fields", "20 years")],
     "Ballymore, by 10 years."),

    # --- cross_company: one metric, two companies --------------------------------
    ("mh-lease-aapl-dal", "cross_company",
     "Which was larger: Apple's fixed lease payment obligations as of September 27, 2025, or Delta Air Lines' minimum operating lease obligations as of December 31, 2025?",
     [P("AAPL_10-K_2025-09-27#14", "Apple, as of September 27, 2025", "$16.8 billion"),
      P("DAL_10-K_2025-12-31#33", "Delta, as of December 31, 2025", "$7.8 billion")],
     "Apple's."),
    ("mh-capex-dal-nvda", "cross_company",
     "Compare Delta Air Lines' capital expenditures for 2024 with NVIDIA's capital expenditures for fiscal year 2026.",
     [P("DAL_10-K_2024-12-31#36", "Delta, 2024"),
      P("NVDA_10-K_2026-01-25#40", "NVIDIA, fiscal year 2026")],
     "NVIDIA's were larger ($6.1 billion vs $5.1 billion)."),
    ("mh-cash-ual-luv", "cross_company",
     "Compare United Airlines' unrestricted cash as of September 30, 2025 with Southwest Airlines' unrestricted cash as of March 31, 2026.",
     [P("UAL_10-Q_2025-09-30#31", "United, as of September 30, 2025", "$13.3 billion"),
      P("LUV_10-Q_2026-03-31#42", "Southwest, as of March 31, 2026")],
     "United's was larger."),
    ("mh-ocf-xom-cost", "cross_company",
     "Which generated more cash from operating activities in the first half of fiscal 2026: ExxonMobil or Costco?",
     [P("XOM_10-Q_2026-06-30#39", "ExxonMobil, first six months of 2026"),
      P("COST_10-Q_2026-02-15#30", "Costco, first half of fiscal 2026", "$7,684 million")],
     "ExxonMobil."),
    ("mh-auth-mrk-wmt", "cross_company",
     "Which had the larger remaining share repurchase authorization: Merck as of December 31, 2025, or Walmart as of July 31, 2025?",
     [P("MRK_10-K_2025-12-31#114", "Merck, as of December 31, 2025", "$7.3 billion"),
      P("WMT_10-Q_2025-07-31#47", "Walmart, as of July 31, 2025", "$5.9 billion")],
     "Merck's."),
    ("mh-buyback-cvx-aapl", "cross_company",
     "How much did Chevron spend repurchasing shares in the third quarter of 2025, and how much common stock did Apple repurchase in the second quarter of fiscal 2026? Which was larger?",
     [P("CVX_10-Q_2025-09-30#63", "Chevron, third quarter of 2025", "$2.6 billion (16.6 million shares)"),
      P("AAPL_10-Q_2026-03-28#23", "Apple, second quarter of fiscal 2026")],
     "Apple's."),
    ("mh-tariff-wmt-tgt", "cross_company",
     "Had Walmart (as of the three months ended April 30, 2026) and Target (as of May 2, 2026) recognized anything from their tariff refund claims?",
     [P("WMT_10-Q_2026-04-30#2", "Walmart, three months ended April 30, 2026",
        "No amounts related to the claims were recognized."),
      P("TGT_10-Q_2026-05-02#3", "Target, as of May 2, 2026",
        "No refunds had been received and no receivable was recorded.")],
     None),

    ("mh-gm-nvda-tgt", "cross_company",
     "Which had the higher gross margin: NVIDIA for the third quarter of fiscal 2026, or Target for the quarter ended August 2, 2025?",
     [P("NVDA_10-Q_2025-10-26#23", "NVIDIA, third quarter of fiscal 2026"),
      P("TGT_10-Q_2025-08-02#12", "Target, quarter ended August 2, 2025")],
     "NVIDIA."),

    # --- aggregation: three or more entities, with a ranking or a roll-up ---------
    ("mh-agg-ocf-q1-2026", "aggregation",
     "Which of Delta Air Lines, Southwest Airlines, and Merck generated the most cash from operating activities in the three months ended March 31, 2026?",
     [P("DAL_10-Q_2026-03-31#23", "Delta, three months ended March 31, 2026"),
      P("LUV_10-Q_2026-03-31#33", "Southwest, three months ended March 31, 2026"),
      P("MRK_10-Q_2026-03-31#84", "Merck, first three months of 2026")],
     "Merck ($3.9 billion)."),
    ("mh-agg-buybacks", "aggregation",
     "Which spent the most on share repurchases: NVIDIA in fiscal year 2026, Walmart in the nine months ended October 31, 2025, or Target in the six months ended August 2, 2025?",
     [P("NVDA_10-K_2026-01-25#36", "NVIDIA, fiscal year 2026"),
      P("WMT_10-Q_2025-10-31#49", "Walmart, nine months ended October 31, 2025"),
      P("TGT_10-Q_2025-08-02#29", "Target, six months ended August 2, 2025", "$251 million")],
     "NVIDIA ($40.4 billion)."),
    ("mh-agg-etr", "aggregation",
     "Which had the highest effective tax rate: Walmart for fiscal 2025, Southwest Airlines for the first nine months of 2025, or Microsoft for the three months ended December 31, 2025?",
     [P("WMT_10-K_2025-01-31#33", "Walmart, fiscal 2025"),
      P("LUV_10-Q_2025-09-30#53", "Southwest, first nine months of 2025"),
      P("MSFT_10-Q_2025-12-31#40", "Microsoft, three months ended December 31, 2025")],
     "Southwest (about 24.6%)."),
    ("mh-agg-dividend", "aggregation",
     "Rank these per-share dividends from highest to lowest: Apple's quarterly cash dividend as of September 27, 2025, the dividend Pfizer declared in June 2026, and the dividend Target paid in the first quarter of 2026.",
     [P("AAPL_10-K_2025-09-27#16", "Apple, as of September 27, 2025"),
      P("PFE_10-Q_2026-06-28#81", "Pfizer, declared June 2026", "$0.43 per share"),
      P("TGT_10-Q_2026-05-02#27", "Target, first quarter of 2026", "$1.14 per share ($526 million)")],
     "Target, then Pfizer, then Apple."),
    ("mh-agg-tariffs", "aggregation",
     "What did Walmart (three months ended April 30, 2026), Target (as of May 2, 2026), and Merck (period ended September 30, 2025) each report about tariffs?",
     [P("WMT_10-Q_2026-04-30#2", "Walmart, three months ended April 30, 2026",
        "Did not recognize any amounts related to tariff refund claims."),
      P("TGT_10-Q_2026-05-02#3", "Target, as of May 2, 2026",
        "No IEEPA tariff refunds had been received and no receivable was recorded."),
      P("MRK_10-Q_2025-09-30#10", "Merck, period ended September 30, 2025",
        "Expected tariffs implemented to date to cost less than $100 million.")],
     None),
    # Hand-authored: no single-hop sample states airline operating income. Only
    # Southwest reports the margin itself; the other two are computed from the
    # quoted operating income and revenue. Delta leads on every reading,
    # including Southwest's 6.7% excluding special items.
    ("mh-agg-airline-margin", "aggregation",
     "Which of the airlines in this corpus had the highest operating margin in the quarter ended June 30, 2026?",
     [G("DAL_10-Q_2026-06-30#0", "Delta, quarter ended June 30, 2026",
        "About 9.6% ($1.9 billion operating income on $19,757 million total operating revenue)",
        ["Our operating income for the June 2026 quarter was $1.9 billion",
         "| Total operating revenue | $ | 19,757 | | $ | 16,648 |"]),
      G("LUV_10-Q_2026-06-30#44", "Southwest, quarter ended June 30, 2026",
        "3.4% as reported (6.7% excluding special items)",
        ["| | Operating margin, as reported | 3.4 | % | | 3.1 | %"]),
      G("UAL_10-Q_2026-06-30#4", "United, quarter ended June 30, 2026",
        "About 6.2% ($1,096 million operating income on $17,672 million operating revenue)",
        ["| Operating revenue | | $ | 17,672 | | | $ | 15,236 |",
         "| Operating income | | 1,096 | | | 1,325 |"])],
     "Delta (about 9.6%), ahead of United (about 6.2%) and Southwest (3.4%)."),
]


def build(sources: dict[str, EvalSample]) -> list[EvalSample]:
    built: list[EvalSample] = []
    for sample_id, kind, query, parts, conclusion in SPEC:
        out_parts: list[dict[str, Any]] = []
        part_spans: list[list[ExpectedSpan]] = []
        doc_ids: set[str] = set()
        for part in parts:
            if "spans" in part:
                spans = [ExpectedSpan(text=t) for t in part["spans"]]
                out_parts.append({
                    "label": part["label"],
                    "answer": part["answer"],
                    "spans": [s.to_json() for s in spans],
                    "source_id": part["source"],
                })
                part_spans.append(spans)
                doc_ids.add(part["source"].split("::")[0])
                continue
            src = sources.get(part["source"])
            if src is None:
                raise SystemExit(f"{sample_id}: unknown source sample {part['source']!r}")
            if not src.expected_spans or not src.expected_answer:
                raise SystemExit(f"{sample_id}: source {src.id!r} lacks spans or an answer")
            out_parts.append({
                "label": part["label"],
                "answer": part["answer"] or src.expected_answer,
                "spans": [s.to_json() for s in src.expected_spans],
                "source_id": src.id,
            })
            part_spans.append(list(src.expected_spans))
            doc_ids.update(src.expected_doc_ids)
        answer = "; ".join(f"{p['label']}: {p['answer']}" for p in out_parts)
        if conclusion:
            answer += f". Conclusion: {conclusion}"
        extra: dict[str, Any] = {"tier": "multihop", "kind": kind, "parts": out_parts}
        if conclusion:
            extra["conclusion"] = conclusion
        built.append(EvalSample(
            id=sample_id,
            query=query,
            # The union, so retrieval_eval can also run on this set: recall there
            # is the share of every part's evidence the single search surfaced.
            expected_spans=[s for spans in part_spans for s in spans],
            expected_doc_ids=sorted(doc_ids),
            expected_answer=answer,
            extra=extra,
        ))
    return built


def check_spans_in_corpus(samples: list[EvalSample]) -> int:
    """Count spans absent from their source document; 0 means every span is findable."""
    if not CORPUS.is_dir():
        print(f"warning: {CORPUS} missing; skipping the span-in-corpus check")
        return 0
    missing = 0
    cache: dict[str, str] = {}
    for sample in samples:
        for part in sample.extra["parts"]:
            doc = part["source_id"].split("::")[0]
            if doc not in cache:
                cache[doc] = normalize((CORPUS / doc).read_text(encoding="utf-8"))
            for span in part["spans"]:
                text = span if isinstance(span, str) else span["text"]
                if normalize(text) not in cache[doc]:
                    missing += 1
                    print(f"{sample.id}: span not in {doc}: {text[:80]!r}")
    return missing


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--check", action="store_true", help="Validate without writing")
    args = parser.parse_args()

    sources = {s.id: s for s in EvalDataset.load(SOURCE)}
    samples = build(sources)

    ids = Counter(s.id for s in samples)
    if dupes := [i for i, n in ids.items() if n > 1]:
        raise SystemExit(f"duplicate ids: {dupes}")
    if check_spans_in_corpus(samples):
        return 1

    kinds = Counter(s.extra["kind"] for s in samples)
    print(f"{len(samples)} samples: " + ", ".join(f"{k}={n}" for k, n in sorted(kinds.items())))
    if not args.check:
        EvalDataset(samples=samples).save(OUTPUT)
        print(f"wrote {OUTPUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
