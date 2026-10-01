#!/usr/bin/env python
"""Build data/eval/edgar_adaptive_set.json: questions whose next search depends on what the last one found.

Milestone 19 phase 4's first stage showed the multi-hop set can't tell an agent
from the pipeline: 34 of its 35 questions name every company and period they
ask about, so one search with the whole question already reaches most of the
evidence, and the oracle showed the 9b combines it well once it has it. What
an agent adds is searching *again, differently*:

* **bridge** -- the question names a fact, not the company (or the period) it
  asks about. The first search finds who; only then can the second search ask
  for the metric, which every company in the corpus reports. Two shapes: an
  identifying fact recent or obscure enough that the model can't know it
  (Owowo, Enflonsia, RECARO seats), and a comparison whose winner the
  follow-up is about.
* **discovery** -- the question names a class ("the airlines in this corpus",
  "the pharmaceutical companies") or asks for the latest period, so which
  companies or which filing to search isn't in the question. One open-set
  question (tariff refunds) was checked against every filing in the corpus.

Each question is judged part by part like the multi-hop set, with a part for
the bridging fact itself (the company named), so an answer that guesses the
right figure from the wrong company fails. Parts reuse verified single-hop
samples (`P`) where one exists; the rest are quoted from the filings (`G`) and
pass the same span-in-corpus check. It is a separate file, not an extension
of the multi-hop set, so results already recorded on that set stay comparable.

Usage:
    python scripts/build_adaptive_set.py            # writes the set
    python scripts/build_adaptive_set.py --check    # validate only
"""

from __future__ import annotations

import sys
from typing import Any

from build_multihop_set import REPO, SPEC as MULTIHOP_SPEC, G, P, Spec, main


def _parts_of(sample_id: str) -> list[dict[str, Any]]:
    """Reuse a multi-hop question's parts, so the two sets can't disagree on a fact."""
    return next(parts for sid, _kind, _q, parts, _c in MULTIHOP_SPEC if sid == sample_id)


OUTPUT = REPO / "data/eval/edgar_adaptive_set.json"

# (id, kind, question, parts, conclusion or None)
SPEC: Spec = [
    # --- bridge: an identifying fact, then a metric every company reports ------
    ("ad-owowo-buyback", "bridge",
     "A company in this corpus holds a working interest in the offshore blocks that contain the Owowo Field. "
     "How much did that company spend repurchasing its shares in the third quarter of 2025?",
     [G("CVX_10-K_2024-12-31#42", "The company holding an interest in the Owowo Field",
        "Chevron (a 27 percent nonoperated working interest in OML 139 and OML 154, which the Owowo Field straddles)",
        ["Chevron also holds a 27 percent nonoperated working interest in OML 139 and OML 154",
         "including the Owowo Field, which straddles OML 139 and OML 154"]),
      P("CVX_10-Q_2025-09-30#63", "Its share repurchases, third quarter of 2025",
        "$2.6 billion (16.6 million shares)")],
     "Chevron spent $2.6 billion."),
    ("ad-enflonsia-ocf", "bridge",
     "How much cash did the company that sells Enflonsia provide from operating activities in the first quarter of 2026?",
     [P("MRK_10-Q_2026-03-31#34", "The company that sells Enflonsia", "Merck"),
      P("MRK_10-Q_2026-03-31#84", "Its cash from operating activities, first quarter of 2026", "$3.9 billion")],
     "Merck, $3.9 billion."),
    ("ad-recaro-cash", "bridge",
     "An airline in this corpus has been retrofitting its aircraft with RECARO seats. "
     "What was its unrestricted cash balance as of March 31, 2026?",
     [P("LUV_10-Q_2026-03-31#8", "The airline retrofitting aircraft with RECARO seats", "Southwest Airlines"),
      P("LUV_10-Q_2026-03-31#42", "Its unrestricted cash, March 31, 2026", "$3.3 billion")],
     "Southwest Airlines, $3.3 billion."),
    ("ad-vepdegestrant-dividend", "bridge",
     "What dividend per share did the company that partnered with Arvinas on vepdegestrant declare in June 2026?",
     [P("PFE_10-Q_2025-09-28#72", "The company that partnered with Arvinas on vepdegestrant", "Pfizer"),
      P("PFE_10-Q_2026-06-28#81", "Its dividend declared in June 2026", "$0.43 per share")],
     "Pfizer, $0.43 per share."),
    ("ad-wellness-ocf", "bridge",
     "The retailer in this corpus that introduced 2,000 new wellness products in January 2025: how much cash did "
     "its operating activities provide in the first quarter of its fiscal 2026?",
     [P("TGT_10-K_2025-02-01#0", "The retailer that introduced 2,000 wellness products", "Target"),
      P("TGT_10-Q_2026-05-02#25", "Its operating cash flow, first quarter of fiscal 2026 (three months ended May 2, 2026)",
        "$0.7 billion")],
     "Target, $0.7 billion."),
    ("ad-refinery-debt", "bridge",
     "One airline in this corpus owns a refinery, which posted an operating loss of $39 million in the March 2026 "
     "quarter. How much did that airline repay on its debt and finance lease obligations in that quarter?",
     [P("DAL_10-Q_2026-03-31#16", "The airline that owns the refinery", "Delta Air Lines"),
      P("DAL_10-Q_2026-03-31#28", "Its debt and finance lease repayments, March 2026 quarter", "$1.6 billion")],
     "Delta Air Lines, $1.6 billion."),
    ("ad-openai-etr", "bridge",
     "What was the effective tax rate, for the three months ended March 31, 2026, of the company that signed a new "
     "definitive agreement with OpenAI in October 2025?",
     [P("MSFT_10-Q_2025-12-31#5", "The company that signed the October 2025 agreement with OpenAI", "Microsoft"),
      P("MSFT_10-Q_2026-03-31#43", "Its effective tax rate, three months ended March 31, 2026", "19%")],
     "Microsoft, 19%."),
    # --- bridge: a comparison, then a follow-up about the winner ---------------
    ("ad-airline-margin-capex", "bridge",
     "Which airline in this corpus had the highest operating margin in the quarter ended June 30, 2026, and what "
     "were that airline's capital expenditures for the six months ended June 30, 2026?",
     [*_parts_of("mh-agg-airline-margin"),
      P("DAL_10-Q_2026-06-30#35", "The winner's capital expenditures, six months ended June 30, 2026",
        "Delta: $2.7 billion")],
     "Delta (about 9.6%, ahead of United at about 6.2% and Southwest at 3.4%), with $2.7 billion of capital "
     "expenditures."),
    ("ad-authorization-ocf", "bridge",
     "Which had the larger remaining share repurchase authorization, Merck as of December 31, 2025 or Walmart as of "
     "July 31, 2025? How much cash did that company provide from operating activities in the first six months of 2026?",
     [P("MRK_10-K_2025-12-31#114", "Merck, as of December 31, 2025", "$7.3 billion"),
      P("WMT_10-Q_2025-07-31#47", "Walmart, as of July 31, 2025", "$5.9 billion"),
      G("MRK_10-Q_2026-06-30#102", "The winner's operating cash flow, first six months of 2026",
        "Merck: $9.3 billion",
        ["Cash provided by operating activities was $9.3 billion in the first six months of 2026"])],
     "Merck ($7.3 billion vs $5.9 billion), which provided $9.3 billion from operating activities."),
    ("ad-apple-buyback-os", "bridge",
     "In which quarter of fiscal 2026 did Apple repurchase more of its common stock, the second or the third, and "
     "which operating system versions did it announce in that quarter?",
     [P("AAPL_10-Q_2026-03-28#23", "Apple, second quarter of fiscal 2026", "$11.0 billion"),
      P("AAPL_10-Q_2026-06-27#22", "Apple, third quarter of fiscal 2026", "$25.8 billion"),
      P("AAPL_10-Q_2026-06-27#4", "Operating systems announced in the larger quarter",
        "The third quarter: iOS 27, macOS 27 Golden Gate, iPadOS 27, watchOS 27, visionOS 27 and tvOS 27")],
     "The third quarter ($25.8 billion vs $11.0 billion), when Apple announced iOS 27, macOS 27 Golden Gate, "
     "iPadOS 27, watchOS 27, visionOS 27 and tvOS 27."),
    # --- discovery: which companies, or which filing, isn't in the question ----
    ("ad-airline-fuel", "discovery",
     "Which airline in this corpus had the largest year-over-year percentage increase in aircraft fuel expense in "
     "the quarter ended June 30, 2026?",
     [G("DAL_10-Q_2026-06-30#10", "Delta, quarter ended June 30, 2026",
        "Up 67% ($4,109 million vs $2,458 million)",
        [{"text": "| Aircraft fuel and related taxes | 4,109 | | 2,458 | | | 1,651 | | 67 | % |",
          "alternatives": ["Aircraft fuel and related taxes increased $1.7 billion compared to the June 2025 quarter"]}]),
      G("LUV_10-Q_2026-06-30#16", "Southwest, quarter ended June 30, 2026",
        "Up 67.0% ($2,215 million vs $1,326 million)",
        [{"text": "| Aircraft fuel and related taxes | 2,215 | | | 1,326 | | | 889 | | | 67.0 |",
          "alternatives": ["| Aircraft fuel and related taxes | 2,215 | 1,326 | 889 | 67.0 |"]}]),
      P("UAL_10-Q_2026-06-30#11", "United, quarter ended June 30, 2026", "Up 84.1% ($2.3 billion)")],
     "United (84.1%), ahead of Delta (67%) and Southwest (67.0%)."),
    ("ad-pharma-ocf", "discovery",
     "Which of the pharmaceutical companies in this corpus generated the most cash from operating activities in the "
     "first six months of 2026?",
     [G("JNJ_10-Q_2026-06-28#45", "Johnson & Johnson, fiscal first six months of 2026", "$11.1 billion",
        [{"text": "11.1\nnet cash generated from operating activities",
          "alternatives": ["| 11.1 | net cash generated from operating activities |"]}]),
      G("MRK_10-Q_2026-06-30#102", "Merck, first six months of 2026", "$9.3 billion",
        ["Cash provided by operating activities was $9.3 billion in the first six months of 2026"]),
      G("PFE_10-Q_2026-06-28#76", "Pfizer, first six months of 2026", "$3,450 million",
        [{"text": "| Operating activities | | $ | 3,450 | | | $ | 1,753 | | |",
          "alternatives": ["| Operating activities | $3,450 | $1,753 |"]}])],
     "Johnson & Johnson ($11.1 billion), ahead of Merck ($9.3 billion) and Pfizer ($3.45 billion)."),
    # Open set, checked by searching every filing for "refund" near "tariff":
    # only these three say they sought refunds. Pfizer mentions the ruling but
    # not a claim; Costco's filings don't mention refunds.
    ("ad-tariff-refunds", "discovery",
     "Which companies in this corpus say they have sought refunds of the tariffs the U.S. Supreme Court struck down "
     "in February 2026, and did any of them report that refunds improved their results?",
     [G("AAPL_10-Q_2026-06-27#6", "Apple",
        "Applied for refunds; recognized refunds received as a reduction of products cost of sales, which helped "
        "raise products gross margin in the third quarter of fiscal 2026",
        ["The Company has applied for a refund of tariffs paid",
         "primarily due to a different mix of products and tariff refunds"]),
      G("WMT_10-Q_2026-04-30#2", "Walmart",
        "Participating in the U.S. Customs and Border Protection refund process; recognized nothing in the three "
        "months ended April 30, 2026",
        ["participating in the process established by the U.S. Customs and Border Protection for refunds of tariffs",
         "the Company did not recognize any amounts related to these claims in the three months ended April 30, 2026"]),
      G("TGT_10-Q_2026-05-02#3", "Target",
        "Filing through the CAPE refund process; nothing received or recorded as of May 2, 2026, and refunds "
        "received after quarter-end were not material",
        ["We incurred tariffs under IEEPA, and are following the established refund filing and validation process",
         "Subsequent to quarter-end, we began receiving refunds, which to date have not been material"])],
     "Apple, Walmart and Target; only Apple reported that refunds improved its results."),
    ("ad-nvda-latest-gm", "discovery",
     "What was NVIDIA's gross margin in the most recent quarter covered by the filings in this corpus?",
     [G("NVDA_10-Q_2026-04-26#22", "NVIDIA, most recent quarter (first quarter of fiscal 2027, ended April 26, 2026)",
        "74.9%",
        [{"text": "Gross margin increased to 74.9% for the first quarter of fiscal year 2027",
          "alternatives": ["| Gross margin | 74.9 | % | | 75.0 | % | | 60.5 | %"]}])],
     None),
    ("ad-wmt-latest-etr", "discovery",
     "What was Walmart's effective income tax rate for the most recent fiscal year covered by the filings in this "
     "corpus, and why did it change from the year before?",
     [G("WMT_10-K_2026-01-31#30", "Walmart, most recent fiscal year (fiscal 2026)",
        "24.4%, up from 23.4% in fiscal 2025",
        ["Our effective income tax rate was 24.4%, 23.4%, and 25.5% for fiscal 2026, 2025 and 2024, respectively."]),
      G("WMT_10-K_2026-01-31#30", "The reason for the change",
        "A share-based compensation charge at its PhonePe subsidiary, which provided no tax benefit",
        ["primarily due to the share-based compensation charge recorded at the Company's PhonePe subsidiary, "
         "which provided no tax benefit"])],
     None),
]


if __name__ == "__main__":
    sys.exit(main(SPEC, OUTPUT, __doc__))
