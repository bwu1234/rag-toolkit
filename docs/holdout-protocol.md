# Fresh grouped holdout protocol (`fresh_v1`)

Status: protocol fixed 2026-10-01, before any question was written. Drafts
exist; **nothing is frozen until human review completes**, and no result on
this set exists yet. This implements steps 1–3 of the
[evaluation rigor plan](evaluation-rigor-plan.md) for the financial-filing
workload and the "untouched confirmation set" of
[Milestone 27](backlog.md#milestone-27--eval-coverage-and-judge-reliability).
`scripts/build_holdout.py` enforces the mechanical rules below.

## What it is for

Every existing EDGAR set has informed a configuration choice, so all of them
are development data. The holdout gives one confirmation measurement of a
*frozen* candidate at the exact serving configuration (retrieval 20/20/5,
the shipped prompt, ingestion through answer) on questions no tuning has
seen. It is not a benchmark to iterate on.

It is drafted from whole filings, not from chunks or retrieved results, so
evidence may cross chunk boundaries. That loss is reported (`audit` warns),
not filtered out.

## Corpus

The questions are drafted against the pooled `edgar_md` + `edgar_md_holdout`
corpora. Evaluate them with `--corpus edgar_md --corpus edgar_md_holdout`, so
each corpus serves as the other's same-sector distractor.

- `edgar_md`: the 61 existing filings, 14 companies.
- `edgar_md_holdout`: 10 filings from 5 new companies (AAL, ALK, AMD, COP,
  HD), fetched 2026-10-01. Ten requested filings had no extractable MD&A.
  Four that did extract were removed by hand: the LOW 10-K overran into
  Item 8, the COP 10-K was truncated, and both BMY 10-Qs start mid-MD&A
  with their revenue discussion missing. See its manifest.

  *Amendment, 2026-10-01, before any BMY question was drafted:* BMY was
  dropped after the protocol commit, once its truncated start was found.
  That removed one holdout group. The seeded draw of existing companies is
  unaffected.

Held-out filings stay searchable; the holdout protects questions and labels,
not documents. Freezing records the sha256 and accession of all 71 filings.
After that, `audit` refuses any corpus that differs.

## Grouping and split

**Group = company.** A fact moves between filings of the same company:
comparatives restate prior periods, and 10-Qs repeat the 10-K's paragraphs.
It does not move across companies. A cross-company question's group is the
set of its companies. A sample whose companies straddle the split is refused.

**Split**, drawn with `random.Random(20261001)` before any authoring:

- **Holdout:** one existing company per sector — AAPL, TGT, LUV, PFE, XOM —
  plus all five new companies.
- **Development (fresh):** COST, CVX, DAL, JNJ, MRK, MSFT, NVDA, UAL, WMT.

Two transfer slices are reported separately and never averaged:

- `new_entity`: companies no configuration has seen.
- `fresh_fact`: new facts about holdout companies whose *other* facts are in
  development sets.

The fresh development split exists because the new task families
(ambiguity, partial answers, revisions) have no development examples. Tune
on it, never on the holdout.

**Leakage rule.** A holdout span may not share its paragraph or table row
with any development-set span. It also may not state a figure that the same
company's development spans state, which catches restatements in other
filings. `audit` treats a violation as an error on holdout samples and a
warning on fresh-development samples.

## Task families and drafting targets

Targets are authoring budgets, set before drafting. The actual counts after
review are recorded at freeze. Every family is reported separately. No
overall score is weighted by expected traffic, because no traffic sample
exists to weight it.

| Family | Expected behavior | Target (holdout / dev) |
|---|---|---|
| `direct_lookup` | answer: one fact, ordinary phrasing, low lexical overlap | 26 / 14 |
| `entity_period` | answer: fiscal vs. calendar, quarter vs. year-to-date, segment vs. company | 16 / 8 |
| `table_calc` | answer: cells with units; calculations carry `calc` (inputs, formula, units, tolerance) | 18 / 10 |
| `distributed` | answer: 2–3 parts across sections or filings, plus a conclusion | 16 / 8 |
| `synthesis` | answer: drivers or factors over a declared scope, with a completeness rubric in parts | 9 / 5 |
| `revision` | answer: a figure or outlook that changed between filings; cite the applicable one | 7 / 3 |
| `ambiguity` | clarify, or answer every reading: a missing period or segment | 9 / 5 |
| `unanswerable` | refuse (absent fact or period), or partial (one part absent) | 17 / 9 |

Conversation (history-dependent turns) is out of scope here: the dataset
schema has no turns. It is owned by
[harness Phase 4c](eval-harness-plan.md#phase-4c--conversational-dataset-and-runner-estimate-pending).

## Sample schema

The runners read the existing fields: `query`, `expected_spans`,
`expected_doc_ids`, `expected_answer`, `matching_mode`, `tier` and `kind`.
Holdout samples add:

- `family`, `tags`, `behavior` (`answer`, `clarify`, `refuse` or `partial`).
- `companies`, `group`, `split` and `transfer`.
- `parts` and `conclusion` for compound questions. A part with
  `available: false` is a part the corpus does not support, and the answer
  must say so.
- `calc` for arithmetic questions.
- `absent_terms` for refusals and unavailable parts: strings the audit
  verifies are absent from the named companies' own filings. Absence is
  scoped to those filings, so another company's use of a word ("guidance")
  does not disprove it. The reviewer still checks for paraphrases and for
  figures that can be derived.
- `origin`, `drafted_with` and `drafted_on`.
- `review`: a verdict, the reviewer, a decision and reason for each rubric
  dimension, and any edits.
- `evidence_locations` (written at freeze): the section heading path of
  each span in the snapshot.

Answerable spans use `span_and_document` matching, so a copy of the text in
another period's filing does not earn credit.

## Drafting record (2026-10-01)

Claude Opus 5.5 drafted the questions from whole filings. For filings
already used by development sets, it drafted from a view that marked every
development-set paragraph and figure, so drafts avoided that material
instead of being screened for it afterwards. The calculation references
were recomputed independently.

**No system has been run on these questions**: no retrieval, no answer and
no judge. Keep it that way until the freeze and the predeclaration in
`holdout_uses`.

| Family | Holdout (target) | Dev (target) |
|---|---|---|
| `direct_lookup` | 23 (26) | 9 (14) |
| `entity_period` | 15 (16) | 14 (8) |
| `table_calc` | 19 (18) | 8 (10) |
| `distributed` | 14 (16) | 6 (8) |
| `synthesis` | 10 (9) | 6 (5) |
| `revision` | 8 (7) | 2 (3) |
| `ambiguity` | 9 (9) | 5 (5) |
| `unanswerable` | 16 (17) | 7 (9) |
| **Total** | **114** (118) | **57** (62) |

Expected behaviors: holdout 89 answer, 9 clarify, 12 refuse and 4 partial;
development 45 answer, 5 clarify, 5 refuse and 2 partial. The holdout has
61 `new_entity` and 53 `fresh_fact` drafts in 15 groups, 5 of them
cross-company. The development split has 11 groups.

Known gaps in the drafts:

- `revision` is thin in development (2 drafts).
- Refusals are all period-absent, near-miss metric or out-of-scope-section
  cases. There are no off-corpus entities, which the existing refusal set
  already covers.
- Every ambiguity is a missing period, segment or basis. None is a genuine
  entity ambiguity.

`audit` reports two spans that the structured chunker splits across chunks.
They are kept, as the protocol requires.

## How to review

```
python scripts/build_holdout.py review                 # every pending draft, with context
python scripts/build_holdout.py review --id fh-aal-...  # one draft
python scripts/build_holdout.py audit                  # re-run every mechanical check
```

Edit `data/eval/drafts/fresh_v1_draft.json` in place. For each draft, fill
in:

- `review.verdict`: `accept` or `reject`;
- `review.reviewer`;
- for each of `answerability`, `usefulness` and `clarity`, a `decision`
  (`pass` or `fail`) and a `reason`;
- `review.edits`: what was changed, if anything.

Reject rather than repair a draft whose question you would not have asked.
Run `audit` after edits; then run `finalize`.

## Review rubric

Following the [harness requirements](eval-harness-plan.md#measurement-quality-requirements),
every accepted *or rejected* draft records a decision and a reason on three
dimensions:

- **Answerability.** The evidence entails the reference answer, and the
  entity, period, unit and version are right. For `refuse`, the fact is
  truly absent: no paraphrase, and nothing derivable from other figures.
- **Usefulness.** An analyst could plausibly ask this.
- **Clarity.** The intended meaning is recoverable. For `ambiguity`, the
  ambiguity is real, and the criteria accept both clarifying and answering
  every reading.

The reviewer also adds alternatives for any other filing stating the same
fact for the same period (`review` lists verbatim copies), and edits the
draft in place, recording what changed under `review.edits`.

**Single-reviewer limitation.** Only one human reviewer is available. The
freeze record states `single_reviewer_limitation: true`, and no agreement
statistic is claimed. If a second reviewer becomes available, double-review
a stratified 20%: every family, both splits, accepts and rejects.

## Freeze and use

1. `audit` passes with no errors on every accepted draft.
2. `finalize` refuses while any verdict or rubric decision is missing. It
   then writes `data/eval/holdout/fresh_v1_{dev,holdout}_{retrieval,answer,multipart}.json`
   and `fresh_v1_freeze.json`. The freeze file records the corpus snapshot,
   file hashes, split, counts and reviewers. Commit them before any run.
3. **Predeclare, then run once.** Before the first holdout run, append to
   `holdout_uses` in the freeze file:
   - the frozen candidate and baseline configs;
   - primary metrics per family (retrieval hit/recall@5 at the final list,
     stage-1 recall@20, answer pass, multi-part completeness, refusal
     correctness and false refusals);
   - the smallest worthwhile effect and the acceptable regressions;
   - the uncertainty method: a paired bootstrap over company groups, with
     exact McNemar reported alongside.
4. Generation repeats: three per variant, as in the harness plan.
5. If holdout failures inform a fix, later results on this set are
   **adaptive**. Record that use, keep the set for regression, and draft
   `fresh_v2` for the next confirmation.

## What it does not cover yet

- **Evidence-support and citation scoring.** The answer judge still compares
  against a reference only. `evidence_locations` and part spans are stored
  so that Phase 4a scorers can use them.
- **Judge calibration.** The 60-answer human fixture is separate work. Until
  it exists, answer-pass rates on this set are uncalibrated.
- **Workload breadth.** It covers MD&A sections of large-cap 10-K/10-Q
  filings only. There are no full filings, PDFs or enterprise documents, and
  five new companies and five holdout groups among existing companies is a
  small number of groups.
