# Judge calibration guide

How to label stored answers so the answer judge can be checked against
people, not against its own earlier pass rate. The procedure implements
"Quality-control human labels" and "Calibrate the judge in Phase 4" in the
[eval harness plan](eval-harness-plan.md#measurement-quality-requirements);
the tooling is `python -m rag.eval.calibrate` (`rag/eval/calibrate.py`).

**Status:** the tooling exists, but no fixture has been labeled yet. The
error bounds and the double-review subset below are still to be chosen by
the reviewers, and nothing has been calibrated.

## What gets labeled

Each fixture item is one **verdict slot**: a whole single-hop or refusal
answer, or one part (or the conclusion) of a multi-hop answer. The item
carries what the judge reads: the question, the reference, the tier or part
label, and the system's answer. Labelers answer the same question the judge
is asked, under the same rubric:

| Item | PASS when | FAIL when |
|---|---|---|
| single-hop (answerable, period, underspecified, table) | the answer is factually consistent with the reference and substantially covers it | it is wrong, materially incomplete, or refuses |
| refusal (`tier: refusal`) | the system declined, said it could not find it, or asserted no specific answer | it asserted a specific figure, date, name or claim as though the corpus supported it |
| multi-hop part | the answer states this part's reference fact, attributed to the right company and period | it omits the part, says it wasn't found, gives a different figure, or misattributes it |
| multi-hop conclusion | the overall comparison or conclusion matches the reference | otherwise |

Judge the answer **against the reference**, not against the passages and not
against what you know of the filings. Whether the answer was grounded in its
passages is a separate metric (Phase 4a), labeled separately later. If the
reference itself looks wrong, label the item against the reference and
add the tag `reference_suspect` with a note. Fix the eval set separately; a
calibration label is not the place to correct it.

**Label blind.** Drafted items don't record the judge's verdict. Don't look it
up in the run's `judgments/` before your label is recorded.

## Rules, with worked examples

These examples are illustrative, written for this guide; they are not
fixture items, and the figures are made up.

1. **Equivalent forms pass.** Rounding and unit changes that preserve the
   value pass: "$1,198 million" against a reference of "$1.2 billion" is a
   PASS. A scale error does not pass: "$1.2 million" against "$1.2 billion"
   FAILs; tag it `wrong_units`.
2. **Wrong period is a FAIL, even when the figure is real.** Reference: "Net
   sales were $391.0 billion in fiscal 2024." Answer: "Net sales were $383.3
   billion" taken from the fiscal 2023 filing. The figure exists in the corpus,
   but it answers for a different period, so FAIL; tag it `wrong_period`.
   These are the false passes a lenient judge is most likely to make, so the
   fixture deliberately includes them.
3. **Partial answers.** For a single-hop question with a two-fact reference
   ("$15.6 billion, with $2.0 billion due within 12 months"), an answer with
   only the headline figure is a PASS if the missing detail is incidental to
   the question, and a FAIL if the question asks for it. Write down which
   reading you used; tag it `partial`. For multi-hop, judge each part on its own;
   omissions in other parts don't affect this one.
4. **Hedged but correct passes.** An answer that gives the reference fact and
   adds "based on the passages provided" or a caveat passes. One that offers
   two candidate figures without committing to the right one FAILs; tag it `hedged`.
5. **Contradictory passages.** If the passages disagree (an original figure
   and a later restatement), label against the reference. An answer that
   reports both and identifies the reference's figure as the one asked for
   passes.
6. **Refusals.** On a refusal item, any specific asserted answer FAILs, even
   when worded tentatively, and so does a figure borrowed from another
   company. A terse decline passes as fully as a long one. On an answerable
   item, a refusal FAILs. A **generation failure**, meaning no answer at all
   or a "could not complete" message, is never a refusal. Drafting leaves
   these out, since they are scored as failures without a judge call.
7. **Context-dependent questions** (the underspecified tier's `implicit` and
   `paraphrase` kinds). The question names the company indirectly ("the
   iPhone maker"). An answer that resolves to the reference's company and
   period and states the fact passes. One that resolves to a different
   company fails; tag it `wrong_entity`.

Tags in use: `wrong_period`, `wrong_units`, `wrong_entity`, `partial`,
`hedged`, `reference_suspect`. Add one when it describes the item, whether
the item passes or fails; reports break error rates down by tag.

## Recording a label

Each review goes into the item's `reviews` list. Never edit an earlier
review; a changed mind is a new entry with a new date:

```json
{"reviewer": "R1", "passed": false, "rationale": "FY2023 figure for a FY2024 question", "at": "2026-10-20"}
```

`reviewer` is a stable pseudonym (R1, R2, …), not a name. The rationale is
required: it is what an adjudicator and a later reader check the label
against.

## Independent review and adjudication

- **Choose the double-reviewed subset before labeling starts**, spanning
  every category. For example: every item whose number is divisible by 3,
  about 20 of 60. Record the rule in the fixture's `double_review` field.
- The second reviewer labels those items without seeing the first review.
- Where the two disagree, an adjudicator records `adjudication`:
  `{"passed": …, "reason": "…", "by": "R1"}`. Calibration refuses an item
  with disagreeing reviews and no adjudication. The original reviews stay.
- **One reviewer only:** label everything as R1 and leave
  `double_review` empty. The report then states that the labels have no
  independent check. Relabeling your own items later doesn't count as a
  second reviewer.

## Building and running the fixture

1. **Draft** from judged answer runs that span the target tiers:

   ```bash
   python -m rag.eval.calibrate draft --runs <answer run> <refusal run> <multihop run> --per-cell 8
   ```

   It samples up to 8 items per (category, judge outcome) cell, so passes and
   failures, refusals, and multi-hop parts and conclusions are all present,
   not just the current system's misses. It writes
   `data/eval/drafts/judge_calibration_draft.json`.
2. **Curate to about 60 items.** Keep the cells balanced, make sure
   `wrong_period` cases are present (the period tier's answers are the likeliest
   source), and save the result as `data/eval/judge_calibration.json`.
3. **Declare the bounds before any calibration run** and commit them: the
   fixture's `bounds` are `{"false_pass_rate": …, "false_fail_rate": …}`,
   the highest error rate a judge may show. Calibration compares each against
   the upper end of the rate's 95% interval, not the point estimate, and
   refuses a fixture without them. Choosing them is the reviewers' decision.
   False passes inflate every score, so their bound is usually the tighter one.
4. **Label, double-review, adjudicate**, as above.
5. **Calibrate:**

   ```bash
   python -m rag.eval.calibrate run --judge-model gemma4:31b-mlx --attempts 3
   ```

   Each attempt judges every item; the report goes to
   `data/eval/results_calibration/` as JSON (every call's inputs and reply)
   and Markdown. It reports agreement, false passes over human FAILs and
   false fails over human PASSes per attempt, each with a Wilson interval,
   then the same by category and tag, then consistency across attempts.
   Consistency measures judge noise; it is not judge accuracy.

Rerun calibration whenever the judge model, its settings, the rubric prompts
or the verdict parser change. The report records the rubric hash and parser
version so a stale calibration is visible. A judge outside its bounds can
still be used, but results that depend on it are exploratory until it is
recalibrated.
