# Evaluation rigor plan

Status: proposed work, 2026-09-28. This document records an assessment of the
existing files and results, not a new evaluation run. Dataset additions,
grader changes, and acceptance gates below are not implemented yet.

## What the current evaluation supports

The EDGAR benchmark supports focused engineering experiments and regression
checks. It has exposed period confusion, sensitivity to query wording,
reranking failures, and incomplete retrieval for comparisons. It does not
establish general RAG quality or production readiness.

| Asset | Current coverage | Appropriate use and limitation |
|---|---|---|
| Baseline corpus | 8 files, roughly 31 chunks; 43 answerable questions | Smoke tests and a small control. Retrieving 20 chunks covers most of the corpus. |
| EDGAR corpus | 61 extracted filings, 14 companies, roughly 4,236 chunks at the recorded fixed chunking config | Meaningful competition among entities and periods, but only extracted MD&A and a narrow document distribution. |
| Generated EDGAR set | 174 questions; 173 have one expected span | Direct lookup regression checks. Questions originate from existing chunks and explicitly name company and period. |
| Period set | 55 questions requiring span and document identity | Provenance checks against repeated paragraphs in different filings. Exact period-end wording makes routing easier than informal dates. |
| Underspecified set | 118 rewrites: 54 implicit-company, 64 paraphrase | Query wording sensitivity. All reuse generated-set facts; implicit descriptions identify a company rather than require clarification. |
| Multi-hop set | 35 questions: 21 cross-period, 8 cross-company, 6 aggregation | Completeness across 2–3 labeled spans. Built from existing single-hop facts except one hand-authored question (`mh-agg-airline-margin`), not independent task coverage. |
| EDGAR refusal set | 15 questions | A small refusal regression control; insufficient coverage for a broad reliability claim. |

The [recorded shipped baseline](measured-results.md#current-shipped-baseline)
has retrieval hit / answer pass of 97.7% / 94.3% on generated, 83.6% / 92.7%
on period, and 82.2% / 76.3% on underspecified. The generated set has only four
retrieval misses left. It offers little room to distinguish improvements.
The older multi-hop result of 15/34 complete answers predates the current
headers and must not be presented as current performance.

Existing strengths should be retained: span labels, explicit period provenance,
paired comparisons, per-tier reporting, configuration fingerprints, recorded
label fixes, and multi-hop scoring that requires every part and conclusion.

## Limits that affect interpretation

- **Questions selected from chunks favor the current representation.** A
  question whose answer fits the source chunk cannot reveal failures that
  require evidence across boundaries. The generator also favors wording
  close to the source. Rephrasing helps but does not add new facts or tasks.
- **Selection and reuse limit independence.** Related questions share facts,
  filings, and templates. Repeated tuning on the same sets, including the
  routing gate, makes them development data. Per-question significance tests
  do not account for adaptive selection or those dependencies.
- **The answer judge does not verify evidence.** It compares question,
  reference, and answer. Identical prose from the wrong period can pass, and
  the multi-hop audit found a false pass on period attribution. Answer pass
  must not be described as citation accuracy or groundedness.
- **Labels need audits of successes too.** A previous review found seven
  question or label defects among 22 retrieval misses. That is a selected
  sample, not an estimated error rate for the whole set. Reviewing only
  misses leaves false successes undetected.
- **Extraction narrows the workload.** EDGAR excludes filings whose MD&A
  extraction fails, including difficult table-heavy cases. These experiments
  do not measure full-filing parsing, messy PDFs, or mixed enterprise content.

Describe a negative finding as: "A benefit sufficient to enable this feature
was not demonstrated on these tasks and configurations." Describe a positive
finding with its corpus, tier, configuration, paired effect, and uncertainty.
Neither wording implies the same result on other workloads. The deterministic
header is a measured implementation choice, not proof that generated context
cannot help elsewhere. CRAG and expansion still need evaluation on relevant
harder tasks before drawing broader conclusions.

## 1. Establish the workload and a fresh holdout

Start with financial-document analysis, the workload the existing corpus can
support. Broader enterprise claims require a separate representative corpus.
Record intended tasks and their expected frequency before choosing examples;
an intentionally difficult challenge set should not masquerade as average
user traffic.

1. Keep every existing set as development/regression data. Randomly splitting
   it now cannot undo earlier tuning or create an untouched test set.
2. Collect fresh questions from realistic analyst tasks, whole documents, and
   available user turns. Log-derived candidates require review and labels;
   thumbs-down feedback is not a reference answer. Sample successful and
   ordinary turns too, so failure mining does not define the whole workload.
3. Group related facts, paraphrases, source filings, revisions, and multi-hop
   components before assigning splits. No underlying fact should cross the
   development/holdout boundary through a rewrite or comparison question.
   Prefer new companies for entity transfer; use a separately declared
   period holdout for temporal transfer and check comparative restatements
   in later filings for leakage.
4. Held-out documents still belong in the searchable evaluation index. The
   holdout protects questions and labels from tuning; it must not make
   answerable tasks unanswerable by removing their evidence.
5. Freeze a versioned corpus snapshot, question set, split assignments,
   grouping keys, and references before the comparison. Choose the candidate
   on development data, then evaluate the frozen candidate on the holdout.
6. If holdout failures inform a fix, label subsequent results as adaptive.
   Keep that version for regression and collect a fresh holdout for the next
   generalization claim. Do not repeatedly inspect the holdout to tune knobs.

Initial planning budget: 150–250 fresh questions across the task families
below, plus roughly 60 human-labeled answers for judge calibration. These are
authoring budgets, not guarantees of statistical power. Use development
variance and the smallest worthwhile improvement to set the final sample
size before inspecting holdout outcomes. Small slices remain diagnostic.

## 2. Cover realistic tasks and failure modes

Assign one primary task family and any secondary tags to each question.
Preserve ordinary lookups alongside harder tasks; report every family
separately. Freeze any deployment-weighted overall metric in advance.

| Task family | What to include | What a successful answer must do |
|---|---|---|
| Direct lookup | Fresh facts, ordinary phrasing, limited lexical overlap | State the correct fact with supporting evidence. |
| Entity and period interpretation | Fiscal versus calendar periods, quarter versus year-to-date, aliases, "last quarter" with an explicit as-of date | Resolve the intended entity and period or ask for clarification. |
| Tables and calculation | Row/column headers, units, footnotes, totals, ratios, percentage changes | Recover the required cells and calculate with declared units and tolerances. |
| Distributed evidence | Facts across sections or documents, comparisons, explanations supported by several passages | Cover every required fact and the requested conclusion. |
| Broad synthesis and search | Drivers, risks, and superlatives over a declared company/period universe | Cover the defined scope; avoid treating a partial search as exhaustive. |
| Conflicts and revisions | Amendments, restatements, competing versions and dates | Apply a stated version policy and cite the applicable source. |
| Genuine ambiguity and conversation | Missing company or period, follow-ups dependent on history | Clarify when needed and preserve conversational constraints. |
| Unanswerable and partially answerable | Existing company but absent fact/period, near-miss metrics, missing comparison component | Refuse unsupported claims while giving supported portions where appropriate. |

Expand refusal coverage beyond the current 15 examples. A missing entity in
the manifest or an absent exact phrase does not prove unanswerability:
review the full corpus for mentions, paraphrases, and facts that can be
derived. Pair near-miss negatives with nearby answerable questions to detect
a system that improves refusal scores simply by refusing everything.

Include context-dependent multi-turn cases as a separate suite; the current
single-query dataset schema does not represent those interactions. Do not
force genuinely ambiguous tasks into a single arbitrary expected document.

## 3. Author and review labels independently of chunking

Write questions from user tasks or whole documents without exposing current
retrieved results or chunk boundaries to the author. LLM drafting can reduce
work, but human review determines acceptance.

For each sample, record stable identity, origin, task tags, split/group keys,
corpus version, expected behavior (answer, clarify, refuse, or partial), and
reviewer decision. Answerable cases also need required facts, reference
answer, entity/period/version constraints, and supporting evidence. Arithmetic
cases need inputs, formula, units, and acceptable tolerance. Synthesis needs
an explicit completeness rubric and search universe.

Anchor evidence to source documents and stable source locations such as
sections or table cells in the frozen snapshot. Retain verbatim quotes and
valid alternatives, but allow evidence to span multiple retrieved chunks.
Do not discard an otherwise valid task merely because the current chunker
splits its evidence. Track parsing loss and chunk-boundary loss separately
from search failures. This requires extending the current span matcher,
which expects each quote to fit within one chunk.

Review that the evidence entails the answer, entity and period are correct,
all valid alternative sources are accepted, and apparent negatives are truly
unsupported. Review successes and failures with the system variant hidden.
Use a second reviewer on a stratified subset and disputed cases; retain
disagreements and their resolution. Report inter-reviewer agreement on that
subset before adjudication (Cohen's κ on accept/reject and on each label
field, with raw percent agreement and n), because κ on a heavily imbalanced
verdict can be low while the reviewers almost always agree. Today every set has
one reviewer, so no agreement figure exists. Version label fixes and rerun both sides
of affected comparisons rather than silently changing one baseline.

## 4. Separate correctness, completeness, and evidence support

| Measurement | Proposed scoring |
|---|---|
| Retrieval | Required-fact evidence recall, all-required-evidence success, ranking quality, and correct entity/period/version provenance. Retain stage-1 versus final-context comparisons. |
| Answer correctness | Facts, numerical values, units, attribution, and calculations; deterministic checks where suitable. |
| Completeness | Every required part and conclusion, plus partial credit reported separately. |
| Evidence support | Each material claim supported by supplied evidence; cited source actually entails the claim and meets provenance constraints. |
| Citation coverage | Required factual claims have valid supporting citations. A correct reference answer alone does not satisfy this. |
| Behavior | Correct answering, clarification, refusal, or partial answer; report false refusals as well as unsupported answers. |
| Operational cost | Latency distribution, retrieval rounds, LLM calls, tokens, and index cost where relevant. |

Give an evidence judge the actual supplied passages and cited source metadata.
Keep correctness and support verdicts separate: an answer can be correct from
model memory yet unsupported by its citations. Check evidence present in the
prompt separately from evidence cited, and retain answers, passages, citations,
judge outputs, model versions, and prompts for audit.

Build the calibration fixture from both passing and failing answers across
tiers, including wrong-period figures, correct numbers with wrong units,
missing comparison parts, unsupported extra claims, and over-refusals. Human
labels are the reference. Report agreement, false-pass and false-fail rates,
their denominators, and uncertainty by category. Recalibrate after judge or
rubric changes; model size and temperature zero do not establish reliability.
Adjudicate consequential disagreements before using them for a default change.

### Context reliance: closed-book and counterfactual controls

A correct answer does not show that retrieval did any work. EDGAR 10-Ks are
public and likely in the generators' training data, and the 27b has already
added a remembered figure to a correct refusal
([parametric leakage](measured-results.md#generator-model-qwen359b-vs-qwen3827b-pre-milestone-19)).
Answer pass rates therefore need two controls that measure whether the answer
depends on the supplied context.

- **Closed-book row.** The same questions with no retrieval, through
  `ClosedBookResponder` (`closed-book / 9b`, `closed-book / 27b` in
  `run_answer_matrix.py`; built for MuSiQue, never run on EDGAR). Its pass
  rate shows which facts the generator already knows. Run it beside every
  answer row on a public corpus.
- **Counterfactual row.** Hand the generator its gold evidence with the asked
  figure replaced by a plausible wrong value. If the answer still reports the
  real value, the model answered from memory and ignored the context. Build it
  on the oracle (`rag/eval/oracle.py`), which already selects each question's
  gold chunks and sends them through the pipeline's own prompt. To keep the
  edit consistent:
  - Restrict it to samples whose span states a figure (124 of the 174
    generated questions). Leave out derived quantities a reader could use to
    recompute the original, or edit them too.
  - Change every occurrence of the figure in all supplied chunks. Pick a
    value with the same units and magnitude that appears nowhere in the
    corpus for that entity, so neither the real value nor any other true
    figure can match by accident. Record the original and substituted values
    and a seed per sample.
  - For agentic rows, apply the same substitution to search results inside
    the tool layer (any returned chunk containing the gold span), so the
    agent can't sidestep it by searching again.
- **Scoring.** Deterministic, with no judge. Classify each answer as
  *follows context* (the substituted value), *parametric override* (the
  original value), *flags the conflict*, *refuses* or *other*. Report the
  override rate both overall and among questions the closed-book row answers
  correctly. Only known facts can leak, so the conditional rate is the
  faithfulness measure, and the unconditional rate also depends on how much
  the model knows. Read the flagged-conflict cases by hand: noticing the
  inconsistency is acceptable behavior, not a failure.
- **Reading it.** A high override rate means answer pass overstates what
  retrieval contributes, and answer-side comparisons between retrieval
  variants are diluted by questions the model can answer without them.
  Expect it to grow with model size. Report it per generator and mode, and
  treat it as a property of the generator and prompt, never a retrieval
  result. It complements the evidence judge: that checks whether claims are
  supported, while this checks whether the model would use contrary
  evidence.

## 5. Compare variants without overstating confidence

- Predeclare the main metric, smallest worthwhile effect, task weights if
  any, acceptable regressions, and latency/cost budget before the final run.
- Use identical corpus, labels, index/config fingerprints, and question IDs
  for paired comparisons. An oracle using gold filters is an upper bound;
  score the real query interpretation path separately.
- Report paired deltas and win/loss counts. Preserve the existing exact
  McNemar check for small binary discordances, but add uncertainty accounting
  for related questions, for example a paired bootstrap over the predeclared
  company or filing-family groups. A small number of groups also limits
  confidence; adding paraphrases does not solve that problem.
- Treat broad variant sweeps as exploratory. Confirm a selected candidate on
  fresh data; predeclare how multiple confirmatory comparisons are handled.
  A nominal p-value after extensive selection is not independent validation.
- Repeat generation for the final answer comparison (initially three runs)
  and report run spread separately from uncertainty across questions.
  Record sampling settings; temperature zero alone is not a reproducibility
  guarantee. Add runs if variability prevents the intended decision.
- Refresh the multi-hop pipeline baseline with current headers and labels
  before comparing agentic retrieval. Do not mix historical baselines into a
  current claim or pool fundamentally different matching modes and tiers.

## 6. Broaden the corpus deliberately

Keep the existing EDGAR snapshot as a stable control. Add new companies,
periods, and same-sector distractors to measure transfer and scale separately.
Then add full filings and table-heavy documents, including cases the current
extractor skips, with extraction quality explicitly reviewed.

For enterprise use, create another corpus representing the intended mix of
PDFs, policies, technical docs, tables, and revisions. Use consistent questions
when measuring added distractors; create new versioned tasks when newly added
evidence changes answerability. Report corpus composition, parsing failures,
duplicates, chunk counts, and index cost. Corpus size alone is not a rigor
metric, and a financial-document result does not establish enterprise quality.

## Delivery order and completion criteria

1. **Protocol and calibration:** freeze the workload, task rubric, grouping
   policy, metrics, and human calibration fixture. Extend saved answer traces
   and scoring so evidence support can be inspected.
2. **Fresh labels and holdout:** collect and review the initial question
   budget, audit split leakage and alternative evidence, and version the
   corpus and labels. Record actual counts and gaps for every task family.
3. **Baselines:** run the shipped pipeline and a pinned vanilla comparator
   on the new suite; refresh multi-hop. Include repeats, costs, human checks
   of passes and failures, and calibrated judge results. Add closed-book
   and counterfactual rows for each generator
   ([context reliance](#context-reliance-closed-book-and-counterfactual-controls))
   so answer pass can be read against what the model already knows.
4. **Targeted comparisons:** choose a candidate on development data for an
   observed failure class, then run the frozen comparison. Revisit CRAG or
   expansion where the tasks plausibly need them; leave defaults unchanged
   until evidence supports the decision.
5. **Corpus transfer and scale:** add the separate workloads above before
   extending conclusions beyond the measured financial tasks.

The first evaluation upgrade is complete when reviewed fresh tasks cover the
declared families, holdout grouping is audited, evidence-aware scoring and
judge calibration are reported, and reproducible current baselines exist.
A default-change decision additionally requires a practically useful effect
with the predeclared uncertainty and regression criteria satisfied. If there
is insufficient evidence, report the limit and retain the current default.

Track implementation under [Milestone 27](backlog.md#milestone-27--eval-coverage-and-judge-reliability),
with parser and table work coordinated through the
[chunking and indexing plan](chunking-indexing-plan.md). Record completed
measurements in [measured results](measured-results.md), including negative
results and the precise scope of each conclusion.
