#!/usr/bin/env python
"""Convert fetched MuSiQue-Ans into a pooled corpus and an eval set.

MuSiQue follow-on, phase B (docs/public-benchmarks-plan.md). Reads the splits
``scripts/fetch_musique.py`` unpacked and writes, for one target:

* ``data/corpora/musique-ans-<target>/documents/corpus.jsonl`` -- every
  paragraph of the target's questions, pooled and deduped, in BEIR's corpus
  format (``_id``, ``title``, ``text``) so `BeirCorpusLoader` indexes it
  unchanged. The id is ``musique-<sha256(title + "\\n" + text)[:16]>``
  (`inventory_musique.paragraph_id`), a pure function of content.
* ``data/eval/musique_ans_<target>.json`` -- one sample per question, in the
  multi-hop set's shape:

  - ``expected_doc_ids``: the supporting paragraphs' ids;
  - ``expected_spans``: their full indexed text (title, space, body; see
    `beir_passage_text`), so under the one-chunk-per-paragraph config evidence
    recall reduces to "was this paragraph retrieved";
  - ``expected_answer`` plus ``answer_aliases``;
  - ``hops`` (2-4), ``kind`` (``2hop``/``3hop``/``4hop``, the stratum) and
    ``hop_type`` (MuSiQue's composition shape, e.g. ``3hop1``);
  - ``parts``: one per decomposition step, with that step's sub-question,
    intermediate answer and supporting paragraph. For per-hop evidence recall
    only: MuSiQue grades the final answer, never the intermediate ones.

Targets:

* ``dev`` -- the confirmatory set: the Ans dev questions over their pooled
  21,100 paragraphs, 2,412 of 2,417 after repeated question texts are dropped
  (`drop_repeated_questions`; the corpus keeps their paragraphs).
* ``train-tune`` -- for tuning and sizing. Eligible questions are Ans train
  questions none of whose paragraphs is in the dev pool. 300 of them, at
  dev's hop mix (train is 72% 2-hop, dev 52%), are the eval set. Their
  paragraphs plus those of further eligible questions, up to dev's 2,417,
  form the corpus, so retrieval faces dev's distractor density. The seed is
  fixed.

The manifest's ``dev_set`` and ``tuning_slice`` record each set's counts and a
hash of its ids, and the run fails if they ever differ.

Malformed data fails the run rather than being dropped: a decomposition step
without a supporting paragraph, a supporting paragraph missing from the pooled
corpus, or a step count that doesn't match the question's hop count. Outputs
are gitignored and regenerated from the pinned archive and this script.

    python scripts/musique_to_eval_set.py dev
    python scripts/musique_to_eval_set.py train-tune
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from inventory_musique import paragraph_id  # noqa: E402

from rag.eval.dataset import EvalDataset, EvalSample, ExpectedSpan  # noqa: E402
from rag.ingestion.loaders import beir_passage_text  # noqa: E402

SOURCE = REPO / "data" / "corpora" / "musique-ans" / "source"
MANIFEST = REPO / "data" / "corpora" / "musique-ans" / "manifest.json"
SEED = 19
#: Questions scored on the tuning slice: enough to estimate per-question
#: variance for sizing the dev sample, small enough for an agent row in hours.
TUNE_EVAL_QUESTIONS = 300


class ConversionError(ValueError):
    """The source data breaks an assumption the eval set depends on."""


def _load(name: str) -> list[dict[str, Any]]:
    with (SOURCE / f"musique_ans_v1.0_{name}.jsonl").open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def hops_of(row: dict[str, Any]) -> int:
    return int(row["id"][0])


def stratified_sample(
    rows: list[dict[str, Any]], size: int, rng: random.Random, shares: dict[int, float]
) -> list[dict[str, Any]]:
    """`size` rows split across hop counts in proportion to `shares` (largest remainder), in source order."""
    strata: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        strata[hops_of(row)].append(row)
    exact = {h: size * shares[h] for h in strata}
    quota = {h: int(share) for h, share in exact.items()}
    for h in sorted(exact, key=lambda h: exact[h] - quota[h], reverse=True)[: size - sum(quota.values())]:
        quota[h] += 1
    if short := {h: (quota[h], len(strata[h])) for h in strata if quota[h] > len(strata[h])}:
        raise ConversionError(f"Not enough questions per hop count for the sample (wanted, available): {short}")
    chosen = {id(r) for h, group in sorted(strata.items()) for r in rng.sample(group, quota[h])}
    return [r for r in rows if id(r) in chosen]


def to_sample(row: dict[str, Any]) -> EvalSample:
    by_idx = {p["idx"]: p for p in row["paragraphs"]}
    hops = hops_of(row)
    steps = row["question_decomposition"]
    if len(steps) != hops:
        raise ConversionError(f"{row['id']}: {len(steps)} decomposition steps for a {hops}-hop question")
    parts: list[dict[str, Any]] = []
    for step in steps:
        paragraph = by_idx.get(step.get("paragraph_support_idx"))
        if paragraph is None or not paragraph["is_supporting"]:
            raise ConversionError(f"{row['id']}: step {step['id']} has no supporting paragraph")
        parts.append({
            "label": step["question"],
            "answer": step["answer"],
            "spans": [beir_passage_text(paragraph["title"], paragraph["paragraph_text"])],
            "source_id": paragraph_id(paragraph),
        })
    supporting = [p for p in row["paragraphs"] if p["is_supporting"]]
    if {paragraph_id(p) for p in supporting} != {part["source_id"] for part in parts}:
        raise ConversionError(f"{row['id']}: supporting paragraphs and decomposition steps disagree")
    return EvalSample(
        id=row["id"],
        query=row["question"],
        expected_doc_ids=sorted({part["source_id"] for part in parts}),
        expected_spans=[ExpectedSpan(text=t) for t in dict.fromkeys(part["spans"][0] for part in parts)],
        expected_answer=row["answer"],
        extra={
            "tier": "musique",
            "kind": f"{hops}hop",
            "hops": hops,
            "hop_type": row["id"].split("__")[0],
            "answer_aliases": list(row["answer_aliases"]),
            "parts": parts,
        },
    )


def corpus_rows(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Every paragraph of `rows`, deduped by content id, in first-seen order."""
    seen: dict[str, dict[str, str]] = {}
    for row in rows:
        for p in row["paragraphs"]:
            seen.setdefault(paragraph_id(p), {"_id": paragraph_id(p), "title": p["title"], "text": p["paragraph_text"]})
    return list(seen.values())


def select(target: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(questions whose paragraphs form the corpus, questions scored)."""
    dev = _load("dev")
    if target == "dev":
        return dev, dev
    dev_pool = {paragraph_id(p) for r in dev for p in r["paragraphs"]}
    eligible = [r for r in _load("train") if not any(paragraph_id(p) in dev_pool for p in r["paragraphs"])]
    # The scored questions follow dev's hop mix, not train's: train is 72% 2-hop
    # against dev's 52%, and the 3- and 4-hop questions are the ones an agent's
    # extra searches are for. Only 244 eligible questions are 4-hop, too few for
    # a dev-sized pool at dev's mix, so the rest of the pool (distractors, whose
    # mix matters less than their number) is drawn at the eligible questions' own.
    shares = {h: n / len(dev) for h, n in Counter(hops_of(r) for r in dev).items()}
    rng = random.Random(SEED)
    scored = stratified_sample(eligible, TUNE_EVAL_QUESTIONS, rng, shares)
    taken = {r["id"] for r in scored}
    rest = [r for r in eligible if r["id"] not in taken]
    natural = {h: n / len(rest) for h, n in Counter(hops_of(r) for r in rest).items()}
    filler = stratified_sample(rest, len(dev) - len(scored), rng, natural)
    return scored + filler, scored


def ids_digest(samples: list[EvalSample]) -> str:
    return hashlib.sha256("\n".join(sorted(s.id for s in samples)).encode()).hexdigest()


def drop_repeated_questions(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Keep the first question with each text; return the kept rows and the dropped ids.

    The system sees only the question text, so a repeat asks the same thing
    twice and counts twice. Dev has five, all with the same answer; one pair
    cites different gold paragraphs, which no system can satisfy twice and the
    oracle can't serve (it looks gold up by question).
    """
    seen: set[str] = set()
    kept: list[dict[str, Any]] = []
    dropped: list[str] = []
    for row in rows:
        if row["question"] in seen:
            dropped.append(row["id"])
            continue
        seen.add(row["question"])
        kept.append(row)
    return kept, dropped


def convert(target: str) -> dict[str, Any]:
    pool, scored = select(target)
    scored, repeated = drop_repeated_questions(scored)
    corpus = corpus_rows(pool)
    corpus_ids = {row["_id"] for row in corpus}
    samples = [to_sample(r) for r in scored]
    for s in samples:
        if missing := [d for d in s.expected_doc_ids if d not in corpus_ids]:
            raise ConversionError(f"{s.id}: supporting paragraph(s) {missing} not in the pooled corpus")

    summary = {
        "corpus_questions": len(pool),
        "corpus_paragraphs": len(corpus),
        "eval_questions": len(samples),
        "eval_by_kind": {k: sum(s.extra["kind"] == k for s in samples) for k in ("2hop", "3hop", "4hop")},
        "eval_ids_sha256": ids_digest(samples),
        "dropped_repeated_questions": repeated,
    }
    # Both sets are pinned: a different question list must fail, not drift.
    pinned = json.loads(MANIFEST.read_text(encoding="utf-8")).get("tuning_slice" if target == "train-tune" else "dev_set")
    if pinned is not None and {k: pinned.get(k) for k in summary} != summary:
        raise ConversionError(f"The {target} set no longer matches the manifest: {summary} vs {pinned}")

    documents = REPO / "data" / "corpora" / f"musique-ans-{target}" / "documents"
    documents.mkdir(parents=True, exist_ok=True)
    with (documents / "corpus.jsonl").open("w", encoding="utf-8") as f:
        for row in corpus:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    EvalDataset(samples=samples).save(REPO / "data" / "eval" / f"musique_ans_{target.replace('-', '_')}.json")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("target", choices=("dev", "train-tune"))
    args = parser.parse_args()
    try:
        print(json.dumps(convert(args.target), indent=2))
    except ConversionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
