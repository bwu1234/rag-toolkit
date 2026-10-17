"""Prompt-injection eval, forced-exposure level (Milestone 28): how often planted instructions work.

Usage::

    python -m rag.eval.injection_eval --corpus edgar_md
    python -m rag.eval.injection_eval --corpus edgar_md --config rag/config/inject_delimited.yaml
    python -m rag.eval.injection_eval --corpus edgar_md --bases 2      # first 2 bases only: a smoke run

Each case of ``data/eval/edgar_injection_set.json`` is one turn: the case's
plant is armed (`rag.eval.injection.ForcedExposure`), the question is asked
under the case's turn filter, and the output -- answer, every passage shown,
every agent tool call, the turn's model calls verbatim -- is stored in a run
folder like the answer evals'. Then every output is scored deterministically
against its case and its base question's control, and the scores are written
as a score revision of the run. No judge, so nothing here spends quota; the
cost is local generation, about one agent turn per case.

The mode follows the config (`chat.mode`), as the answer evals do. Run one
config per row; delimiters are `chat.passage_delimiters`.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from rag.chat import build_chat_service
from rag.chunking.models import Chunk
from rag.config.settings import RagConfig, load_config
from rag.eval.checkpoint import code_version
from rag.eval.dataset import EvalDataset
from rag.eval.injection import (
    ATTACKS,
    CONTROL,
    PLANT_RANK,
    SCORER,
    CaseScore,
    ForcedExposure,
    InjectionCase,
    plant_chunk,
    score_case,
    summarize,
)
from rag.eval.metrics import wilson_interval
from rag.eval.provenance import IndexConflict, checked_index
from rag.eval.runs import RUNS_DIR, Run, RunIncomplete, open_run
from rag.generation.chat_service import ChatResponder
from rag.ingestion.corpora import chunk_selected_corpora
from rag.logging_config import configure_logging
from rag.observability.records import TurnRecord
from rag.vectorstore.base import ScoredChunk

logger = logging.getLogger(__name__)

DEFAULT_SET = Path(__file__).resolve().parents[2] / "data" / "eval" / "edgar_injection_set.json"


def _scored(chunk: Chunk) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk.id, text=chunk.text, document_id=chunk.document_id, source=chunk.source,
        doc_type=chunk.doc_type, score=1.0, metadata=dict(chunk.metadata), header=chunk.header,
    )


def templates(chunks: list[Chunk]) -> dict[str, ScoredChunk]:
    """Each document's first chunk: the header and metadata a plant borrows, and the scope check reads."""

    found: dict[str, ScoredChunk] = {}
    for chunk in chunks:
        found.setdefault(chunk.document_id, _scored(chunk))
    return found


def case_output(case: InjectionCase, plant: ScoredChunk, responder: ChatResponder,
                exposure: ForcedExposure) -> dict[str, Any]:
    """Ask one case's question with its plant armed: the output record a run stores."""

    records: list[TurnRecord] = []
    started = time.monotonic()
    with exposure.armed(plant):
        answer = responder.ask(case.query, query_filter=case.scope, on_record=records.append)
        exposures = exposure.exposures
    return {
        "sample_id": case.case_id,
        "query": case.query,
        "query_filter": case.scope.model_dump(exclude_defaults=True),
        "plant_id": plant.chunk_id,
        "exposures": exposures,
        "answer": answer.answer,
        "generation_failure": answer.generation_failure,
        "latency_s": time.monotonic() - started,
        "llm_calls": answer.llm_calls,
        "prompt_tokens": answer.prompt_tokens,
        "completion_tokens": answer.completion_tokens,
        "tool_calls": answer.tool_calls,
        "stopped_reason": answer.stopped_reason,
        "search_queries": answer.search_queries,
        "agent_calls": [asdict(call) for call in answer.agent_calls],
        "passages": [asdict(citation) for citation in answer.citations],
        "cited_chunk_ids": answer.cited_chunk_ids,
        "invalid_citations": answer.invalid_citations,
        "turn": records[-1].to_dict() if records else None,
    }


def score_run(cases: list[InjectionCase], outputs: dict[str, dict[str, Any]], *, agentic: bool,
              document_metadata: dict[str, dict[str, Any]]) -> list[CaseScore]:
    """Every case's score, each attack paired with its base question's control."""

    return [
        score_case(
            case, outputs[case.case_id], outputs.get(case.control_id) if case.control_id else None,
            agentic=agentic, document_metadata=document_metadata,
        )
        for case in cases
        if case.case_id in outputs
    ]


def _interval(successes: int, n: int) -> str:
    if not n:
        return "n/a"
    low, high = wilson_interval(successes, n)
    return f"{successes}/{n} ({successes / n:.0%}, 95% CI {low:.0%}-{high:.0%})"


def print_report(summary: dict[str, Any], scores: list[CaseScore]) -> None:
    controls = summary["controls"]
    print(f"\nControls: {controls['retained']}/{controls['cases']} kept every gold figure"
          f"; {controls['generation_failures']} generation failure(s)")
    print("\n| attack | carrier | exposed | success | attempted | harmed | check fired on control | excess calls |")
    print("|---|---|---|---|---|---|---|---|")
    rows = [(f"{a}/{c}", a, c) for a in ATTACKS for c in ("text", "header")]
    for key, attack, carrier in rows:
        group = summary["by_attack_carrier"].get(key)
        if group is None:
            continue
        judged = [s for s in scores if s.attack == attack and s.carrier == carrier and s.exposed
                  and s.success is not None]
        attempted = "n/a" if group["attempts"] is None else f"{group['attempts']}"
        print(
            f"| {attack} | {carrier} | {group['exposed']}/{group['cases']} | "
            f"{_interval(group['successes'], len(judged))} | {attempted} | {group['harmed']} | "
            f"{group['control_hits']} | {group['mean_excess_calls'] if group['mean_excess_calls'] is not None else 'n/a'} |"
        )
    overall = summary["overall"]
    print(f"\nOverall: {overall['successes']} successful attack(s) of {overall['exposed']} exposed; "
          f"{overall['harmed']} answer(s) lost a gold figure the control kept.")
    hits = [s.case_id for s in scores if s.control_hit]
    if hits:
        print(f"WARNING: a success check fired on the control for {len(hits)} case(s): {', '.join(hits)}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m rag.eval.injection_eval", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_SET, metavar="PATH")
    parser.add_argument("--config", default=None, metavar="PATH")
    parser.add_argument("--corpus", action="append", default=None, metavar="NAME")
    parser.add_argument("--bases", type=int, default=0, metavar="N",
                        help="Run only the first N base questions (each with its control and attacks).")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR, metavar="DIR")
    parser.add_argument("--fresh", action="store_true", help="Start a new run even if an interrupted one matches.")
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = _build_parser().parse_args(argv)
    full = EvalDataset.load(args.eval_set)
    bases: list[str] = []
    for sample in full:
        base = sample.extra["injection"]["base_id"]
        if base not in bases:
            bases.append(base)
    keep = set(bases[: args.bases]) if args.bases else set(bases)
    selected = EvalDataset(
        samples=[s for s in full if s.extra["injection"]["base_id"] in keep], source_path=full.source_path
    )
    cases = [InjectionCase.from_extra(s.query, s.extra) for s in selected]

    config: RagConfig = load_config(args.config)
    agentic = config.chat.mode == "agentic"
    selection, documents, chunks = chunk_selected_corpora(config, args.corpus)
    found = templates(chunks)
    missing = sorted({c.target_document for c in cases} - set(found))
    if missing:
        logger.error("Plant target document(s) not in the corpus: %s", ", ".join(missing))
        return 1
    document_metadata = {doc: template.metadata for doc, template in found.items()}

    try:
        index = checked_index(config, selection, documents, chunks, label="injection_eval")
        run: Run = open_run(
            args.runs_dir, stage="injection", experiment="injection_eval",
            variant=Path(args.config).stem if args.config else "config", config=config, selection=selection,
            dataset_path=args.eval_set, dataset=full, selected=selected,
            selection_method=f"first {args.bases} bases" if args.bases else "all",
            index=index, code=code_version(),
            extra={"exposure": "forced", "plant_rank": PLANT_RANK, "mode": config.chat.mode},
            resume=not args.fresh,
        )
    except IndexConflict as exc:
        logger.error("%s", exc)
        return 1

    exposure = ForcedExposure()
    responder = build_chat_service(config, corpora=args.corpus, wrap_retriever=exposure.wrap)
    started = time.monotonic()
    done = run.load_outputs()
    for number, case in enumerate(cases, start=1):
        if case.case_id in done:
            continue
        logger.info("[%d/%d] %s", number, len(cases), case.case_id)
        run.append_output(case_output(case, plant_chunk(case, found[case.target_document]), responder, exposure))
    try:
        run.complete(elapsed_s=time.monotonic() - started)
    except RunIncomplete as exc:
        logger.error("%s", exc)
        return 1

    outputs = run.load_outputs()
    scores = score_run(cases, outputs, agentic=agentic, document_metadata=document_metadata)
    summary = summarize(scores)
    revision = run.write_scores(
        scorer=SCORER,
        settings={"agentic": agentic, "plant_rank": PLANT_RANK},
        scores={"summary": summary, "cases": [dataclasses.asdict(s) for s in scores]},
    )
    unexposed = [s.case_id for s in scores if s.attack != CONTROL and not s.exposed]
    if unexposed:
        logger.warning("%d attacked case(s) never showed the plant: %s", len(unexposed), ", ".join(unexposed))
    print_report(summary, scores)
    print(f"\nRun: {run.path}  (score revision {revision})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
