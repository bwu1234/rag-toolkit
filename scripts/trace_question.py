"""Replay one eval question and write its whole turn to a Markdown file.

    python scripts/trace_question.py --id ad-airline-fuel --corpus edgar_md
    python scripts/trace_question.py --id ad-airline-fuel --corpus edgar_md \\
        --variant "agentic react / 27b, think=low"
    python scripts/trace_question.py --id ad-airline-fuel --raw

Finds the question by id in `data/eval/*.json` (or `--eval-set`), answers it
through the configured responder -- or as one row of `run_answer_matrix.py`
runs it, with `--variant` -- and writes the trace: summary, answer next to the
expected one, timeline, the agent's tool calls, retrieval scores, and every LLM
call verbatim, system prompt and tool results included
(`rag.observability.transcript`). A `.json` `--out` keeps the raw record instead.

`--raw` runs the agent's model in raw mode (`llm.raw`): the trace then also
shows each call's rendered prompt (chat template and special tokens) -- whole
the first time, then only what each call added -- and the model's unparsed
output, thinking and tool-call markup included. It needs a qwen3.8 model as
`agent.llm` (the 27b), and adds a one-token `/api/chat` check per step.

Nothing is judged and nothing goes to the turn log, as with the eval runners:
this is for reading one turn, not scoring it. It runs the real models, so an
agent variant on the 27b takes a minute or two per question.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag.config.settings import RagConfig, load_config  # noqa: E402
from rag.chat import build_chat_service  # noqa: E402
from rag.logging_config import configure_logging  # noqa: E402
from rag.observability.records import TurnRecord  # noqa: E402
from rag.observability.transcript import write_trace  # noqa: E402

EVAL_DIR = Path("data/eval")
DEFAULT_OUT_DIR = Path("data/traces")
EDGAR_EVAL_CORPUS = "edgar_md"


def find_sample(sample_id: str, eval_sets: list[Path]) -> tuple[dict[str, Any], Path]:
    """The raw sample with `sample_id` and the file it's in; raises if none or several match."""

    found: list[tuple[dict[str, Any], Path]] = []
    for path in eval_sets:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(raw, list):
            found += [(item, path) for item in raw if isinstance(item, dict) and item.get("id") == sample_id]
    if not found:
        raise SystemExit(f"error: no sample with id {sample_id!r} in the {len(eval_sets)} eval set(s) searched")
    if len(found) > 1:
        raise SystemExit(
            f"error: {sample_id!r} is in several sets ({', '.join(str(p) for _, p in found)}); pick one with --eval-set"
        )
    return found[0]


def variant_overrides(family: str, name: str) -> dict[str, Any]:
    """The config overrides of one `run_answer_matrix.py` row, so a replay matches what the matrix ran."""

    import run_answer_matrix as matrix

    variants = matrix.FAMILIES[family]
    variant = next((v for v in variants if v.name == name), None)
    if variant is None:
        names = "\n  ".join(v.name for v in variants)
        raise SystemExit(f"error: no variant {name!r} in --family {family}. Its rows:\n  {names}")
    if variant.oracle or variant.closed_book:
        raise SystemExit(f"error: {name!r} is an oracle or closed-book row, which doesn't run through ask()")
    return variant.overrides


def with_raw_agent_llm(config: RagConfig) -> RagConfig:
    """`config` with the agent's model in raw mode; raises unless the turn will run on that model."""

    if config.chat.mode != "agentic":
        raise SystemExit("error: --raw covers the agent's model, and this config runs chat.mode: pipeline")
    agent_llm = config.agent.llm or config.llm
    if agent_llm.provider != "ollama":
        raise SystemExit(f"error: --raw needs an Ollama model; the agent's model is on {agent_llm.provider!r}")
    raw_llm = agent_llm.model_copy(update={"raw": True})
    return config.model_copy(update={"agent": config.agent.model_copy(update={"llm": raw_llm})})


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--id", required=True, help="The eval sample id, e.g. ad-airline-fuel")
    parser.add_argument("--eval-set", type=Path, action="append", default=None,
                        help=f"Look only in this set (repeatable). Default: every {EVAL_DIR}/*.json")
    parser.add_argument("--config", default=None, help="Config YAML (default: rag/config/config.yaml)")
    parser.add_argument("--corpus", action="append", default=None, metavar="NAME",
                        help="Corpus to answer from, overriding corpora.active (repeatable). "
                        f"Default for an edgar_* set: {EDGAR_EVAL_CORPUS}")
    parser.add_argument("--variant", default=None,
                        help="Run as this row of scripts/run_answer_matrix.py (e.g. 'agentic react / 27b')")
    parser.add_argument("--family", default="m19", help="The matrix family --variant is from (default: m19)")
    parser.add_argument("--raw", action="store_true",
                        help="Render the agent model's prompts client-side and show the exact prompt and output "
                        "text of every call (qwen3.8 on Ollama only)")
    parser.add_argument("--out", type=Path, default=None,
                        help=f"Where to write it (.md, or .json for the raw record). Default: {DEFAULT_OUT_DIR}/<id>.md")
    args = parser.parse_args()
    configure_logging()

    sample, source = find_sample(args.id, args.eval_set or sorted(EVAL_DIR.glob("*.json")))
    corpora = args.corpus
    if corpora is None and source.name.startswith("edgar_"):
        # `corpora.active` is the small baseline corpus, which can't answer an
        # EDGAR question; edgar_md is what the EDGAR evals run on.
        corpora = [EDGAR_EVAL_CORPUS]
        print(f"{source.name} is an EDGAR set: answering from --corpus {EDGAR_EVAL_CORPUS}")
    config = load_config(args.config)
    if args.variant is not None:
        import run_answer_matrix as matrix

        config = matrix.row_config(config, variant_overrides(args.family, args.variant))
    if args.raw:
        config = with_raw_agent_llm(config)
    responder = build_chat_service(config, corpora=corpora)

    out = args.out or DEFAULT_OUT_DIR / (
        args.id + (f"__{_slug(args.variant)}" if args.variant else "") + ("__raw" if args.raw else "") + ".md"
    )
    context = [
        ("Sample", f"{args.id} ({source})"),
        ("Config", str(args.config or "rag/config/config.yaml")),
        ("Corpus", ", ".join(config.corpus_selection(corpora).names)),
        ("Mode", config.chat.mode),
    ]
    if args.variant:
        context.append(("Variant", f"{args.variant} (--family {args.family})"))
    if sample.get("expected_doc_ids"):
        context.append(("Expected documents", ", ".join(str(d) for d in sample["expected_doc_ids"])))

    print(f"Answering {args.id}: {sample['query']}")
    records: list[TurnRecord] = []
    try:
        responder.ask(sample["query"], on_record=records.append)
    finally:
        if records:
            write_trace(
                records[0], out, title=f"Trace: {args.id}", context=context,
                expected_answer=sample.get("expected_answer"),
            )
            print(f"Trace written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
