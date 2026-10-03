"""Pipeline answer eval with only the generator swapped, judge held fixed.

run_answer_matrix.py judges with the generating model, so a model swap there
changes the judge too. Here the judge is a third model from another family.

FROZEN RECORD, not maintained tooling. This is the script behind the
"Generator model: qwen3.5:9b vs qwen3.8:27b" section of
docs/measured-results.md, committed so those numbers can be audited and rerun.
It uses private names from rag.eval / rag.generation and will break as
Milestone 19 changes those interfaces; Phase 0 of docs/milestone-19-plan.md
replaces it with a proper harness. Do not extend it -- build that instead.

Run from the repo root with an `edgar` index built and the models pulled in
Ollama. Results go to data/eval/results/probe_2026-09_pipeline_generator_swap.json.

Usage: python pipeline_generator_swap.py [MODEL ...]   (ONLY=answerable|refusals
to run one set). The 27b answerable run lost its per-sample detail to a crash
before per-phase checkpointing was added; only its aggregate survives, in
measured-results.md.
"""
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from rag.config.settings import load_config  # noqa: E402
from rag.eval.answer_eval import run_answer_eval, subsample  # noqa: E402
from rag.eval.dataset import EvalDataset  # noqa: E402
from rag.chat import build_chat_service  # noqa: E402
from rag.llm.factory import get_llm_client  # noqa: E402
from rag.llm.ollama_llm import OllamaLLMClient  # noqa: E402
from rag.logging_config import configure_logging  # noqa: E402

# Another Ollama client on this machine can force model reloads that drop an
# in-flight request; retry rather than lose a 30-minute run.
_orig = OllamaLLMClient.generate
def _retrying(self, prompt, *, system=None):
    for attempt in range(4):
        try:
            return _orig(self, prompt, system=system)
        except RuntimeError:
            if attempt == 3:
                raise
            time.sleep(10)
OllamaLLMClient.generate = _retrying

JUDGE = "gemma4:31b-mlx"
ONLY = os.environ.get("ONLY")  # "answerable" | "refusals" | None
GENERATORS = sys.argv[1:] or ["qwen3.5:9b-mlx", "qwen3.8:27b-mlx"]
OUT = REPO / "data/eval/results/probe_2026-09_pipeline_generator_swap.json"

configure_logging()
base = load_config()
judge_cfg = base.llm.model_copy(update={"model": JUDGE, "temperature": 0.0, "think": False})
judge = get_llm_client(judge_cfg)
answerable = subsample(EvalDataset.load(REPO / "data/eval/edgar_eval_set.json"), 40)
refusals = EvalDataset.load(REPO / "data/eval/edgar_refusal_set.json")

results = json.loads(OUT.read_text()) if OUT.exists() else {}
for model in GENERATORS:
    cfg = base.model_copy(deep=True)
    cfg.llm.model = model
    svc = build_chat_service(cfg, corpora=["edgar"])
    rec = results.get(model, {})
    for name, ds in [("answerable", answerable), ("refusals", refusals)]:
        if ONLY and name != ONLY:
            continue
        t = time.monotonic()
        rep = run_answer_eval(ds, svc, judge)
        rec[name] = {
            "passed": rep.num_passed, "n": rep.num_evaluated,
            "unparseable": rep.num_unparseable, "pass_rate": round(rep.pass_rate, 3),
            "elapsed_s": round(time.monotonic() - t),
            "per_sample": [{"id": r.sample_id, "passed": r.passed, "answer": r.actual_answer[:400],
                            "judge": r.judge_output[:200]} for r in rep.sample_results],
        }
        results[model] = rec
        OUT.write_text(json.dumps(results, indent=1))
        print(model, name, rec[name]["passed"], "/", rec[name]["n"], rec[name]["elapsed_s"], "s", flush=True)
    results[model] = rec
    OUT.write_text(json.dumps(results, indent=1))
