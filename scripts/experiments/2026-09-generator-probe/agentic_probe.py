"""Minimal agentic loop over the shipped Retriever, per model.

Throwaway prototype of Milestone 19's shape -- one tool (`rag_search`) wrapping
the existing hybrid+bge-v2-m3 Retriever, passages numbered globally across
calls, a hard cap on tool calls. Measures what the plan depends on: does the
model call the tool sensibly, decompose multi-entity questions, and stay
grounded -- and what it costs.

FROZEN RECORD, not maintained tooling. This is the script behind the
"Generator model: qwen3.5:9b vs qwen3.8:27b" section of
docs/measured-results.md, committed so those numbers can be audited and rerun.
It uses private names from rag.eval / rag.generation and will break as
Milestone 19 changes those interfaces; Phase 0 of docs/milestone-19-plan.md
replaces it with a proper harness. Do not extend it -- build that instead.

Run from the repo root with an `edgar` index built and the models pulled in
Ollama. Results go to data/eval/results/probe_2026-09_agentic.json.

Usage: python agentic_probe.py [MODEL ...]
"""
import json
import sys
import time
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from rag.config.settings import load_config  # noqa: E402
from rag.eval.answer_eval import (  # noqa: E402
    _JUDGE_SYSTEM_PROMPT,
    _REFUSAL_JUDGE_SYSTEM_PROMPT,
    _judge_prompt,
    _parse_verdict,
    _refusal_judge_prompt,
)
from rag.eval.dataset import EvalDataset  # noqa: E402
from rag.generation.builder import build_chat_service  # noqa: E402
from rag.generation.ollama_llm import OllamaLLMClient  # noqa: E402
from rag.generation.prompts import _citation_label  # noqa: E402
from rag.logging_config import configure_logging  # noqa: E402
from rag.retrieval.builder import build_retriever  # noqa: E402

MODELS = sys.argv[1:] or ["qwen3.5:9b-mlx", "qwen3.8:27b-mlx"]
JUDGE = "gemma4:31b-mlx"
MAX_TOOL_CALLS = 8
OUT = REPO / "data/eval/results/probe_2026-09_agentic.json"

SYSTEM = (
    "You answer questions about a corpus of SEC 10-K/10-Q MD&A sections from "
    "large US public companies (fiscal 2024-2026). You have one tool, rag_search, "
    "which returns numbered passages. Search before answering any question about "
    "the filings. A question about several companies or periods needs a separate "
    "search for each; keep the company name and period in every query. You may "
    "search again based on what earlier results showed. Answer only from the "
    "passages, citing them inline as [n]. If the passages don't support an "
    "answer, or only cover part of it, say so plainly instead of guessing."
)
TOOL = {"type": "function", "function": {
    "name": "rag_search",
    "description": "Hybrid keyword+semantic search over the filings, reranked. Returns the 5 best passages, each labeled with its source file (TICKER_FORM_PERIOD-END.md).",
    "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "Standalone search query; include company and period."}}, "required": ["query"]},
}}

MULTI = [
    ("mh-airlines-rev", "Compare Delta Air Lines' and United Airlines' total operating revenue for fiscal year 2025.", ["delta", "united"]),
    ("mh-lease", "Which reported larger fixed lease payment obligations in its most recent 10-K, Apple or Microsoft?", ["apple", "microsoft"]),
    ("mh-retail-comps", "How did Walmart's and Target's comparable sales change in their most recent reported quarter?", ["walmart", "target"]),
    ("mh-pharma-ira", "What did Merck, Pfizer, and Johnson & Johnson each say about Medicare drug price negotiation under the Inflation Reduction Act?", ["merck", "pfizer", "johnson"]),
    ("mh-luv-fuel", "How did Southwest Airlines' average fuel cost per gallon change between fiscal 2024 and fiscal 2025?", ["southwest"]),
    ("mh-superlative", "Which of the companies in this corpus had the highest operating margin last quarter?", []),
]
MULTI_JUDGE = (
    "You grade an answer to a question that spans several companies or periods, "
    "given the exact passages the system retrieved. Output exactly one word first: "
    "PASS if (a) every specific figure or claim in the answer is supported by the "
    "passages, and (b) the answer addresses every company/period asked about, or "
    "explicitly states which parts the passages did not cover. FAIL if it asserts "
    "anything unsupported (including a ranking or comparison the passages cannot "
    "establish) or silently omits part of the question. Brief reason after the verdict."
)


def chat(client, model, messages, tools=True):
    body = {"model": model, "messages": messages, "stream": False, "think": False,
            "options": {"temperature": 0.2, "num_predict": 1024, "num_ctx": 32768}}
    if tools:
        body["tools"] = [TOOL]
    for attempt in range(4):
        try:
            r = client.post("/api/chat", json=body, timeout=600)
            r.raise_for_status()
            return r.json()
        except httpx.HTTPError:
            if attempt == 3:
                raise
            time.sleep(10)


def run_agent(client, retriever, model, question):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    passages, queries, bad_calls, tokens, prompt_tokens, llm_calls = [], [], 0, 0, 0, 0
    t0 = time.monotonic()
    for _ in range(MAX_TOOL_CALLS + 1):
        budget_left = MAX_TOOL_CALLS - len(queries)
        resp = chat(client, model, msgs, tools=budget_left > 0)
        tokens += resp.get("eval_count", 0)
        prompt_tokens += resp.get("prompt_eval_count", 0)
        llm_calls += 1
        msg = resp["message"]
        msgs.append(msg)
        calls = msg.get("tool_calls") or []
        if not calls:
            break
        for c in calls:
            fn = c.get("function", {})
            q = (fn.get("arguments") or {}).get("query")
            if fn.get("name") != "rag_search" or not isinstance(q, str) or not q.strip():
                bad_calls += 1
                msgs.append({"role": "tool", "tool_name": fn.get("name", "?"), "content": "Error: call rag_search with a non-empty 'query' string."})
                continue
            if len(queries) >= MAX_TOOL_CALLS:
                msgs.append({"role": "tool", "tool_name": "rag_search", "content": "Search budget exhausted; answer from the passages you have."})
                continue
            queries.append(q)
            chunks = retriever.retrieve(q).chunks
            start = len(passages) + 1
            passages.extend(chunks)
            body = "\n\n".join(f"Passage [{start + i}] (source: {_citation_label(ch)}):\n{ch.text}" for i, ch in enumerate(chunks))
            msgs.append({"role": "tool", "tool_name": "rag_search", "content": body or "No results."})
    return {
        "answer": msgs[-1].get("content", ""), "queries": queries, "bad_calls": bad_calls,
        "docs": [ch.document_id for ch in passages], "passages": passages,
        "gen_tokens": tokens, "prompt_tokens": prompt_tokens, "llm_calls": llm_calls, "latency_s": round(time.monotonic() - t0, 1),
    }


def judge_verdict(judge, system, prompt):
    out = judge.generate(prompt, system=system)
    return _parse_verdict(out), out.strip()[:200]


def main():
    configure_logging()
    base = load_config()
    retriever = build_retriever(base, corpora=["edgar"])
    # 120s default is too short for an 11k-token multi-hop grading prompt on a shared GPU.
    judge = OllamaLLMClient(JUDGE, base.llm.base_url, temperature=0.0, max_tokens=1024, think=False, timeout=900)
    client = httpx.Client(base_url=base.llm.base_url, trust_env=False)

    # Different stride offset from Exp A's 40 so the two don't share questions.
    full = EvalDataset.load(REPO / "data/eval/edgar_eval_set.json")
    single = [full.samples[int(i * len(full) / 10) + 3] for i in range(10)]
    refusal_ids = {"neg-entity-tesla-margin", "neg-period-aapl-2015", "neg-scope-wmt-board", "neg-cross-corpus-acs-auth"}
    refusals = [s for s in EvalDataset.load(REPO / "data/eval/edgar_refusal_set.json").samples if s.id in refusal_ids]

    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    for model in MODELS:
        rows = []
        for s in single:
            r = run_agent(client, retriever, model, s.query)
            r["passed"], r["judge"] = judge_verdict(judge, _JUDGE_SYSTEM_PROMPT, _judge_prompt(s.query, s.expected_answer, r["answer"]))
            r["doc_hit"] = any(d in s.expected_doc_ids for d in r["docs"])
            results[model] = rows
            OUT.write_text(json.dumps(results, indent=1, default=str))
            rows.append({"id": s.id, "group": "single", **{k: v for k, v in r.items() if k != "passages"}})
        for s in refusals:
            r = run_agent(client, retriever, model, s.query)
            r["passed"], r["judge"] = judge_verdict(judge, _REFUSAL_JUDGE_SYSTEM_PROMPT, _refusal_judge_prompt(s.query, s.expected_answer, r["answer"]))
            rows.append({"id": s.id, "group": "refusal", **{k: v for k, v in r.items() if k != "passages"}})

        # Multi-hop: agentic vs. the shipped single-pass pipeline, same model.
        cfg = base.model_copy(deep=True)
        cfg.llm.model = model
        pipeline = build_chat_service(cfg, corpora=["edgar"])
        for qid, q, entities in MULTI:
            r = run_agent(client, retriever, model, q)
            joined = " ".join(r["queries"]).lower()
            r["entity_coverage"] = (sum(e in joined for e in entities) / len(entities)) if entities else None
            ctx = "\n\n".join(f"[{i}] ({ch.document_id}) {ch.text}" for i, ch in enumerate(r["passages"], 1))
            r["passed"], r["judge"] = judge_verdict(judge, MULTI_JUDGE, f"Question: {q}\n\nRetrieved passages:\n{ctx}\n\nSystem answer: {r['answer']}\n\nVerdict:")
            t0 = time.monotonic()
            p = pipeline.ask(q)
            pctx = "\n\n".join(f"[{i}] ({c.document_id}) {c.text}" for i, c in enumerate(p.citations, 1))
            pv, pj = judge_verdict(judge, MULTI_JUDGE, f"Question: {q}\n\nRetrieved passages:\n{pctx}\n\nSystem answer: {p.answer}\n\nVerdict:")
            rows.append({"id": qid, "group": "multi", **{k: v for k, v in r.items() if k != "passages"},
                         "pipeline": {"answer": p.answer, "passed": pv, "judge": pj, "latency_s": round(time.monotonic() - t0, 1)}})
        results[model] = rows
        OUT.write_text(json.dumps(results, indent=1, default=str))
        for g in ("single", "refusal", "multi"):
            rs = [x for x in rows if x["group"] == g]
            print(model, g, sum(bool(x["passed"]) for x in rs), "/", len(rs),
                  "calls", sum(len(x["queries"]) for x in rs), "bad", sum(x["bad_calls"] for x in rs),
                  "s", round(sum(x["latency_s"] for x in rs)), flush=True)


if __name__ == "__main__":
    main()
