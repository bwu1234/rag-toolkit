"""Tests for rag/eval/musique_eval.py and the MuSiQue rows of scripts/run_answer_matrix.py.

No model runs: the chat service and extractor are fakes. What these pin is what
would silently misreport the outside check: an extraction of NONE scored as
a match, per-hop evidence credited to the wrong step, a sampled run whose
questions change between rows (they must pair), and a closed-book row that
reaches retrieval or reports no cost.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from rag.eval.dataset import EvalDataset, EvalSample, ExpectedSpan
from rag.eval.musique_eval import (
    ClosedBookResponder,
    extract_answer,
    score_sample,
    stratified_subset,
    summarize,
)
from rag.generation.chat_service import ChatAnswer, ChatService, Citation
from rag.llm.base import LLMClient, LLMUsage
from rag.observability.usage import metered_client

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import run_answer_matrix as matrix  # noqa: E402

P1, P2 = "Grant Green performer paragraph", "Steve Hillage spouse paragraph"


def _sample(sid: str = "2hop__1_2", kind: str = "2hop") -> EvalSample:
    return EvalSample(
        id=sid, query="Who is the spouse of the Green performer?",
        expected_spans=[ExpectedSpan(P1), ExpectedSpan(P2)], expected_answer="Miquette Giraudy",
        extra={"kind": kind, "answer_aliases": ["Miquette"], "parts": [
            {"label": "Green >> performer", "answer": "Steve Hillage", "spans": [P1]},
            {"label": "#1 >> spouse", "answer": "Miquette Giraudy", "spans": [P2]},
        ]},
    )


class _Extractor(LLMClient):
    def __init__(self, reply: str) -> None:
        self.reply = reply

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        return self.reply


def _answer(text: str, passages: list[str]) -> ChatAnswer:
    return ChatAnswer(answer=text, citations=[Citation(chunk_id=p, document_id=p, text=p, score=1.0) for p in passages])


def test_a_committed_answer_is_scored_against_the_aliases() -> None:
    r = score_sample(_sample(), _answer("It is Miquette Giraudy [2].", [P2]), _Extractor("Miquette"), 1.0)
    assert (r.extracted, r.em, r.f1, r.contains) == ("Miquette", 1, 1.0, True)
    # Only step 2's paragraph came back: recall 0.5, credited to hop 2.
    assert r.evidence_recall == 0.5 and r.hop_found == [False, True]


def test_no_committed_answer_scores_zero_and_is_counted_apart() -> None:
    r = score_sample(_sample(), _answer("The passages don't say.", []), _Extractor("NONE."), 1.0)
    assert (r.extracted, r.em, r.f1, r.contains) == ("", 0, 0.0, False)
    assert summarize([r]).num_none == 1
    # An empty response never reaches the extractor.
    assert extract_answer(_Extractor("Paris"), "q", "   ") == ""


def test_a_stratified_sample_keeps_hop_shares_and_is_the_same_every_time() -> None:
    ds = EvalDataset(samples=[_sample(f"{k}__{i}", k) for k, n in (("2hop", 60), ("3hop", 30), ("4hop", 10))
                              for i in range(n)])
    picked = stratified_subset(ds, 20)
    kinds = [s.extra["kind"] for s in picked]
    assert (kinds.count("2hop"), kinds.count("3hop"), kinds.count("4hop")) == (12, 6, 2)
    assert [s.id for s in picked] == [s.id for s in stratified_subset(ds, 20)]
    assert stratified_subset(ds, 0) is ds


class _Metered(LLMClient):
    def generate(self, prompt: str, *, system: str | None = None) -> str:
        return self.generate_with_usage(prompt, system=system)[0]

    def generate_with_usage(self, prompt: str, *, system: str | None = None) -> tuple[str, LLMUsage | None]:
        return f"answer to {prompt}", LLMUsage(prompt_tokens=7, completion_tokens=3)


def test_the_closed_book_responder_answers_without_retrieval_and_is_metered() -> None:
    a = ClosedBookResponder(metered_client(_Metered())).ask("Who?")
    assert a.answer == "answer to Who?" and a.citations == [] and a.retrieval_attempts == 0
    assert (a.llm_calls, a.prompt_tokens, a.completion_tokens) == (1, 7, 3)


# ---------------------------------------------------------------------------
# The matrix, end to end with fakes
# ---------------------------------------------------------------------------


class _Chat(ChatService):
    def __init__(self) -> None:
        self.asked: list[str] = []

    def ask(self, query: str) -> ChatAnswer:  # type: ignore[override]
        self.asked.append(query)
        return _answer("Miquette Giraudy", [P1, P2])


def test_the_musique_family_runs_its_set_with_a_closed_book_row(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    eval_set = tmp_path / "musique.json"
    eval_set.write_text(json.dumps([_sample().to_dict()]))
    chat = _Chat()
    built: list[str] = []

    def build_chat_service(config: Any, corpora: Any = None, retriever: Any = None) -> _Chat:
        built.append(config.chat.mode)
        return chat

    monkeypatch.setattr(matrix, "build_chat_service", build_chat_service)
    monkeypatch.setattr(matrix, "get_llm_client", lambda config: _Extractor("Miquette Giraudy"))
    monkeypatch.setattr(sys, "argv", [
        "run_answer_matrix.py", "--family", "musique", "--musique", str(eval_set),
        "--variant", "pipeline / 9b", "--variant", "closed-book / 9b",
        "--results-dir", str(tmp_path / "results"),
    ])
    assert matrix.main() == 0
    [results_file] = (tmp_path / "results").glob("*.json")
    rows = {r["variant"]: r for r in json.loads(results_file.read_text())["results"]}

    # The closed-book row never built a chat service (no retrieval).
    assert built == ["pipeline"] and chat.asked == [_sample().query]
    assert rows["closed-book / 9b"]["retrieval"] == "none"
    assert rows["closed-book / 9b"]["musique"]["evidence_recall"] == 0.0
    assert rows["pipeline / 9b"]["musique"]["em"] == 1.0
    assert rows["pipeline / 9b"]["musique"]["evidence_by_hop"] == [1.0, 1.0]
    assert set(rows) == {"pipeline / 9b", "closed-book / 9b"} and "answerable" not in rows["pipeline / 9b"]
    [markdown] = (tmp_path / "results").glob("*.md")
    assert "MuSiQue-Ans" in markdown.read_text()
