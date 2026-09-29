"""Tests for the oracle row (Milestone 19 phase 4): gold evidence in place of retrieval.

The oracle is only a ceiling if it hands over every gold span and nothing else,
so these pin which chunks it picks, and that the pipeline's own prompt and
citations run on them unchanged.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.chunking.models import Chunk
from rag.config.settings import ChatConfig, CragConfig, RagConfig
from rag.eval.dataset import EvalSample, ExpectedSpan
from rag.eval.oracle import OracleRetriever, build_oracle_retriever, gold_chunks
from rag.generation.builder import build_chat_service
from rag.generation.chat_service import ChatService
from rag.generation.llm import LLMClient


def _chunk(chunk_id: str, text: str, doc: str = "A.md") -> Chunk:
    return Chunk(id=f"{doc}::{chunk_id}", text=text, document_id=doc, source=Path(doc),
                 doc_type="markdown", header=f"{doc} header")


def _sample(*spans: str | ExpectedSpan, docs: tuple[str, ...] = ("A.md",)) -> EvalSample:
    return EvalSample(
        id="s1", query="q?", expected_doc_ids=list(docs),
        expected_spans=[s if isinstance(s, ExpectedSpan) else ExpectedSpan(text=s) for s in spans],
        expected_answer="a",
    )


def test_picks_the_first_chunk_holding_each_span_in_span_order() -> None:
    chunks = [_chunk("c0", "Revenue was $5 billion."), _chunk("c1", "Income was $1 billion."),
              _chunk("c2", "Again: revenue was $5 billion.")]
    found, unfound = gold_chunks(_sample("income was $1 billion", "Revenue  was $5 billion"), chunks)
    assert [c.id for c in found] == ["A.md::c1", "A.md::c0"]
    assert unfound == []


def test_one_chunk_holding_two_spans_is_passed_once() -> None:
    chunks = [_chunk("c0", "Revenue was $5 billion. Income was $1 billion.")]
    found, _ = gold_chunks(_sample("Revenue was $5 billion", "Income was $1 billion"), chunks)
    assert [c.id for c in found] == ["A.md::c0"]


def test_ignores_the_same_sentence_in_another_filing() -> None:
    # A span-and-document sample: the prior year's filing repeats the sentence.
    chunks = [_chunk("c0", "Revenue was $5 billion.", doc="OLD.md"), _chunk("c3", "Revenue was $5 billion.")]
    found, _ = gold_chunks(_sample("Revenue was $5 billion"), chunks)
    assert [c.id for c in found] == ["A.md::c3"]


def test_an_alternative_quote_counts_like_in_evidence_recall() -> None:
    span = ExpectedSpan(text="not in the corpus", alternatives=("Revenue was $5 billion",))
    found, unfound = gold_chunks(_sample(span), [_chunk("c0", "Revenue was $5 billion.")])
    assert [c.id for c in found] == ["A.md::c0"] and unfound == []


def test_reports_a_span_no_chunk_holds() -> None:
    found, unfound = gold_chunks(_sample("Revenue was $5 billion", "missing"), [_chunk("c0", "Revenue was $5 billion.")])
    assert [c.id for c in found] == ["A.md::c0"]
    assert unfound == ["missing"]


def test_refuses_a_sample_without_gold_spans() -> None:
    with pytest.raises(ValueError, match="no gold spans"):
        gold_chunks(_sample(docs=()), [_chunk("c0", "x")])


def test_retriever_returns_gold_chunks_with_their_headers_and_refuses_unknown_queries() -> None:
    retriever = OracleRetriever({"q?": [_chunk("c0", "Revenue was $5 billion.")]})
    result = retriever.retrieve("q?")
    assert [(c.chunk_id, c.header) for c in result.chunks] == [("A.md::c0", "A.md header")]
    assert result.candidate_count == 1
    with pytest.raises(KeyError):
        retriever.retrieve("another question")


class _EchoLLM(LLMClient):
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        self.prompts.append(prompt)
        return "Revenue was $5 billion [1]."


def test_the_pipeline_answers_from_the_oracle_chunks() -> None:
    llm = _EchoLLM()
    service = ChatService(retriever=OracleRetriever({"q?": [_chunk("c0", "Revenue was $5 billion.")]}),
                          llm_client=llm)
    answer = service.ask("q?")
    assert "Revenue was $5 billion." in llm.prompts[0]
    assert [c.chunk_id for c in answer.citations] == ["A.md::c0"]


def test_the_oracle_refuses_crag_and_the_agent() -> None:
    with pytest.raises(ValueError, match="crag.enabled: false"):
        build_oracle_retriever(RagConfig(crag=CragConfig(enabled=True)), [], None)
    with pytest.raises(ValueError, match="pipeline only"):
        build_chat_service(RagConfig(chat=ChatConfig(mode="agentic")), retriever=OracleRetriever({}))
