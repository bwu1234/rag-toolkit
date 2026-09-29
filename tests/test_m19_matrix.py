"""Tests for the Milestone 19 phase 4 matrix in scripts/run_answer_matrix.py.

No model runs: the chat service, judge and oracle are fakes. What these pin is
what would silently corrupt the pipeline-vs-agent comparison: a row whose
config isn't what its name says, the oracle scoring an empty prompt on the
refusal set, repeats overwriting each other, and groundedness verdicts
miscounted against the judge.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from rag.config.settings import LLMConfig, RagConfig
from rag.eval.dataset import EvalDataset
from rag.generation.chat_service import ChatAnswer, ChatService

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import run_answer_matrix as matrix  # noqa: E402


def _variant(name: str) -> matrix.Variant:
    return next(v for v in matrix.M19_VARIANTS if v.name == name)


def test_a_dict_override_builds_an_unset_model_field_from_defaults() -> None:
    config = matrix.apply_overrides(RagConfig(), {"agent.llm": {"model": "big", "think": "low"}})
    assert config.agent.llm == LLMConfig(model="big", think="low")


def test_a_dict_override_merges_into_a_set_model_field() -> None:
    base = RagConfig(llm=LLMConfig(model="small", temperature=0.7))
    config = matrix.apply_overrides(base, {"llm": {"max_tokens": 4096}})
    assert (config.llm.model, config.llm.temperature, config.llm.max_tokens) == ("small", 0.7, 4096)
    assert base.llm.max_tokens == 1024  # the base config is untouched


def test_every_row_is_what_its_name_says() -> None:
    rows = {v.name: matrix.apply_overrides(RagConfig(), v.overrides) for v in matrix.M19_VARIANTS}
    for name, config in rows.items():
        expected_mode = "agentic" if name.startswith("agentic") else "pipeline"
        assert config.chat.mode == expected_mode, name
        assert matrix.generator_of(config).endswith("27b-mlx" if "27b" in name else "9b-mlx"), name
        assert config.crag.enabled == ("groundedness" in name), name
    assert rows["agentic planned / 9b"].agent.strategy == "planned"
    assert rows["agentic react / 27b, think=low"].agent.llm.think == "low"  # type: ignore[union-attr]
    grounded = rows["agentic react / 27b + groundedness"].crag
    assert grounded.check_groundedness and not grounded.grade_documents and grounded.max_retries == 0
    assert [v.name for v in matrix.M19_VARIANTS if v.oracle] == ["oracle / 9b"]


def test_agent_rows_differ_from_each_other_in_one_factor() -> None:
    # 9b vs 27b must be the model alone: same token cap and timeout.
    llms = [matrix.apply_overrides(RagConfig(), v.overrides).agent.llm
            for v in matrix.M19_VARIANTS if v.name.startswith("agentic")]
    assert {(llm.max_tokens, llm.timeout_s) for llm in llms if llm} == {(4096, 600.0)}


def test_the_oracle_row_checkpoints_apart_from_the_pipeline_row() -> None:
    config = matrix.apply_overrides(RagConfig(), _variant("pipeline / 9b").overrides)
    dataset = EvalDataset.from_dicts([{"id": "s", "query": "q", "expected_answer": "a"}])
    plain = matrix.checkpoint_fingerprint(config, config.llm, dataset, "edgar", "c")
    oracle = matrix.checkpoint_fingerprint(config, config.llm, dataset, "edgar", "c", oracle=True)
    assert plain != oracle and "retrieval" not in plain


def test_repeats_get_their_own_names_and_share_a_base_name() -> None:
    runs = matrix.repeated([_variant("oracle / 9b")], 3)
    assert [r.name for r in runs] == ["oracle / 9b #1", "oracle / 9b #2", "oracle / 9b #3"]
    assert all(r.oracle for r in runs)
    assert {matrix.base_name(r.name) for r in runs} == {"oracle / 9b"}
    assert matrix.repeated([_variant("oracle / 9b")], 1)[0].name == "oracle / 9b"


def test_groundedness_counts_line_up_verdicts_with_the_judge() -> None:
    counts = matrix.groundedness_counts([(False, False), (False, True), (True, False), (True, True), (None, False)])
    assert counts == {"checked": 4, "unchecked": 1, "ungrounded": 2,
                      "ungrounded_and_failed": 1, "grounded_and_failed": 1}
    assert matrix.groundedness_counts([(None, True)]) is None


def test_the_spread_table_counts_each_run() -> None:
    def run(name: str, complete: float) -> dict[str, Any]:
        return {"variant": name, "multihop": {"complete_rate": complete, "num_evaluated": 35}}

    table = matrix.render_repeats([run("pipeline / 9b #1", 15 / 35), run("pipeline / 9b #2", 13 / 35)])
    assert "| `pipeline / 9b` | multihop complete | 15/35, 13/35 | 14.0 | 2 |" in table
    assert matrix.render_repeats([run("pipeline / 9b", 0.5)]) == ""


# ---------------------------------------------------------------------------
# End to end, with fakes
# ---------------------------------------------------------------------------


class _Chat(ChatService):
    def __init__(self) -> None:
        self.asked: list[str] = []

    def ask(self, query: str) -> ChatAnswer:  # type: ignore[override]
        self.asked.append(query)
        return ChatAnswer(answer=f"answer to {query}", citations=[], llm_calls=1, grounded=False)


class _PassJudge:
    def generate(self, prompt: str, *, system: str | None = None) -> str:
        return "PASS fine"


def _write_sets(tmp_path: Path) -> tuple[Path, Path]:
    multihop = tmp_path / "multihop.json"
    multihop.write_text(json.dumps([
        {"id": f"mh-{i}", "query": f"q{i}", "expected_answer": "A: 1", "kind": "aggregation",
         "expected_spans": ["a1"], "parts": [{"label": "A", "answer": "1", "spans": ["a1"]}]}
        for i in range(2)
    ]))
    refusals = tmp_path / "refusals.json"
    refusals.write_text(json.dumps([
        {"id": "neg-0", "query": "r0", "expected_answer": "Must decline", "tier": "refusal"},
    ]))
    return multihop, refusals


def _run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, chat: _Chat, *extra: str) -> list[dict[str, Any]]:
    multihop, refusals = _write_sets(tmp_path)
    built: list[dict[str, Any]] = []

    def build_chat_service(config: RagConfig, corpora: Any = None, retriever: Any = None) -> _Chat:
        built.append({"mode": config.chat.mode, "retriever": retriever})
        return chat

    oracle_sets: list[list[str]] = []

    def build_oracle(config: RagConfig, samples: Any, corpora: Any) -> tuple[str, list[Any]]:
        oracle_sets.append([s.id for s in samples])
        return "oracle-retriever", []

    monkeypatch.setattr(matrix, "build_chat_service", build_chat_service)
    monkeypatch.setattr(matrix, "build_oracle_retriever", build_oracle)
    monkeypatch.setattr(matrix, "get_llm_client", lambda config: _PassJudge())
    monkeypatch.setattr(sys, "argv", [
        "run_answer_matrix.py", "--family", "m19", "--sets", "refusals,multihop",
        "--multihop", str(multihop), "--refusals", str(refusals),
        "--results-dir", str(tmp_path / "results"), *extra,
    ])
    assert matrix.main() == 0
    [results_file] = (tmp_path / "results").glob("*.json")
    results: list[dict[str, Any]] = json.loads(results_file.read_text())["results"]
    for r in results:
        r["_built"] = built
        r["_oracle_sets"] = oracle_sets
    return results


def test_the_oracle_skips_the_refusal_set_and_gets_only_gold_questions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    chat = _Chat()
    [row] = _run(monkeypatch, tmp_path, chat, "--variant", "oracle / 9b")

    assert "refusals" not in row and row["multihop"]["num_evaluated"] == 2
    assert chat.asked == ["q0", "q1"]
    assert row["_oracle_sets"] == [["mh-0", "mh-1"]]
    assert row["_built"] == [{"mode": "pipeline", "retriever": "oracle-retriever"}]
    assert (row["retrieval"], row["mode"]) == ("oracle", "pipeline")


def test_repeats_are_separate_rows_with_a_spread_table(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rows = _run(monkeypatch, tmp_path, _Chat(), "--variant", "agentic react / 27b", "--repeat", "2")

    assert [r["variant"] for r in rows] == ["agentic react / 27b #1", "agentic react / 27b #2"]
    assert all(r["generator"] == "ollama:qwen3.8:27b-mlx" and r["mode"] == "agentic:react" for r in rows)
    assert all(r["refusals"]["num_evaluated"] == 1 for r in rows)
    [markdown] = (tmp_path / "results").glob("*.md")
    table = markdown.read_text()
    assert "Spread across repeats" in table


def test_groundedness_verdicts_are_saved_and_counted(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    [row] = _run(monkeypatch, tmp_path, _Chat(), "--variant", "agentic react / 27b + groundedness")

    assert row["multihop"]["groundedness"] == {
        "checked": 2, "unchecked": 0, "ungrounded": 2, "ungrounded_and_failed": 0, "grounded_and_failed": 0,
    }
    assert [s["grounded"] for s in row["refusals"]["samples"]] == [False]


def test_an_unknown_variant_name_is_refused(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sys, "argv", ["run_answer_matrix.py", "--family", "m19", "--variant", "crag=off",
                                      "--results-dir", str(tmp_path)])
    with pytest.raises(SystemExit):
        matrix.main()


# ---------------------------------------------------------------------------
# The hosted reference pair
# ---------------------------------------------------------------------------


def test_hosted_rows_take_their_settings_from_the_model_config_file() -> None:
    from rag.config.settings import load_config

    reference = load_config(REPO / "rag/config/gemini-3.5-flash-lite.yaml").llm
    pipeline, agent = (matrix.apply_overrides(RagConfig(), v.overrides) for v in matrix.M19_HOSTED_VARIANTS)
    assert pipeline.llm == reference
    assert agent.agent.llm == reference.model_copy(update={"max_tokens": 4096, "timeout_s": 600.0})
    assert agent.llm.provider == "ollama"  # the utility calls stay local
    assert reference.requests_per_day is not None  # the budget guard travels with the rows


def test_a_plain_m19_run_never_includes_a_hosted_row() -> None:
    hosted = {v.name for v in matrix.M19_HOSTED_VARIANTS}
    assert not hosted & {v.name for v in matrix.M19_VARIANTS}
    assert matrix.DEFAULT_RESULTS_DIRS["m19-hosted"] == matrix.DEFAULT_RESULTS_DIRS["m19"]


def test_the_hosted_family_refuses_an_unbounded_answerable_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run_answer_matrix.py", "--family", "m19-hosted", "--limit", "0",
                                      "--results-dir", str(tmp_path)])
    with pytest.raises(SystemExit):
        matrix.main()


def test_the_local_baseline_stays_the_first_row_next_to_hosted_rows() -> None:
    names = ["agentic react / flash-lite", "pipeline / flash-lite", "oracle / 9b", "pipeline / 9b #2", "pipeline / 9b #1"]
    ordered = sorted(names, key=lambda n: (matrix._ROW_ORDER.get(matrix.base_name(n), 999), n))
    assert ordered == ["pipeline / 9b #1", "pipeline / 9b #2", "oracle / 9b",
                       "pipeline / flash-lite", "agentic react / flash-lite"]


class _SpentChat(ChatService):
    """Answers once, then finds the day's request budget spent."""

    def __init__(self) -> None:
        self.asked: list[str] = []

    def ask(self, query: str) -> ChatAnswer:  # type: ignore[override]
        from rag.generation.daily_budget import DailyRequestBudgetSpent

        if self.asked:
            raise DailyRequestBudgetSpent("spent")
        self.asked.append(query)
        return ChatAnswer(answer="a", citations=[], llm_calls=1)


def test_a_spent_budget_stops_the_run_and_keeps_its_checkpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    multihop, _ = _write_sets(tmp_path)
    chat = _SpentChat()
    monkeypatch.setattr(matrix, "build_chat_service", lambda config, corpora=None, retriever=None: chat)
    monkeypatch.setattr(matrix, "get_llm_client", lambda config: _PassJudge())
    monkeypatch.setattr(sys, "argv", [
        "run_answer_matrix.py", "--family", "m19-hosted", "--sets", "multihop",
        "--variant", "pipeline / flash-lite", "--multihop", str(multihop),
        "--results-dir", str(tmp_path / "results"),
    ])

    assert matrix.main() == 2
    [partial] = (tmp_path / "results" / ".partial").glob("*.jsonl")
    assert len(partial.read_text().splitlines()) == 2  # the header and the one finished sample
    assert not list((tmp_path / "results").glob("*.json"))
