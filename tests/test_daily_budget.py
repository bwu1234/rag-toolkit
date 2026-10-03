"""Tests for the per-model daily request counter (the free tier's budget guard).

What they protect: a caller stops before the provider's per-day quota is gone
for everyone, the count survives across processes and resets with the day, and
a refused request is never sent.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from rag.config.settings import REPO_ROOT, LLMConfig
from rag.llm.daily_budget import DailyRequestBudgetSpent, DailyRequestCounter
from rag.llm.factory import get_llm_client
from rag.llm.gemini_llm import GeminiLLMClient


def _counter(path: Path, limit: int = 3, key: str = "flash", day: list[str] | None = None) -> DailyRequestCounter:
    day = day or ["2026-09-29"]
    return DailyRequestCounter(path, key=key, limit=limit, today=lambda: day[0])


def test_counts_up_to_the_limit_then_refuses_without_recording_more(tmp_path: Path) -> None:
    counter = _counter(tmp_path / "log.json")
    assert [counter.take() for _ in range(3)] == [1, 2, 3]
    with pytest.raises(DailyRequestBudgetSpent, match="3 of this machine's 3"):
        counter.take()
    assert counter.used() == 3


def test_the_count_is_shared_across_counters_on_one_file(tmp_path: Path) -> None:
    # Two processes (an eval run and the API) spend the same day.
    path = tmp_path / "log.json"
    _counter(path).take()
    _counter(path).take()
    assert _counter(path).used() == 2


def test_each_model_has_its_own_count(tmp_path: Path) -> None:
    path = tmp_path / "log.json"
    _counter(path, key="flash").take()
    _counter(path, key="gemma").take()
    assert json.loads(path.read_text())["counts"] == {"flash": 1, "gemma": 1}


def test_a_new_pacific_day_starts_every_count_over(tmp_path: Path) -> None:
    day = ["2026-09-29"]
    counter = _counter(tmp_path / "log.json", limit=1, day=day)
    counter.take()
    day[0] = "2026-09-30"
    assert counter.used() == 0
    assert counter.take() == 1


def test_a_refused_request_is_never_sent(tmp_path: Path) -> None:
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, json={"candidates": [{"content": {"role": "model", "parts": [{"text": "ok"}]}}]})

    client = GeminiLLMClient(model="flash", base_url="https://fake", api_key="k",
                             daily_counter=_counter(tmp_path / "log.json", limit=1))
    client._client = httpx.Client(base_url=client.base_url, transport=httpx.MockTransport(handler))
    assert client.generate("q") == "ok"
    with pytest.raises(DailyRequestBudgetSpent):
        client.generate("q")
    assert len(sent) == 1


def test_config_sets_the_limit_to_the_quota_minus_the_reserve(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    client = get_llm_client(LLMConfig(provider="gemini", model="flash", requests_per_day=500,
                                      requests_per_day_reserve=50))
    assert isinstance(client, GeminiLLMClient) and client.daily_counter is not None
    assert (client.daily_counter.limit, client.daily_counter.key) == (450, "flash")
    assert client.daily_counter.path == (REPO_ROOT / "data/logs/llm_daily_requests.json").resolve()
    unbudgeted = get_llm_client(LLMConfig(provider="gemini", model="flash"))
    assert isinstance(unbudgeted, GeminiLLMClient) and unbudgeted.daily_counter is None


def test_config_refuses_a_budget_it_cannot_apply() -> None:
    with pytest.raises(ValueError, match="has none"):
        LLMConfig(provider="ollama", requests_per_day=500)
    with pytest.raises(ValueError, match="leaves nothing"):
        LLMConfig(provider="gemini", model="flash", requests_per_day=50, requests_per_day_reserve=50)
