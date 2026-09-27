"""Tests for the Gemini `LLMClient` adapter.

Hermetic like `test_llm.py`: requests go through `httpx.MockTransport`, and the
client's clock and sleep are replaced by a fake clock, so pacing and retry
waits are asserted as numbers instead of being slept through.
"""

from __future__ import annotations

import json

import httpx
import pytest

from rag.config.settings import GEMINI_BASE_URL, LLMConfig
from rag.generation.factory import get_llm_client
from rag.generation.gemini_llm import GeminiDailyQuotaExhausted, GeminiLLMClient


class _FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def _client(handler, **kwargs) -> tuple[GeminiLLMClient, _FakeClock]:
    client = GeminiLLMClient(model="gemma-4-31b-it", base_url="https://fake-gemini", api_key="k", **kwargs)
    client._client = httpx.Client(
        base_url=client.base_url, transport=httpx.MockTransport(handler), headers=client._client.headers
    )
    clock = _FakeClock()
    client._clock = clock
    client._sleep = clock.sleep
    return client, clock


def _ok(text: str = "PASS", total_tokens: int | None = None) -> httpx.Response:
    body: dict = {"candidates": [{"content": {"role": "model", "parts": [{"text": text}]}}]}
    if total_tokens is not None:
        body["usageMetadata"] = {
            "promptTokenCount": total_tokens - 1,
            "candidatesTokenCount": 1,
            "totalTokenCount": total_tokens,
        }
    return httpx.Response(200, json=body)


def _quota_429(quota_id: str, retry_delay: str = "34s") -> httpx.Response:
    # Shape of a real free-tier 429: a per-day one still carries a short
    # retryDelay, which is why the quotaId is what decides.
    return httpx.Response(429, json={"error": {
        "code": 429,
        "status": "RESOURCE_EXHAUSTED",
        "message": "You exceeded your current quota",
        "details": [
            {"@type": "type.googleapis.com/google.rpc.QuotaFailure",
             "violations": [{"quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_requests",
                             "quotaId": quota_id}]},
            {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry_delay},
        ],
    }})


# ---------------------------------------------------------------------------
# Request and response translation
# ---------------------------------------------------------------------------


def test_generate_sends_prompt_system_and_options_with_key_in_header() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _ok()

    client, _ = _client(handler, temperature=0.0, max_tokens=256)
    client.generate("the question", system="be a judge")

    request = seen[0]
    assert request.url.path == "/v1beta/models/gemma-4-31b-it:generateContent"
    assert request.headers["x-goog-api-key"] == "k"
    assert "key=" not in str(request.url)
    body = json.loads(request.read())
    assert body["contents"] == [{"role": "user", "parts": [{"text": "the question"}]}]
    assert body["systemInstruction"] == {"parts": [{"text": "be a judge"}]}
    assert body["generationConfig"] == {"temperature": 0.0, "maxOutputTokens": 256}


def test_generate_omits_system_instruction_when_not_given() -> None:
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.read()))
        return _ok()

    client, _ = _client(handler)
    client.generate("q")

    assert "systemInstruction" not in bodies[0]


def test_generate_joins_text_parts_and_skips_thoughts() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [
            {"text": "let me think...", "thought": True},
            {"text": "PASS"},
            {"text": " - matches"},
        ]}}]})

    client, _ = _client(handler)

    assert client.generate("q") == "PASS - matches"


def test_generate_with_usage_reports_token_counts() -> None:
    client, _ = _client(lambda request: _ok(total_tokens=50))

    text, usage = client.generate_with_usage("q")

    assert text == "PASS"
    assert usage is not None
    assert (usage.prompt_tokens, usage.completion_tokens) == (49, 1)


def test_generate_raises_on_blocked_prompt() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"promptFeedback": {"blockReason": "SAFETY"}})

    client, _ = _client(handler)

    with pytest.raises(RuntimeError, match="no candidates.*SAFETY"):
        client.generate("q")


# ---------------------------------------------------------------------------
# 429s: per-minute retries, per-day stops
# ---------------------------------------------------------------------------


def test_per_minute_429_is_retried_after_its_retry_delay() -> None:
    responses = [_quota_429("GenerateRequestsPerMinutePerProjectPerModel-FreeTier", "7s"), _ok("PASS")]
    client, clock = _client(lambda request: responses.pop(0))

    assert client.generate("q") == "PASS"
    assert clock.sleeps == [7.0]


def test_per_day_429_raises_without_retrying_despite_a_retry_delay() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return _quota_429("GenerateRequestsPerDayPerProjectPerModel-FreeTier", "34s")

    client, clock = _client(handler)

    with pytest.raises(GeminiDailyQuotaExhausted, match="daily quota"):
        client.generate("q")
    assert calls == 1
    assert clock.sleeps == []


def test_429_without_quota_details_is_not_reported_as_daily() -> None:
    # Only a named per-day quota counts as the day being spent; a bare 429 is
    # treated as transient and gives up as an ordinary error.
    client, _ = _client(lambda request: httpx.Response(429, json={"error": {"message": "slow down"}}), max_retries=2)

    with pytest.raises(RuntimeError, match="HTTP 429.*3 attempt") as excinfo:
        client.generate("q")
    assert not isinstance(excinfo.value, GeminiDailyQuotaExhausted)


def test_server_error_backs_off_then_succeeds() -> None:
    responses = [httpx.Response(503, json={"error": {"message": "overloaded"}}), _ok()]
    client, clock = _client(lambda request: responses.pop(0))

    assert client.generate("q") == "PASS"
    assert clock.sleeps == [1.0]


def test_client_error_is_not_retried() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(400, json={"error": {"message": "bad model"}})

    client, _ = _client(handler)

    with pytest.raises(RuntimeError, match="HTTP 400.*bad model"):
        client.generate("q")
    assert calls == 1


# ---------------------------------------------------------------------------
# Per-minute pacing
# ---------------------------------------------------------------------------


def test_paces_under_the_tokens_per_minute_limit() -> None:
    # The first call is counted at the 600 tokens the API reported, not its
    # tiny estimate. The second is estimated at ~401 tokens, which would put the
    # window at 1001 > 1000/min, so it waits for the first to age out.
    client, clock = _client(lambda request: _ok(total_tokens=600), tokens_per_minute=1000)

    client.generate("x" * 35)
    clock.now += 10
    client.generate("x" * 1400)

    assert clock.sleeps == [50.0]


def test_paces_under_the_requests_per_minute_limit() -> None:
    client, clock = _client(lambda request: _ok(total_tokens=1), requests_per_minute=2)

    for _ in range(3):
        client.generate("q")

    assert clock.sleeps == [60.0]


def test_no_limits_means_no_pacing() -> None:
    client, clock = _client(lambda request: _ok(total_tokens=10_000))

    for _ in range(5):
        client.generate("q")

    assert clock.sleeps == []


def test_prompt_larger_than_the_token_limit_fails_fast() -> None:
    client, _ = _client(lambda request: _ok(), tokens_per_minute=100)

    with pytest.raises(RuntimeError, match="can never be sent"):
        client.generate("x" * 1000)


# ---------------------------------------------------------------------------
# Config and factory
# ---------------------------------------------------------------------------


def test_gemini_config_defaults_to_the_gemini_endpoint() -> None:
    assert LLMConfig(provider="gemini", model="gemma-4-31b-it").base_url == GEMINI_BASE_URL
    assert LLMConfig(provider="gemini", model="m", base_url="https://proxy").base_url == "https://proxy"
    assert LLMConfig().base_url == "http://localhost:11434"


def test_factory_builds_gemini_client_with_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_GEMINI_KEY", "secret")
    config = LLMConfig(
        provider="gemini", model="gemma-4-31b-it", api_key_env="MY_GEMINI_KEY",
        requests_per_minute=30, tokens_per_minute=16000, timeout_s=300,
    )

    client = get_llm_client(config)

    assert isinstance(client, GeminiLLMClient)
    assert client.base_url == GEMINI_BASE_URL
    assert client._client.headers["x-goog-api-key"] == "secret"
    assert (client.requests_per_minute, client.tokens_per_minute) == (30, 16000)
    assert client._client.timeout.read == 300


def test_factory_names_the_env_var_when_the_key_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(ValueError, match=r"\$GEMINI_API_KEY"):
        get_llm_client(LLMConfig(provider="gemini", model="gemma-4-31b-it"))
