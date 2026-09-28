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
from rag.generation.llm import AssistantTurn, ChatMessage, ToolCall, ToolCallingLLM, ToolDefinition, ToolResult


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
# Thinking models (Gemini 3+ Flash-Lite)
# ---------------------------------------------------------------------------


def test_thinking_level_is_sent_as_the_upper_case_enum() -> None:
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.read()))
        return _ok()

    client, _ = _client(handler, thinking_level="minimal")
    client.generate("q")

    assert bodies[0]["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "MINIMAL"}


def test_no_thinking_config_is_sent_unless_asked() -> None:
    # Hosted Gemma rejects thinkingConfig outright, so the default must omit it.
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.read()))
        return _ok()

    client, _ = _client(handler)
    client.generate("q")

    assert "thinkingConfig" not in bodies[0]["generationConfig"]


def test_thought_tokens_count_as_completion_tokens() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "candidates": [{"content": {"parts": [{"text": "PASS"}]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 5,
                              "thoughtsTokenCount": 40, "totalTokenCount": 145},
        })

    client, _ = _client(handler)

    _, usage = client.generate_with_usage("q")

    assert usage is not None
    assert (usage.prompt_tokens, usage.completion_tokens) == (100, 45)


def test_missing_output_counts_stay_unknown_rather_than_zero() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "candidates": [{"content": {"parts": [{"text": "PASS"}]}}],
            "usageMetadata": {"promptTokenCount": 100},
        })

    client, _ = _client(handler)

    _, usage = client.generate_with_usage("q")

    assert usage is not None
    assert usage.completion_tokens is None


@pytest.mark.parametrize("candidate", [
    # Thoughts used the whole cap: the API sends the candidate with no parts.
    {"content": {"role": "model"}, "finishReason": "MAX_TOKENS"},
    {"content": {"parts": [{"text": "hmm", "thought": True}]}, "finishReason": "MAX_TOKENS"},
])
def test_hitting_max_tokens_with_no_answer_raises_and_names_the_fix(candidate: dict) -> None:
    client, _ = _client(lambda request: httpx.Response(200, json={"candidates": [candidate]}), max_tokens=64)

    with pytest.raises(RuntimeError, match=r"max_tokens=64.*thinking_level"):
        client.generate("q")


def test_an_answer_cut_short_by_max_tokens_is_still_returned(caplog: pytest.LogCaptureFixture) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"candidates": [
            {"content": {"parts": [{"text": "Revenue was"}]}, "finishReason": "MAX_TOKENS"}
        ]})

    client, _ = _client(handler)

    with caplog.at_level("WARNING"):
        assert client.generate("q") == "Revenue was"
    assert "truncated" in caplog.text


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


def test_factory_passes_the_thinking_level(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "secret")

    client = get_llm_client(LLMConfig(provider="gemini", model="gemini-3.5-flash-lite", thinking_level="low"))

    assert isinstance(client, GeminiLLMClient)
    assert client.thinking_level == "low"


def test_thinking_level_is_refused_for_a_provider_that_would_ignore_it() -> None:
    with pytest.raises(ValueError, match="thinking_level is a Gemini setting"):
        LLMConfig(provider="ollama", thinking_level="low")


def test_factory_names_the_env_var_when_the_key_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(ValueError, match=r"\$GEMINI_API_KEY"):
        get_llm_client(LLMConfig(provider="gemini", model="gemma-4-31b-it"))


# ---------------------------------------------------------------------------
# Tool calling (chat)
# ---------------------------------------------------------------------------

SEARCH = ToolDefinition(
    name="rag_search",
    description="Search the filings.",
    parameters={
        "type": "object",
        "properties": {"query": {"type": "string"}, "top_k": {"anyOf": [{"type": "integer"}, {"type": "null"}], "default": None}},
        "required": ["query"],
    },
)

#: Recorded from gemini-3.5-flash-lite (thinking_level minimal): a signed call
#: with an id, and no text part.
FLASH_CALL_PARTS = [
    {
        "functionCall": {"name": "rag_search", "args": {"query": "Apple total net sales fiscal 2025"}, "id": "call_204087"},
        "thoughtSignature": "EmAKXgFpFH0T",
    }
]

#: Recorded from gemma-4-31b-it: a thought part before the signed call.
GEMMA_CALL_PARTS = [
    {"text": "I need to find Apple's total net sales for fiscal 2025.", "thought": True},
    {
        "functionCall": {"name": "rag_search", "args": {"query": "Apple total net sales fiscal 2025"}, "id": "call_199867"},
        "thoughtSignature": "EiYKJGUyNDgz",
    },
]


def _parts_response(parts: list, finish: str = "STOP") -> httpx.Response:
    return httpx.Response(200, json={
        "candidates": [{"content": {"role": "model", "parts": parts}, "finishReason": finish}],
        "usageMetadata": {"promptTokenCount": 436, "candidatesTokenCount": 25, "totalTokenCount": 461},
    })


def _recording(sent: list[dict], response: httpx.Response):
    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.read()))
        return response

    return handler


def test_gemini_client_is_a_tool_calling_llm() -> None:
    assert isinstance(GeminiLLMClient(model="m", base_url="https://x", api_key="k"), ToolCallingLLM)


@pytest.mark.parametrize(("parts", "call_id", "signature"), [
    (FLASH_CALL_PARTS, "call_204087", "EmAKXgFpFH0T"),
    (GEMMA_CALL_PARTS, "call_199867", "EiYKJGUyNDgz"),
])
def test_chat_parses_signed_function_calls_and_skips_thoughts(parts: list, call_id: str, signature: str) -> None:
    client, _ = _client(_recording([], _parts_response(parts)))

    turn = client.chat([ChatMessage("user", "Apple net sales?")], [SEARCH])

    assert turn.content == ""  # the thought part is not answer text
    assert turn.tool_calls == (
        ToolCall("rag_search", {"query": "Apple total net sales fiscal 2025"}, id=call_id, signature=signature),
    )
    assert turn.usage is not None and (turn.usage.prompt_tokens, turn.usage.completion_tokens) == (436, 25)


def test_chat_declares_tools_with_json_schema_and_omits_them_when_none() -> None:
    sent: list[dict] = []
    client, _ = _client(_recording(sent, _ok("Answer [1].")))

    client.chat([ChatMessage("user", "q")], [SEARCH])
    client.chat([ChatMessage("user", "q")])

    # `parametersJsonSchema` takes the pydantic schema as-is (`anyOf`,
    # `default`); the OpenAPI-subset `parameters` field would not.
    assert sent[0]["tools"] == [{"functionDeclarations": [
        {"name": "rag_search", "description": "Search the filings.", "parametersJsonSchema": SEARCH.parameters}
    ]}]
    assert "tools" not in sent[1]


def test_chat_sends_the_signature_back_on_the_call_it_came_with() -> None:
    # Without it Gemini 3 answers 400 "Function call is missing a thought_signature".
    sent: list[dict] = []
    client, _ = _client(_recording(sent, _ok("Answer [1].")))
    call = ToolCall("rag_search", {"query": "Apple"}, id="call_1", signature="SIG")

    client.chat([
        ChatMessage("system", "Search first."),
        ChatMessage("user", "Apple net sales?"),
        AssistantTurn(content="", tool_calls=(call,)),
        ToolResult(call, "Passage [1] ..."),
    ])

    assert sent[0]["systemInstruction"] == {"parts": [{"text": "Search first."}]}
    assert sent[0]["contents"] == [
        {"role": "user", "parts": [{"text": "Apple net sales?"}]},
        {"role": "model", "parts": [
            {"functionCall": {"name": "rag_search", "args": {"query": "Apple"}, "id": "call_1"}, "thoughtSignature": "SIG"}
        ]},
        {"role": "user", "parts": [
            {"functionResponse": {"name": "rag_search", "response": {"result": "Passage [1] ..."}, "id": "call_1"}}
        ]},
    ]


def test_parallel_results_go_back_together_after_all_calls() -> None:
    # Gemini requires FC1, FC2, FR1, FR2; interleaving is a 400.
    sent: list[dict] = []
    client, _ = _client(_recording(sent, _ok("Both [1][2].")))
    first = ToolCall("rag_search", {"query": "Delta"}, id="a", signature="SIG")
    second = ToolCall("rag_search", {"query": "United"}, id="b")

    client.chat([
        ChatMessage("user", "Compare Delta and United."),
        AssistantTurn(content="Searching both.", tool_calls=(first, second)),
        ToolResult(first, "Delta passage"),
        ToolResult(second, "United passage"),
        AssistantTurn(content="One more.", tool_calls=(first,)),
        ToolResult(first, "Delta again"),
    ])

    contents = sent[0]["contents"]
    assert [c["role"] for c in contents] == ["user", "model", "user", "model", "user"]
    assert contents[1]["parts"][0] == {"text": "Searching both."}
    assert [p["functionCall"]["id"] for p in contents[1]["parts"][1:]] == ["a", "b"]
    assert "thoughtSignature" not in contents[1]["parts"][2]  # only the first call is signed
    assert [p["functionResponse"]["id"] for p in contents[2]["parts"]] == ["a", "b"]
    assert [p["functionResponse"]["id"] for p in contents[4]["parts"]] == ["a"]  # a new step, a new turn


def test_chat_refuses_a_system_message_after_the_conversation_starts() -> None:
    client, _ = _client(_recording([], _ok()))

    with pytest.raises(ValueError, match="only before the first"):
        client.chat([ChatMessage("user", "q"), ChatMessage("system", "late")])


@pytest.mark.parametrize("reason", ["MALFORMED_FUNCTION_CALL", "UNEXPECTED_TOOL_CALL"])
def test_chat_raises_when_gemini_rejects_the_models_call(reason: str) -> None:
    response = httpx.Response(200, json={"candidates": [{"finishReason": reason}]})
    client, _ = _client(_recording([], response))

    with pytest.raises(RuntimeError, match=reason):
        client.chat([ChatMessage("user", "q")], [SEARCH])


def test_a_call_cut_at_max_tokens_is_still_a_turn() -> None:
    # MAX_TOKENS with no text raises for `generate`, but a turn that got its
    # call out has something to act on.
    client, _ = _client(_recording([], _parts_response(FLASH_CALL_PARTS, finish="MAX_TOKENS")))

    turn = client.chat([ChatMessage("user", "q")], [SEARCH])

    assert len(turn.tool_calls) == 1


def test_a_user_message_after_tool_results_folds_into_the_last_response() -> None:
    # Gemini 3.x documents inline instructions inside the function response;
    # the agent's forced-synthesis instruction arrives as a user message.
    sent: list[dict] = []
    client, _ = _client(_recording(sent, _ok("Answer [1].")))
    first = ToolCall("rag_search", {"query": "Delta"}, id="a")
    second = ToolCall("rag_search", {"query": "United"}, id="b")

    client.chat([
        ChatMessage("user", "Compare Delta and United."),
        AssistantTurn(content="", tool_calls=(first, second)),
        ToolResult(first, "Delta passage"),
        ToolResult(second, "United passage"),
        ChatMessage("user", "Answer now."),
    ])

    contents = sent[0]["contents"]
    assert [c["role"] for c in contents] == ["user", "model", "user"]
    responses = [p["functionResponse"]["response"]["result"] for p in contents[2]["parts"]]
    assert responses == ["Delta passage", "United passage\n\nAnswer now."]


def test_a_user_message_after_a_model_turn_stays_its_own_turn() -> None:
    sent: list[dict] = []
    client, _ = _client(_recording(sent, _ok("Sure.")))
    client.chat([ChatMessage("user", "Hi"), AssistantTurn(content="Hello"), ChatMessage("user", "Again")])
    assert sent[0]["contents"][-1] == {"role": "user", "parts": [{"text": "Again"}]}
