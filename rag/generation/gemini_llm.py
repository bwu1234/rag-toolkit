"""Gemini API-backed `LLMClient` adapter.

Talks to the Gemini API's REST `generateContent` endpoint over `httpx` (already
a dependency) rather than the `google-genai` SDK -- one POST doesn't justify a
second HTTP stack. The same endpoint serves Gemini and hosted Gemma models
(`gemma-4-31b-it`), which is what makes it usable as the eval judge: the fixed
judge is Gemma 4 31B, and running it here frees the local GPU.

Two things a naive client gets wrong against the free tier:

- **The binding limit is tokens per minute, not requests.** Gemma 4 31B allows
  30 requests/min but 16k tokens/min, and one judge prompt carrying five
  passages is ~2-3k tokens. So the client paces itself before sending, keeping
  a rolling 60s window of tokens spent, instead of firing and eating 429s.
- **A 429 means two different things.** A per-minute 429 clears in seconds; a
  per-day 429 doesn't clear until the quota resets. Both carry a
  `RetryInfo.retryDelay` of a few seconds, so trusting that alone retries a
  spent day forever. Only the `QuotaFailure` violation's `quotaId` tells them
  apart (`GenerateRequestsPerDayPerProjectPerModel-FreeTier` vs
  `...PerMinute...`) -- the same test Google's gemini-cli uses. A per-day 429
  raises `GeminiDailyQuotaExhausted`, and nothing else is reported as one.

Gemini 3+ Flash-Lite models think, and can't be told not to. Their thought
tokens are drawn from `maxOutputTokens` and reported apart from the answer's
(`thoughtsTokenCount`), so this adapter sends `thinking_level` when set, counts
thoughts as completion tokens (as Ollama's `eval_count` already does for qwen),
and raises when thinking spent the whole budget instead of returning "".

It is also a `ToolCallingLLM`, for the Milestone 19 agent. Gemini's function
calling has two rules a port of the Ollama adapter would break. A Gemini 3
model attaches a `thoughtSignature` to a step's first `functionCall`, and the
next request fails with a 400 unless it goes back unchanged, so it travels
on `ToolCall.signature`. And all of a step's results go back in one user turn
of `functionResponse` parts, after all of its calls: interleaving calls and
responses is also a 400.

The API key is read from the environment (`api_key_env`), never from config.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections import deque
from collections.abc import Callable, Sequence
from typing import Any

import httpx

from rag.generation.daily_budget import DailyRequestCounter
from rag.generation.llm import (
    AssistantTurn,
    ChatMessage,
    LLMUsage,
    Message,
    ToolCall,
    ToolCallingLLM,
    ToolDefinition,
    ToolResult,
)

logger = logging.getLogger(__name__)

#: Rough chars-per-token for pacing before the provider has counted anything.
#: Deliberately low (so the estimate runs high): underestimating is what causes
#: a 429. The window is corrected to the reported count after each call.
_CHARS_PER_TOKEN = 3.5

_WINDOW_S = 60.0

#: Status codes worth retrying: per-minute rate limiting and transient server
#: trouble (500, 503 "model overloaded", 504).
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class GeminiDailyQuotaExhausted(RuntimeError):
    """The API reported this model's per-day quota as spent. Retrying won't help until it resets."""


class GeminiLLMClient(ToolCallingLLM):
    """Generates responses via the Gemini API's `models/{model}:generateContent` endpoint.

    `requests_per_minute` / `tokens_per_minute` are the project's free-tier
    limits for `model`, read from AI Studio; `None` disables that half of the
    pacing. `max_retries` bounds retries of per-minute 429s and 5xx errors.
    `daily_counter`, if given, is charged one request per HTTP attempt and
    refuses once this machine's daily budget for the model is spent.
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        *,
        api_key: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        timeout: float = 120.0,
        requests_per_minute: int | None = None,
        tokens_per_minute: int | None = None,
        max_retries: int = 5,
        thinking_level: str | None = None,
        daily_counter: DailyRequestCounter | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("GeminiLLMClient needs a non-empty api_key")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self.max_retries = max_retries
        self.thinking_level = thinking_level
        self.daily_counter = daily_counter
        # The key goes in a header, not the `?key=` query string, so it can't
        # end up in a logged URL or an httpx error message.
        self._client = httpx.Client(
            base_url=self.base_url, timeout=timeout, headers={"x-goog-api-key": api_key}
        )
        # (timestamp, tokens) per request sent in the last `_WINDOW_S`.
        self._window: deque[tuple[float, int]] = deque()
        # Injectable so tests can drive the pacing without real waits.
        self._clock: Callable[[], float] = time.monotonic
        self._sleep: Callable[[float], None] = time.sleep

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        return self.generate_with_usage(prompt, system=system)[0]

    def generate_with_usage(self, prompt: str, *, system: str | None = None) -> tuple[str, LLMUsage | None]:
        estimate = int(len(prompt + (system or "")) / _CHARS_PER_TOKEN) + 1
        payload, usage = self._generate([{"role": "user", "parts": [{"text": prompt}]}], system, [], estimate)
        return _response_parts(payload, max_tokens=self.max_tokens)[0], usage

    def chat(self, messages: Sequence[Message], tools: Sequence[ToolDefinition] = ()) -> AssistantTurn:
        system, contents = _to_contents(messages)
        declarations = [_declaration(t) for t in tools]
        estimate = int(len(json.dumps(contents) + (system or "") + json.dumps(declarations)) / _CHARS_PER_TOKEN) + 1
        payload, usage = self._generate(contents, system, declarations, estimate)
        text, calls = _response_parts(payload, max_tokens=self.max_tokens)
        return AssistantTurn(content=text, tool_calls=calls, usage=usage)

    def _generate(
        self,
        contents: list[dict[str, Any]],
        system: str | None,
        declarations: list[dict[str, Any]],
        estimate: int,
    ) -> tuple[dict[str, object], LLMUsage]:
        generation_config: dict[str, object] = {
            "temperature": self.temperature,
            "maxOutputTokens": self.max_tokens,
        }
        if self.thinking_level is not None:
            # The REST enum is upper case (MINIMAL, LOW, MEDIUM, HIGH).
            generation_config["thinkingConfig"] = {"thinkingLevel": self.thinking_level.upper()}
        body: dict[str, object] = {"contents": contents, "generationConfig": generation_config}
        if system is not None:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        # Omitted when empty, as in the Ollama adapter: a turn offered no tools
        # must answer in text, which is how the agent forces synthesis.
        if declarations:
            body["tools"] = [{"functionDeclarations": declarations}]

        payload = self._post_with_retries(body, estimate)

        usage_meta = payload.get("usageMetadata")
        usage_meta = usage_meta if isinstance(usage_meta, dict) else {}
        answer_tokens = _optional_int(usage_meta.get("candidatesTokenCount"))
        thought_tokens = _optional_int(usage_meta.get("thoughtsTokenCount"))
        # Thoughts are billed and rate-limited as output; leaving them out
        # would make a thinking model look cheaper than it is.
        completion_tokens = (
            None if answer_tokens is None and thought_tokens is None
            else (answer_tokens or 0) + (thought_tokens or 0)
        )
        usage = LLMUsage(
            prompt_tokens=_optional_int(usage_meta.get("promptTokenCount")),
            completion_tokens=completion_tokens,
        )
        # Replace the estimate with what the API counted, so pacing tracks reality.
        reported = _optional_int(usage_meta.get("totalTokenCount"))
        if reported is not None and self._window:
            stamp, _ = self._window.pop()
            self._window.append((stamp, reported))
        return payload, usage

    def _post_with_retries(self, body: dict[str, object], estimate: int) -> dict[str, object]:
        path = f"/v1beta/models/{self.model}:generateContent"
        attempt = 0
        while True:
            self._wait_for_capacity(estimate)
            if self.daily_counter is not None:
                self.daily_counter.take()  # raises DailyRequestBudgetSpent at the limit
            self._window.append((self._clock(), estimate))
            try:
                response = self._client.post(path, json=body)
            except httpx.HTTPError as exc:
                raise RuntimeError(
                    f"Failed to reach the Gemini API at {self.base_url} (model={self.model!r}): {exc}"
                ) from exc

            if response.status_code == 200:
                payload = response.json()
                if not isinstance(payload, dict):
                    raise RuntimeError(f"Unexpected response shape from Gemini generateContent: {payload!r}")
                return payload

            if response.status_code == 429 and _is_daily_quota(response):
                raise GeminiDailyQuotaExhausted(
                    f"Gemini daily quota for {self.model!r} is spent; it resets daily "
                    f"(midnight Pacific). Details: {_error_message(response)}"
                )
            if response.status_code not in _RETRYABLE_STATUS or attempt >= self.max_retries:
                raise RuntimeError(
                    f"Gemini generateContent failed with HTTP {response.status_code} "
                    f"(model={self.model!r}) after {attempt + 1} attempt(s): {_error_message(response)}"
                )
            delay = _retry_delay(response) or min(2.0**attempt, 30.0)
            logger.warning(
                "Gemini HTTP %d for %s; retrying in %.1fs (attempt %d/%d)",
                response.status_code, self.model, delay, attempt + 1, self.max_retries,
            )
            self._sleep(delay)
            attempt += 1

    def _wait_for_capacity(self, tokens: int) -> None:
        """Block until sending `tokens` more stays inside both per-minute limits."""
        if self.tokens_per_minute is not None and tokens > self.tokens_per_minute:
            raise RuntimeError(
                f"Prompt is ~{tokens} tokens, over the {self.tokens_per_minute}/min limit for "
                f"{self.model!r}; it can never be sent under this quota"
            )
        while True:
            now = self._clock()
            while self._window and now - self._window[0][0] >= _WINDOW_S:
                self._window.popleft()
            over_requests = (
                self.requests_per_minute is not None and len(self._window) >= self.requests_per_minute
            )
            over_tokens = (
                self.tokens_per_minute is not None
                and sum(t for _, t in self._window) + tokens > self.tokens_per_minute
            )
            if not (over_requests or over_tokens):
                return
            # Wait for the oldest entry to age out, then re-check.
            wait = _WINDOW_S - (now - self._window[0][0])
            logger.debug("Pacing %s under its per-minute quota: waiting %.1fs", self.model, wait)
            self._sleep(max(wait, 0.01))

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "GeminiLLMClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def api_key_from_env(var: str) -> str:
    """Read the API key from `var`, failing with the fix if it's unset."""
    key = os.environ.get(var, "").strip()
    if not key:
        raise ValueError(
            f"The gemini LLM provider reads its API key from ${var}, which is unset. "
            f"Export it (the key comes from AI Studio); it never goes in config.yaml."
        )
    return key


def _to_contents(messages: Sequence[Message]) -> tuple[str | None, list[dict[str, Any]]]:
    """Translate a conversation into Gemini's `systemInstruction` and `contents`.

    Consecutive `ToolResult`s share one user turn: Gemini wants a step's
    responses together, after all of its calls. A user message right after
    them is folded into the last response's text, which is where Gemini 3.x
    documents inline instructions go ("appended directly to the response
    text") -- the agent's forced-synthesis instruction arrives this way.
    """
    system: list[str] = []
    contents: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, ChatMessage) and message.role == "system":
            if contents:
                # Gemini has one system instruction, outside the turn list, so a
                # system message partway through would silently move to the top.
                raise ValueError("Gemini takes system messages only before the first user/assistant turn")
            system.append(message.content)
        elif isinstance(message, ChatMessage):
            previous = contents[-1] if contents else None
            if previous and previous["role"] == "user" and all("functionResponse" in p for p in previous["parts"]):
                result = previous["parts"][-1]["functionResponse"]["response"]
                result["result"] = f"{result['result']}\n\n{message.content}"
            else:
                contents.append({"role": "user", "parts": [{"text": message.content}]})
        elif isinstance(message, ToolResult):
            response: dict[str, Any] = {"name": message.call.name, "response": {"result": message.content}}
            if message.call.id is not None:
                response["id"] = message.call.id
            part = {"functionResponse": response}
            previous = contents[-1] if contents else None
            if previous and previous["role"] == "user" and all("functionResponse" in p for p in previous["parts"]):
                previous["parts"].append(part)
            else:
                contents.append({"role": "user", "parts": [part]})
        else:
            parts: list[dict[str, Any]] = [{"text": message.content}] if message.content else []
            for call in message.tool_calls:
                function_call: dict[str, Any] = {"name": call.name, "args": call.arguments}
                if call.id is not None:
                    function_call["id"] = call.id
                call_part: dict[str, Any] = {"functionCall": function_call}
                if call.signature is not None:
                    call_part["thoughtSignature"] = call.signature
                parts.append(call_part)
            contents.append({"role": "model", "parts": parts or [{"text": ""}]})
    return ("\n\n".join(system) if system else None), contents


def _declaration(tool: ToolDefinition) -> dict[str, Any]:
    # `parametersJsonSchema` takes standard JSON Schema, which is what pydantic
    # derives for our tools; the older `parameters` field takes only an
    # OpenAPI subset and would need a schema rewrite.
    return {"name": tool.name, "description": tool.description, "parametersJsonSchema": tool.parameters}


def _response_parts(payload: dict[str, object], *, max_tokens: int) -> tuple[str, tuple[ToolCall, ...]]:
    """The answer text and function calls of the first candidate, skipping thought-summary parts.

    Raises when the token cap left neither an answer nor a call -- with a
    thinking model, usually because thoughts used it up -- since "" would read
    downstream as a refusal. A cap that cut the answer short only warns: the
    text is still the best the model produced.
    """
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        feedback = payload.get("promptFeedback")
        raise RuntimeError(f"Gemini returned no candidates (promptFeedback={feedback!r})")
    candidate = candidates[0]
    content = candidate.get("content") if isinstance(candidate, dict) else None
    parts = content.get("parts") if isinstance(content, dict) else None
    reason = candidate.get("finishReason") if isinstance(candidate, dict) else None
    valid_parts = [p for p in (parts if isinstance(parts, list) else []) if isinstance(p, dict)]
    text = "".join(
        part["text"] for part in valid_parts if isinstance(part.get("text"), str) and not part.get("thought")
    )
    calls = tuple(_parse_call(part) for part in valid_parts if "functionCall" in part)
    if reason in ("MALFORMED_FUNCTION_CALL", "UNEXPECTED_TOOL_CALL"):
        raise RuntimeError(f"Gemini rejected the model's own function call (finishReason={reason}): {candidate!r}")
    if reason == "MAX_TOKENS":
        # Thoughts can use the whole cap, and then the candidate has no parts at all.
        if not text.strip() and not calls:
            raise RuntimeError(
                f"Gemini hit max_tokens={max_tokens} before writing any answer. With a thinking model "
                "the thoughts count toward that cap: raise llm.max_tokens or lower llm.thinking_level."
            )
        logger.warning("Gemini answer truncated at max_tokens=%d", max_tokens)
    elif not isinstance(parts, list):
        raise RuntimeError(f"Gemini candidate has no content (finishReason={reason!r}): {candidate!r}")
    return text, calls


def _parse_call(part: dict[str, Any]) -> ToolCall:
    function_call = part["functionCall"]
    name = function_call.get("name") if isinstance(function_call, dict) else None
    args = function_call.get("args", {}) if isinstance(function_call, dict) else None
    if not isinstance(name, str) or not isinstance(args, dict):
        raise RuntimeError(f"Unexpected functionCall shape from Gemini: {part!r}")
    call_id = function_call.get("id")
    signature = part.get("thoughtSignature")
    return ToolCall(
        name=name,
        arguments=args,
        id=call_id if isinstance(call_id, str) else None,
        signature=signature if isinstance(signature, str) else None,
    )


def _error_details(response: httpx.Response) -> list[dict[str, object]]:
    try:
        error = response.json().get("error", {})
    except (ValueError, AttributeError):
        return []
    details = error.get("details") if isinstance(error, dict) else None
    return [d for d in details if isinstance(d, dict)] if isinstance(details, list) else []


def _is_daily_quota(response: httpx.Response) -> bool:
    """True only when a QuotaFailure violation names a per-day quota."""
    for detail in _error_details(response):
        if not str(detail.get("@type", "")).endswith("google.rpc.QuotaFailure"):
            continue
        violations = detail.get("violations")
        for violation in violations if isinstance(violations, list) else []:
            quota_id = str(violation.get("quotaId", "")) if isinstance(violation, dict) else ""
            if "PerDay" in quota_id or "Daily" in quota_id:
                return True
    return False


def _retry_delay(response: httpx.Response) -> float | None:
    """Seconds from a RetryInfo detail's `retryDelay` ("34s"), else the Retry-After header."""
    for detail in _error_details(response):
        if str(detail.get("@type", "")).endswith("google.rpc.RetryInfo"):
            match = re.fullmatch(r"(\d+(?:\.\d+)?)s", str(detail.get("retryDelay", "")))
            if match:
                return float(match.group(1))
    header = response.headers.get("retry-after", "")
    return float(header) if header.replace(".", "", 1).isdigit() else None


def _error_message(response: httpx.Response) -> str:
    try:
        error = response.json().get("error", {})
        return str(error.get("message") or error)
    except (ValueError, AttributeError):
        return response.text[:500]


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None
