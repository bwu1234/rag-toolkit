"""Ollama-backed `LLMClient` adapter.

Talks to a local Ollama daemon's `/api/chat` endpoint over HTTP -- the chat
counterpart to `OllamaEmbedder`'s use of `/api/embed`. Using `/api/chat`
(messages array) rather than `/api/generate` (raw prompt string) lets the
system prompt and user prompt stay cleanly separated, matching how
instruction-tuned chat models are actually trained to be steered.

The same endpoint serves tool calling (`chat()`), so one adapter implements
both `generate` and `ToolCallingLLM.chat`.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import httpx

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

#: Share of `num_ctx` a prompt may fill before a call logs a warning -- an
#: early sign that the next, longer prompt of the conversation will overflow.
CONTEXT_WARN_FRACTION = 0.9

#: `think` as Ollama takes it: on/off, or a reasoning level for models that
#: accept one (the 27b takes `low`/`medium`/`xhigh`).
ThinkSetting = bool | str


class ContextOverflowError(RuntimeError):
    """The prompt is longer than the context window the request asked for.

    Raised instead of letting Ollama truncate. A caller holding a growing
    conversation (the agent) catches it to stop adding and answer with what it
    has, rather than carrying on with a prompt that lost its beginning.
    """


class OllamaLLMClient(ToolCallingLLM):
    """Generates chat responses via a local Ollama daemon's `/api/chat` endpoint.

    `temperature`/`max_tokens` are bound at construction (from `LLMConfig`)
    rather than threaded through `generate` -- the chat service shouldn't need
    to know provider-specific option names (`num_predict` vs `max_tokens`,
    etc.); that translation is this adapter's job.

    `num_ctx` is the context window to request. `None` sends nothing and gets
    the daemon's default, which is what the pipeline has always done. The agent
    sets it, because its prompts run ~6x the pipeline's. With it set, requests
    also send `truncate: false`: by default Ollama's llama.cpp engine cuts an
    over-long prompt to fit and reports only the tokens it kept, so the loss is
    invisible in the response. With truncation off it answers 400 instead,
    which this adapter raises as `ContextOverflowError`. (Measured on Ollama
    0.34.4: a 6.9k-token prompt under `num_ctx: 4096` came back as 2,050
    prompt tokens and a wrong answer. The MLX engine ignores per-request
    `num_ctx` and did not truncate even a 43k-token prompt.)
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        think: ThinkSetting = False,
        num_ctx: int | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.think = think
        self.num_ctx = num_ctx
        # `trust_env=False`: see `OllamaEmbedder` -- a loopback connection to
        # Ollama should never go through the system proxy.
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout, trust_env=False)

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        return self.generate_with_usage(prompt, system=system)[0]

    def generate_with_usage(self, prompt: str, *, system: str | None = None) -> tuple[str, LLMUsage | None]:
        messages: list[dict[str, Any]] = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = self._post_chat(messages)
        message = payload["message"]
        return message["content"], self._usage(payload)

    def chat(self, messages: Sequence[Message], tools: Sequence[ToolDefinition] = ()) -> AssistantTurn:
        payload = self._post_chat([_to_wire(m) for m in messages], [_tool_to_wire(t) for t in tools])
        message = payload["message"]
        thinking = message.get("thinking")
        return AssistantTurn(
            content=message["content"],
            tool_calls=tuple(_parse_tool_call(raw, payload) for raw in message.get("tool_calls") or ()),
            thinking=thinking if isinstance(thinking, str) and thinking else None,
            usage=self._usage(payload),
        )

    def _post_chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        options: dict[str, Any] = {"temperature": self.temperature, "num_predict": self.max_tokens}
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": self.think,
            "options": options,
        }
        if self.num_ctx is not None:
            options["num_ctx"] = self.num_ctx
            # Fail rather than silently drop the start of the prompt; see the
            # class docstring. Only alongside `num_ctx`, so the pipeline's
            # requests stay exactly as they were measured.
            body["truncate"] = False
        # Omitted rather than sent empty when there are none: a turn offered no
        # tools must answer in text, which is how the agent forces synthesis.
        if tools:
            body["tools"] = tools

        try:
            response = self._client.post("/api/chat", json=body)
            if response.status_code == 400 and "exceed" in response.text and "context" in response.text:
                raise ContextOverflowError(
                    f"Prompt exceeds the {self.num_ctx}-token context window requested for "
                    f"model={self.model!r}: {response.text}"
                )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"Failed to get a chat completion from Ollama at {self.base_url} "
                f"(model={self.model!r}): {exc}"
            ) from exc

        payload = response.json()
        message = payload.get("message") if isinstance(payload, dict) else None
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise RuntimeError(
                f"Unexpected response shape from Ollama /api/chat: expected a "
                f"{{'message': {{'content': str}}}} object, got {payload!r}"
            )
        return payload

    def _usage(self, payload: dict[str, Any]) -> LLMUsage:
        # Ollama reports prompt/output token counts on every non-streaming
        # response (`prompt_eval_count`, `eval_count`). Read defensively: a
        # missing count is recorded as unknown rather than as zero tokens.
        usage = LLMUsage(
            prompt_tokens=_optional_int(payload.get("prompt_eval_count")),
            completion_tokens=_optional_int(payload.get("eval_count")),
        )
        if (
            self.num_ctx is not None
            and usage.prompt_tokens is not None
            and usage.prompt_tokens >= CONTEXT_WARN_FRACTION * self.num_ctx
        ):
            logger.warning(
                "Prompt used %d of %d context tokens (model=%s); a longer one will raise "
                "ContextOverflowError rather than be truncated",
                usage.prompt_tokens,
                self.num_ctx,
                self.model,
            )
        return usage

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "OllamaLLMClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def _to_wire(message: Message) -> dict[str, Any]:
    """Translate one provider-neutral message into Ollama's `/api/chat` shape."""

    if isinstance(message, ChatMessage):
        return {"role": message.role, "content": message.content}
    if isinstance(message, ToolResult):
        # Ollama pairs a result with its call by `tool_name`.
        return {"role": "tool", "tool_name": message.call.name, "content": message.content}
    wire: dict[str, Any] = {"role": "assistant", "content": message.content}
    if message.thinking is not None:
        wire["thinking"] = message.thinking
    if message.tool_calls:
        wire["tool_calls"] = [_call_to_wire(call) for call in message.tool_calls]
    return wire


def _call_to_wire(call: ToolCall) -> dict[str, Any]:
    wire: dict[str, Any] = {"function": {"name": call.name, "arguments": call.arguments}}
    if call.id is not None:
        wire["id"] = call.id
    return wire


def _tool_to_wire(tool: ToolDefinition) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {"name": tool.name, "description": tool.description, "parameters": tool.parameters},
    }


def _parse_tool_call(raw: object, payload: object) -> ToolCall:
    """Read one entry of `message.tool_calls`, which Ollama returns with arguments as an object."""

    function = raw.get("function") if isinstance(raw, dict) else None
    name = function.get("name") if isinstance(function, dict) else None
    arguments = function.get("arguments", {}) if isinstance(function, dict) else None
    if not isinstance(name, str) or not isinstance(arguments, dict):
        raise RuntimeError(
            f"Unexpected tool call shape from Ollama /api/chat: expected "
            f"{{'function': {{'name': str, 'arguments': object}}}}, got {raw!r} in {payload!r}"
        )
    call_id = raw.get("id") if isinstance(raw, dict) else None
    return ToolCall(name=name, arguments=arguments, id=call_id if isinstance(call_id, str) else None)


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None
