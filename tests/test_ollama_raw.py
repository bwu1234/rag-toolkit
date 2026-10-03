"""The client-side port of Ollama's qwen3.8 renderer and qwen3.5 parser (`rag.llm.ollama_raw`).

The renderer cases marked "golden" are copied verbatim from Ollama v0.35.1's
own `model/renderers/qwen38_test.go` (`TestQwen38RendererMatchesReferenceFlows`),
which Ollama verifies against the model's reference Jinja template: matching
them is matching the server byte for byte. The rest pin what this repo's
agent sends -- pydantic tool schemas, JSON-encoded filter strings -- and the
property a raw transcript relies on: each prompt starts with the previous
prompt plus the model's output.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx
import pytest

from rag.llm import ollama_raw
from rag.llm.base import AssistantTurn, ChatMessage, RawCompletion, ToolCall, ToolDefinition, ToolResult
from rag.llm.ollama_llm import OllamaLLMClient, _to_wire, _tool_to_wire

XHIGH = (
    "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, "
    "consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer."
)
LOW = (
    "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the "
    "conclusion without unnecessary elaboration."
)


def _weather_tool(name: str, description: str) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "required": ["city"], "properties": {"city": {"type": "string"}}},
        },
    }


WEATHER = [_weather_tool("get_weather", "Get weather"), _weather_tool("get_uv", "Get UV index")]

TOOL_HEADER = """<|im_start|>system
# Tools

You have access to the following functions:

<tools>
{"type": "function", "function": {"name": "get_weather", "description": "Get weather", "parameters": {"type": "object", "required": ["city"], "properties": {"city": {"type": "string"}}}}}
{"type": "function", "function": {"name": "get_uv", "description": "Get UV index", "parameters": {"type": "object", "required": ["city"], "properties": {"city": {"type": "string"}}}}}
</tools>

If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>
</tool_call>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags
- Required parameters MUST be specified
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
</IMPORTANT><|im_end|>
"""

GOLDEN = [
    pytest.param(
        [{"role": "user", "content": "Hello"}], [], None,
        f"<|im_start|>system\n{XHIGH}<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n",
        id="default xhigh injects system guidance",
    ),
    pytest.param(
        [{"role": "system", "content": "Base policy."}, {"role": "developer", "content": "Use Go."},
         {"role": "user", "content": "Hello"}], [], None,
        f"<|im_start|>system\n{XHIGH}\n\nBase policy.\n\nUse Go.<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n",
        id="system and developer instructions merge",
    ),
    pytest.param(
        [{"role": "user", "content": "Hello"}], [], "low",
        f"<|im_start|>system\n{LOW}<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n",
        id="low effort injects concise guidance",
    ),
    pytest.param(
        [{"role": "user", "content": "Hello"}], [], False,
        "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        id="thinking disabled emits explicit empty block",
    ),
    pytest.param(
        [{"role": "user", "content": "First"},
         {"role": "assistant", "thinking": "Plan", "content": "<think>literal</think>\nAnswer"},
         {"role": "user", "content": "Next"}], [], None,
        f"<|im_start|>system\n{XHIGH}<|im_end|>\n<|im_start|>user\nFirst<|im_end|>\n<|im_start|>assistant\n"
        "<think>\nPlan\n</think>\n\n<think>literal</think>\nAnswer<|im_end|>\n<|im_start|>user\nNext<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n",
        id="preserves reasoning metadata without extracting content tags",
    ),
    pytest.param(
        [{"role": "developer", "content": "Use tools when requested."}, {"role": "user", "content": "Check the weather."}],
        WEATHER, True,
        TOOL_HEADER.removesuffix("<|im_end|>\n") + "\n\nUse tools when requested.<|im_end|>\n"
        "<|im_start|>user\nCheck the weather.<|im_end|>\n<|im_start|>assistant\n<think>\n",
        id="developer instruction with tools",
    ),
    pytest.param(
        [{"role": "user", "content": "Weather?"},
         {"role": "assistant", "thinking": "Need current data.", "content": "I'll check.",
          "tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "Montréal"}}}]},
         {"role": "tool", "content": '{"temp": 18}'}],
        WEATHER, True,
        TOOL_HEADER + "<|im_start|>user\nWeather?<|im_end|>\n<|im_start|>assistant\n<think>\nNeed current data.\n"
        "</think>\n\nI'll check.\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nMontréal\n</parameter>\n"
        "</function>\n</tool_call><|im_end|>\n<|im_start|>user\n<tool_response>\n{\"temp\": 18}\n</tool_response>"
        "<|im_end|>\n<|im_start|>assistant\n<think>\n",
        id="tool call result and next generation prompt",
    ),
]


@pytest.mark.parametrize(("messages", "tools", "think", "want"), GOLDEN)
def test_renderer_matches_ollamas_golden_prompts(messages, tools, think, want) -> None:
    assert ollama_raw.render(messages, tools, think) == want


def test_renderer_rejects_a_conversation_with_no_user_query() -> None:
    with pytest.raises(ValueError, match="no user query"):
        ollama_raw.render([{"role": "tool", "content": "x"}], [], None)


# -- tool schemas, as Ollama's Go structs keep them -----------------------------


def test_tool_schema_keeps_only_the_fields_ollama_decodes() -> None:
    # The shape pydantic gives the agent's `filters` argument: a $ref inside
    # anyOf, plus title/default that Ollama's ToolProperty has no field for.
    tool = {
        "type": "function",
        "function": {
            "name": "rag_search",
            "description": "Search <docs> & more.",
            "parameters": {
                "$defs": {"Range": {"title": "Range", "type": "object", "properties": {"lte": {"type": "integer"}}}},
                "properties": {
                    "query": {"title": "Query", "type": "string", "description": "What to find."},
                    "filters": {"anyOf": [{"$ref": "#/$defs/Range"}, {"type": "null"}], "default": None},
                },
                "required": ["query"],
                "type": "object",
            },
        },
    }

    line = ollama_raw.render([{"role": "user", "content": "q"}], [tool], "low").split("<tools>\n", 1)[1].split("\n", 1)[0]

    assert line == (
        '{"type": "function", "function": {"name": "rag_search", "description": "Search \\u003cdocs\\u003e '
        '\\u0026 more.", "parameters": {"type": "object", "$defs": {"Range": {"properties": {"lte": {"type": '
        '"integer"}}, "title": "Range", "type": "object"}}, "required": ["query"], "properties": {"query": '
        '{"type": "string", "description": "What to find."}, "filters": {"anyOf": [{}, {"type": "null"}]}}}}}'
    )


def test_go_number_formatting() -> None:
    assert [ollama_raw._go_v_float(v) for v in (5, 0.5, 123456, 1234567, 20250630, 1e-05, 0.0001)] == [
        "5", "0.5", "123456", "1.234567e+06", "2.025063e+07", "1e-05", "0.0001",
    ]
    assert [ollama_raw._go_json_float(v) for v in (20250630, 1e21, 1e-7, 0.000001, 2.5)] == [
        "20250630", "1e+21", "1e-7", "0.000001", "2.5",
    ]


def test_tool_call_arguments_render_as_go_prints_them() -> None:
    # A number that comes back from JSON is a float64 to Go, so `%v` prints a
    # date-sized integer in exponent form; nested objects come out key-sorted.
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "", "tool_calls": [{"function": {
            "name": "f", "arguments": {"z": 20250630, "a": {"b": 1, "a": "x<y"}, "flag": True},
        }}]},
        {"role": "tool", "content": "r"},
    ]

    prompt = ollama_raw.render(messages, [], False)

    assert (
        "<function=f>\n<parameter=z>\n2.025063e+07\n</parameter>\n<parameter=a>\n"
        '{"a":"x\\u003cy","b":1}\n</parameter>\n<parameter=flag>\ntrue\n</parameter>\n</function>'
    ) in prompt


# -- think --------------------------------------------------------------------


def test_resolve_think_falls_back_to_the_models_default_for_an_unlisted_level() -> None:
    spec = ollama_raw.ThinkingSpec.from_show({"thinking": {"values": [False, "low", "medium", "xhigh"], "default": "medium"}})

    assert ollama_raw.resolve_think("low", spec) == "low"
    assert ollama_raw.resolve_think("high", spec) == "medium"
    assert ollama_raw.resolve_think(None, spec) == "medium"
    assert ollama_raw.resolve_think(False, spec) is False


# -- the parser -----------------------------------------------------------------


def test_parse_splits_thinking_content_and_a_tool_call() -> None:
    output = (
        "Need current data.\n</think>\n\nI'll check.\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\n"
        "Montréal\n</parameter>\n</function>\n</tool_call>"
    )

    parsed = ollama_raw.parse(output, WEATHER, "low")

    assert parsed.thinking == "Need current data."
    assert parsed.content == "I'll check."
    assert parsed.tool_calls == (ollama_raw.ParsedCall("get_weather", {"city": "Montréal"}),)


def test_parse_types_parameters_by_the_tool_schema_as_ollama_does() -> None:
    tool = {"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {
        "n": {"type": "integer"},
        "x": {"type": "number"},
        "on": {"type": "boolean"},
        "tags": {"type": "array"},
        # A $ref Ollama drops leaves only "null": the value stays a string.
        "filters": {"anyOf": [{"$ref": "#/$defs/F"}, {"type": "null"}]},
        "free": {},
    }}}}
    output = (
        "</think>\n\n<tool_call>\n<function=f>\n<parameter=n>\n5\n</parameter>\n<parameter=x>\n2.0\n</parameter>\n"
        "<parameter=on>\nTrue\n</parameter>\n<parameter=tags>\n[\"a\"]\n</parameter>\n"
        "<parameter=filters>\n{\"equals\": {\"ticker\": \"DAL\"}}\n</parameter>\n"
        "<parameter=free>\nnull\n</parameter>\n</function>\n</tool_call>"
    )

    [call] = ollama_raw.parse(output, [tool], True).tool_calls

    assert call.arguments == {
        "n": 5, "x": 2, "on": True, "tags": ["a"], "filters": '{"equals": {"ticker": "DAL"}}', "free": None,
    }


def test_parse_keeps_markup_characters_inside_a_parameter() -> None:
    output = "</think>\n\n<tool_call>\n<function=calculator>\n<parameter=expression>\n(5 > 2) & 1 < 3\n</parameter>\n</function>\n</tool_call>"

    [call] = ollama_raw.parse(output, [], True).tool_calls

    assert call.arguments == {"expression": "(5 > 2) & 1 < 3"}


def test_parse_closes_thinking_at_a_tool_call_when_the_model_forgets_think_close() -> None:
    output = "Plan it.\n<tool_call>\n<function=get_uv>\n<parameter=city>\nOslo\n</parameter>\n</function>\n</tool_call>"

    parsed = ollama_raw.parse(output, WEATHER, True)

    assert (parsed.thinking, parsed.content) == ("Plan it.", "")
    assert parsed.tool_calls == (ollama_raw.ParsedCall("get_uv", {"city": "Oslo"}),)


def test_parse_returns_an_unclosed_tool_call_as_content() -> None:
    parsed = ollama_raw.parse("ok</think>\n\nSure.\n<tool_call>\n<function=get_uv>", WEATHER, True)

    assert parsed.content == "Sure.<tool_call>\n<function=get_uv>"
    assert parsed.tool_calls == ()


def test_parse_with_thinking_off_is_all_content() -> None:
    parsed = ollama_raw.parse("The answer is 4.", [], False)

    assert (parsed.thinking, parsed.content, parsed.tool_calls) == ("", "The answer is 4.", ())


def test_each_prompt_starts_with_the_previous_prompt_and_output() -> None:
    """What lets a raw transcript show only the new text: re-rendering the parsed reply reproduces it."""

    question = [{"role": "system", "content": "You answer from filings."}, {"role": "user", "content": "Weather?"}]
    first = ollama_raw.render(question, WEATHER, "low")
    output = (
        "Need current data.\n</think>\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nOslo\n"
        "</parameter>\n</function>\n</tool_call>"
    )
    parsed = ollama_raw.parse(output, WEATHER, "low")
    reply = {"role": "assistant", "content": parsed.content, "thinking": parsed.thinking, "tool_calls": [
        {"function": {"name": c.name, "arguments": c.arguments}} for c in parsed.tool_calls
    ]}

    second = ollama_raw.render([*question, reply, {"role": "tool", "content": "rain"}], WEATHER, "low")

    assert second.startswith(first + output)
    assert second[len(first + output):] == (
        "<|im_end|>\n<|im_start|>user\n<tool_response>\nrain\n</tool_response><|im_end|>\n"
        "<|im_start|>assistant\n<think>\n"
    )


# -- OllamaLLMClient(raw=True) --------------------------------------------------

_SHOW = {
    "modelfile": "FROM qwen3.8:27b-mlx\nTEMPLATE {{ .Prompt }}\nRENDERER qwen3.8\nPARSER qwen3.5\n",
    "thinking": {"values": [False, "low", "medium", "xhigh"], "default": "medium"},
}
_TOOL = ToolDefinition(
    name="get_uv", description="Get UV index",
    parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
)
_OUTPUT = "Look it up.\n</think>\n\n<tool_call>\n<function=get_uv>\n<parameter=city>\nOslo\n</parameter>\n</function>\n</tool_call>"


def _raw_client(handler, **kwargs) -> OllamaLLMClient:
    client = OllamaLLMClient(model="qwen3.8:27b-mlx", base_url="http://fake-ollama:11434", raw=True, think="low", **kwargs)
    client._client = httpx.Client(base_url=client.base_url, transport=httpx.MockTransport(handler))
    return client


def _ollama(show: dict[str, Any] = _SHOW, chat_tokens: int = 321):
    seen: dict[str, list[dict[str, Any]]] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.read()) if request.method == "POST" else {}
        seen.setdefault(request.url.path, []).append(body)
        if request.url.path == "/api/show":
            return httpx.Response(200, json=show)
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": ollama_raw.PORTED_FROM_OLLAMA})
        if request.url.path == "/api/generate":
            return httpx.Response(200, json={
                "response": _OUTPUT, "done_reason": "stop", "prompt_eval_count": 321, "eval_count": 40,
            })
        return httpx.Response(200, json={
            "message": {"role": "assistant", "content": ""}, "prompt_eval_count": chat_tokens, "eval_count": 1,
        })

    return handler, seen


def test_raw_chat_sends_the_rendered_prompt_and_parses_the_raw_output() -> None:
    handler, seen = _ollama()
    messages = [ChatMessage("system", "Be brief."), ChatMessage("user", "UV in Oslo?")]

    turn = _raw_client(handler).chat(messages, [_TOOL])

    [generate] = seen["/api/generate"]
    want = ollama_raw.render([_to_wire(m) for m in messages], [_tool_to_wire(_TOOL)], "low")
    assert (generate["prompt"], generate["raw"], generate["stream"]) == (want, True, False)
    assert "think" not in generate and "messages" not in generate
    assert turn.tool_calls == (ToolCall(name="get_uv", arguments={"city": "Oslo"}),)
    assert (turn.thinking, turn.content, turn.stop_reason) == ("Look it up.", "", "stop")
    assert turn.raw is not None
    assert (turn.raw.prompt, turn.raw.output, turn.raw.chat_prompt_tokens) == (want, _OUTPUT, 321)
    # The check: the same messages through /api/chat, one token, Ollama rendering.
    [check] = seen["/api/chat"]
    assert check["options"]["num_predict"] == 1 and check["think"] == "low"
    assert check["messages"] == [_to_wire(m) for m in messages]


def test_raw_chat_reads_the_model_once() -> None:
    handler, seen = _ollama()
    client = _raw_client(handler)
    reply = client.chat([ChatMessage("user", "UV in Oslo?")], [_TOOL])

    client.chat([ChatMessage("user", "UV in Oslo?"), reply, ToolResult(call=reply.tool_calls[0], content="3")], [_TOOL])

    assert len(seen["/api/show"]) == 1
    assert len(seen["/api/generate"]) == 2


def test_raw_chat_warns_when_the_chat_endpoint_counts_different_tokens(caplog: pytest.LogCaptureFixture) -> None:
    handler, _ = _ollama(chat_tokens=330)

    with caplog.at_level(logging.WARNING, logger="rag.llm.ollama_llm"):
        turn = _raw_client(handler).chat([ChatMessage("user", "UV in Oslo?")], [_TOOL])

    assert turn.raw is not None and turn.raw.chat_prompt_tokens == 330
    assert "drifted" in caplog.text


def test_raw_chat_refuses_a_model_with_another_renderer() -> None:
    handler, seen = _ollama(show={"modelfile": "FROM x\nRENDERER qwen3.5\nPARSER qwen3.5\n"})

    with pytest.raises(RuntimeError, match="RENDERER qwen3.8"):
        _raw_client(handler).chat([ChatMessage("user", "hi")])

    assert "/api/generate" not in seen


def test_generate_is_unaffected_by_raw_mode() -> None:
    handler, seen = _ollama()

    _raw_client(handler).generate("hi")

    assert "/api/generate" not in seen and len(seen["/api/chat"]) == 1


def test_assistant_turn_raw_is_not_sent_back() -> None:
    with_raw = AssistantTurn(content="a", raw=RawCompletion(prompt="p", output="o"))

    assert _to_wire(with_raw) == _to_wire(AssistantTurn(content="a"))
