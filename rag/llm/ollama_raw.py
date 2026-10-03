"""Client-side port of Ollama's qwen3.8 prompt renderer and qwen3.5 output parser.

What `OllamaLLMClient(raw=True)` uses to call `/api/generate` with
`raw: true`: we render the exact prompt text the model sees, and get back the
exact text it generated -- `<think>`, `<tool_call>` and all -- which `/api/chat`
never exposes (it renders server-side and returns only the parsed pieces).

The model's Modelfile names its renderer and parser (`RENDERER qwen3.8`,
`PARSER qwen3.5`); both are Go code compiled into Ollama, not a template. This
module ports them from Ollama v0.35.1 so that, for the same messages, the
prompt is byte-identical to what `/api/chat` would build:

- renderer: `model/renderers/qwen35.go` (the `qwen35Renderer38` variant) and
  `formatToolCallArgument` / `marshalWithSpaces` beside it;
- parser: `model/parsers/qwen35.go` delegating to `qwen3coder.go`;
- tool schemas: `api/types.go` -- the server decodes our tool JSON into Go
  structs before rendering, which drops every JSON-Schema keyword it has no
  field for (`title`, `default`, `$ref`, `additionalProperties`, ...) and
  sorts the keys of free-form values (`$defs`, `items`). `normalize_tool`
  reproduces that round trip, and the parser reads parameter types from it.

Go's `encoding/json` differs from Python's in ways that change bytes: it
escapes `<`, `>` and `&`, prints float64s its own way, and `fmt`'s `%v`
prints a JSON number that came back as a float64 (e.g. 20250630) as
`2.025063e+07`. The `_go_*` helpers reproduce those.

Not ported, because nothing here sends them: images, a trailing assistant
message used as a prefill, and Ollama's front-of-conversation truncation
(which only runs when a prompt overflows `num_ctx`, and the MLX engine
ignores `num_ctx`). The live check in `OllamaLLMClient` -- the same messages'
prompt token count from `/api/chat` -- is what catches a drift from Ollama's
own rendering, e.g. after an Ollama upgrade.
"""

from __future__ import annotations

import json
import math
import re
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

#: The Ollama release this port was read from.
PORTED_FROM_OLLAMA = "0.35.1"
#: The `RENDERER` / `PARSER` a model's Modelfile must name for this port to apply.
RENDERER = "qwen3.8"
PARSER = "qwen3.5"

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
TOOL_OPEN = "<tool_call>"
TOOL_CLOSE = "</tool_call>"

_XHIGH_INSTRUCTIONS = (
    "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, "
    "consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer."
)
_LOW_INSTRUCTIONS = (
    "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the "
    "conclusion without unnecessary elaboration."
)

_TOOL_POSTAMBLE = """
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
</IMPORTANT>"""

#: Go's `unicode.IsSpace`. Python's `str.isspace` also counts \x1c-\x1f, which Go doesn't.
_GO_SPACE = "\t\n\v\f\r \x85\xa0                　"

#: The `think` value an Ollama request carries: unset, on/off, or a level name.
Think = bool | str | None


def _trim(text: str) -> str:
    return text.strip(_GO_SPACE)


def _ltrim(text: str) -> str:
    return text.lstrip(_GO_SPACE)


def _rtrim(text: str) -> str:
    return text.rstrip(_GO_SPACE)


# -- Go's encoding/json and fmt, as far as rendering needs them ---------------


def _go_string(text: str) -> str:
    """`text` as Go's `json.Marshal` quotes it: HTML-safe, non-ASCII left as UTF-8."""

    out = ['"']
    for char in text:
        code = ord(char)
        if char == '"':
            out.append('\\"')
        elif char == "\\":
            out.append("\\\\")
        elif char == "\n":
            out.append("\\n")
        elif char == "\r":
            out.append("\\r")
        elif char == "\t":
            out.append("\\t")
        elif char == "\b":
            out.append("\\b")
        elif char == "\f":
            out.append("\\f")
        elif code < 0x20 or char in "<>&  ":
            out.append(f"\\u{code:04x}")
        elif 0xD800 <= code <= 0xDFFF:
            out.append("�")  # a lone surrogate isn't valid UTF-8; Go substitutes U+FFFD
        else:
            out.append(char)
    out.append('"')
    return "".join(out)


def _shortest_digits(value: float) -> tuple[str, int]:
    """The shortest round-tripping digits of `abs(value)` and the decimal point's position.

    `value == 0.d1d2... * 10**point`, as Go's `strconv` represents it. Python's
    `repr` is the same shortest round-trip algorithm.
    """

    sign_digits_exp = Decimal(repr(abs(value))).as_tuple()
    digits = "".join(str(d) for d in sign_digits_exp.digits)
    exponent = int(sign_digits_exp.exponent)
    stripped = digits.rstrip("0")
    exponent += len(digits) - len(stripped)
    digits = stripped.lstrip("0")
    return digits, len(digits) + exponent


def _format_f(digits: str, point: int) -> str:
    if point <= 0:
        return "0." + "0" * -point + digits
    if point >= len(digits):
        return digits + "0" * (point - len(digits))
    return digits[:point] + "." + digits[point:]


def _format_e(digits: str, point: int) -> str:
    exponent = point - 1
    mantissa = digits[0] + ("." + digits[1:] if len(digits) > 1 else "")
    return f"{mantissa}e{'-' if exponent < 0 else '+'}{abs(exponent):02d}"


def _go_json_float(value: float) -> str:
    """A float64 as Go's `json.Marshal` prints it (ES6-style: `%e` only outside [1e-6, 1e21))."""

    if math.isnan(value) or math.isinf(value):
        raise ValueError(f"Go's encoding/json can't encode {value!r}")
    if value == 0:
        return "-0" if math.copysign(1, value) < 0 else "0"
    sign = "-" if value < 0 else ""
    digits, point = _shortest_digits(value)
    if abs(value) < 1e-6 or abs(value) >= 1e21:
        text = _format_e(digits, point)
        # Go trims "e-07" to "e-7" (but leaves "e+21").
        text = re.sub(r"e-0(\d)$", r"e-\1", text)
        return sign + text
    return sign + _format_f(digits, point)


def _go_v_float(value: float) -> str:
    """A float64 as `fmt.Sprintf("%v", f)` prints it: `%g` with Go's shortest-precision rule."""

    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "+Inf" if value > 0 else "-Inf"
    if value == 0:
        return "-0" if math.copysign(1, value) < 0 else "0"
    sign = "-" if value < 0 else ""
    digits, point = _shortest_digits(value)
    exponent = point - 1
    if exponent < -4 or exponent >= 6:  # strconv: shortest %g decides at precision 6
        return sign + _format_e(digits, point)
    return sign + _format_f(digits, point)


class _Ordered(dict):  # type: ignore[type-arg]
    """A mapping Go keeps in insertion order (a struct, or Ollama's ordered maps)."""


def _go_json(value: Any) -> str:
    """Compact JSON as Go's `json.Marshal` writes it.

    A plain `dict` is a Go `map[string]any` -- keys sorted; an `_Ordered` is a
    struct or an ordered map -- keys as given. Every number is a float64, as
    Go decodes one into `any`.
    """

    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str):
        return _go_string(value)
    if isinstance(value, (int, float)):
        return _go_json_float(float(value))
    if isinstance(value, _Ordered):
        return "{" + ",".join(f"{_go_string(k)}:{_go_json(v)}" for k, v in value.items()) + "}"
    if isinstance(value, Mapping):
        return "{" + ",".join(f"{_go_string(k)}:{_go_json(value[k])}" for k in sorted(value)) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_go_json(v) for v in value) + "]"
    raise TypeError(f"can't encode {type(value).__name__} as Go JSON")


def _add_json_spaces(text: str) -> str:
    """Ollama's `addJSONSpaces`: a space after every `:` and `,` outside a string."""

    out: list[str] = []
    in_string = escaped = False
    for char in text:
        out.append(char)
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char in ":,":
            out.append(" ")
    return "".join(out)


def _format_tool_call_argument(value: Any) -> str:
    """Ollama's `formatToolCallArgument`, on a value as Go decoded it from JSON."""

    if value is None:
        return "null"
    if isinstance(value, str):
        return value
    if isinstance(value, (Mapping, list, tuple)):
        return _go_json(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return _go_v_float(float(value))
    return str(value)


# -- tool schemas, as the server's Go structs keep them ----------------------


def _property_type(raw: Any) -> list[str]:
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list) and all(isinstance(t, str) for t in raw):
        return list(raw)
    return []


def _type_json(types: list[str]) -> Any:
    return types[0] if len(types) == 1 else list(types)


def _normalize_properties(raw: Any) -> _Ordered | None:
    if not isinstance(raw, Mapping):
        return None
    return _Ordered((name, _normalize_property(prop)) for name, prop in raw.items())


def _normalize_property(raw: Any) -> _Ordered:
    """`api.ToolProperty`: only its seven fields survive, in struct order, empty ones omitted."""

    prop = raw if isinstance(raw, Mapping) else {}
    out = _Ordered()
    any_of = prop.get("anyOf")
    if isinstance(any_of, list) and any_of:
        out["anyOf"] = [_normalize_property(p) for p in any_of]
    types = _property_type(prop.get("type"))
    if types:
        out["type"] = _type_json(types)
    if prop.get("items") is not None:
        out["items"] = prop["items"]
    if prop.get("description"):
        out["description"] = prop["description"]
    if isinstance(prop.get("enum"), list) and prop["enum"]:
        out["enum"] = prop["enum"]
    properties = _normalize_properties(prop.get("properties"))
    if properties is not None:
        out["properties"] = properties
    if isinstance(prop.get("required"), list) and prop["required"]:
        out["required"] = prop["required"]
    return out


def normalize_tool(tool: Mapping[str, Any]) -> _Ordered:
    """One `/api/chat` tool definition after the server's decode-into-`api.Tool` round trip."""

    function = tool.get("function") or {}
    parameters = function.get("parameters") or {}
    params = _Ordered(type=parameters.get("type") or "")
    if parameters.get("$defs") is not None:
        params["$defs"] = parameters["$defs"]
    if parameters.get("items") is not None:
        params["items"] = parameters["items"]
    if isinstance(parameters.get("required"), list) and parameters["required"]:
        params["required"] = parameters["required"]
    params["properties"] = _normalize_properties(parameters.get("properties"))

    fn = _Ordered(name=function.get("name") or "")
    if function.get("description"):
        fn["description"] = function["description"]
    fn["parameters"] = params

    out = _Ordered(type=tool.get("type") or "")
    if tool.get("items") is not None:
        out["items"] = tool["items"]
    out["function"] = fn
    return out


# -- think --------------------------------------------------------------------


@dataclass(frozen=True)
class ThinkingSpec:
    """The thinking values a model accepts and its default, as `/api/show` reports them."""

    values: tuple[bool | str, ...]
    default: bool | str

    @classmethod
    def from_show(cls, show: Mapping[str, Any]) -> ThinkingSpec | None:
        thinking = show.get("thinking")
        if not isinstance(thinking, Mapping) or not isinstance(thinking.get("values"), list):
            return None
        default = thinking.get("default")
        if not isinstance(default, (bool, str)):
            return None
        return cls(values=tuple(thinking["values"]), default=default)


def resolve_think(requested: Think, spec: ThinkingSpec | None) -> Think:
    """Ollama's `renderers.ResolveThinking`: a level the model doesn't list becomes its default."""

    if spec is None:
        return requested
    if isinstance(requested, bool):
        return requested
    if isinstance(requested, str) and requested in spec.values:
        return requested
    return spec.default


def _think_enabled(think: Think) -> bool:
    """`ThinkValue.Bool()`: on for `True` or any level name, off for `False` or unset."""

    return think is True or isinstance(think, str)


def _reasoning_instructions(think: Think) -> str:
    if think is None:
        return _XHIGH_INSTRUCTIONS
    if not _think_enabled(think):
        return ""
    level = "medium" if think is True else think
    if level == "low":
        return _LOW_INSTRUCTIONS
    if level == "medium":
        return ""
    return _XHIGH_INSTRUCTIONS


# -- the renderer ---------------------------------------------------------------


def _decode_like_go(value: Any) -> Any:
    """A JSON value as Go decodes it into `any`: maps lose their order (marshaled sorted)."""

    if isinstance(value, Mapping):
        return {k: _decode_like_go(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_decode_like_go(v) for v in value]
    return value


def _normalize_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """`normalizeQwen38Messages`: fold every system/developer message into one leading system turn."""

    msgs = [dict(m, role=str(m.get("role", "")).lower()) for m in messages]
    instructions = [m for m in msgs if m["role"] in ("system", "developer")]
    if not instructions or (len(instructions) == 1 and msgs[0]["role"] == "system"):
        return msgs
    merged = "\n\n".join(
        content for content in (_trim(m.get("content") or "") for m in instructions) if content
    )
    return [{"role": "system", "content": merged}] + [m for m in msgs if m["role"] not in ("system", "developer")]


def _is_tool_response(content: str) -> bool:
    content = _trim(content)
    return content.startswith("<tool_response>") and content.endswith("</tool_response>")


def render(messages: Sequence[Mapping[str, Any]], tools: Sequence[Mapping[str, Any]], think: Think) -> str:
    """The prompt Ollama's `qwen3.8` renderer builds for an `/api/chat` request.

    `messages` and `tools` are the request's own JSON (`role`, `content`,
    `thinking`, `tool_calls`; `{"type": "function", "function": {...}}`), and
    `think` the value after `resolve_think`.
    """

    msgs = _normalize_messages(messages)
    if not msgs:
        raise ValueError("no messages provided")
    if not any(m["role"] == "user" and not _is_tool_response(m.get("content") or "") for m in msgs):
        raise ValueError("no user query found in messages")

    is_thinking = True if think is None else _think_enabled(think)
    instructions = _reasoning_instructions(think)
    out: list[str] = []

    if tools:
        out.append(IM_START + "system\n")
        if instructions:
            out.append(instructions + "\n\n")
        out.append("# Tools\n\nYou have access to the following functions:\n\n<tools>")
        for tool in tools:
            out.append("\n" + _add_json_spaces(_go_json(normalize_tool(tool))))
        out.append(_TOOL_POSTAMBLE)
        if msgs[0]["role"] == "system":
            system = _trim(msgs[0].get("content") or "")
            if system:
                out.append("\n\n" + system)
        out.append(IM_END + "\n")
    elif msgs[0]["role"] == "system":
        system = _trim(msgs[0].get("content") or "")
        if system or instructions:
            out.append(IM_START + "system\n")
            if instructions:
                out.append(instructions)
                if system:
                    out.append("\n\n")
            out.append(system + IM_END + "\n")
    elif instructions:
        out.append(IM_START + "system\n" + instructions + IM_END + "\n")

    for i, message in enumerate(msgs):
        role = message["role"]
        content = _trim(message.get("content") or "")
        last = i == len(msgs) - 1
        prefill = last and role == "assistant"
        if prefill:
            raise ValueError("a trailing assistant message (prefill) isn't ported")

        if role == "user" or (role == "system" and i != 0):
            out.append(IM_START + role + "\n" + content + IM_END + "\n")
        elif role == "assistant":
            # qwen3.8 always renders the think block, from the `thinking` field
            # only (tags inside `content` stay literal).
            reasoning = _trim(message.get("thinking") or "")
            out.append(IM_START + "assistant\n" + THINK_OPEN + "\n" + reasoning + "\n" + THINK_CLOSE + "\n\n" + content)
            for j, call in enumerate(message.get("tool_calls") or ()):
                if j == 0:
                    if _trim(content):
                        out.append("\n\n")
                else:
                    out.append("\n")
                function: Mapping[str, Any] = call.get("function") or {}
                arguments: Mapping[str, Any] = function.get("arguments") or {}
                out.append(TOOL_OPEN + "\n<function=" + str(function.get("name", "")) + ">\n")
                for name, value in arguments.items():
                    out.append("<parameter=" + name + ">\n")
                    out.append(_format_tool_call_argument(_decode_like_go(value)))
                    out.append("\n</parameter>\n")
                out.append("</function>\n" + TOOL_CLOSE)
            out.append(IM_END + "\n")
        elif role == "tool":
            if i == 0 or msgs[i - 1]["role"] != "tool":
                out.append(IM_START + "user")
            out.append("\n<tool_response>\n" + content + "\n</tool_response>")
            if last or msgs[i + 1]["role"] != "tool":
                out.append(IM_END + "\n")
        elif role != "system":
            out.append(IM_START + role + "\n" + content + IM_END + "\n")

        if last:
            out.append(IM_START + "assistant\n")
            out.append(THINK_OPEN + "\n" if is_thinking else THINK_OPEN + "\n\n" + THINK_CLOSE + "\n\n")

    return "".join(out)


# -- the parser -----------------------------------------------------------------


@dataclass(frozen=True)
class ParsedCall:
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ParsedOutput:
    """Raw model output split the way Ollama's `/api/chat` returns it."""

    content: str
    thinking: str
    tool_calls: tuple[ParsedCall, ...]


def _overlap(text: str, tag: str) -> int:
    """The longest suffix of `text` that is a prefix of `tag`."""

    for size in range(min(len(tag), len(text)), 0, -1):
        if text.endswith(tag[:size]):
            return size
    return 0


def _split_thinking(output: str) -> tuple[str, str]:
    """`Qwen35Parser` in its thinking state, on the whole output at once: (thinking, rest)."""

    buffer = output
    trimmed = _ltrim(buffer)
    if trimmed.startswith(THINK_OPEN):
        # Some checkpoints repeat the <think> the prompt already opened.
        buffer = _ltrim(trimmed[len(THINK_OPEN):])
    elif THINK_OPEN.startswith(trimmed):
        return "", ""  # nothing but (part of) an opening tag

    close_at, tool_at = buffer.find(THINK_CLOSE), buffer.find(TOOL_OPEN)
    if tool_at != -1 and (close_at == -1 or tool_at < close_at):
        # The model can forget </think> before a tool call; the streaming parser
        # closes thinking at the first <tool_call> it sees.
        before, after = buffer.split(TOOL_OPEN, 1)
        buffer = _rtrim(before) + THINK_CLOSE + TOOL_OPEN + _ltrim(after)
    if THINK_CLOSE in buffer:
        thinking, rest = buffer.split(THINK_CLOSE, 1)
        return _rtrim(thinking), _ltrim(rest)

    # No close at all: everything is thinking, minus what the streaming parser
    # holds back and never flushes -- a partial closing tag and trailing space.
    held = max(_overlap(buffer, THINK_CLOSE), _overlap(buffer, TOOL_OPEN))
    return _rtrim(buffer[: len(buffer) - held]), ""


_TAG = re.compile(r"<(\w+)=([^>]+)>", re.ASCII)
_XML_TAG = re.compile(r'</?(?:function|parameter)(?:\s+name="[^"]*")?>')


def _xml_attr(value: str) -> str:
    """Go's `xml.EscapeText`."""

    table = {'"': "&#34;", "'": "&#39;", "&": "&amp;", "<": "&lt;", ">": "&gt;", "\t": "&#x9;", "\n": "&#xA;", "\r": "&#xD;"}
    return "".join(table.get(char, char) for char in value)


def _to_xml(raw: str) -> str:
    """`transformToXML`: `<function=x>` becomes `<function name="x">`, other text is escaped."""

    transformed = _TAG.sub(lambda m: f'<{m.group(1)} name="{_xml_attr(m.group(2))}">', raw)
    out: list[str] = []
    last = 0
    for match in _XML_TAG.finditer(transformed):
        out.append(transformed[last:match.start()].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
        out.append(match.group(0))
        last = match.end()
    out.append(transformed[last:].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
    return "".join(out)


_INT = re.compile(r"[+-]?[0-9]+")
_FLOAT = re.compile(r"[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?|[+-]?(?i:inf|infinity|nan)")


def _strict_json(raw: str) -> Any:
    def reject(name: str) -> Any:
        raise ValueError(name)

    return json.loads(raw, parse_constant=reject)


def _parse_value(raw: str, types: list[str]) -> Any:
    """`parseValue`: the parameter's text, coerced by the declared types in Ollama's precedence."""

    raw = raw.removeprefix("\n").removesuffix("\n")
    if raw.lower() == "null":
        return None
    if not types:
        return raw
    kinds = set(types)
    only = len(types) == 1
    if "boolean" in kinds:
        if raw.lower() in ("true", "false"):
            return raw.lower() == "true"
        if only:
            return False
    if "integer" in kinds:
        if _INT.fullmatch(raw) and -(2**63) <= int(raw) < 2**63:
            return int(raw)
        if only:
            return raw
    if "number" in kinds:
        if _FLOAT.fullmatch(raw):
            number = float(raw)
            if math.isfinite(number) and number == math.trunc(number):
                return int(number)
            return number
        if only:
            return raw
    for kind, shape in (("array", list), ("object", dict)):
        if kind in kinds:
            try:
                parsed = _strict_json(raw)
            except ValueError:
                parsed = None
            if isinstance(parsed, shape):
                return parsed
            if only:
                return raw
    return raw


def _parameter_types(tool: Mapping[str, Any] | None, name: str) -> list[str]:
    if tool is None:
        return []
    properties = tool["function"]["parameters"].get("properties") or {}
    prop = properties.get(name)
    if prop is None:
        return []
    if prop.get("anyOf"):
        return [t for option in prop["anyOf"] for t in _property_type(option.get("type"))]
    return _property_type(prop.get("type"))


def _parse_tool_call(raw: str, tools: Sequence[_Ordered]) -> ParsedCall:
    """`parseToolCall`: one `<function=...>` block, parameter values typed by the tool's schema."""

    xml = _to_xml(raw)
    # Go's decoder skips text before the first element and stops at its end;
    # ElementTree rejects both, so cut down to the <function> element.
    start, end = xml.find("<function"), xml.rfind("</function>")
    if start != -1 and end > start:
        xml = xml[start:end + len("</function>")]
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as exc:
        raise ValueError(f"tool call isn't well-formed: {exc}: {raw!r}") from exc
    if root.tag != "function":
        raise ValueError(f"expected element type <function> but have <{root.tag}>: {raw!r}")
    name = root.get("name", "")
    tool = next((t for t in tools if t["function"]["name"] == name), None)
    arguments: dict[str, Any] = {}
    for parameter in root.findall("parameter"):
        text = (parameter.text or "") + "".join(child.tail or "" for child in parameter)
        arguments[parameter.get("name", "")] = _parse_value(text, _parameter_types(tool, parameter.get("name", "")))
    return ParsedCall(name=name, arguments=arguments)


def _split_tool_calls(content: str, tools: Sequence[_Ordered]) -> tuple[str, tuple[ParsedCall, ...]]:
    """`Qwen3CoderParser` on the whole post-thinking text with `done`: (content, calls)."""

    text: list[str] = []
    calls: list[ParsedCall] = []
    rest = content
    while TOOL_OPEN in rest:
        before, after = rest.split(TOOL_OPEN, 1)
        text.append(_rtrim(before))
        if TOOL_CLOSE not in after:
            # Unclosed when generation stopped: Ollama hands it back as content.
            text.append(TOOL_OPEN + after)
            return "".join(text), tuple(calls)
        raw_call, rest = after.split(TOOL_CLOSE, 1)
        calls.append(_parse_tool_call(raw_call, tools))
        rest = _ltrim(rest)
    text.append(rest)
    return "".join(text), tuple(calls)


def parse(output: str, tools: Sequence[Mapping[str, Any]], think: Think) -> ParsedOutput:
    """Split raw model output into content, thinking and tool calls, as `/api/chat` would.

    `think` decides the starting state, as it did in the prompt: with
    thinking on, the output begins inside the `<think>` block the prompt opened.
    """

    normalized = [normalize_tool(tool) for tool in tools]
    if think is None or _think_enabled(think):
        thinking, rest = _split_thinking(output)
    else:
        thinking, rest = "", output
    content, calls = _split_tool_calls(rest, normalized)
    return ParsedOutput(content=content, thinking=thinking, tool_calls=calls)
