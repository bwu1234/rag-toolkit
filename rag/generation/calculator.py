"""A calculator the agent can call, so arithmetic over retrieved figures is computed, not guessed.

The agent's answers often hinge on a number the filings don't print: a
percentage change, a margin, a ratio between two companies. A model doing that
division in its reasoning gets multi-digit arithmetic wrong often enough to
matter, and nothing checks it. `evaluate` does it exactly instead.

A safe evaluator, not Python: the expression is parsed with `ast` and only
numbers, `+ - * / // % **`, parentheses, and `abs`, `round`, `min`, `max` are
allowed. Anything else -- a name, an attribute, a string -- is rejected with a
message the model can act on, as is the thousands separator it most likely
copied from a filing (`4,109` parses as a tuple in Python). Exponents and
input length are bounded, so no expression can run long or build a huge
integer.

Offered to the agent only (`agent.tools`), not over MCP: the MCP server serves
this repo's retrieval, and an outside agent brings its own arithmetic.
"""

from __future__ import annotations

import ast
import math
import operator
import re
from collections.abc import Callable
from typing import Any

from rag.generation.llm import ToolDefinition

CALCULATOR_TOOL = "calculator"

MAX_EXPRESSION_CHARS = 500
MAX_EXPONENT = 64
_MAX_MAGNITUDE = 1e100
#: A digit, a comma, three digits: a thousands separator. Inside `max(4,109)` it
#: would parse as two arguments and give a wrong answer silently, so it's
#: refused before parsing; `max(4, 109)`, with the space, is two arguments.
_THOUSANDS = re.compile(r"\d,\d{3}(?!\d)")
_THOUSANDS_MESSAGE = (
    "remove thousands separators: write 4109, not 4,109 (separate function arguments "
    "with a comma and a space, e.g. max(4, 109))"
)

CALCULATOR_DEFINITION = ToolDefinition(
    name=CALCULATOR_TOOL,
    description=(
        "Evaluate an arithmetic expression exactly and return the result. Use it for any "
        "calculation over figures from the passages -- a difference, a ratio, a percentage "
        "change, a margin -- instead of computing in your head. Supports + - * / // % **, "
        "parentheses, and abs(), round(x, digits), min(), max(). Write plain numbers: no "
        "thousands separators, currency symbols or percent signs. Example: percentage change "
        "from 2775 to 5110 is (5110 - 2775) / 2775 * 100."
    ),
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The expression to evaluate, e.g. (5110 - 2775) / 2775 * 100",
            }
        },
        "required": ["expression"],
    },
)

Number = int | float

_BINARY: dict[type[ast.operator], Callable[[Any, Any], Number]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY: dict[type[ast.unaryop], Callable[[Any], Number]] = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCTIONS: dict[str, Callable[..., Any]] = {"abs": abs, "round": round, "min": min, "max": max}


class CalculatorError(ValueError):
    """An expression the calculator won't evaluate; the message is what the model is shown."""


def evaluate(expression: str) -> Number:
    """The value of `expression`, or `CalculatorError` saying what to fix."""

    text = expression.strip()
    if not text:
        raise CalculatorError("the expression is empty")
    if len(text) > MAX_EXPRESSION_CHARS:
        raise CalculatorError(f"the expression is longer than {MAX_EXPRESSION_CHARS} characters")
    if "$" in text:
        raise CalculatorError("remove currency symbols; write plain numbers")
    if _THOUSANDS.search(text):
        raise CalculatorError(_THOUSANDS_MESSAGE)
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        # `%` is modulo here, so a percent sign usually lands as a syntax error.
        hint = "; write percentages as plain numbers, without %" if "%" in text else ""
        raise CalculatorError(f"not a valid expression ({exc.msg}){hint}") from None
    value = _eval(tree.body)
    if isinstance(value, float) and not math.isfinite(value):
        raise CalculatorError("the result is not a finite number")
    return value


def format_result(value: Number) -> str:
    """A result as the model reads it: integers exact, floats to 12 significant digits."""

    if isinstance(value, int):
        return str(value)
    if value.is_integer() and abs(value) < 1e15:
        return str(int(value))
    return format(value, ".12g")


def _eval(node: ast.expr) -> Number:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int | float):
            raise CalculatorError(f"{node.value!r} is not a number")
        return node.value
    if isinstance(node, ast.Tuple):
        raise CalculatorError(_THOUSANDS_MESSAGE)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise CalculatorError(f"exponents are limited to {MAX_EXPONENT}")
        try:
            result = _BINARY[type(node.op)](left, right)
        except ZeroDivisionError:
            raise CalculatorError("division by zero") from None
        except OverflowError:
            raise CalculatorError("the result is too large") from None
        if isinstance(result, complex):
            raise CalculatorError("the result is not a real number")
        if abs(result) > _MAX_MAGNITUDE:
            raise CalculatorError("the result is too large")
        return result
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTIONS:
            name = node.func.id if isinstance(node.func, ast.Name) else "that"
            raise CalculatorError(f"unknown function {name!r}; the functions are {', '.join(_FUNCTIONS)}")
        if node.keywords:
            raise CalculatorError("pass function arguments by position, e.g. round(x, 1)")
        args = [_eval(arg) for arg in node.args]
        try:
            value: Number = _FUNCTIONS[node.func.id](*args)
            return value
        except TypeError as exc:
            raise CalculatorError(f"{node.func.id}(): {exc}") from None
    if isinstance(node, ast.Name):
        raise CalculatorError(f"{node.id!r} is not a number; use only numbers and operators")
    raise CalculatorError(f"{type(node).__name__} isn't supported; use numbers, + - * / // % **, and parentheses")
