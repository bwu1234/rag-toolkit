"""The agent's calculator (`rag.agent.calculator`): exact where it answers, explicit where it won't.

What these pin: arithmetic a filing question needs comes out right, and every
input a model plausibly copies from a filing -- thousands separators, `$`, `%`
-- is refused with a message rather than evaluated into a wrong number.
"""

from __future__ import annotations

import pytest

from rag.agent.calculator import CalculatorError, evaluate, format_result


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("(5110 - 2775) / 2775 * 100", "84.1441441441"),
        ("round((4109 - 2458) / 2458 * 100, 1)", "67.2"),
        ("10 / 4", "2.5"),
        ("10 // 4", "2"),
        ("10 % 4", "2"),
        ("2 ** 10", "1024"),
        ("-(3 - 5)", "2"),
        ("abs(-1.5)", "1.5"),
        ("max(1, 2, 3) - min(4, 5)", "-1"),
        ("0.1 + 0.2", "0.3"),
        ("4.0 * 2", "8"),
    ],
)
def test_evaluates_arithmetic(expression: str, expected: str) -> None:
    assert format_result(evaluate(expression)) == expected


@pytest.mark.parametrize(
    ("expression", "message"),
    [
        ("4,109 - 2,458", "thousands separators"),
        ("max(4,109)", "thousands separators"),  # would otherwise be max(4, 109) = 109
        ("$5 + 1", "currency symbols"),
        ("67% * 2", "without %"),
        ("1 / 0", "division by zero"),
        ("2 ** 64 ** 64", "too large"),
        ("2 ** 65", "exponents are limited"),
        ("10 ** 60 * 10 ** 60", "too large"),
        ("(-8) ** 0.5", "not a real number"),
        ("x + 1", "'x' is not a number"),
        ("__import__('os')", "unknown function"),
        ("(1).real", "isn't supported"),
        ("round(x=1)", "by position"),
        ("'a' * 3", "is not a number"),
        ("True + 1", "is not a number"),
        ("min()", r"min\(\): min expected"),
        ("", "empty"),
        ("1 +", "not a valid expression"),
        ("1" * 501, "longer than"),
    ],
)
def test_refuses_with_a_message_the_model_can_act_on(expression: str, message: str) -> None:
    with pytest.raises(CalculatorError, match=message):
        evaluate(expression)
