"""Live smoke test of tool calling against a real Ollama daemon.

Everything else tests the adapter against a mocked transport, which proves we
send and parse the shapes we *think* Ollama uses. This proves the daemon agrees,
through the same build path the agent will use (`build_agent_llm`): one search
turn, one tool-free answer turn. Skipped when Ollama or the model is absent.

The overflow test pins the daemon behaviour `ContextOverflowError` depends on:
llama.cpp-engine (GGUF) models truncate an over-long prompt silently unless the
request says `truncate: false`.
"""

from __future__ import annotations

import httpx
import pytest

from rag.config.settings import load_config
from rag.agent.builder import build_agent_llm
from rag.llm.base import ChatMessage, ToolDefinition, ToolResult
from rag.llm.ollama_llm import ContextOverflowError, OllamaLLMClient
from rag.observability.usage import metered

pytestmark = pytest.mark.live

SEARCH = ToolDefinition(
    name="rag_search",
    description="Search SEC filings. Returns numbered passages.",
    parameters={
        "type": "object",
        "properties": {"query": {"type": "string", "description": "Standalone query; include company and period."}},
        "required": ["query"],
    },
)


def _require_model(base_url: str, model: str) -> None:
    try:
        tags = httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=2, trust_env=False).json()
    except httpx.HTTPError:
        pytest.skip(f"Ollama not reachable at {base_url}")
    if model not in {m.get("name") for m in tags.get("models", [])}:
        pytest.skip(f"model {model!r} not pulled")


def test_tool_call_round_trip_against_real_ollama() -> None:
    config = load_config()
    llm_config = config.agent.llm or config.llm
    if llm_config.provider != "ollama":
        pytest.skip("agent model is not served by Ollama")
    _require_model(llm_config.base_url, llm_config.model)
    client = build_agent_llm(config)
    num_ctx = config.agent.num_ctx

    messages: list = [
        ChatMessage("system", "Search with rag_search before answering questions about filings."),
        ChatMessage("user", "What was Apple's total net sales in fiscal 2025?"),
    ]
    with metered() as meter:
        turn = client.chat(messages, [SEARCH])
        assert turn.tool_calls, f"expected a search, got text: {turn.content!r}"
        call = turn.tool_calls[0]
        assert call.name == "rag_search"
        assert isinstance(call.arguments.get("query"), str) and call.arguments["query"].strip()

        messages += [turn, ToolResult(call, "Passage [1] (source: AAPL_10-K_2025-09-27.md): Total net sales were $416.2 billion.")]
        answer = client.chat(messages)  # no tools offered: must answer in text

    assert answer.tool_calls == ()
    assert answer.content.strip()
    assert meter.calls == 2
    for t in (turn, answer):
        # Ollama truncates past num_ctx silently, so a reported prompt at or
        # above the window would mean the request's num_ctx never took effect.
        assert t.usage is not None and t.usage.prompt_tokens is not None
        assert 0 < t.usage.prompt_tokens < num_ctx


#: A GGUF build, served by Ollama's llama.cpp engine: the one that truncates.
#: The MLX builds the config ships with ignore per-request `num_ctx`.
GGUF_MODEL = "qwen3.5:9b"


def test_prompt_over_num_ctx_raises_instead_of_being_truncated() -> None:
    base_url = load_config().llm.base_url
    _require_model(base_url, GGUF_MODEL)
    client = OllamaLLMClient(GGUF_MODEL, base_url, num_ctx=2048, max_tokens=5, timeout=300)
    too_long = " ".join(f"item{i} is blue." for i in range(1000))  # ~6.9k tokens

    with pytest.raises(ContextOverflowError):
        client.chat([ChatMessage("user", too_long)])


def test_a_request_cut_off_by_the_deadline_returns_on_time_and_frees_the_model() -> None:
    """The cancellation `rag.deadline` advertises for Ollama, against the real daemon.

    Two checks. The caller gets `DeadlineExceeded` close to the deadline, not
    when the generation would have finished. And a follow-up request is answered
    promptly: Ollama stops generating when its client disconnects, so the
    abandoned 2,048-token generation isn't still occupying the model. That
    second check can't fail falsely, but with parallel request slots
    (`OLLAMA_NUM_PARALLEL` > 1) it passes even if generation continued, so it
    is evidence, not proof; the daemon's source is (`mlxrunner/pipeline.go`).
    """

    import time

    from rag.deadline import DeadlineExceeded, deadline_scope

    config = load_config()
    if config.llm.provider != "ollama":
        pytest.skip("llm is not served by Ollama")
    _require_model(config.llm.base_url, config.llm.model)
    client = OllamaLLMClient(config.llm.model, config.llm.base_url, max_tokens=2048, timeout=120.0)
    long_job = [ChatMessage("user", "Count from 1 to 3000, one number per line, with no other text.")]
    client.chat([ChatMessage("user", "Say ok.")])  # loaded, so the deadline measures generation only

    started = time.monotonic()
    with deadline_scope(time.monotonic() + 3.0, label="turn"), pytest.raises(DeadlineExceeded):
        client.chat(long_job)
    assert time.monotonic() - started < 5.0

    short = OllamaLLMClient(config.llm.model, config.llm.base_url, max_tokens=1, timeout=120.0)
    started = time.monotonic()
    short.chat([ChatMessage("user", "Say ok.")])
    assert time.monotonic() - started < 10.0
