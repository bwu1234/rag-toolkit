#!/usr/bin/env python
"""Pull the Ollama models the effective config names, so a fresh Ollama is ready.

Reads the models from `load_config()` -- the YAML plus any `RAG__SECTION__KEY`
environment overrides -- rather than a hardcoded list, so the docker-compose
setup (which overrides `llm.model`) and a local install both pull exactly what
they will use. Components whose provider isn't `ollama` are skipped.

Usage:
    python scripts/pull_models.py
    RAG__LLM__MODEL=qwen3.5:9b python scripts/pull_models.py

Pulling a model that is already present only re-checks its manifest, so this
is safe to run on every `docker compose up`.
"""

from __future__ import annotations

import logging
import sys

import httpx

from rag.config.settings import load_config
from rag.logging_config import configure_logging

logger = logging.getLogger(__name__)

# Model downloads are multi-GB and the non-streaming endpoint answers only once
# the pull finishes, so there is deliberately no read timeout.
_PULL_TIMEOUT = httpx.Timeout(connect=10.0, read=None, write=10.0, pool=10.0)


def ollama_models() -> dict[str, set[str]]:
    """Map each Ollama base URL in the effective config to the models it must serve."""

    config = load_config()
    wanted: dict[str, set[str]] = {}
    for component in (config.embedding, config.llm):
        if component.provider == "ollama":
            wanted.setdefault(component.base_url.rstrip("/"), set()).add(component.model)
    return wanted


def pull(base_url: str, model: str) -> None:
    logger.info("Pulling %s from %s", model, base_url)
    # trust_env=False for the same reason as the Ollama adapters: this is a
    # direct connection to a local/compose-network daemon, not the internet.
    with httpx.Client(base_url=base_url, timeout=_PULL_TIMEOUT, trust_env=False) as client:
        response = client.post("/api/pull", json={"model": model, "stream": False})
    if response.status_code != 200 or response.json().get("status") != "success":
        raise RuntimeError(f"Pulling {model} from {base_url} failed: {response.status_code} {response.text}")
    logger.info("Ready: %s", model)


def main() -> int:
    configure_logging()
    wanted = ollama_models()
    if not wanted:
        logger.info("No component uses the ollama provider; nothing to pull")
        return 0
    try:
        for base_url, models in wanted.items():
            for model in sorted(models):
                pull(base_url, model)
    except (httpx.HTTPError, RuntimeError) as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
