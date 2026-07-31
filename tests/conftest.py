"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from rag.config.settings import RagConfig


@pytest.fixture()
def default_config() -> RagConfig:
    """A RagConfig built entirely from defaults — no file I/O, safe for unit tests."""

    return RagConfig()
