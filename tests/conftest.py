"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from rag.config.settings import RagConfig


@pytest.fixture()
def default_config() -> RagConfig:
    """A RagConfig built entirely from defaults — no file I/O, safe for unit tests."""

    return RagConfig()


@pytest.fixture(autouse=True)
def _runs_stay_in_tmp(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point every loaded module's default runs directory at a temp folder.

    The eval runners write a run folder unless told otherwise, and their
    default is the real ``data/eval/runs``. A test that forgets ``--runs-dir``
    would otherwise leave fake runs among measured ones. Modules imported
    ``RUNS_DIR`` by name, so each loaded copy is patched, not only the original.
    """
    import sys

    from rag.eval import runs

    real, tmp = runs.RUNS_DIR, tmp_path_factory.mktemp("runs")
    for module in list(sys.modules.values()):
        if getattr(module, "RUNS_DIR", None) == real:
            monkeypatch.setattr(module, "RUNS_DIR", tmp)
