"""`python -m` entrypoints of the answer runners.

`-m` runs a runner's file as `__main__`, while `rag.eval.score` imports it under
its package name, so the module exists twice. Each runner's `main` checks the
stored report's class, which only holds if `main` itself comes from the package
module -- these pin that the `__main__` block hands off to it.
"""

from __future__ import annotations

import importlib
import runpy
import warnings

import pytest


@pytest.mark.parametrize("module", ["rag.eval.answer_eval", "rag.eval.multihop_eval"])
def test_dash_m_runs_the_package_modules_main(module: str, monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[str] = []
    monkeypatch.setattr(importlib.import_module(module), "main", lambda: called.append(module) or 7)

    with warnings.catch_warnings():
        # runpy warns that the module is already imported -- the situation under test.
        warnings.simplefilter("ignore", RuntimeWarning)
        with pytest.raises(SystemExit) as exit_info:
            runpy.run_module(module, run_name="__main__")

    assert called == [module]
    assert exit_info.value.code == 7
