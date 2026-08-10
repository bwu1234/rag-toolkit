"""Centralized logging setup.

Call `configure_logging()` once at process start (CLI entrypoints, the FastAPI
app, etc.). Modules elsewhere should just do `logging.getLogger(__name__)`.
"""

from __future__ import annotations

import logging
import sys
from typing import TextIO

_CONFIGURED = False

DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def configure_logging(level: int | str = logging.INFO, stream: TextIO | None = None) -> None:
    """Idempotently configure the root logger with a single stream handler.

    `stream` defaults to stdout, which is right for the CLI, API, and UI. The
    MCP stdio server passes stderr instead: there, stdout carries the JSON-RPC
    protocol and a single log line written to it is a parse error that ends
    the session.
    """

    global _CONFIGURED
    if _CONFIGURED:
        return

    handler = logging.StreamHandler(stream=stream if stream is not None else sys.stdout)
    handler.setFormatter(logging.Formatter(DEFAULT_FORMAT))

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)

    _CONFIGURED = True
