"""stdio entrypoint: `python -m rag.mcp`.

Point an MCP client at this module to get `rag_search` and `rag_list_corpora`
over stdio. Uses the official SDK when the optional `mcp` extra is installed
and the dependency-free fallback when it isn't -- the tool contract is the
same either way, so a client cannot tell which one answered.
"""

from __future__ import annotations

import argparse
import logging
import sys

from rag.logging_config import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m rag.mcp",
        description="Serve this project's retrieval tools over MCP (stdio).",
    )
    parser.add_argument(
        "--config", default=None, help="Path to a config YAML (defaults to rag/config/config.yaml)"
    )
    parser.add_argument(
        "--log-level", default="WARNING", help="Logging level (default: WARNING)"
    )
    parser.add_argument(
        "--no-sdk",
        action="store_true",
        help="Force the dependency-free JSON-RPC transport even if the `mcp` SDK is installed",
    )
    args = parser.parse_args()

    # stderr, always: stdout is the JSON-RPC wire. Both transports also point
    # fd 1 at stderr while serving, so this is belt and braces -- but it also
    # covers anything logged before serving starts.
    configure_logging(args.log_level, stream=sys.stderr)
    logger = logging.getLogger(__name__)

    if not args.no_sdk:
        try:
            from rag.mcp.server import run_stdio
        except ImportError:
            logger.warning(
                "The `mcp` SDK is not installed; serving with the built-in JSON-RPC "
                "fallback. Install it with: pip install -e '.[mcp]'"
            )
        else:
            run_stdio(args.config)
            return

    from rag.mcp.fallback import run_stdio as run_stdio_fallback

    run_stdio_fallback(args.config)


if __name__ == "__main__":
    main()
