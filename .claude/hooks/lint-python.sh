#!/usr/bin/env bash
# PostToolUse hook: lint a just-edited Python file the same way CI will.
#
# CI runs `ruff check .`, `mypy --ignore-missing-imports rag`, and `pytest`.
# The first two are cheap enough to run per-edit, and catching them here avoids
# a red CI run for a lint error that was fixable the moment it was written.
#
# mypy is deliberately NOT run on tests/ -- CI scopes it to `rag` because the
# suite uses structurally-compatible fakes rather than ABC subclasses, which
# mypy's nominal typing flags without there being a real bug. Mirroring that
# scope here keeps the hook from reporting errors CI would never fail on.
#
# Exits 2 with findings on stderr, which Claude Code feeds back to the model.
# Any other failure (tool missing, unreadable file) exits 0: a broken linter
# must not block editing.

set -uo pipefail

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_DIR" || exit 0

# Prefer the project venv -- the hook's shell does not inherit an activated one.
if [ -x "$PROJECT_DIR/.venv/bin/ruff" ]; then
  RUFF="$PROJECT_DIR/.venv/bin/ruff"
else
  RUFF="$(command -v ruff || true)"
fi
if [ -x "$PROJECT_DIR/.venv/bin/mypy" ]; then
  MYPY="$PROJECT_DIR/.venv/bin/mypy"
else
  MYPY="$(command -v mypy || true)"
fi

file_path=$(jq -r '.tool_input.file_path // .tool_response.filePath // empty' 2>/dev/null)
[ -n "$file_path" ] || exit 0
[ -f "$file_path" ] || exit 0

# Normalize to a repo-relative path so the rag/ vs tests/ test is reliable
# whether the tool reported an absolute or relative path.
rel="${file_path#"$PROJECT_DIR"/}"

case "$rel" in
  rag/*.py|tests/*.py|rag/**/*.py|scripts/*.py) ;;
  *) exit 0 ;;
esac

findings=""

if [ -n "$RUFF" ]; then
  if ! ruff_out=$("$RUFF" check --force-exclude "$rel" 2>&1); then
    findings+="ruff:"$'\n'"$ruff_out"$'\n'
  fi
fi

# mypy only for the package, matching CI's scope (see header).
case "$rel" in
  rag/*)
    if [ -n "$MYPY" ]; then
      if ! mypy_out=$("$MYPY" --ignore-missing-imports "$rel" 2>&1); then
        findings+="mypy:"$'\n'"$mypy_out"$'\n'
      fi
    fi
    ;;
esac

if [ -n "$findings" ]; then
  printf 'Lint failures in %s (CI runs the same checks):\n%s' "$rel" "$findings" >&2
  exit 2
fi

exit 0
