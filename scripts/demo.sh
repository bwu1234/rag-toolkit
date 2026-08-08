#!/usr/bin/env bash
#
# End-to-end walkthrough: ingest -> chunk -> index -> retrieve -> chat.
#
# Usage:
#   scripts/demo.sh                                  # the configured active corpus
#   scripts/demo.sh --corpus edgar                   # one corpus, isolated
#   scripts/demo.sh --corpus baseline --corpus edgar # both, pooled
#   scripts/demo.sh --question "What changed in Q3?"
#
# The document directories are resolved from `corpora` in config.yaml rather
# than hardcoded here -- an earlier version of this script gated on a literal
# `data/corpus`, which the corpora registry moved out from under it and left the
# demo failing on its first check.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CORPORA=()
QUESTION="Summarize the key ideas in this repository"

while [ $# -gt 0 ]; do
  case "$1" in
    --corpus)
      [ $# -ge 2 ] || { echo "[demo] --corpus needs a NAME" >&2; exit 2; }
      CORPORA+=("$2")
      shift 2
      ;;
    --question)
      [ $# -ge 2 ] || { echo "[demo] --question needs a QUESTION" >&2; exit 2; }
      QUESTION="$2"
      shift 2
      ;;
    -h|--help)
      sed -n '3,9p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "[demo] Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

# Flags forwarded to every rag.cli invocation below.
CLI_ARGS=()
for name in ${CORPORA+"${CORPORA[@]}"}; do
  CLI_ARGS+=(--corpus "$name")
done

# Ask the config which corpora are selected and where their documents live.
# First line is the human-readable description; the rest are directories.
if ! RESOLVED=$(python - ${CORPORA+"${CORPORA[@]}"} <<'PY'
import sys

from rag.config.settings import load_config

names = sys.argv[1:] or None
try:
    selection = load_config().corpus_selection(names)
except ValueError as exc:
    print(exc, file=sys.stderr)
    raise SystemExit(1)

print(selection.describe())
for directory in selection.document_dirs:
    print(directory)
PY
); then
  echo "[demo] Could not resolve the corpus selection (see error above)." >&2
  exit 1
fi

DESCRIPTION=$(printf '%s\n' "$RESOLVED" | head -1)
DOC_DIRS=()
while IFS= read -r line; do
  [ -n "$line" ] && DOC_DIRS+=("$line")
done < <(printf '%s\n' "$RESOLVED" | tail -n +2)

for directory in ${DOC_DIRS+"${DOC_DIRS[@]}"}; do
  if [ ! -d "$directory" ]; then
    echo "[demo] Documents directory not found: $directory" >&2
    echo "[demo] Corpora other than 'baseline' are gitignored — reproduce this one" >&2
    echo "[demo] from its manifest (e.g. python scripts/fetch_edgar.py) first." >&2
    exit 1
  fi
  if ! find "$directory" -type f ! -name '.*' 2>/dev/null | grep -q .; then
    echo "[demo] No documents in $directory" >&2
    echo "[demo] Add at least one PDF, Markdown, or text file before running the demo." >&2
    exit 1
  fi
done

echo "[demo] Corpus: $DESCRIPTION"

echo "[demo] Step 1/5: ingest a small preview"
python -m rag.cli ingest --show 3 ${CLI_ARGS+"${CLI_ARGS[@]}"}

echo "[demo] Step 2/5: chunk the corpus into preview-sized windows"
python -m rag.cli chunk --show 3 ${CLI_ARGS+"${CLI_ARGS[@]}"}

echo "[demo] Step 3/5: build the vector index"
python -m rag.cli index ${CLI_ARGS+"${CLI_ARGS[@]}"}

echo "[demo] Step 4/5: retrieve relevant passages"
python -m rag.cli retrieve "$QUESTION" ${CLI_ARGS+"${CLI_ARGS[@]}"}

echo "[demo] Step 5/5: ask a grounded question end to end"
python -m rag.cli chat "$QUESTION" ${CLI_ARGS+"${CLI_ARGS[@]}"}
