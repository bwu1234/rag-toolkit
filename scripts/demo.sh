#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ ! -d "data/corpus" ]; then
  echo "[demo] data/corpus directory not found. Add documents before running the index step."
  exit 1
fi

if ! find "data/corpus" -type f 2>/dev/null | grep -q .; then
  echo "[demo] data/corpus is empty. Add at least one PDF, Markdown, or text document before running the demo."
  exit 1
fi

echo "[demo] Step 1/5: ingest a small preview"
python -m rag.cli ingest --show 3

echo "[demo] Step 2/5: chunk the corpus into preview-sized windows"
python -m rag.cli chunk --show 3

echo "[demo] Step 3/5: build the vector index"
python -m rag.cli index

echo "[demo] Step 4/5: retrieve relevant passages"
python -m rag.cli retrieve "Summarize the key ideas in this repository"

echo "[demo] Step 5/5: ask a grounded question end to end"
python -m rag.cli chat "Summarize the key ideas in this repository"
