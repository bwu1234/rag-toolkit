#!/usr/bin/env python
"""Fetch a pinned BEIR dataset and lay it out as a named corpus.

Public benchmarks plan, phase 1. The committed manifest
(``data/corpora/beir-<name>/manifest.json``, written in phase 0) is the
provenance record: this script downloads the archive it names, refuses it
unless the SHA-256 matches, and refuses every extracted file unless it matches
the hash the manifest's inventory recorded. Nothing is committed; the data is
reproduced from the manifest, as with EDGAR.

Layout, so that only passages can ever reach an index::

    data/corpora/beir-<name>/
      manifest.json            committed provenance
      documents/corpus.jsonl   the corpus directory the registry points at
      source/queries.jsonl     queries and relevance labels, outside it,
      source/qrels/<split>.tsv   read by scripts/beir_to_eval_set.py

Usage::

    python scripts/fetch_beir.py scifact          # or beir-scifact
    python scripts/fetch_beir.py fiqa --force     # re-download and overwrite

Dependencies: ``httpx`` (already a project dependency) and the standard
library's ``zipfile``/``hashlib``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import tempfile
import zipfile
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from rag.logging_config import configure_logging  # noqa: E402

logger = logging.getLogger(__name__)

CORPORA_DIR = REPO / "data" / "corpora"


class FetchError(RuntimeError):
    """The archive or a file in it does not match the pinned manifest."""


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_name(name: str) -> str:
    """``beir-scifact`` or ``scifact`` -> ``scifact``."""
    return name.removeprefix("beir-")


def expected_files(manifest: dict, root: Path) -> dict[str, tuple[Path, str]]:
    """Archive member name -> (destination, pinned sha256), from the manifest's inventory.

    The archive's top-level directory is the BEIR dataset name.
    """
    name = dataset_name(manifest["name"])
    inventory = manifest["inventory"]
    files = {
        f"{name}/corpus.jsonl": (root / "documents" / "corpus.jsonl", inventory["corpus"]["sha256"]),
        f"{name}/queries.jsonl": (root / "source" / "queries.jsonl", inventory["queries"]["sha256"]),
    }
    for split, qrels in inventory["qrels"].items():
        files[f"{name}/qrels/{split}.tsv"] = (root / "source" / "qrels" / f"{split}.tsv", qrels["sha256"])
    return files


def already_fetched(files: dict[str, tuple[Path, str]]) -> bool:
    return all(dest.exists() and sha256_of(dest) == sha for dest, sha in files.values())


def download(url: str, dest: Path) -> None:
    with httpx.stream("GET", url, follow_redirects=True, timeout=60.0) as response:
        response.raise_for_status()
        with dest.open("wb") as f:
            for block in response.iter_bytes(1 << 20):
                f.write(block)


def extract(archive: Path, files: dict[str, tuple[Path, str]]) -> None:
    """Write each expected member to its destination, verifying it before it lands.

    Members are written by fixed destination rather than archive path, so a
    crafted member name cannot escape the corpus directory. Members the
    manifest does not name are ignored and logged.
    """
    with zipfile.ZipFile(archive) as zf:
        members = set(zf.namelist())
        missing = sorted(set(files) - members)
        if missing:
            raise FetchError(f"archive lacks {', '.join(missing)}")
        ignored = sorted(m for m in members - set(files) if not m.endswith("/"))
        if ignored:
            logger.info("Ignoring %d archive member(s) the manifest does not name: %s", len(ignored), ignored)
        for member, (dest, pinned) in files.items():
            dest.parent.mkdir(parents=True, exist_ok=True)
            partial = dest.with_name(dest.name + ".partial")
            with zf.open(member) as src, partial.open("wb") as out:
                while block := src.read(1 << 20):
                    out.write(block)
            actual = sha256_of(partial)
            if actual != pinned:
                partial.unlink()
                raise FetchError(f"{member}: sha256 {actual} does not match the manifest's {pinned}")
            partial.replace(dest)
            logger.info("wrote %s (%.1f MB)", dest, dest.stat().st_size / 1e6)


def fetch(name: str, *, force: bool = False) -> Path:
    """Fetch dataset ``name`` into ``data/corpora/beir-<name>/``; returns that directory."""
    root = CORPORA_DIR / f"beir-{dataset_name(name)}"
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FetchError(f"no manifest at {manifest_path.relative_to(REPO)}; phase 0 pins one per dataset")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = expected_files(manifest, root)

    if not force and already_fetched(files):
        logger.info("%s already fetched and verified; --force re-downloads", root.relative_to(REPO))
        return root

    pinned = manifest["archive"]["sha256"]
    with tempfile.TemporaryDirectory(dir=root) as tmp:
        archive = Path(tmp) / f"{dataset_name(name)}.zip"
        logger.info("Downloading %s", manifest["source"])
        download(manifest["source"], archive)
        actual = sha256_of(archive)
        if actual != pinned:
            raise FetchError(f"archive sha256 {actual} does not match the manifest's {pinned}")
        logger.info("Archive verified (sha256 %s…)", actual[:16])
        extract(archive, files)
    return root


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch a pinned BEIR dataset as a named corpus.")
    parser.add_argument("name", help="Dataset: fiqa, scifact or nfcorpus (the beir- prefix is optional).")
    parser.add_argument("--force", action="store_true", help="Re-download and overwrite verified files.")
    args = parser.parse_args()
    configure_logging()
    try:
        root = fetch(args.name, force=args.force)
    except FetchError as exc:
        logger.error("%s", exc)
        return 1
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    logger.info(
        "Corpus %s ready: %d passages in documents/, queries and qrels (%s) in source/",
        manifest["name"],
        manifest["inventory"]["corpus"]["documents"],
        ", ".join(manifest["inventory"]["qrels"]),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
