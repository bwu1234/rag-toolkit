#!/usr/bin/env python
"""Fetch the pinned MuSiQue v1.0 archive and unpack the MuSiQue-Ans splits this repo uses.

MuSiQue follow-on, phase B (docs/public-benchmarks-plan.md). The committed
manifest (``data/corpora/musique-ans/manifest.json``, phase A) is the
provenance record: the archive must match its SHA-256, and each extracted
file must match the hash its inventory recorded. Only the Ans train and dev
splits are unpacked: test labels are hidden, and MuSiQue-Full can't be run
pooled (see the manifest's ``decisions``). Nothing here is committed;
``scripts/musique_to_eval_set.py`` builds the corpora and eval sets from it.

    data/corpora/musique-ans/
      manifest.json                          committed provenance
      source/musique_ans_v1.0_{train,dev}.jsonl

Usage::

    python scripts/fetch_musique.py           # download, verify, unpack
    python scripts/fetch_musique.py --force   # re-download and overwrite
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from fetch_beir import FetchError, download, sha256_of  # noqa: E402

from rag.logging_config import configure_logging  # noqa: E402

logger = logging.getLogger(__name__)

ROOT = REPO / "data" / "corpora" / "musique-ans"
MANIFEST = ROOT / "manifest.json"
SOURCE = ROOT / "source"
#: The splits unpacked; the rest of the archive is never read after the inventory.
FILES = ("musique_ans_v1.0_train.jsonl", "musique_ans_v1.0_dev.jsonl")


def expected_files(manifest: dict) -> dict[str, tuple[Path, str]]:
    """Archive member -> (destination, pinned sha256)."""
    pinned = manifest["inventory"]["files"]
    return {f"data/{name}": (SOURCE / name, pinned[name]["sha256"]) for name in FILES}


def fetch(*, force: bool = False) -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    files = expected_files(manifest)
    if not force and all(dest.exists() and sha256_of(dest) == sha for dest, sha in files.values()):
        logger.info("Already fetched and verified: %s", SOURCE)
        return
    SOURCE.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / manifest["archive"]["file"]
        logger.info("Downloading %s (%d bytes)", manifest["archive"]["file"], manifest["archive"]["bytes"])
        download(manifest["download"], archive)
        actual = sha256_of(archive)
        if actual != manifest["archive"]["sha256"]:
            raise FetchError(f"Archive sha256 {actual} does not match the manifest's {manifest['archive']['sha256']}")
        with zipfile.ZipFile(archive) as zf:
            for member, (dest, sha) in files.items():
                # Written by fixed destination, never by member path, so a crafted
                # name can't escape the corpus directory; verified before it lands.
                partial = dest.with_suffix(".partial")
                with zf.open(member) as src, partial.open("wb") as out:
                    while block := src.read(1 << 20):
                        out.write(block)
                if (got := sha256_of(partial)) != sha:
                    partial.unlink()
                    raise FetchError(f"{member}: sha256 {got} does not match the manifest's {sha}")
                partial.replace(dest)
                logger.info("Verified %s", dest.relative_to(REPO))


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--force", action="store_true", help="Re-download and overwrite")
    args = parser.parse_args()
    configure_logging()
    try:
        fetch(force=args.force)
    except FetchError as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
