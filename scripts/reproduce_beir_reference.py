#!/usr/bin/env python
"""Recreate the pinned Pyserini reference runs and score them (public benchmarks plan, phase 3).

Runs **inside the isolated reference environment**, never the repo's own
(``docs/beir-reference-protocol.md``, "Recreating the reference environment"):
Pyserini 2.4.0, Java 21 and ``faiss-cpu`` are reproduction dependencies, not
repo dependencies. Standard library only, so it needs nothing from ``rag``.

For each dataset and reference system it runs the exact 2CR search command
recorded in ``data/corpora/beir-<name>/manifest.json`` (with ``--output``
redirected), then the manifest's ``trec_eval`` commands, and compares the
result with the pinned score and the frozen tolerance. It writes::

    data/benchmarks/reference/<dataset>/<system>.trec    the full run
    data/benchmarks/reference/<dataset>/results.json     scores, commands,
                                                         index identity, environment

Pyserini downloads each prebuilt index to ``~/.cache/pyserini/indexes`` and
checks it against the md5 the manifest pins; the index directory name carries
that md5, and it is recorded.

Usage::

    .venv-pyserini/bin/python scripts/reproduce_beir_reference.py scifact nfcorpus fiqa
    .venv-pyserini/bin/python scripts/reproduce_beir_reference.py scifact --system bm25-flat

On macOS, ``faiss-cpu`` and ``torch`` each bundle an OpenMP runtime, and the
dense search dies with SIGSEGV in ``libomp`` unless ``OMP_NUM_THREADS=1``.
Thread counts don't change results; the value is recorded.

``JAVA_HOME`` must point at a Java 21 runtime (Homebrew's ``openjdk@21``:
``$(brew --prefix openjdk@21)/libexec/openjdk.jdk/Contents/Home``). Exit status 1 if any cell is
outside the tolerance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shlex
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
CORPORA_DIR = REPO / "data" / "corpora"
OUT_DIR = REPO / "data" / "benchmarks" / "reference"
INDEX_CACHE = Path(os.environ.get("PYSERINI_CACHE", Path.home() / ".cache" / "pyserini")) / "indexes"

#: trec_eval measure name -> the manifest's metric name.
MEASURES = {"ndcg_cut_10": "nDCG@10", "recall_100": "R@100"}


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def with_output(command: str, output: Path) -> list[str]:
    """The manifest command as argv, writing its run to ``output``, run by this interpreter."""
    argv = shlex.split(command)
    if argv[:2] != ["python", "-m"]:
        raise ValueError(f"unexpected reference command: {command}")
    argv[0] = sys.executable
    argv[argv.index("--output") + 1] = str(output)
    return argv


def trec_eval(command: str, run: Path) -> dict[str, float]:
    argv = shlex.split(command.replace("RUN", str(run)))
    argv[0] = sys.executable
    out = subprocess.run(argv, capture_output=True, text=True, check=True).stdout
    scores = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[1] == "all" and parts[0] in MEASURES:
            scores[MEASURES[parts[0]]] = float(parts[2])
    return scores


def index_identity(name: str, md5: str) -> dict[str, object]:
    matches = sorted(INDEX_CACHE.glob(f"*{md5}*"))
    if not matches:
        return {"name": name, "md5": md5, "cache_dir": None}
    return {"name": name, "md5": md5, "cache_dir": str(matches[0]), "downloaded": True}


def environment() -> dict[str, str]:
    java = subprocess.run(
        [str(Path(os.environ["JAVA_HOME"]) / "bin" / "java"), "-version"], capture_output=True, text=True
    ).stderr.splitlines()[0]
    versions = {pkg: metadata.version(pkg) for pkg in ("pyserini", "faiss-cpu", "torch", "transformers")}
    return {
        **versions,
        "java": java,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", ""),
    }


def reproduce(dataset: str, systems: list[str] | None) -> bool:
    manifest = json.loads((CORPORA_DIR / f"beir-{dataset}" / "manifest.json").read_text(encoding="utf-8"))
    reference = manifest["reference"]
    tolerance = reference["tolerance_abs"]
    out_dir = OUT_DIR / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.json"
    results: dict[str, Any] = (
        json.loads(results_path.read_text(encoding="utf-8")) if results_path.exists() else {"runs": {}}
    )
    results.update(dataset=dataset, tolerance_abs=tolerance, environment=environment())

    ok = True
    for system, spec in reference["runs"].items():
        if systems and system not in systems:
            continue
        run_path = out_dir / f"{system}.trec"
        argv = with_output(spec["command"], run_path)
        print(f"[{dataset}/{system}] {shlex.join(argv[1:])}", flush=True)
        start = time.monotonic()
        subprocess.run(argv, check=True)
        seconds = time.monotonic() - start

        scores: dict[str, float] = {}
        for command in reference["eval_commands"]:
            scores.update(trec_eval(command, run_path))
        cells = {}
        for metric in MEASURES.values():
            diff = scores[metric] - spec[metric]
            cells[metric] = {
                "reproduced": scores[metric],
                "published": spec[metric],
                "diff": round(diff, 4),
                "within_tolerance": abs(diff) <= tolerance + 1e-9,
            }
            ok &= cells[metric]["within_tolerance"]
            print(f"  {metric}: {scores[metric]:.4f} vs published {spec[metric]:.4f} ({diff:+.4f})", flush=True)
        results["runs"][system] = {
            "command": spec["command"],
            "argv": argv[1:],
            "eval_commands": reference["eval_commands"],
            "index": index_identity(spec["index"]["name"], spec["index"]["md5"]),
            "run_file": str(run_path.relative_to(REPO)),
            "run_sha256": sha256_of(run_path),
            "search_seconds": round(seconds, 1),
            "scores": cells,
        }
        results_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("datasets", nargs="+", help="fiqa, scifact, nfcorpus (with or without beir-)")
    parser.add_argument("--system", action="append", help="only this reference system (repeatable)")
    args = parser.parse_args()
    if "JAVA_HOME" not in os.environ:
        parser.error("set JAVA_HOME to a Java 21 runtime")
    ok = all([reproduce(re.sub(r"^beir-", "", d), args.system) for d in args.datasets])
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
