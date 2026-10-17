"""What produced an eval result: the config, the data, the index and the code.

A metric with no record of what produced it can't be reproduced or safely
compared, and the record has to fail closed: the old results fingerprint
(`scripts/run_matrix.py`'s ``FINGERPRINTED``) hashed an allowlist of fields, so
every field added since was silently left out and runs differing only in, say,
``chunking.strategy`` looked comparable. Here the identities are:

* `config_hash` -- the whole effective `RagConfig` minus `CONFIG_HASH_DENYLIST`,
  fields shown not to affect what an eval measures, each with its reason. A new
  field counts until someone argues otherwise; ``tests/test_provenance.py``
  fails until every field is classified one way or the other.
* `dataset_hash` -- every sample in full, extra fields included (``tier``
  picks the refusal rubric, ``parts`` defines a multi-hop judgment), and
  `selection_hash` -- the ordered ids actually evaluated, so a ``--limit``
  run can't pass for a full one.
* `verify_index` -- the index recipe and corpus hashes, the build id the index
  manifest records, and a check of the live index against the corpus
  (`rag.index_manifest`, `rag.index_report.inspect_index`).
* `code_version` -- the commit plus a hash of uncommitted changes.

Hash equality is a reuse condition (may this checkpoint be resumed?), not the
rule for whether two variants may be compared: variants differ in config on
purpose. See docs/eval-harness-plan.md, "Provenance and identity".
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import subprocess
import types
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Union, get_args, get_origin

from pydantic import BaseModel

from rag.chunking.models import Chunk
from rag.config.settings import CorpusSelection, LLMConfig, RagConfig
from rag.eval.dataset import EvalDataset
from rag.hashing import stable_hash
from rag.index_manifest import (
    corpus_hash,
    index_hash,
    index_manifest_path,
    index_recipe,
    inventory_digest,
)
from rag.index_report import inspect_index
from rag.ingestion.models import Document

logger = logging.getLogger(__name__)

_REPO = Path(__file__).resolve().parents[2]

#: Config leaves left out of `config_hash`, each with why it can't change what
#: an eval measures. ``*`` matches any key of a mapping (a corpus name). Keep
#: this short: an excluded field that does matter makes stale work look
#: reusable, while an included one that doesn't only costs a rerun. Endpoints,
#: timeouts and concurrency stay in -- they can change the backend, the
#: failures or the measured latency.
CONFIG_HASH_DENYLIST: dict[str, str] = {
    "observability.turn_log.provider": "Whether finished turns are also appended to a log; "
    "the answer is complete before the sink sees it.",
    "observability.turn_log.path": "Where that log is written.",
    "observability.turn_log.otel.endpoint": "Where finished turns are also exported as traces; "
    "the answer is complete before the sink sees it, and the eval runners pass no sink.",
    "observability.turn_log.otel.protocol": "How those traces are sent.",
    "observability.turn_log.otel.service_name": "The name those traces are filed under.",
    "observability.turn_log.otel.capture_content": "Whether those traces carry text.",
    "llm.daily_request_log": "Path of the daily quota ledger, which decides whether a call may "
    "start, never what it returns.",
    "agent.llm.daily_request_log": "As llm.daily_request_log.",
    "eval.judge.daily_request_log": "As llm.daily_request_log.",
    "llm.api_key_env": "The name of the environment variable holding a key, not a setting of "
    "the model; provider and model are hashed.",
    "agent.llm.api_key_env": "As llm.api_key_env.",
    "eval.judge.api_key_env": "As llm.api_key_env.",
    "corpora.registry.*.description": "Free text for people; nothing reads it.",
    "chunking.contextual.concurrency": "How fast contexts are generated at index time; what they "
    "say is covered by the index identity.",
    "chunking.contextual.cache": "Whether generated contexts are reused across rebuilds; covered "
    "by the index identity like concurrency.",
}


def digest(value: Any) -> str:
    """Short, stable hash of any JSON-serializable value, as checkpoint fingerprints use it.

    Kept byte-for-byte as it was in `rag.eval.checkpoint`, so checkpoints and
    code versions recorded before this module existed still compare equal.
    New identities use `rag.hashing.stable_hash`.
    """
    blob = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def code_version() -> str:
    """The git commit, plus a hash of uncommitted changes to the code; "unknown" outside git."""
    try:
        commit = _git("rev-parse", "HEAD")
        diff = _git("diff", "HEAD", "--", "rag", "scripts")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{commit[:12]}+{digest(diff)}" if diff else commit[:12]


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=_REPO, capture_output=True, text=True, check=True
    ).stdout.strip()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _nested_model(annotation: Any) -> type[BaseModel] | None:
    """The model a field holds directly or optionally (``LLMConfig | None``), if any."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    if get_origin(annotation) in (Union, types.UnionType):
        models = [a for a in get_args(annotation) if isinstance(a, type) and issubclass(a, BaseModel)]
        return models[0] if len(models) == 1 else None
    return None


def config_leaves(model: type[BaseModel] = RagConfig, prefix: str = "") -> Iterator[str]:
    """Every leaf setting of ``model``'s schema as a dotted path, ``*`` for mapping keys."""
    for name, field in model.model_fields.items():
        path = f"{prefix}{name}"
        annotation = field.annotation
        if (nested := _nested_model(annotation)) is not None:
            yield from config_leaves(nested, f"{path}.")
        elif get_origin(annotation) is dict and (value := _nested_model(get_args(annotation)[1])):
            yield from config_leaves(value, f"{path}.*.")
        else:
            yield path


def _drop(tree: Any, parts: list[str]) -> None:
    """Delete the leaf at ``parts`` from a dumped config, wherever it exists."""
    if not isinstance(tree, dict):
        return  # an unset optional model (`agent.llm: null`)
    head, rest = parts[0], parts[1:]
    keys = list(tree) if head == "*" else [head] if head in tree else []
    for key in keys:
        if rest:
            _drop(tree[key], rest)
        else:
            del tree[key]


def hashed_config(config: RagConfig) -> dict[str, Any]:
    """``config`` as JSON, minus `CONFIG_HASH_DENYLIST`: exactly what `config_hash` covers."""
    return without_denylisted(config.model_dump(mode="json"))


#: Optional sections whose every leaf is denylisted, dropped whole: otherwise the
#: section's bare presence (`null`, or `{}` once its leaves go) would enter the
#: hash, and adding one would re-key every config hashed before it.
DENYLISTED_SECTIONS = ("observability.turn_log.otel",)


def without_denylisted(dump: dict[str, Any]) -> dict[str, Any]:
    """A copy of a dumped config (as a run manifest stores it) minus `CONFIG_HASH_DENYLIST`."""
    dump = copy.deepcopy(dump)
    for path in (*CONFIG_HASH_DENYLIST, *DENYLISTED_SECTIONS):
        _drop(dump, path.split("."))
    return dump


def config_hash(config: RagConfig) -> str:
    return stable_hash(hashed_config(config))


def judge_hash(judge: LLMConfig) -> str:
    """The judge's settings, hashed the same way; it is chosen apart from the row config."""
    return stable_hash(judge.model_dump(mode="json"))


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


def sample_dicts(dataset: EvalDataset) -> list[dict[str, Any]]:
    """Every sample in its canonical `EvalSample.to_dict` form; refuses duplicate ids."""
    samples = [sample.to_dict() for sample in dataset]
    counts = Counter(sample["id"] for sample in samples)
    if duplicates := sorted(sample_id for sample_id, n in counts.items() if n > 1):
        raise ValueError(f"Eval set has duplicate sample ids: {', '.join(duplicates)}")
    return samples


def dataset_hash(dataset: EvalDataset) -> str:
    """Hash of every sample in full, in file order; refuses duplicate ids.

    Hashes the canonical form `EvalSample.to_dict` writes, extra fields
    included, so a reformatted but unchanged file keeps its hash.
    """
    return stable_hash(sample_dicts(dataset))


def selection_hash(sample_ids: list[str]) -> str:
    """Hash of the ordered ids a run evaluated."""
    return stable_hash(list(sample_ids))


def write_dataset_snapshot(directory: Path, dataset: EvalDataset) -> Path:
    """Save ``dataset`` as ``<directory>/<dataset_hash>.json`` unless already there.

    Named by content, so every row recorded against the same labels shares one
    file, and a label fix produces a new one instead of rewriting history.
    """
    samples = sample_dicts(dataset)
    path = directory / f"{stable_hash(samples)}.json"
    if not path.exists():
        directory.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(samples, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IndexProvenance:
    """Which index a run measured, and whether that is the index its config describes.

    Attributes:
        index_hash: The recipe the config describes (`rag.index_manifest.index_recipe`).
        corpus_hash: The corpus as loaded and cleaned now.
        build_id: The index manifest's build id, when it has a complete one.
        conflicts: Evidence the index does not hold what the config and corpus
            describe: a different recipe, a changed corpus, missing or stale
            chunks. A run with conflicts measures something else, and the
            matrix scripts refuse it.
        unverified: Why the index can't be confirmed when nothing conflicts --
            typically an index built before build records existed.
    """

    index_hash: str
    corpus_hash: str
    build_id: str | None
    conflicts: tuple[str, ...] = ()
    unverified: str | None = None

    @property
    def verified(self) -> bool:
        return self.build_id is not None and not self.conflicts and self.unverified is None

    def as_record(self) -> dict[str, Any]:
        return {
            "index_hash": self.index_hash,
            "corpus_hash": self.corpus_hash,
            "index_build_id": self.build_id,
            "verified": self.verified,
            **({"conflicts": list(self.conflicts)} if self.conflicts else {}),
            **({"unverified": self.unverified} if self.unverified else {}),
        }


def flatten(tree: Any, prefix: str = "") -> dict[str, Any]:
    """A nested mapping as ``{"dotted.path": leaf}``."""
    if not isinstance(tree, dict):
        return {prefix.rstrip("."): tree}
    flat: dict[str, Any] = {}
    for key, value in tree.items():
        flat.update(flatten(value, f"{prefix}{key}."))
    return flat


def _recipe_differences(stored: dict[str, Any], current: dict[str, Any]) -> list[str]:
    old, new = flatten(stored), flatten(current)
    return [f"{key}: {old.get(key)!r} -> {new.get(key)!r}"
            for key in sorted(old.keys() | new.keys()) if old.get(key) != new.get(key)]


def verify_index(
    config: RagConfig, selection: CorpusSelection, documents: list[Document], chunks: list[Chunk]
) -> IndexProvenance:
    """Check that the built index for ``selection`` holds what ``config`` would build from ``documents``.

    ``documents`` and ``chunks`` are `rag.ingestion.corpora.chunk_selected_corpora`'s
    output for ``config``: what `index` would write now.
    """
    expected_index = index_hash(config, selection)
    expected_corpus = corpus_hash(config, selection, documents)
    state = inspect_index(config, selection, chunks)
    if state is None:
        return IndexProvenance(
            expected_index, expected_corpus, None,
            conflicts=(f"no index is built for {selection.describe()} at {selection.index_dir}",),
        )

    conflicts: list[str] = []
    if state.missing or state.stale or state.changed:
        conflicts.append(
            f"the index differs from the corpus: {state.missing} chunk(s) missing, "
            f"{state.stale} stale, {state.changed} with changed text"
        )
    conflicts += [f"manifest {line}" for line in state.manifest_differences]

    build = state.build
    if build is None:
        return IndexProvenance(
            expected_index, expected_corpus, None, tuple(conflicts),
            unverified=(
                f"{index_manifest_path(selection.index_dir, selection.slug).name} has no build "
                "record; run `python -m rag.cli index` with this config to record one (it "
                "re-embeds nothing when the index is current)"
            ),
        )
    if build["status"] != "complete":
        return IndexProvenance(
            expected_index, expected_corpus, None, tuple(conflicts),
            unverified=f"the last `rag.cli index` run is {build['status']}; re-run it",
        )
    if build["index_hash"] != expected_index:
        conflicts.append(
            "the index was built from a different recipe: "
            + "; ".join(_recipe_differences(build["recipe"] or {}, index_recipe(config, selection)))
        )
    if build["corpus_hash"] != expected_corpus:
        conflicts.append("the corpus has changed since the index was built")
    elif build["inventory_digest"] != inventory_digest(chunks):
        # Same recipe and corpus but different chunks: the chunker's code changed.
        conflicts.append("the chunks the corpus produces now differ from the build's")
    return IndexProvenance(expected_index, expected_corpus, build["build_id"], tuple(conflicts))


class IndexConflict(RuntimeError):
    """The built index does not hold what the run's config and corpus describe."""


def checked_index(
    config: RagConfig,
    selection: CorpusSelection,
    documents: list[Document],
    chunks: list[Chunk],
    *,
    label: str,
) -> IndexProvenance:
    """`verify_index`, raising on a conflict and warning when the index is only unverified.

    A conflict means the run would measure a different index than its config
    names -- the kind of result that looks fine and isn't -- so it stops the
    run. An index that merely predates build records is still measured, with
    ``verified: false`` in the result, rather than forcing every existing index
    to be rebuilt before anything can run.

    Raises:
        IndexConflict: naming each conflict, prefixed with ``label``.
    """
    index = verify_index(config, selection, documents, chunks)
    if index.conflicts:
        raise IndexConflict(
            f"{label}: the index at {selection.index_dir} is not what this config describes:\n  "
            + "\n  ".join(index.conflicts)
            + "\nRun `python -m rag.cli index` with the same config (and --reset if it asks)."
        )
    if not index.verified:
        logger.warning("%s: index unverified, recorded as such: %s", label, index.unverified)
    return index


# ---------------------------------------------------------------------------
# The record a result carries
# ---------------------------------------------------------------------------


def provenance_record(
    *,
    config: RagConfig,
    dataset_path: Path,
    dataset: EvalDataset,
    selected: EvalDataset,
    selection: str,
    code: str,
    index: IndexProvenance | None,
    judge: LLMConfig | None = None,
) -> dict[str, Any]:
    """The ``provenance`` block one eval set's result carries.

    Args:
        dataset: The whole eval set as loaded from ``dataset_path``.
        selected: The samples actually evaluated, in order.
        selection: How ``selected`` was chosen, e.g. ``"all"``, ``"evenly spaced, --limit 40"``.
        index: None when the run read no index (oracle, closed-book).
        judge: The answer judge, for answer evals.
    """
    return {
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code_version": code,
        "config_hash": config_hash(config),
        **({"judge_hash": judge_hash(judge)} if judge is not None else {}),
        "dataset": {
            "path": str(dataset_path),
            "dataset_hash": dataset_hash(dataset),
            "num_samples": len(dataset),
            "selection": selection,
            "selection_hash": selection_hash([s.id for s in selected]),
            "num_selected": len(selected),
        },
        "index": index.as_record() if index is not None else None,
    }
