"""Records which settings an index was built with, and refuses to mix them.

The incremental indexer skips a chunk when its *text* hash is unchanged, so a
config change that alters the vectors without altering chunk text -- a new
embedding model, contextual chunking toggled or re-tuned -- would otherwise go
unnoticed: new chunks would be embedded one way and the untouched rest another,
all in one collection. A query embedded with a different model than the index
has the same problem from the other side. Neither fails loudly on its own
(Chroma only complains if the *dimensions* differ).

So each index carries a small sidecar manifest of the settings that shape its
contents, and a mismatch is an error naming what changed, with `--reset` as the
fix. It is a sidecar file rather than vector-store metadata because it
describes the vector and BM25 indexes together and shouldn't depend on which
`VectorStore` backend is configured.

Chunking parameters are deliberately absent: they change chunk text (and ids),
which the content hash and stale-chunk purge in `rag.cli index` already handle
incrementally.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rag.chunking.contextualizer import CONTEXT_SYSTEM_PROMPT, build_context_prompt
from rag.config.settings import RagConfig

logger = logging.getLogger(__name__)

_MANIFEST_VERSION = 1


class IndexManifestMismatch(RuntimeError):
    """The configured settings differ from those the index was built with."""


def index_manifest_path(index_dir: Path, slug: str) -> Path:
    """Sidecar path for the index of corpus selection ``slug``, beside its BM25 file."""

    return index_dir / f"index_manifest__{slug}.json"


def _context_prompt_hash() -> str:
    """Short hash of the context prompt's wording, so editing it counts as a change."""

    template = build_context_prompt("{document}", "{chunk}")
    return hashlib.sha256(f"{CONTEXT_SYSTEM_PROMPT}\n{template}".encode()).hexdigest()[:16]


@dataclass(frozen=True)
class IndexManifest:
    """The settings that determine what an index contains.

    Attributes:
        embedding: What turns text into vectors -- must match at query time too.
        contextual: How chunks were enriched before embedding; index-time only.
    """

    embedding: dict[str, Any]
    contextual: dict[str, Any]

    @classmethod
    def from_config(cls, config: RagConfig) -> IndexManifest:
        embedding = {
            "provider": config.embedding.provider,
            "model": config.embedding.model,
            "dimensions": config.embedding.dimensions,
        }
        ctx = config.chunking.contextual
        contextual: dict[str, Any] = {"enabled": ctx.enabled}
        if ctx.enabled:
            # `concurrency` and `cache` change how fast blurbs are made, not
            # what they say, so they are left out.
            contextual.update(
                llm_provider=config.llm.provider,
                llm_model=config.llm.model,
                max_document_chars=ctx.max_document_chars,
                max_context_chars=ctx.max_context_chars,
                prompt_hash=_context_prompt_hash(),
            )
        return cls(embedding=embedding, contextual=contextual)

    def differences(self, other: IndexManifest, *, sections: tuple[str, ...]) -> list[str]:
        """Human-readable ``section.key: stored -> configured`` lines, for ``sections``."""

        lines: list[str] = []
        for section in sections:
            mine: dict[str, Any] = getattr(self, section)
            theirs: dict[str, Any] = getattr(other, section)
            for key in sorted(mine.keys() | theirs.keys()):
                if mine.get(key) != theirs.get(key):
                    lines.append(f"{section}.{key}: {mine.get(key)!r} -> {theirs.get(key)!r}")
        return lines


def read_index_manifest(path: Path) -> IndexManifest | None:
    """Load a manifest, or ``None`` if the index predates manifests (or has none yet)."""

    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    return IndexManifest(embedding=raw["embedding"], contextual=raw["contextual"])


def write_index_manifest(path: Path, manifest: IndexManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": _MANIFEST_VERSION, **asdict(manifest)}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def prepare_for_indexing(path: Path, config: RagConfig, *, index_is_empty: bool) -> None:
    """Check the index can be extended under ``config``, then record ``config``'s manifest.

    An empty index (fresh, or just ``--reset``) takes whatever is configured. A
    non-empty index with no manifest predates this check: it can't be verified,
    so it is adopted with a warning rather than forcing an expensive rebuild on
    every existing index.

    Raises:
        IndexManifestMismatch: the index was built with different settings.
    """

    current = IndexManifest.from_config(config)
    stored = read_index_manifest(path)
    if stored is not None and not index_is_empty:
        diffs = stored.differences(current, sections=("embedding", "contextual"))
        if diffs:
            raise IndexManifestMismatch(
                f"The index at {path.parent} was built with different settings:\n  "
                + "\n  ".join(diffs)
                + "\nAdding to it would mix incompatible chunks. Re-run with --reset "
                "to rebuild it, or restore the old settings."
            )
    elif stored is None and not index_is_empty:
        logger.warning(
            "Index at %s has no manifest (built before manifests existed); assuming it "
            "matches the current config. If the embedding model or contextual settings "
            "changed since it was built, re-run with --reset.",
            path.parent,
        )
    write_index_manifest(path, current)


def check_queryable(path: Path, config: RagConfig) -> None:
    """Refuse to query an index embedded by a different model than ``config`` names.

    An index without a manifest is not checked (see `prepare_for_indexing`).

    Raises:
        IndexManifestMismatch: the embedding settings differ.
    """

    stored = read_index_manifest(path)
    if stored is None:
        return
    diffs = stored.differences(IndexManifest.from_config(config), sections=("embedding",))
    if diffs:
        raise IndexManifestMismatch(
            f"The index at {path.parent} was embedded with different settings than "
            "configured, so query vectors would not be comparable:\n  "
            + "\n  ".join(diffs)
            + "\nRestore the old embedding settings, or rebuild with `rag.cli index --reset`."
        )
