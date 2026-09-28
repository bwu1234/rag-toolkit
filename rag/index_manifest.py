"""Records which settings an index was built with, and refuses to mix them.

The incremental indexer skips a chunk when its *text* hash is unchanged, so a
config change that alters the vectors without altering chunk text -- a new
embedding model, contextual chunking toggled or re-tuned, a chunk header
template changed -- would otherwise go unnoticed: new chunks would be embedded one way and the untouched rest another,
all in one collection. A query embedded with a different model than the index
has the same problem from the other side. Neither fails loudly on its own
(Chroma only complains if the *dimensions* differ).

So each index carries a small sidecar manifest of the settings that shape its
contents, and a mismatch is an error naming what changed, with `--reset` as the
fix. It is a sidecar file rather than vector-store metadata because it
describes the vector and BM25 indexes together and shouldn't depend on which
`VectorStore` backend is configured.

Chunk sizes are deliberately absent: they change chunk text (and ids), which
the content hash and stale-chunk purge in `rag.cli index` already handle
incrementally. The header template and carried metadata are present because
they don't: both change what is stored beside an unchanged text.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rag.chunking.contextualizer import CONTEXT_SYSTEM_PROMPT, build_context_prompt
from rag.config.settings import DEFAULT_CARRY_METADATA, RagConfig

logger = logging.getLogger(__name__)

_MANIFEST_VERSION = 1

#: Every section that describes an index's contents, and so must match to extend it.
INDEX_SECTIONS = ("embedding", "contextual", "chunk_fields")

#: `chunk_fields` as it was before manifests recorded it: the chunker carried a
#: hardcoded allowlist and wrote no header. An older manifest is read as this,
#: so an index built then isn't refused under settings that match it.
_LEGACY_CHUNK_FIELDS: dict[str, Any] = {
    "carry_metadata": list(DEFAULT_CARRY_METADATA),
    "header_template": None,
}


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
        chunk_fields: What is stored beside each chunk's text: the metadata
            keys carried from its document, and the header template.
    """

    embedding: dict[str, Any]
    contextual: dict[str, Any]
    chunk_fields: dict[str, Any]

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
        chunk_fields = {
            "carry_metadata": list(config.chunking.carry_metadata),
            "header_template": config.chunking.header.template,
        }
        return cls(embedding=embedding, contextual=contextual, chunk_fields=chunk_fields)

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
    return IndexManifest(
        embedding=raw["embedding"],
        contextual=raw["contextual"],
        chunk_fields=raw.get("chunk_fields", _LEGACY_CHUNK_FIELDS),
    )


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
        diffs = stored.differences(current, sections=INDEX_SECTIONS)
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
            "matches the current config. If the embedding model, contextual settings or chunk fields "
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
