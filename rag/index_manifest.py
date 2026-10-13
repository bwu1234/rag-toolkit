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

The manifest also carries a **build record** (`IndexBuild`), which eval
provenance reads to say exactly which index a run measured. It is a different
job from the check above, so it is kept apart from it: the check decides
whether an index *can be extended*, and leaves chunk sizes out because
re-chunking is incremental; the build record describes what the index *holds*,
so its recipe includes them. A build gets a new id whenever `index` writes or
deletes anything, and a run that stops partway leaves the record marked
`building`, which no eval accepts as verified.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from rag.chunking.contextualizer import CONTEXT_SYSTEM_PROMPT, build_context_prompt
from rag.chunking.models import Chunk, content_hash
from rag.config.settings import DEFAULT_CARRY_METADATA, CorpusSelection, RagConfig
from rag.hashing import stable_hash
from rag.ingestion.models import Document

logger = logging.getLogger(__name__)

#: 2 added the `build` record; version 1 manifests read as having none.
_MANIFEST_VERSION = 2

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
        # Only when set, so the manifests of existing (and all Ollama) indexes
        # are written exactly as before the field existed.
        if config.embedding.revision is not None:
            embedding["revision"] = config.embedding.revision
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


def write_index_manifest(path: Path, manifest: IndexManifest, build: IndexBuild | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"version": _MANIFEST_VERSION, **asdict(manifest)}
    if build is not None:
        payload["build"] = asdict(build)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


BuildStatus = Literal["building", "complete", "incomplete"]


@dataclass(frozen=True)
class IndexBuild:
    """Which index an eval measured: an id that changes whenever its contents do.

    Attributes:
        status: ``building`` from the moment `index` may write until it
            finishes; ``complete`` after a clean run; ``incomplete`` when files
            failed to load, so stale chunks were kept.
        build_id: Random, new on every run that wrote or deleted anything;
            None while building.
        index_hash: ``stable_hash(recipe)``.
        recipe: `index_recipe` at the time of the build.
        corpus_hash: `corpus_hash` of the documents the build indexed.
        inventory_digest: `inventory_digest` of the chunks it holds.
        num_chunks: How many chunks that is.
        recorded_at: UTC time the record was written.
        previous: While building, the record being replaced, so a run that
            changes nothing can restore it with its id.
    """

    status: BuildStatus
    build_id: str | None = None
    index_hash: str | None = None
    recipe: dict[str, Any] | None = None
    corpus_hash: str | None = None
    inventory_digest: str | None = None
    num_chunks: int | None = None
    recorded_at: str | None = None
    previous: IndexBuild | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> IndexBuild:
        previous = raw.get("previous")
        return cls(**{**raw, "previous": cls.from_dict(previous) if previous else None})


def index_recipe(config: RagConfig, selection: CorpusSelection) -> dict[str, Any]:
    """Every setting that shapes what an index for ``selection`` holds.

    A superset of `IndexManifest`: it adds the corpora and their cleaning, and
    the chunker. Query-time settings (`hnsw_ef_search`, the reranker, the
    query instruction) are left out, since one index serves all of them, and
    so is the sparse backend: its file is checked against the corpus directly
    (`rag.index_report.inspect_index`), which also covers a backend switch.
    """

    chunking = config.chunking
    chunker: dict[str, Any] = {"strategy": chunking.strategy}
    # `none` ignores the sizes, so they would only make equal indexes differ.
    if chunking.strategy != "none":
        chunker.update(chunk_size=chunking.chunk_size, chunk_overlap=chunking.chunk_overlap)
    if chunking.strategy == "structured":
        chunker["structured"] = chunking.structured.model_dump(mode="json")
    registry = config.corpora.registry
    return {
        "corpora": {
            name: {"clean": registry[name].clean if name in registry else True}
            for name in selection.names
        },
        "chunker": chunker,
        **asdict(IndexManifest.from_config(config)),
        "vector_store": config.vector_store.provider,
    }


def index_hash(config: RagConfig, selection: CorpusSelection) -> str:
    return stable_hash(index_recipe(config, selection))


def corpus_hash(config: RagConfig, selection: CorpusSelection, documents: list[Document]) -> str:
    """Hash of the loaded, cleaned documents and how each corpus was cleaned.

    Hashing the cleaned text rather than the files means a change to a loader
    or cleaner counts as a corpus change too. `Document.source` is left out:
    it is an absolute path, so two checkouts of the same files would differ.
    """

    registry = config.corpora.registry
    return stable_hash({
        "corpora": {name: registry[name].clean if name in registry else True for name in selection.names},
        "documents": sorted(
            ([d.id, d.doc_type, d.text, d.metadata] for d in documents), key=lambda item: str(item[0])
        ),
    })


def inventory_digest(chunks: list[Chunk]) -> str:
    """Hash of the chunk ids and content hashes an index holds -- what `index` compares."""

    return stable_hash(sorted([chunk.id, content_hash(chunk.text)] for chunk in chunks))


def read_index_build(path: Path) -> IndexBuild | None:
    """The build record, or None when the index predates build records (or has no manifest)."""

    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8")).get("build")
    return IndexBuild.from_dict(raw) if raw else None


def finish_index_build(
    path: Path,
    config: RagConfig,
    selection: CorpusSelection,
    documents: list[Document],
    chunks: list[Chunk],
    *,
    modified: bool,
    complete: bool,
) -> IndexBuild:
    """Record the build `index` just finished, replacing the ``building`` marker.

    A run that wrote nothing keeps the previous build's id when everything it
    describes still matches, so re-running `index` to check it is current does
    not make every earlier eval row look like it measured a different index.
    """

    manifest = read_index_manifest(path)
    if manifest is None:
        raise RuntimeError(f"{path} is missing; prepare_for_indexing writes it before any build")
    stored = read_index_build(path)
    previous = stored.previous if stored is not None else None
    recipe = index_recipe(config, selection)
    record = IndexBuild(
        status="complete" if complete else "incomplete",
        build_id=uuid.uuid4().hex,
        index_hash=stable_hash(recipe),
        recipe=recipe,
        corpus_hash=corpus_hash(config, selection, documents),
        inventory_digest=inventory_digest(chunks),
        num_chunks=len(chunks),
        recorded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    unchanged = (
        not modified
        and previous is not None
        and previous.status == record.status == "complete"
        and (previous.index_hash, previous.corpus_hash, previous.inventory_digest)
        == (record.index_hash, record.corpus_hash, record.inventory_digest)
    )
    if unchanged:
        assert previous is not None
        record = previous
    write_index_manifest(path, manifest, record)
    return record


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
    build = read_index_build(path)
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
    # Marked as building before the first write, so a run killed partway
    # leaves a record no eval will accept rather than the last good one. After
    # such a run nothing says what it changed (regenerated contexts leave no
    # trace in the chunk hashes), so the next build can't reuse the old id.
    previous = None if build is not None and build.status == "building" else build
    write_index_manifest(path, current, IndexBuild(status="building", previous=previous))


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
