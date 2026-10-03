"""Query transformations that run *before* retrieval: HyDE and multi-query expansion.

Both techniques attack the same weakness from opposite directions. A user's
question and the passage that answers it are different kinds of text -- the
question is short and interrogative, the passage is long and declarative -- so
embedding the question and comparing it to passage embeddings compares things
that never looked alike to begin with. And a single phrasing of a question only
ever probes one region of the embedding space, so vocabulary the corpus happens
to use ("rate limiting", "throttling", "429") decides whether retrieval works.

- **HyDE** closes the shape gap: ask the LLM to *write the passage that would
  answer this*, then embed that hypothetical passage instead of the question.
  The invented specifics don't need to be true -- they only need to look like
  the real passage, so that passage-to-passage similarity does the work
  question-to-passage similarity was doing badly.
- **Multi-query** closes the vocabulary gap: ask the LLM for several rephrasings,
  retrieve for each, and fuse the ranked lists. A chunk that any phrasing finds
  gets a chance; a chunk several phrasings agree on rises.

Both produce *more than one search query*, which is why they share an interface.
`ExpandedQuery` keeps a separate list per consumer -- embedder, BM25,
cross-encoder -- because each accepts different text. BM25 matches literal
terms, so a hypothetical document full of invented specifics ("1,000 requests
per minute") makes it retrieve on words that may appear nowhere in the corpus;
a cross-encoder is trained on (question, passage) pairs, so a passage in the
question slot is off-distribution. HyDE's invention therefore reaches the
embedder alone, and both other stages keep the user's real question.

Like `QueryCondenser`, every expander fails open -- an LLM error or an
unparseable reply falls back to the original query rather than failing the turn.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

from rag.llm.base import LLMClient

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExpandedQuery:
    """The search queries retrieval should actually run, per retriever type.

    Three lists rather than one, because each consumer wants different text:

    - `dense` — embedded and compared to chunk embeddings. Free-form: a
      hypothetical passage belongs here.
    - `sparse` — fed to BM25, which matches literal terms. Invented specifics
      are actively harmful (see the module docstring on HyDE).
    - `rerank` — scored pairwise against candidates by a cross-encoder. These
      must be **question-shaped**: cross-encoders are trained on (query,
      passage) pairs, so handing one a generated passage as the "query" is off
      the distribution it was trained for. HyDE therefore keeps its invention
      out of this list entirely, while multi-query's rephrasings -- still real
      questions -- belong in it.

    An expander with no such asymmetry puts the same strings in all three.
    """

    dense: list[str]
    sparse: list[str]
    rerank: list[str]

    @classmethod
    def unchanged(cls, query: str) -> "ExpandedQuery":
        """The identity expansion -- what every expander falls back to."""
        return cls(dense=[query], sparse=[query], rerank=[query])

    @property
    def is_expanded(self) -> bool:
        """True when this produced anything beyond a single unchanged query."""
        return (
            len(self.dense) > 1
            or len(self.sparse) > 1
            or len(self.rerank) > 1
            or self.dense != self.sparse
        )

    def all_queries(self) -> list[str]:
        """Every distinct query text, dense first, for logging and display."""
        seen: list[str] = []
        for query in [*self.dense, *self.sparse, *self.rerank]:
            if query not in seen:
                seen.append(query)
        return seen


class QueryExpander(ABC):
    """Turns one user query into the set of queries retrieval will run.

    To add a strategy: subclass this, implement `expand`, and register it in
    `rag.retrieval.factory.get_query_expander`. `Retriever` only knows this
    interface -- it runs whatever queries it's handed and fuses the results, so
    a new expander needs no retriever changes.
    """

    @abstractmethod
    def expand(self, query: str) -> ExpandedQuery:
        """Return the queries to search for. Must never raise -- fail open instead."""
        raise NotImplementedError


class NoOpQueryExpander(QueryExpander):
    """Pass-through used when `retrieval.expansion.provider = "none"`.

    Selecting it makes retrieval exactly what it was before expansion existed:
    one query, one embedding call, no LLM in the retrieval path.
    """

    def expand(self, query: str) -> ExpandedQuery:
        return ExpandedQuery.unchanged(query)


HYDE_SYSTEM_PROMPT = (
    "You write short reference passages. Given a question, write the passage "
    "that would answer it, as though excerpted from internal documentation. "
    "Write in plain declarative prose, the way a document would -- never "
    "address the reader, never restate the question. "
    "Invent plausible specifics (names, numbers, procedures) wherever you don't "
    "know them: this passage is never shown to anyone and is never treated as "
    "true. It is used only to find real documents that look like it, so a "
    "confident, specific, wrong passage works far better than a hedge. "
    "Never write 'I don't know' or refuse. Reply with 2-4 sentences and nothing "
    "else -- no preamble, no heading, no quotation marks."
)

MULTI_QUERY_SYSTEM_PROMPT = (
    "You rewrite a search question into several alternative phrasings for a "
    "document search. Vary the vocabulary and the angle -- use synonyms, "
    "domain jargon, and both specific and general framings -- so that together "
    "they cover wording the source documents might plausibly use. Each "
    "alternative must be a standalone question answerable on its own. "
    "Reply with one phrasing per line and nothing else: no numbering, no "
    "bullets, no preamble, no blank lines."
)


def _clean_line(line: str) -> str:
    """Strip the numbering and bullets models add despite being asked not to."""

    cleaned = line.strip().lstrip("-*•").strip()
    # Leading "1." / "2)" / "3 -" style numbering.
    if cleaned[:1].isdigit():
        for separator in (".", ")", ":", "-"):
            head, found, tail = cleaned.partition(separator)
            if found and head.strip().isdigit():
                cleaned = tail.strip()
                break
    return cleaned.strip().strip('"').strip()


def parse_query_lines(reply: str, *, limit: int) -> list[str]:
    """Parse a model's line-per-query reply into at most `limit` clean queries.

    Deduplicates case-insensitively -- a model asked for several rephrasings
    will sometimes return the same one twice, and a duplicate query is pure
    cost: an extra embedding round trip and an extra identical ranked list
    that skews the fusion toward whatever it found.
    """

    queries: list[str] = []
    seen: set[str] = set()
    for line in reply.splitlines():
        cleaned = _clean_line(line)
        if not cleaned or cleaned.casefold() in seen:
            continue
        seen.add(cleaned.casefold())
        queries.append(cleaned)
        if len(queries) >= limit:
            break
    return queries


class HyDEQueryExpander(QueryExpander):
    """Embeds a hypothetical answer passage instead of (or alongside) the question.

    `num_documents > 1` generates several independent passages, each embedded
    and searched separately, with the ranked lists fused downstream -- the
    original paper's variance-reduction trick, since any single generated
    passage may wander somewhere unhelpful in embedding space.

    `include_original` keeps the user's real question in the dense list as
    well, which is the safe default: if a generated passage misses badly, the
    plain question is still there to retrieve on.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        *,
        num_documents: int = 1,
        include_original: bool = True,
    ) -> None:
        self._llm_client = llm_client
        self.num_documents = num_documents
        self.include_original = include_original

    def expand(self, query: str) -> ExpandedQuery:
        documents = []
        for index in range(self.num_documents):
            document = self._generate(query, index)
            if document:
                documents.append(document)

        if not documents:
            logger.warning("HyDE produced no usable passages; falling back to the original query")
            return ExpandedQuery.unchanged(query)

        # The original goes last: fusion is rank-based, so ordering within the
        # dense list doesn't affect scoring, but keeping generated text first
        # makes the trace read the way the technique works.
        dense = [*documents, query] if self.include_original else documents

        # BM25 and the reranker both keep the real question: one matches literal
        # terms, the other expects question-shaped input, and the generated
        # passage is wrong for both. HyDE is a *dense-retrieval* technique only.
        return ExpandedQuery(dense=dense, sparse=[query], rerank=[query])

    def _generate(self, query: str, index: int) -> str | None:
        try:
            reply = self._llm_client.generate(query, system=HYDE_SYSTEM_PROMPT)
        except Exception:  # noqa: BLE001 -- fail open
            logger.exception("HyDE generation %d failed for query %r", index, query)
            return None

        passage = " ".join(reply.split()).strip().strip('"').strip()
        if not passage:
            logger.warning("HyDE generation %d returned an empty passage for query %r", index, query)
            return None

        logger.info("HyDE passage %d for %r: %r", index, query, passage)
        return passage


class MultiQueryExpander(QueryExpander):
    """Retrieves for several rephrasings of the question and fuses the results.

    The original query is always included and always first: the rewrites are
    speculative, and a rewrite that drifts shouldn't be able to displace what
    the user actually asked. Both retrievers get the full set -- unlike HyDE,
    a rephrasing is still a real question, so BM25 benefits from the extra
    vocabulary rather than being poisoned by it.
    """

    def __init__(self, llm_client: LLMClient, *, num_queries: int = 3) -> None:
        self._llm_client = llm_client
        self.num_queries = num_queries

    def expand(self, query: str) -> ExpandedQuery:
        variants = self._generate_variants(query)
        if not variants:
            return ExpandedQuery.unchanged(query)

        queries = [query]
        seen = {query.casefold()}
        for variant in variants:
            if variant.casefold() not in seen:
                seen.add(variant.casefold())
                queries.append(variant)

        logger.info("Multi-query expanded %r into %d query/queries", query, len(queries))
        # Every consumer gets the full set: a rephrasing is a real question, so
        # it's valid input for BM25 and for the cross-encoder alike.
        return ExpandedQuery(dense=list(queries), sparse=list(queries), rerank=list(queries))

    def _generate_variants(self, query: str) -> list[str]:
        prompt = (
            f"Question: {query}\n\n"
            f"Write {self.num_queries} alternative phrasings, one per line:"
        )
        try:
            reply = self._llm_client.generate(prompt, system=MULTI_QUERY_SYSTEM_PROMPT)
        except Exception:  # noqa: BLE001 -- fail open
            logger.exception("Multi-query expansion failed for query %r", query)
            return []

        variants = parse_query_lines(reply, limit=self.num_queries)
        if not variants:
            logger.warning("Multi-query expansion returned no usable phrasings for %r", query)
        return variants
