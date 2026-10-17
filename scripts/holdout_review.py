"""Local review server for the fresh holdout drafts (`build_holdout.py review --serve`).

The protocol has a person record, for every draft, a verdict, a reviewer, and
a decision and reason on each rubric dimension. Typing those into a 171-entry
JSON file by hand is where review records go wrong (a misspelt verdict, a
missing reason), so this serves one draft per screen with its evidence in
context and a form that writes the ``review`` block back into the draft file.

What it guards:

- **Only the review block changes.** Each save re-reads the file, replaces one
  record's ``review`` and writes the file back in its existing format
  (``indent=2``, trailing newline), atomically. Hand edits to any other field,
  or to other records, survive.
- **No lost updates.** The client sends the review it last saw; if the file
  now holds something else (a hand edit, a second tab), the save is refused
  and the page shows the current record.
- **The rubric is enforced at save time** by :func:`build_holdout.review_problems`,
  the same rule ``finalize`` applies.
- **Localhost only.** It binds 127.0.0.1, checks the Host header (DNS
  rebinding) and requires a JSON content type on writes, which a cross-site
  form cannot send without a CORS preflight this server never answers.

Content edits (question, answer, alternative quotes) are still made in the
file; reload the page to see them, and record what changed under ``edits``.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import build_holdout as bh

PAGE = Path(__file__).with_name("holdout_review.html")

#: Units of source text shown either side of the evidence.
CONTEXT = 2


class Conflict(Exception):
    """The record on disk is not the one the reviewer was editing."""

    def __init__(self, current: dict[str, Any]) -> None:
        super().__init__("review changed on disk")
        self.current = current


# --------------------------------------------------------------------------
# Evidence in context
# --------------------------------------------------------------------------


def _is_row(unit: str) -> bool:
    return unit.lstrip().startswith("|")


def _cells(row: str) -> list[str]:
    return [c.strip() for c in row.strip().strip("|").split("|")]


def _is_separator(row: str) -> bool:
    return all(set(c) <= set("-: ") for c in _cells(row))


def passage(units: list[str], hits: set[int], quote: str) -> list[dict[str, Any]]:
    """The source around the first cluster of ``hits``, as renderable blocks.

    Table rows are grouped into tables, with the table's header row brought
    along when the context window starts below it; prose paragraphs carry the
    quote split out for highlighting (or ``hit`` when it is not verbatim in
    one paragraph). ``gap`` marks skipped units.
    """
    first = min(hits)
    cluster = sorted(h for h in hits if h - first <= 6)
    start, end = max(0, first - CONTEXT), min(len(units), cluster[-1] + CONTEXT + 1)
    keep = list(range(start, end))
    if _is_row(units[first]):
        top = first
        while top > 0 and _is_row(units[top - 1]):
            top -= 1
        if top < start:
            keep = [top, *keep]
    blocks: list[dict[str, Any]] = []
    prev: int | None = None
    for i in keep:
        unit = units[i]
        if prev is not None and i != prev + 1:
            if _is_row(unit) and blocks and blocks[-1]["kind"] == "table":
                blocks[-1]["rows"].append({"cells": ["…"], "hit": False, "gap": True})  # same table
            else:
                blocks.append({"kind": "gap"})
        prev = i
        if _is_row(unit):
            if _is_separator(unit):
                continue
            if not blocks or blocks[-1]["kind"] != "table":
                blocks.append({"kind": "table", "rows": []})
            blocks[-1]["rows"].append({"cells": _cells(unit), "hit": i in hits})
            continue
        at = unit.find(quote) if i in hits else -1
        parts = [unit[:at], quote, unit[at + len(quote):]] if at >= 0 else [unit, "", ""]
        blocks.append({"kind": "p", "parts": parts, "hit": i in hits and at < 0})
    return blocks


def _units_for(corpus: bh.Corpus, doc_id: str, quote: str) -> set[int]:
    hits = corpus.units_containing(doc_id, quote)
    if hits:
        return hits
    # A quote shorter than the overlap window that crosses a paragraph break.
    needle = bh.normalize(quote)
    return {i for i, u in enumerate(corpus.paragraphs.get(doc_id, [])) if u and (needle[:20] in u or needle[-20:] in u)}


def build_payload(path: Path, corpus: bh.Corpus, dev_index: bh.DevIndex) -> list[dict[str, Any]]:
    """Every draft with the context ``review`` prints, plus the source passage."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    samples = bh.EvalDataset.load(path).samples
    units = {d: bh.split_units(doc.text) for d, doc in corpus.documents.items()}
    out: list[dict[str, Any]] = []
    for sample, record in zip(samples, raw, strict=True):
        x = sample.extra
        spans = []
        for primary, quotes in bh.all_quotes(sample):
            locations = []
            for doc_id in sample.expected_doc_ids:
                if bh.normalize(primary) not in corpus.normalized.get(doc_id, ""):
                    continue
                hits = _units_for(corpus, doc_id, primary)
                locations.append({
                    "doc": doc_id,
                    "section": corpus.section_of(doc_id, primary),
                    "blocks": passage(units[doc_id], hits, primary) if hits else [],
                })
            spans.append({
                "primary": primary,
                "alts": quotes[1:],
                "locations": locations,
                "also": sorted(set(corpus.containing(primary)) - set(sample.expected_doc_ids)),
                "leaks": dev_index.overlaps(corpus, primary),
            })
        out.append({
            "id": sample.id, "query": sample.query, "answer": sample.expected_answer,
            "docs": sample.expected_doc_ids, "split": x["split"], "family": x["family"],
            "behavior": x["behavior"], "transfer": x["transfer"], "group": x["group"],
            "tags": x.get("tags") or [], "parts": x.get("parts") or [],
            "conclusion": x.get("conclusion"), "calc": x.get("calc"),
            "absent": x.get("absent_terms") or [], "note": x.get("note"),
            "review": record["review"], "spans": spans,
        })
    return out


# --------------------------------------------------------------------------
# Saving a review
# --------------------------------------------------------------------------


def _text(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"expected text, got {type(value).__name__}")
    return value.strip() or None


def normalize_review(review: Any) -> dict[str, Any]:
    """The review block in the draft file's shape: blank text becomes null."""
    if not isinstance(review, dict):
        raise ValueError("review must be an object")
    expected = {"verdict", "reviewer", "edits", *bh.REVIEW_DIMENSIONS}
    if set(review) != expected:
        raise ValueError(f"review must have exactly the keys {sorted(expected)}")
    out: dict[str, Any] = {"verdict": review["verdict"], "reviewer": _text(review["reviewer"])}
    for dim in bh.REVIEW_DIMENSIONS:
        block = review[dim]
        if not isinstance(block, dict) or set(block) != {"decision", "reason"}:
            raise ValueError(f"{dim} must be {{decision, reason}}")
        out[dim] = {"decision": _text(block["decision"]), "reason": _text(block["reason"])}
    out["edits"] = _text(review["edits"])
    return out


def _write_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise


def save_review(path: Path, sample_id: str, base: Any, review: Any) -> dict[str, Any]:
    """Replace one record's review block, if it still equals ``base``.

    Raises ``KeyError`` for an unknown id, :class:`Conflict` when the file
    changed under the reviewer, and ``ValueError`` when the review breaks the
    rubric. Returns the review as written.
    """
    new = normalize_review(review)
    problems = bh.review_problems(new)
    if problems:
        raise ValueError("; ".join(problems))
    records = json.loads(path.read_text(encoding="utf-8"))
    matches = [r for r in records if r["id"] == sample_id]
    if len(matches) != 1:
        raise KeyError(sample_id)
    record = matches[0]
    if record["review"] != base:
        raise Conflict(record["review"])
    record["review"] = new
    _write_atomic(path, records)
    return new


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------


class ReviewServer(ThreadingHTTPServer):
    def __init__(self, port: int, path: Path, corpus: bh.Corpus, dev_index: bh.DevIndex) -> None:
        super().__init__(("127.0.0.1", port), Handler)
        self.draft_path = path
        self.corpus = corpus
        self.dev_index = dev_index
        self.lock = threading.Lock()  # one read-modify-write of the draft file at a time

    @property
    def allowed_hosts(self) -> set[str]:
        port = self.server_address[1]
        return {f"127.0.0.1:{port}", f"localhost:{port}"}


class Handler(BaseHTTPRequestHandler):
    server: ReviewServer

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        pass  # saves are reported in do_POST; per-request lines are noise

    def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: HTTPStatus, data: Any) -> None:
        self._send(status, json.dumps(data, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def _host_ok(self) -> bool:
        if self.headers.get("Host") in self.server.allowed_hosts:
            return True
        self._json(HTTPStatus.FORBIDDEN, {"error": "unexpected Host header"})
        return False

    def do_GET(self) -> None:  # noqa: N802 - stdlib name
        if not self._host_ok():
            return
        if self.path == "/":
            self._send(HTTPStatus.OK, PAGE.read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/drafts":
            with self.server.lock:
                drafts = build_payload(self.server.draft_path, self.server.corpus, self.server.dev_index)
            self._json(HTTPStatus.OK, {"source": str(self.server.draft_path), "drafts": drafts})
        else:
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802 - stdlib name
        if not self._host_ok():
            return
        if self.path != "/api/review":
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        if self.headers.get_content_type() != "application/json":
            self._json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "expected application/json"})
            return
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
            sample_id, base, review = body["id"], body["base"], body["review"]
        except (ValueError, KeyError, TypeError):
            self._json(HTTPStatus.BAD_REQUEST, {"error": "expected {id, base, review}"})
            return
        try:
            with self.server.lock:
                saved = save_review(self.server.draft_path, sample_id, base, review)
        except KeyError:
            self._json(HTTPStatus.NOT_FOUND, {"error": f"no draft {sample_id!r}"})
        except Conflict as c:
            self._json(HTTPStatus.CONFLICT, {"error": "this draft's review changed on disk", "review": c.current})
        except ValueError as e:
            self._json(HTTPStatus.UNPROCESSABLE_ENTITY, {"error": str(e)})
        else:
            print(f"saved {sample_id}: {saved['verdict']}", flush=True)
            self._json(HTTPStatus.OK, {"review": saved})


def serve(path: Path, config_path: str | None, port: int, open_browser: bool = True) -> int:
    _, corpus, _, dev_index = bh._corpus_and_index(config_path)
    server = ReviewServer(port, path, corpus, dev_index)
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"Reviewing {path}\nServing {url}  (Ctrl-C to stop)", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0
