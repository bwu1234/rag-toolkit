"""Tests for the holdout review server (`scripts/holdout_review.py`).

What review relies on: a save changes one record's review block and nothing
else, a stale form can't overwrite a newer record, the rubric is enforced
before anything is written, and the server only answers its own origin.
"""

from __future__ import annotations

import http.client
import json
import sys
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_holdout  # noqa: E402
import holdout_review  # noqa: E402
from build_holdout import Corpus, DevIndex, assemble_sample, assign_splits, review_problems  # noqa: E402
from holdout_review import Conflict, passage, save_review  # noqa: E402

from rag.ingestion.models import Document  # noqa: E402

FUEL = "Aircraft fuel expense was $4.9 billion in the second quarter of 2026, up 83.3%."
ROW = "| Passenger load factor (percent) | 83.2 | 84.7 | (1.5)pts |"
DOC = Document(
    id="AAL_10-Q_2026-06-30.md",
    text=f"# AAL 10-Q\n\n## Results\n\nIntro paragraph.\n\n{FUEL} More text.\n\n"
         f"| Metric | 2026 | 2025 | Change |\n|---|---|---|---|\n| ASMs | 1 | 2 | 3 |\n| Yield | 4 | 5 | 6 |\n{ROW}\n",
    source=Path("AAL_10-Q_2026-06-30.md"), doc_type="markdown", metadata={},
)


def _draft(sample_id: str, span: str = FUEL) -> dict[str, Any]:
    return assemble_sample({"id": sample_id, "family": "direct_lookup", "behavior": "answer", "query": "q?",
                            "answer": "a", "docs": [DOC.id], "spans": [span]},
                           assign_splits(), drafted_with="t", drafted_on="d")


def _review(verdict: str = "accept", decision: str | None = "pass", reason: str | None = "ok",
            reviewer: str | None = "R") -> dict[str, Any]:
    review: dict[str, Any] = {"verdict": verdict, "reviewer": reviewer, "edits": None}
    for dim in build_holdout.REVIEW_DIMENSIONS:
        review[dim] = {"decision": decision, "reason": reason}
    return review


@pytest.fixture
def draft(tmp_path: Path) -> Path:
    path = tmp_path / "draft.json"
    path.write_text(json.dumps([_draft("a"), _draft("b", ROW)], indent=2, ensure_ascii=False) + "\n")
    return path


# -- rubric ------------------------------------------------------------------


def test_pending_may_hold_partial_notes() -> None:
    review = _review("pending", decision=None, reason=None, reviewer=None)
    review["clarity"] = {"decision": "fail", "reason": None}
    assert review_problems(review) == []


def test_a_verdict_needs_reviewer_and_a_decision_and_reason_everywhere() -> None:
    assert review_problems(_review()) == []
    assert review_problems(_review(reviewer=None)) == ["needs a reviewer"]
    problems = review_problems(_review(decision=None, reason=None))
    assert "answerability: needs a decision" in problems and "clarity: needs a reason" in problems


def test_verdict_must_agree_with_the_decisions() -> None:
    mixed = _review("accept")
    mixed["usefulness"]["decision"] = "fail"
    assert review_problems(mixed) == ["an accept must pass every dimension"]
    assert review_problems(_review("reject")) == ["a reject must fail at least one dimension"]
    mixed["verdict"] = "reject"
    assert review_problems(mixed) == []


def test_unknown_verdict_or_decision_is_refused() -> None:
    assert review_problems(_review("maybe")) == ["verdict must be one of pending, accept, reject"]
    assert "usefulness: decision must be pass or fail" in review_problems(_review(decision="ok"))


def test_finalize_lists_what_each_incomplete_review_lacks(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    raw = _draft("f")
    raw["review"] = _review(reason=None)
    path = tmp_path / "d.json"
    path.write_text(json.dumps([raw]))
    assert build_holdout.cmd_finalize(path, None) == 1
    out = capsys.readouterr().out
    assert "1 without a complete review record" in out and "f: answerability: needs a reason" in out


# -- saving ------------------------------------------------------------------


def test_save_changes_only_that_records_review_block(draft: Path) -> None:
    before = json.loads(draft.read_text())
    saved = save_review(draft, "a", before[0]["review"], _review(reason="  fine  "))
    after = json.loads(draft.read_text())
    assert saved["answerability"]["reason"] == "fine"  # trimmed
    assert after[0]["review"] == saved
    assert {k: v for k, v in after[0].items() if k != "review"} == {k: v for k, v in before[0].items() if k != "review"}
    assert after[1] == before[1]
    assert draft.read_text() == json.dumps(after, indent=2, ensure_ascii=False) + "\n"  # format kept


def test_blank_text_is_stored_as_null(draft: Path) -> None:
    base = json.loads(draft.read_text())[0]["review"]
    saved = save_review(draft, "a", base, _review("pending", decision=None, reason="", reviewer=" "))
    assert saved["reviewer"] is None and saved["usefulness"]["reason"] is None


def test_stale_base_is_a_conflict_and_writes_nothing(draft: Path) -> None:
    base = json.loads(draft.read_text())[0]["review"]
    save_review(draft, "a", base, _review())
    on_disk = draft.read_text()
    with pytest.raises(Conflict) as err:
        save_review(draft, "a", base, _review("reject", decision="fail"))
    assert err.value.current["verdict"] == "accept"
    assert draft.read_text() == on_disk


def test_rubric_violation_and_bad_shape_write_nothing(draft: Path) -> None:
    on_disk = draft.read_text()
    base = json.loads(on_disk)[0]["review"]
    with pytest.raises(ValueError, match="needs a reviewer"):
        save_review(draft, "a", base, _review(reviewer=None))
    with pytest.raises(ValueError, match="exactly the keys"):
        save_review(draft, "a", base, {**_review(), "note": "x"})
    with pytest.raises(KeyError):
        save_review(draft, "zzz", base, _review())
    assert draft.read_text() == on_disk


# -- evidence in context ------------------------------------------------------


def test_passage_brings_the_table_header_and_marks_the_row() -> None:
    corpus = Corpus({DOC.id: DOC})
    units = build_holdout.split_units(DOC.text)
    blocks = passage(units, corpus.units_containing(DOC.id, ROW), ROW)
    assert [b["kind"] for b in blocks] == ["table"]  # one table, even across the skipped rows
    table = blocks[0]
    assert table["rows"][0]["cells"][0] == "Metric"  # header carried along
    assert table["rows"][1].get("gap") is True
    assert [r["hit"] for r in table["rows"]][-1] is True
    assert all(r["cells"][0] != "---" for r in table["rows"])  # separator dropped


def test_passage_splits_the_quote_out_of_its_paragraph() -> None:
    corpus = Corpus({DOC.id: DOC})
    units = build_holdout.split_units(DOC.text)
    blocks = passage(units, corpus.units_containing(DOC.id, FUEL), FUEL)
    hit = next(b for b in blocks if b["kind"] == "p" and b["parts"][1])
    assert hit["parts"] == ["", FUEL, " More text."]


# -- HTTP ----------------------------------------------------------------------


@pytest.fixture
def server(draft: Path) -> Iterator[holdout_review.ReviewServer]:
    corpus = Corpus({DOC.id: DOC})
    srv = holdout_review.ReviewServer(0, draft, corpus, DevIndex.build([], corpus))
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv
    srv.shutdown()
    srv.server_close()


def _request(srv: holdout_review.ReviewServer, method: str, path: str, body: Any = None,
             headers: dict[str, str] | None = None) -> tuple[int, Any]:
    port = srv.server_address[1]
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    sent = {"Host": f"127.0.0.1:{port}", "Content-Type": "application/json", **(headers or {})}
    conn.request(method, path, body=None if body is None else json.dumps(body), headers=sent)
    res = conn.getresponse()
    data = res.read()
    conn.close()
    is_json = res.getheader("Content-Type", "").startswith("application/json")
    return res.status, json.loads(data) if is_json else data


def test_serves_the_page_and_drafts_with_context(server: holdout_review.ReviewServer) -> None:
    status, page = _request(server, "GET", "/")
    assert status == 200 and b"Holdout Draft Review" in page
    status, body = _request(server, "GET", "/api/drafts")
    assert status == 200 and [d["id"] for d in body["drafts"]] == ["a", "b"]
    assert body["drafts"][0]["spans"][0]["locations"][0]["section"] == "Results"


def test_post_saves_and_a_stale_post_conflicts(server: holdout_review.ReviewServer, draft: Path) -> None:
    base = json.loads(draft.read_text())[0]["review"]
    status, body = _request(server, "POST", "/api/review", {"id": "a", "base": base, "review": _review()})
    assert status == 200 and body["review"]["verdict"] == "accept"
    status, body = _request(server, "POST", "/api/review", {"id": "a", "base": base, "review": _review()})
    assert status == 409 and body["review"]["verdict"] == "accept"
    status, body = _request(server, "POST", "/api/review",
                            {"id": "b", "base": base, "review": _review(reviewer=None)})
    assert status == 422 and "needs a reviewer" in body["error"]


@pytest.mark.parametrize("headers, expected", [
    ({"Host": "evil.example:80"}, 403),           # DNS rebinding
    ({"Content-Type": "text/plain"}, 415),        # cross-site form without preflight
])
def test_post_refuses_other_origins(server: holdout_review.ReviewServer, draft: Path,
                                    headers: dict[str, str], expected: int) -> None:
    before = draft.read_text()
    base = json.loads(before)[0]["review"]
    status, _ = _request(server, "POST", "/api/review", {"id": "a", "base": base, "review": _review()}, headers)
    assert status == expected
    assert draft.read_text() == before
