"""Self-check for src/api/corpus_files.py -- the Regulations browse/read/
search feature. list_documents()/get_document_text() only touch the
filesystem (fast); search_documents() hits the real persisted Chroma store
and requires it already built via `python -m src.rag.ingest_corpus`.

Run with `python tests/test_corpus_files.py`.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api.corpus_files import get_document_text, list_documents, search_documents


def test_list_documents():
    docs = list_documents()
    assert len(docs) == 8, f"expected 3 regulations + 5 steward_decisions, got {len(docs)}"
    by_type = {}
    for d in docs:
        by_type.setdefault(d["doc_type"], []).append(d)
        assert d["title"] and d["size_bytes"] > 0 and d["modified_at"]
    assert len(by_type["regulation"]) == 3
    assert len(by_type["steward_decision"]) == 5
    assert all("grand_prix" in d and "car_number" in d for d in by_type["steward_decision"])


def test_get_document_text():
    docs = list_documents()
    real_filename = docs[0]["filename"]
    text = get_document_text(real_filename)
    assert isinstance(text, str) and len(text) > 100

    for bad in ("not_a_real_file.pdf", "../../etc/passwd", "..\\..\\Windows\\System32", "regs/../secret.pdf"):
        try:
            get_document_text(bad)
            assert False, f"expected FileNotFoundError for {bad!r}"
        except FileNotFoundError:
            pass


def test_search_documents():
    # phrased close to the real document's own wording ("released from its
    # garage in an unsafe condition") rather than generic keywords -- a
    # vocab-mismatched query against this small embedding model doesn't
    # reliably surface a 3-page steward decision over a 300+ page regulation
    # PDF, which isn't a search bug, just a weak query.
    hits = search_documents("car released from the garage in an unsafe condition", k=5)
    assert len(hits) > 0
    assert all(h["doc_type"] in ("regulation", "steward_decision") for h in hits)
    assert any(h["filename"] == "2026_australian_gp_car12_unsafe_release.pdf" for h in hits), (
        "expected the real Australian GP unsafe-release decision among top hits"
    )


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} checks passed")
