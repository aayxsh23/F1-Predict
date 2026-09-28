"""The regulations index (src/rag/corpus.py): listing, reading, keyword search,
article numbers and circuit filtering. Needs data/corpus/index.json
(`python -m src.rag.corpus`). Run with `python tests/test_corpus.py`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rag.corpus import circuit_summary, get_document_text, list_documents, search, search_documents


def test_list_documents():
    docs = list_documents()
    by_type = {}
    for d in docs:
        by_type.setdefault(d["doc_type"], []).append(d)
        assert d["title"] and d["size_bytes"] > 0 and "text" not in d
    assert len(by_type["regulation"]) == 3 and len(by_type["steward_decision"]) == 5
    assert all("grand_prix" in d and "car_number" in d for d in by_type["steward_decision"])


def test_get_document_text_is_an_exact_lookup():
    assert len(get_document_text(list_documents()[0]["filename"])) > 100
    # traversal attempts, unknown names, and circuit write-ups (not browsable) all miss
    for bad in ("not_a_real_file.pdf", "../../etc/passwd", r"..\..\Windows", "baku.txt"):
        try:
            get_document_text(bad)
            raise AssertionError(f"expected FileNotFoundError for {bad!r}")
        except FileNotFoundError:
            pass


def test_keyword_search_finds_the_ruling_and_cites_articles():
    hits = search_documents("unsafe release", k=5)
    assert any(h["filename"] == "2026_australian_gp_car12_unsafe_release.pdf" for h in hits)
    reg = search("grid penalty power unit elements", k=3, doc_types={"regulation"})
    assert reg and all(h["article"] for h in reg), "regulation hits carry their article number"


def test_circuit_write_ups_stay_on_their_circuit():
    hits = search("overtaking tyre degradation safety car", k=5, circuit="Baku")
    assert all(h["circuit"] == "Baku" for h in hits if h["doc_type"] == "circuit_summary")
    assert len(circuit_summary("Baku")) > 200 and circuit_summary("Nowhere") == ""


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} checks passed")
