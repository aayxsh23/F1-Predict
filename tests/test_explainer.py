"""Self-check for the Phase 4+5 RAG explainer: runs the full local chain
(prediction -> SHAP -> retrieval -> local LLM generation) end-to-end on a real
row and checks it actually produced grounded output, not just "didn't crash".
Heavy (loads the 4-bit Llama 3.2 1B + its LoRA adapter, several seconds) --
run explicitly with `python tests/test_explainer.py`, not part of a fast loop.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rag.explain import explain
from src.rag.ingest_corpus import load_vector_store

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "model_matrix.parquet"


def test_vector_store_is_populated():
    store = load_vector_store()
    results = store.similarity_search("overtaking difficulty and safety car frequency", k=3)
    assert len(results) == 3
    assert all(r.page_content.strip() for r in results)


_STOPWORDS = {"the", "this", "how", "a", "an", "of", "to", "at", "on", "and", "or",
              "is", "are", "their", "its", "how", "for", "with", "in"}


def _distinctive_keyword(phrase: str) -> str:
    """The longest non-stopword in a FEATURE_PHRASES entry -- many phrases
    start with "the"/"this"/"how", so checking only the first word (as an
    earlier version of this test did) passes for almost any English text and
    can't actually catch a rambling or off-topic explanation."""
    words = [w.strip(".,") for w in phrase.lower().split() if w.strip(".,") not in _STOPWORDS]
    return max(words, key=len) if words else phrase.lower()


def test_explain_produces_grounded_output():
    df = pd.read_parquet(DATA_PATH)
    row = df[df["target_finish_position"].notna()].sort_values("race_date").tail(1)

    result = explain(row, target="finish_position")
    print(f"  prediction={result['prediction']:.1f} sources={result['sources']}")

    assert isinstance(result["prediction"], float)
    assert len(result["sources"]) > 0, "retrieval returned no sources"
    explanation = result["explanation"].strip()
    assert len(explanation) > 40, f"explanation suspiciously short: {explanation!r}"

    # require at least 2 of the top-3 SHAP features' distinctive keywords to
    # actually show up -- not proof of correctness, but a much harder bar to
    # clear by accident than matching any single common word
    top3 = result["top_features"][:3]
    keywords = [_distinctive_keyword(f["phrase"]) for f in top3]
    matches = sum(1 for kw in keywords if kw in explanation.lower())
    assert matches >= 2, f"explanation references only {matches}/3 top SHAP features (keywords={keywords}): {explanation!r}"

    # the fine-tuned style is a handful of tight sentences; a rambling
    # explanation (the base model's documented failure mode) reads much
    # longer than this in practice
    word_count = len(explanation.split())
    assert word_count <= 150, f"explanation is unusually long/rambling ({word_count} words): {explanation!r}"


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} checks passed")
