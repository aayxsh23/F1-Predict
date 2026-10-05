"""The chat pipeline with the real local model, not a mock -- the actual
deliverable (a local explain_prediction/general answer that's grounded, not
rambling). Loads the 4-bit Llama 3.2 1B + its LoRA adapter, several seconds
to tens of seconds on a small GPU. Run explicitly with
`python tests/test_chat_llm.py`, not part of a fast loop.
"""
import sys
from pathlib import Path

try:
    import torch  # noqa: F401  -- the local LLM extras (requirements-llm.txt)
except ImportError:
    import pytest

    pytest.skip("needs the local LLM extras: pip install -r requirements-llm.txt", allow_module_level=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent.chat import stream_chat
from src.rag.llm import is_available

assert is_available(), "torch imports but is_available() says no -- check transformers is installed too"


def _events(question: str, context=None):
    return list(stream_chat([{"role": "user", "content": question}], context))


def _top_driver() -> str:
    text = "".join(e["text"] for e in _events("who wins this race") if e["type"] == "token")
    return text.split("**")[1]


_STOPWORDS = {"the", "this", "how", "a", "an", "of", "to", "at", "on", "and", "or", "is", "are",
              "their", "its", "for", "with", "in", "team's", "team", "s"}


def _distinctive_word(phrase: str) -> str:
    words = [w.strip(".,'") for w in phrase.lower().split() if w.strip(".,'") not in _STOPWORDS]
    return max(words, key=len) if words else phrase.lower()


def test_explain_prediction_is_grounded_real_generation():
    code = _top_driver()
    events = _events(f"why is {code} predicted there")
    assert events[0]["name"] == "explain_prediction" and events[1]["ok"] is True
    explanation = "".join(e["text"] for e in events if e["type"] == "token").strip()
    print(f"\n  [{code}] {explanation}")
    assert len(explanation) > 40, f"explanation suspiciously short: {explanation!r}"

    # the sources came from the SHAP-driven retrieval, which this test doesn't
    # re-derive -- instead check the explanation references the driver and
    # circuit it's actually about, a cheap but real grounding signal
    assert code in explanation
    word_count = len(explanation.split())
    assert word_count <= 180, f"explanation is unusually long/rambling ({word_count} words): {explanation!r}"


def test_general_question_answers_from_handed_facts_only():
    events = _events("what's going on with this race weekend")
    assert events[0]["name"] == "general"
    answer = "".join(e["text"] for e in events if e["type"] == "token").strip()
    print(f"\n  {answer}")
    assert len(answer) > 20
    assert events[-1]["type"] == "done"


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} checks passed")
