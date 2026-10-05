"""stream_chat()'s event contract (src/agent/chat.py): the SSE shape the
frontend renders, for both deterministic intents (no model at all) and the
two LLM-backed intents with the local model unavailable (the fast, always-on
path -- a real generation is tested separately in test_chat_llm.py, which
needs the local-LLM extras and is slow). Run with `python tests/test_chat.py`."""
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent.chat import stream_chat


def _events(question: str, context=None):
    return list(stream_chat([{"role": "user", "content": question}], context))


def test_deterministic_question_is_tool_start_end_token_done():
    events = _events("who wins this race")
    kinds = [e["type"] for e in events]
    assert kinds[0] == "tool_start" and events[0]["name"] == "race_forecast"
    assert kinds[1] == "tool_end" and events[1]["ok"] is True
    assert "token" in kinds and kinds[-1] == "done"


def test_search_rules_emits_sources_before_the_answer():
    events = _events("what's the penalty for an unsafe release")
    kinds = [e["type"] for e in events]
    assert "sources" in kinds
    sources = next(e for e in events if e["type"] == "sources")["items"]
    assert sources and sources[0]["filename"]


def test_a_failed_lookup_is_reported_not_crashed():
    events = _events("why is this predicted that way")  # no driver named, no context
    end = next(e for e in events if e["type"] == "tool_end")
    assert end["ok"] is False
    assert events[-1]["type"] == "done"
    text = "".join(e["text"] for e in events if e["type"] == "token")
    assert "driver" in text.lower()


def _top_driver() -> str:
    text = "".join(e["text"] for e in _events("who wins this race") if e["type"] == "token")
    return text.split("**")[1]


@patch("src.rag.llm.is_available", return_value=False)
def test_explain_prediction_falls_back_without_the_local_model(_mock):
    top_driver = _top_driver()
    events = _events(f"why is {top_driver} predicted there")
    assert events[0]["name"] == "explain_prediction"
    assert any(e["type"] == "sources" for e in events)
    text = "".join(e["text"] for e in events if e["type"] == "token")
    assert top_driver in text and "effect" in text  # the plain-list fallback, not prose
    assert events[-1]["type"] == "done"


@patch("src.rag.llm.is_available", return_value=False)
def test_general_falls_back_without_the_local_model(_mock):
    events = _events("what colour is the sky")
    assert events[0]["name"] == "general"
    text = "".join(e["text"] for e in events if e["type"] == "token")
    assert "forecast" in text.lower() or "strategy" in text.lower()  # the capability-list fallback
    assert events[-1]["type"] == "done"


@patch("src.rag.llm.is_available", return_value=False)
def test_context_resolves_the_selected_driver_without_naming_them(_mock):
    # a driver already selected on screen should answer without the question naming them
    code = _top_driver()
    events = _events("why is he predicted there", context={"driver": code})
    end = next(e for e in events if e["type"] == "tool_end")
    assert end["ok"] is True


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} checks passed")
