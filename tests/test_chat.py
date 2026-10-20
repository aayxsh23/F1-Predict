"""stream_chat()'s event contract (src/agent/chat.py): the SSE shape the
frontend renders, for both deterministic intents (no model at all) and the
two model-backed intents with the local model unavailable (the fast, always-on
path), and the lookup loop with a scripted model. Run with `python tests/test_chat.py`."""
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
    events = _events("what does article B1.6.2 say")
    kinds = [e["type"] for e in events]
    assert "sources" in kinds
    sources = next(e for e in events if e["type"] == "sources")["items"]
    assert sources and sources[0]["filename"]


def test_a_failed_lookup_is_reported_not_crashed():
    events = _events("why is he predicted there")  # no driver named, no context
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


def test_invented_numbers_and_names_are_caught():
    from src.agent.chat import ungrounded

    # both from real local-model answers
    facts = "Current race: Singapore Grand Prix\nTop of the forecast: ANT (Mercedes), VER (Red Bull)\nsafety cars (0.9), pit lane (24s)"
    assert ungrounded("Singapore sits over 300m above sea level.", facts) == ["300"]
    assert ungrounded("The current driver to beat is António Félix da Costa.", facts) == ["António", "Félix", "Costa"]
    assert ungrounded("Safety cars are common (0.9) here. ANT of Mercedes leads VER.", facts) == []


def test_agent_loop_runs_lookups_and_checks_the_answer():
    """A scripted model: one lookup, then an answer; then one that invents a number."""
    import json

    from src.agent.chat import agent_answer

    replies = iter(["<tool_call>\n<function=season_schedule>\n</function>\n</tool_call>", "There are {n} rounds this season."])
    rounds = {}

    def fake_generate(prompt, max_tokens=700):
        reply = next(replies)
        if "{n}" in reply:
            rounds["n"] = json.loads(prompt.split("<tool_response>\n")[-1].split("\n</tool_response>")[0])["rounds"]
            reply = reply.format(n=len(rounds["n"]))
        return reply

    with patch("src.rag.llm.generate", side_effect=fake_generate):
        events = list(agent_answer([{"role": "user", "content": "how many races this year?"}]))
    assert [e["type"] for e in events] == ["tool_start", "tool_end", "answer"]
    assert events[-1]["grounded"] and str(len(rounds["n"])) in events[-1]["text"]

    with patch("src.rag.llm.generate", return_value="Verstappen has 412 wins."):
        answer = list(agent_answer([{"role": "user", "content": "how many wins does max have?"}]))[-1]
    assert answer["grounded"] is False


def test_bracket_citations_become_plain_words():
    from src.agent.chat import _plain_sources

    assert _plain_sources("Six times【source: Jolpica-F1 race results】.") == "Six times (source: Jolpica-F1 race results)."
    assert _plain_sources("Six times【】.") == "Six times."


def test_a_question_about_another_race_names_it():
    from src.agent import tools
    from src.agent.chat import _on_screen

    now = tools.forecast()
    ask = [{"role": "user", "content": "who won here?"}]
    assert _on_screen(ask, {"season": now["season"], "round": now["round"]}) == ask, "the current weekend needs no note"
    other = _on_screen(ask, {"season": 2026, "round": 3, "race_name": "Japanese Grand Prix"})
    assert other[-1]["content"] == "who won here? (I'm looking at the 2026 Japanese Grand Prix.)"
    assert _on_screen(ask, None) == ask


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} checks passed")
