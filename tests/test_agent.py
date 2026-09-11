"""Self-check for the Phase 6 agent's deterministic logic (routing +
answer-templating) -- no network or LLM needed, since these are exactly the
pieces that had real bugs on first real testing (see PROGRESS.md/LEARNING.md):
"championship" alone misclassifying to title_scenario instead of
constructor_standings, and an unrelated follow-up wrongly inheriting stale
intent. Run with `python tests/test_agent.py`.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import HumanMessage

from src.agent.graph import (
    _extract_driver_code,
    _format_constructor_standings_answer,
    _format_driver_standings_answer,
    _format_title_scenario_answer,
    classify_node,
)

DRIVER_STANDINGS = [
    {"code": "ANT", "given_name": "Andrea Kimi", "family_name": "Antonelli", "team": "Mercedes", "points": 267.0, "wins": 7, "position": 1},
    {"code": "RUS", "given_name": "George", "family_name": "Russell", "team": "Mercedes", "points": 201.0, "wins": 2, "position": 2},
    {"code": "NOR", "given_name": "Lando", "family_name": "Norris", "team": "McLaren", "points": 171.0, "wins": 2, "position": 4},
]
CONSTRUCTOR_STANDINGS = [
    {"name": "Mercedes", "points": 468.0, "wins": 9, "position": 1},
    {"name": "Ferrari", "points": 346.0, "wins": 2, "position": 2},
]


def test_constructor_keyword_beats_championship_keyword():
    # regression: "championship" alone used to win this race and misroute to
    # title_scenario -- "constructor" must take priority
    state = {"messages": [HumanMessage(content="Who is leading the constructors championship?")]}
    assert classify_node(state)["intent"] == "constructor_standings"


def test_title_scenario_needs_a_specific_phrase():
    state = {"messages": [HumanMessage(content="What does Norris need to do to win the title this weekend?")]}
    assert classify_node(state)["intent"] == "title_scenario"


def test_unrelated_followup_does_not_inherit_stale_intent():
    # regression: a completely unrelated question used to inherit whatever
    # intent the conversation last had, just because it matched no keyword
    state = {"messages": [HumanMessage(content="When did F1 start using turbo hybrid engines?")], "intent": "constructor_standings"}
    assert classify_node(state)["intent"] != "constructor_standings"


def test_explicit_continuation_does_inherit_previous_intent():
    state = {"messages": [HumanMessage(content="And what about the gap to the team in 2nd?")], "intent": "constructor_standings"}
    assert classify_node(state)["intent"] == "constructor_standings"


def test_driver_code_extraction_matches_code_or_surname():
    assert _extract_driver_code("How many points does Norris have?", DRIVER_STANDINGS) == "NOR"
    assert _extract_driver_code("what about NOR", DRIVER_STANDINGS) == "NOR"
    assert _extract_driver_code("who's leading", DRIVER_STANDINGS) is None


def test_driver_standings_answer_is_grounded_in_real_numbers():
    answer = _format_driver_standings_answer({"driver_standings": DRIVER_STANDINGS}, "How many points does Norris have?")
    assert "171" in answer and "Norris" in answer


def test_constructor_gap_is_actually_computed():
    # regression: this used to fall through to a generic top-5 list instead
    # of computing the actual point gap the question asked for
    answer = _format_constructor_standings_answer({"constructor_standings": CONSTRUCTOR_STANDINGS}, "what's the gap to 2nd?")
    assert "122" in answer  # 468 - 346


def test_title_scenario_answer_reports_real_points_behind():
    data = {
        "driver": DRIVER_STANDINGS[2], "rival": DRIVER_STANDINGS[0],
        "scenario": {
            "still_mathematically_in_contention": True, "points_behind": 96.0,
            "remaining_races": 10, "remaining_sprints": 1, "max_points_available": 258,
            "points_needed_if_rival_scores_zero": 97.0,
        },
    }
    answer = _format_title_scenario_answer(data)
    assert "96" in answer and "97" in answer


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} checks passed")
