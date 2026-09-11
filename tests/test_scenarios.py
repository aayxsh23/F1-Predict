"""Self-check for the agent's championship scenario math (src/agent/scenarios.py)
-- deterministic, no network or LLM needed. Run with `python tests/test_scenarios.py`.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent.scenarios import closest_rival, max_points_available, remaining_rounds, title_scenario

SCHEDULE = [{"round": r, "race_name": f"Round {r}", "date": "2026-01-01", "is_sprint": r in (14, 18)} for r in range(1, 24)]


def test_remaining_rounds_excludes_completed():
    remaining = remaining_rounds(SCHEDULE, completed_through_round=13)
    assert len(remaining) == 10
    assert all(r["round"] > 13 for r in remaining)


def test_max_points_available_counts_sprints():
    remaining = remaining_rounds(SCHEDULE, completed_through_round=13)  # rounds 14-23, sprints at 14 and 18
    pool = max_points_available(remaining)
    assert pool == 10 * 25 + 2 * 8  # 10 races * race win, 2 of them also sprint wins


def test_closest_rival_works_for_leader_and_chaser():
    standings = [
        {"code": "ANT", "points": 267.0, "position": 1},
        {"code": "RUS", "points": 201.0, "position": 2},
        {"code": "NOR", "points": 171.0, "position": 4},
    ]
    # the leader's closest rival is whoever is 2nd
    assert closest_rival(standings, "ANT")["code"] == "RUS"
    # a chaser's closest rival is the actual leader, not whoever's directly ahead of them
    assert closest_rival(standings, "NOR")["code"] == "ANT"


def test_title_scenario_eliminated_driver():
    remaining = remaining_rounds(SCHEDULE, completed_through_round=20)  # only 3 rounds left, no sprints
    pool = max_points_available(remaining)  # 75
    result = title_scenario(driver_points=100.0, rival_points=100.0 + pool + 1, remaining=remaining)
    assert result["still_mathematically_in_contention"] is False
    # can't even draw level with a static rival, let alone one who's still racing
    assert result["points_needed_if_rival_scores_zero"] is None


def test_title_scenario_already_clinched():
    remaining = remaining_rounds(SCHEDULE, completed_through_round=20)
    pool = max_points_available(remaining)
    # driver already so far ahead the rival can't matter regardless of score
    result = title_scenario(driver_points=500.0, rival_points=100.0, remaining=remaining)
    assert result["still_mathematically_in_contention"] is True
    assert result["points_needed_if_rival_scores_zero"] == 0
    assert pool > 0  # sanity: this isn't trivially true because there are no races left


def test_title_scenario_realistic_midseason_battle():
    remaining = remaining_rounds(SCHEDULE, completed_through_round=13)  # 10 rounds, 2 sprints -> pool 266
    result = title_scenario(driver_points=171.0, rival_points=267.0, remaining=remaining)
    pool = 10 * 25 + 2 * 8
    assert result["max_points_available"] == pool
    assert result["points_behind"] == 96.0
    assert result["still_mathematically_in_contention"] is True  # 171+266=437 > 267
    # to draw level assuming the rival scores nothing more: close a 96-point gap
    assert result["points_needed_if_rival_scores_zero"] == 97.0


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} checks passed")
