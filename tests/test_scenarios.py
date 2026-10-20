"""Self-check for the agent's championship scenario math (src/agent/scenarios.py)
-- deterministic, no network or LLM needed. Run with `python tests/test_scenarios.py`.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent.scenarios import closest_rival, max_points_available, remaining_rounds, title_odds, title_scenario

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


def test_title_odds_sum_to_one_and_respect_elimination():
    remaining = remaining_rounds(SCHEDULE, completed_through_round=20)  # 3 races, 75 points left
    points = [300.0, 280.0, 100.0]  # the third driver can no longer catch the leader
    odds = title_odds(points, [1.0, 1.5, 2.0], remaining, tau=2.0, p_dnf=0.05, groups=["A", "B", "A"], n=3000)
    assert abs(odds["champion"].sum() - 1) < 1e-9
    assert odds["champion"][2] == 0 and odds["champion"][0] > odds["champion"][1] > 0
    assert abs(sum(odds["team_champion"].values()) - 1) < 1e-9


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} checks passed")


def test_title_odds_strength_ignores_this_weekends_grid(monkeypatch):
    """After qualifying the shown forecast includes the grid; a grid penalty
    must not count in every remaining round (Singapore 2026)."""
    from src.agent import tools

    payload = {"season": 2026, "round": 17, "drivers": [
        {"driver": "RUS", "predicted_finish_position": 9.5, "season_strength": 5.0},
        {"driver": "VER", "predicted_finish_position": 1.7, "season_strength": 4.8}]}
    assert tools.season_strength(payload) == {"RUS": 5.0, "VER": 4.8}

    old = {"season": 2026, "round": 17, "drivers": [{"driver": "RUS", "predicted_finish_position": 9.5},
                                                     {"driver": "VER", "predicted_finish_position": 1.7}]}
    timeline = [{"label": "Before practice", "stage": "pre_weekend", "drivers": [
        {"driver": "RUS", "predicted_finish_position": 4.97}, {"driver": "VER", "predicted_finish_position": 4.83}]},
        {"label": "Official grid", "stage": "race_day", "drivers": []}]
    monkeypatch.setattr(tools, "get_json", lambda name: timeline)
    assert tools.season_strength(old) == {"RUS": 4.97, "VER": 4.83}, "older forecasts: the weekend's pre-practice snapshot"


def test_title_ties_go_to_fia_countback():
    import numpy as np

    level = [100, 100]
    by_wins = title_odds(level, [1, 2], [], tau=1.0, places=np.array([[2, 0, 0, 0, 0], [1, 5, 0, 0, 0]]), n=50)
    assert list(by_wins["champion"]) == [1.0, 0.0], "more wins takes a tie on points"
    by_seconds = title_odds(level, [1, 2], [], tau=1.0, places=np.array([[1, 3, 0, 0, 0], [1, 5, 0, 0, 0]]), n=50)
    assert list(by_seconds["champion"]) == [0.0, 1.0], "then more second places"
    assert list(title_odds(level, [1, 2], [], tau=1.0, n=50)["champion"]) == [0.5, 0.5], "nothing to separate them: split"


def test_team_drift_widens_late_season_uncertainty():
    rounds = [{"round": i, "is_sprint": False} for i in range(10)]
    steady = title_odds([0, 0, 0, 0], [1, 2, 3, 4], rounds, tau=1.5, groups=["A", "A", "B", "B"], n=4000)
    drifting = title_odds([0, 0, 0, 0], [1, 2, 3, 4], rounds, tau=1.5, groups=["A", "A", "B", "B"], n=4000, drift=1.0)
    assert drifting["champion"][0] < steady["champion"][0]
