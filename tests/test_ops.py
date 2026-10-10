"""The operational checks: the freshness alarm, and scoring what the app
published. Synthetic inputs, no network. Run with `python tests/test_ops.py`."""
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.check_freshness import check
from src.models.live_record import score_snapshot

SINGAPORE = [{"RoundNumber": 17, "Location": "Marina Bay",
              "Session1": "Practice 1", "Session1DateUtc": "2026-10-09T08:30:00",
              "Session2": "Sprint Qualifying", "Session2DateUtc": "2026-10-09T12:30:00",
              "Session3": "Sprint", "Session3DateUtc": "2026-10-10T09:00:00",
              "Session4": "Qualifying", "Session4DateUtc": "2026-10-10T13:30:00",
              "Session5": "Race", "Session5DateUtc": "2026-10-11T12:00:00", "RaceStartUtc": "2026-10-11T12:00:00"}]


def _at(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def test_freshness_flags_a_session_missing_from_the_forecast():
    latest = {"season": 2026, "round": 17, "sessions": ["FP1", "SQ", "S"], "generated_at": "x"}
    stale, reason = check(SINGAPORE, latest, _at("2026-10-10T19:00:00"))  # qualifying ended hours ago
    assert stale and "Q" in reason
    assert not check(SINGAPORE, {**latest, "sessions": ["FP1", "SQ", "S", "Q"]}, _at("2026-10-10T19:00:00"))[0]
    assert not check(SINGAPORE, latest, _at("2026-10-10T15:00:00"))[0], "data can take an hour or two: not stale yet"


def test_freshness_wants_the_right_round_and_is_quiet_between_weekends():
    assert check(SINGAPORE, {"season": 2026, "round": 16, "sessions": []}, _at("2026-10-09T10:00:00"))[0]
    assert not check(SINGAPORE, {"season": 2026, "round": 16, "sessions": []}, _at("2026-10-05T10:00:00"))[0]


def test_live_record_scores_what_was_published():
    snap = {"label": "After qualifying", "stage": "post_quali", "generated_at": "t", "drivers": [
        {"driver": "AAA", "predicted_finish_position": 1.2, "predicted_qualifying_gap_pct": 0.0},
        {"driver": "BBB", "predicted_finish_position": 2.5, "predicted_qualifying_gap_pct": 0.2},
        {"driver": "CCC", "predicted_finish_position": 3.1, "predicted_qualifying_gap_pct": 0.4}]}
    res = pd.DataFrame({"driver": ["AAA", "BBB", "CCC"], "finish_position": [2.0, 1.0, 3.0], "dnf": [False] * 3,
                        "quali_gap_pct": [0.0, 0.1, 0.5], "quali_position": [1.0, 2.0, 3.0], "grid_position": [1.0, 2.0, 3.0]})
    s = score_snapshot(snap, res)
    assert s["winner_called"] is False and s["podium_called"] == 3
    assert abs(s["finish_mae"] - (0.8 + 1.5 + 0.1) / 3) < 1e-3 and abs(s["grid_baseline_mae"] - 2 / 3) < 1e-3
    assert "quali_mae" not in s, "qualifying is only scored on forecasts made before it"


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} checks passed")
