"""Self-check for the OpenF1 fallback (src/data/openf1.py), no network. Run
with `python tests/test_openf1.py`.

Checked against FastF1 by ingesting the same races both ways (2025 Austin,
2026 Baku, Kuala Lumpur and Singapore, 2026-10-19): grid, results, points,
qualifying, practice and long-run pace, safety-car laps and the lap table
agree, except lap 1 (OpenF1 times it about 0.4 s longer), stints changed
under a red flag, and pit-lane starters' first lap.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.openf1 import DELETED, lap_status, track_status  # noqa: E402


def _rc(rows):
    return pd.DataFrame(rows, columns=["date", "category", "flag", "scope", "sector", "message"])


def test_track_status_follows_race_control_like_fastf1():
    rc = _rc([
        ("2026-10-11T12:00:00+00:00", "Flag", "GREEN", "Track", None, "GREEN LIGHT - PIT EXIT OPEN"),
        ("2026-10-11T12:10:00+00:00", "Flag", "YELLOW", "Sector", 5, "YELLOW IN TRACK SECTOR 5"),
        ("2026-10-11T12:10:20+00:00", "Flag", "CLEAR", "Sector", 5, "CLEAR IN TRACK SECTOR 5"),
        ("2026-10-11T12:16:35+00:00", "SafetyCar", None, None, None, "VSC DEPLOYED"),
        ("2026-10-11T12:17:40+00:00", "SafetyCar", None, None, None, "VSC ENDING"),
        ("2026-10-11T12:17:54+00:00", "Flag", "CLEAR", "Track", None, "TRACK CLEAR"),
        ("2026-10-11T12:30:00+00:00", "SafetyCar", None, None, None, "SAFETY CAR DEPLOYED"),
        ("2026-10-11T12:33:00+00:00", "SafetyCar", None, None, None, "SAFETY CAR IN THIS LAP"),
        ("2026-10-11T12:35:00+00:00", "Flag", "GREEN", "Track", None, "TRACK CLEAR"),
        ("2026-10-11T12:50:00+00:00", "Flag", "RED", "Track", None, "RED FLAG"),
    ])
    events = track_status(rc)
    assert events["status"].tolist() == ["1", "2", "1", "6", "7", "1", "4", "1", "5"], events
    t = lambda hm: pd.Timestamp(f"2026-10-11T12:{hm}:00", tz="UTC")  # noqa: E731
    laps = lap_status(pd.Series([t("01"), t("09"), t("16"), t("31"), t("40"), t("49")]),
                      pd.Series([t("03"), t("11"), t("18"), t("32"), t("42"), t("51")]), events)
    # green, a yellow, VSC -> ending -> green, under the safety car, green, red flag
    assert laps == ["1", "121", "1671", "4", "1", "15"], laps


def test_deleted_laps_are_read_with_their_time():
    msg = pd.Series(["CAR 6 (HAD) TIME 1:32.048 DELETED - TRACK LIMITS AT TURN 19 LAP 15 21:36:26",
                     "CAR 44 (HAM) LAP DELETED - TRACK LIMITS AT TURN 16 LAP 12 21:21:42 (PIT)",
                     "WAVED BLUE FLAG FOR CAR 77 (BOT) TIMED AT 20:38:16"])
    found = msg.str.extract(DELETED).dropna().to_numpy().tolist()
    assert found == [["6", "1", "32.048"]], found


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} checks passed")
