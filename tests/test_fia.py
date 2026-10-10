"""FIA documents: the grid and upgrade parsers read the PDFs' text layout
correctly, and upgrade features only ever look at earlier races.
Run with `python tests/test_fia.py`."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.fia import parse_grid, parse_upgrades
from src.features.team_features import _upgrades_recent

# text as pypdf extracts it: a car with no lap time is followed straight by its team line
GRID = """Final Starting Grid
1 1 Max VERSTAPPEN
Oracle Red Bull Racing
1:22.207
10 22 Yuki TSUNODA
Oracle Red Bull Racing
12 55 Carlos SAINZ
Atlassian Williams Racing
1:23.042
2 4 Lando NORRIS
"""

UPGRADES = """Car Presentation - Miami Grand Prix
ORACLE RED BULL RACING
1 Floor Edge Reliability A support has been removed.
MCLAREN FORMULA 1 TEAM
1 Front Wing Performance -
Flow Conditioning New front wing.
2 Beam Wing Circuit specific -
Drag Range Offloaded beam wing.
3 Floor Body Performance - Local Load
VISA CASH APP RB FORMULA ONE TEAM
No updates submitted for this event.
"""


def test_grid_lines_never_run_into_the_next_car():
    assert parse_grid(GRID) == {1: 1, 22: 10, 55: 12, 4: 2}


def test_upgrades_count_performance_parts_per_team():
    assert parse_upgrades(UPGRADES) == {"Red Bull Racing": 0, "McLaren": 2, "Racing Bulls": 0}


def test_recent_upgrades_exclude_this_race():
    df = pd.DataFrame({"team": ["A"] * 4, "season": 2025, "round": [1, 2, 3, 4],
                       "race_date": pd.date_range("2025-03-01", periods=4, freq="14D"),
                       "performance_upgrades": [5.0, 0.0, 2.0, 9.0]})
    assert _upgrades_recent(df).tolist()[1:] == [5.0, 5.0, 7.0]  # round 4 sees 1-3, never its own 9
    assert pd.isna(_upgrades_recent(df).iloc[0])


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} checks passed")
