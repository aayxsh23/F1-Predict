"""Feature rows for a real upcoming race, not a historical replay.

Nothing is reimplemented: a live prediction is the historical raw table with
one more race appended, built from whatever session data exists right now, run
through the exact same feature layers as training. Sessions that haven't
happened yet simply leave their columns NaN.

Two FIA documents join in when published (src/data/fia.py): the teams' car
upgrades (Friday, before practice) and the official starting grid with
penalties applied (race morning). Until the grid is out, the qualifying order
stands in for it, as it does in training at that stage. `grid_overrides`
still lets a known grid be forced.
"""
import fastf1
import numpy as np
import pandas as pd

from src.data import fia
from src.data.fastf1_client import event_schedule, load_session
from src.data.ingest import add_sprint, practice_features, quali_features, race_start
from src.data.weather import race_forecast
from src.features import build_dataset
from src.features.circuit_reference import coords
from src.models.catalog import TEAM_LINEAGE

# a session that hasn't happened yet is the normal case here, not an error
fastf1.set_log_level("CRITICAL")


def _current_lineup(raw: pd.DataFrame) -> pd.DataFrame:
    """(driver, team, car number) from the latest race we have. Use
    `lineup_overrides` for a known substitution."""
    last = raw.sort_values("race_date").iloc[-1]
    mask = (raw["season"] == last["season"]) & (raw["round"] == last["round"])
    return raw.loc[mask, ["driver", "team", "driver_number"]].drop_duplicates("driver").reset_index(drop=True)


def event_info(season: int, round_number: int) -> pd.Series:
    sched = event_schedule(season)
    event = sched[sched["RoundNumber"] == round_number]
    if event.empty:
        raise ValueError(f"round {round_number} not found in the {season} schedule")
    return event.iloc[0]


def build_live_rows(
    season: int,
    round_number: int,
    lineup_overrides: dict[str, str] | None = None,
    grid_overrides: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """One feature row per driver for (season, round), plus the sessions that
    have data so far, in weekend order (e.g. ["FP1", "FP2"] or ["FP1", "SQ", "S"])."""
    raw = build_dataset.load_raw()
    event = event_info(season, round_number)
    location, start = event["Location"], race_start(event)

    lineup = _current_lineup(raw)
    for driver, team in (lineup_overrides or {}).items():
        lineup.loc[lineup["driver"] == driver, "team"] = team

    try:
        practice, practice_fastest, sessions = practice_features(season, round_number)
    except Exception:
        practice, practice_fastest, sessions = pd.DataFrame(columns=["Driver"]), np.nan, []
    # sprint weekends: Sprint Qualifying and the Sprint, NaN until each has run
    sprint = add_sprint(lineup[["driver"]], season, round_number, event["EventFormat"])
    sessions = sessions + [code for code, col in (("SQ", "sprint_quali_gap_pct"), ("S", "sprint_finish_position"))
                           if sprint[col].notna().any()]
    try:
        quali = quali_features(season, round_number)
    except Exception:
        quali = pd.DataFrame(columns=["Abbreviation", "quali_gap_pct"])
    if quali["quali_gap_pct"].notna().any():  # a not-yet-run session loads fine, just empty
        sessions = sessions + ["Q"]
    weather = race_forecast(*coords(location, season), start)
    quali_pos = quali.set_index("Abbreviation").get("quali_position", pd.Series(dtype=float))
    grid = lineup["driver"].map(quali_pos.to_dict())
    official = fia.starting_grid(season, event["EventName"]) if "Q" in sessions else None
    if official:
        numbers = pd.to_numeric(lineup["driver_number"], errors="coerce")
        grid = numbers.map(official)
        grid[grid.isna()] = len(official) + grid.isna().cumsum()[grid.isna()]  # pit-lane starters: the back
    grid = grid.where(~lineup["driver"].isin(list(grid_overrides or {})), lineup["driver"].map(grid_overrides or {}))
    upgrades = fia.upgrades(season, event["EventName"]) or {}

    live = lineup.assign(
        grid_position=grid, grid_official=1.0 if official else 0.0,
        performance_upgrades=lineup["team"].replace(TEAM_LINEAGE).map({TEAM_LINEAGE.get(k, k): v for k, v in upgrades.items()})
        if upgrades else np.nan,
        finish_position=np.nan, points=np.nan, status=np.nan, dnf=False, race_pace_pct=np.nan, num_pit_stops=np.nan,
        practice_fastest_s=practice_fastest, season=season, round=round_number, location=location,
        race_date=pd.Timestamp(event["EventDate"]), race_start_utc=pd.Timestamp(start), **weather,
    )
    live = live.merge(quali.rename(columns={"Abbreviation": "driver"}), on="driver", how="left")
    live = live.merge(practice.rename(columns={"Driver": "driver"}), on="driver", how="left")
    live = live.merge(sprint, on="driver", how="left")

    full = build_dataset.build(raw=pd.concat([raw, live.astype({"dnf": bool})], ignore_index=True))
    rows = full[(full["season"] == season) & (full["round"] == round_number)].reset_index(drop=True)
    return rows.merge(lineup[["driver", "driver_number"]], on="driver", how="left"), sessions


if __name__ == "__main__":
    import sys

    from src.models.predict import predict_all

    rows, sessions = build_live_rows(int(sys.argv[1]), int(sys.argv[2]))
    out = predict_all(rows).sort_values("predicted_finish_position")
    print("sessions so far:", sessions or "none")
    print(out[["driver", "team", "grid_position", "predicted_qualifying_gap_pct", "predicted_finish_position",
               "predicted_race_gap_pct"]].to_string(index=False))
