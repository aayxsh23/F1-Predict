"""Predict a real upcoming race — not a historical replay.

The design principle: don't reimplement anything. Historical rows already
carry every leakage-safe rolling stat computed by the Phase 1 feature layers;
a live prediction is just those same layers re-run on the historical table
with one extra race's worth of rows appended on the end, built from whatever
real session data genuinely exists for it right now. Sessions that haven't
happened yet fail to load and their columns stay NaN — there's no separate
"stage" system to maintain, the model's native NaN handling already does
this (see mask_for_stage in predict.py, which does the same thing for
testing on a historical row instead of live data).

Two columns can never be filled in early no matter what's happened, because
they genuinely don't exist yet: grid_position before qualifying resolves
(including any post-quali penalties) and starting_tire_compound before it's
announced on race day. Both accept manual overrides for exactly that reason
— e.g. once stewards confirm a grid penalty, or a compound choice leaks pre-
race, you can supply the real value instead of the automatic proxy.
"""
from pathlib import Path

import fastf1
import numpy as np
import pandas as pd

from src.data.fastf1_client import event_schedule, load_session
from src.data.ingest import _practice_pace, _quali_features, _weather_features
from src.features import build_dataset
from src.models.predict import load_model, predict

# a session that hasn't happened yet is the expected, common case here (not
# an error) — FastF1 logs it as a full traceback by default, which is noisy
# for something we already handle via try/except below
fastf1.set_log_level("CRITICAL")

NAN_WEATHER = dict(air_temp=np.nan, track_temp=np.nan, rain_probability=np.nan, wind_speed=np.nan, wet_track_probability=np.nan)


def _current_lineup(raw: pd.DataFrame) -> pd.DataFrame:
    """(driver, team) pairing from the most recent race we have data for —
    driver lineups rarely change mid-season. Use `lineup_overrides` for a
    known substitution."""
    last = raw.sort_values("race_date").iloc[-1]
    mask = (raw["season"] == last["season"]) & (raw["round"] == last["round"])
    return raw.loc[mask, ["driver", "team"]].drop_duplicates().reset_index(drop=True)


def _latest_available_weather(season: int, round_number: int) -> dict:
    """Weather firms up through the weekend (plan's own framing) — use
    whichever real session happened most recently."""
    for code in ("Q", "FP3", "FP2", "FP1"):
        try:
            s = load_session(season, round_number, code, laps=False, weather=True)
            w = _weather_features(s)
            if not all(pd.isna(v) for v in w.values()):
                return w
        except Exception:
            continue
    return dict(NAN_WEATHER)


def _quali_grid_proxy(season: int, round_number: int) -> dict:
    """Best-known starting order before the real grid is published: current
    qualifying classification. Doesn't reflect grid penalties applied after
    quali — pass `grid_overrides` once those are known."""
    try:
        q = load_session(season, round_number, "Q", laps=False, weather=False)
        return q.results.set_index("Abbreviation")["Position"].to_dict()
    except Exception:
        return {}


def build_live_rows(
    season: int,
    round_number: int,
    lineup_overrides: dict[str, str] | None = None,
    grid_overrides: dict[str, float] | None = None,
    compound_overrides: dict[str, str] | None = None,
) -> pd.DataFrame:
    """One row per driver for (season, round_number), in the exact feature
    shape the trained model expects — ready to pass straight to `predict()`."""
    raw = build_dataset.load_raw()
    event = event_schedule(season)
    event = event[event["RoundNumber"] == round_number]
    if event.empty:
        raise ValueError(f"round {round_number} not found in the {season} schedule")
    location, race_date = event.iloc[0]["Location"], event.iloc[0]["EventDate"]

    lineup = _current_lineup(raw)
    for driver, team in (lineup_overrides or {}).items():
        lineup.loc[lineup["driver"] == driver, "team"] = team

    try:
        practice = _practice_pace(season, round_number)
    except Exception:
        practice = pd.DataFrame(columns=["Driver", "practice_pace"])
    try:
        quali = _quali_features(season, round_number)
    except Exception:
        quali = pd.DataFrame(columns=["Abbreviation", "quali_gap_to_pole", "q1_time", "q2_time", "q3_time"])
    weather = _latest_available_weather(season, round_number)
    grid_lookup = {**_quali_grid_proxy(season, round_number), **(grid_overrides or {})}
    compound_lookup = compound_overrides or {}

    rows = []
    for _, r in lineup.iterrows():
        driver, team = r["driver"], r["team"]
        row = {
            "driver": driver, "driver_number": np.nan, "team": team,
            "grid_position": grid_lookup.get(driver, np.nan),
            "finish_position": np.nan, "points": np.nan, "status": np.nan, "dnf": False,
            "race_pace_pct": np.nan, "num_pit_stops": np.nan,
            "starting_compound": compound_lookup.get(driver, np.nan),
            "quali_gap_to_pole": np.nan, "q1_time": np.nan, "q2_time": np.nan, "q3_time": np.nan,
            "practice_pace": np.nan,
            **weather,
            "season": season, "round": round_number, "location": location, "race_date": pd.Timestamp(race_date),
        }
        if not quali.empty:
            match = quali.loc[quali["Abbreviation"] == driver]
            if not match.empty:
                for col in ("quali_gap_to_pole", "q1_time", "q2_time", "q3_time"):
                    row[col] = match.iloc[0][col]
        if not practice.empty:
            match = practice.loc[practice["Driver"] == driver]
            if not match.empty:
                row["practice_pace"] = match.iloc[0]["practice_pace"]
        rows.append(row)

    live_raw = pd.DataFrame(rows)
    combined = pd.concat([raw, live_raw], ignore_index=True)
    full_matrix = build_dataset.build(raw=combined)
    live_matrix = full_matrix[(full_matrix["season"] == season) & (full_matrix["round"] == round_number)]
    return live_matrix.reset_index(drop=True)


def predict_upcoming_race(season: int, round_number: int, **overrides) -> pd.DataFrame:
    rows = build_live_rows(season, round_number, **overrides)
    model = load_model()
    rows = rows.copy()
    rows["predicted_finish_position"] = predict(model, rows).round(2)
    display_cols = ["driver", "team", "grid_position", "quali_gap_to_pole", "practice_pace", "predicted_finish_position"]
    return rows.sort_values("predicted_finish_position")[display_cols].reset_index(drop=True)


if __name__ == "__main__":
    import sys

    season, round_number = int(sys.argv[1]), int(sys.argv[2])
    result = predict_upcoming_race(season, round_number)
    known = result[["grid_position", "quali_gap_to_pole", "practice_pace"]].notna().any().to_dict()
    print(f"season {season} round {round_number} - sessions with real data so far: "
          f"practice={known['practice_pace']}, quali={known['quali_gap_to_pole']}, grid={known['grid_position']}")
    print(result.to_string(index=False))
