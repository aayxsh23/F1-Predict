"""Merge the circuit -> driver -> team -> relative -> weather/strategy layers
into the final model matrix (data/processed/model_matrix.parquet)."""
from pathlib import Path

import pandas as pd

from src.features import circuit_reference, driver_features, relative_features, strategy_features, team_features
from src.models.catalog import LOCATION_ALIASES, TEAM_LINEAGE

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "races"
UPGRADES_DIR = RAW_DIR.parent / "upgrades"  # src/data/fia.py
TESTING_DIR = RAW_DIR.parent / "testing"  # src/data/ingest.py --testing
OUT_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "model_matrix.parquet"

CIRCUIT_COLS = [
    "overtaking_difficulty", "is_street_circuit", "pit_lane_loss_time", "safety_car_frequency",
    "dnf_rate", "longest_straight_m", "braking_zone_count", "tyre_degradation_level",
    "rain_race_frequency", "track_length_km",
]
# forecasts for race start (src/data/weather.py), not measured race weather:
# a live prediction only ever has the forecast
WEATHER_COLS = ["air_temp_forecast", "rain_mm_forecast", "wind_kph_forecast"]
# race-level facts carried through for the lap-time and race-duration estimates
RACE_INFO_COLS = ["race_start_utc", "practice_fastest_s", "quali_pole_s", "race_winner_time_s", "race_laps", "sc_laps"]
# sprint weekends only (NaN elsewhere): Sprint Qualifying gap to its pole as %
# of the pole lap, Sprint finishing position, Sprint green-flag pace (% over
# the fastest lap). Same-weekend facts known before the Grand Prix qualifying.
SPRINT_COLS = ["sprint_quali_gap_pct", "sprint_finish_position", "sprint_race_pace_pct"]
KEY = ["season", "round", "driver"]
# the first season loaded: the models' history and (see train.Target.first_season)
# their training rows. 2021 was added on 2026-10-10 after a tuning-window test;
# raw files from earlier seasons may sit in data/raw/races without being used.
FIRST_SEASON = 2021


def _load_dir(path: Path) -> pd.DataFrame:
    files = sorted(path.glob("*.parquet"))
    return pd.concat((pd.read_parquet(f) for f in files), ignore_index=True) if files else pd.DataFrame()


def load_raw() -> pd.DataFrame:
    raw = _load_dir(RAW_DIR)
    if raw.empty:
        raise FileNotFoundError(f"No raw race data in {RAW_DIR} — run `python -m src.data.ingest` first.")
    return raw[raw["season"] >= FIRST_SEASON].reset_index(drop=True)


def load_upgrades() -> pd.DataFrame:
    return _load_dir(UPGRADES_DIR)


def load_testing() -> pd.DataFrame:
    return _load_dir(TESTING_DIR)


def build(raw: pd.DataFrame | None = None, upgrades: pd.DataFrame | None = None,
          testing: pd.DataFrame | None = None) -> pd.DataFrame:
    """raw: pre-loaded raw table to use instead of reading data/raw/races/ --
    live prediction passes historical rows plus one appended not-yet-happened
    race here, so the exact same feature layers below (including every
    leakage-safe rolling stat) run identically for training and live rows.
    upgrades / testing default to the committed tables; pass an empty frame
    to build without them."""
    raw = raw if raw is not None else load_raw()
    upgrades = upgrades if upgrades is not None else load_upgrades()
    testing = testing if testing is not None else load_testing()
    team_name = raw["team"]
    # history follows the venue and the constructor, not the name on the entry
    # list: "Miami Gardens" is Miami, "RB" is AlphaTauri renamed
    raw = raw.assign(location=raw["location"].replace(LOCATION_ALIASES), team=raw["team"].replace(TEAM_LINEAGE))
    # FastF1 records a pit-lane start as grid 0: that's the back, not pole
    n_cars = raw.groupby(["season", "round"])["driver"].transform("count")
    raw["grid_position"] = raw["grid_position"].mask(raw["grid_position"] == 0, n_cars)
    if not upgrades.empty:  # a live race may already carry its own (src/data/fia.py)
        ups = upgrades.assign(team=upgrades["team"].replace(TEAM_LINEAGE)).groupby(["season", "round", "team"], as_index=False)["performance_upgrades"].sum()
        table = raw[["season", "round", "team"]].merge(ups, on=["season", "round", "team"], how="left")["performance_upgrades"]
        raw["performance_upgrades"] = raw.reindex(columns=["performance_upgrades"])["performance_upgrades"].fillna(table.set_axis(raw.index))
    raw = circuit_reference.resolve(raw)

    driver = driver_features.build(raw)
    team = team_features.build(raw, testing if not testing.empty else None)
    relative = relative_features.build(raw.merge(driver[KEY + ["driver_recent_form"]], on=KEY))
    strategy = strategy_features.build(raw)

    out = raw.reindex(columns=KEY + ["team", "location", "race_date"] + CIRCUIT_COLS + WEATHER_COLS + RACE_INFO_COLS + SPRINT_COLS).copy()
    out["team"] = team_name.loc[out.index]  # the name they raced under, for display
    out["is_sprint_weekend"] = raw.reindex(columns=["sprint_weekend"])["sprint_weekend"].fillna(False).astype(float)
    # coarse steps: hourly forecast wobble shouldn't count as new information
    for col, step in zip(WEATHER_COLS, (1.0, 0.5, 5.0)):
        out[col] = (out[col] / step).round() * step
    # not model inputs: the qualifying order stands in for the grid until the
    # official one (penalties applied) is out (features.mask_for_stage), and
    # retirements are left to the odds sampler (train.py trains on finishers)
    out["quali_position"] = raw.reindex(columns=["quali_position"])["quali_position"]
    out["grid_official"] = raw.reindex(columns=["grid_official"])["grid_official"].fillna(1.0)  # history: always official
    out["dnf"] = raw["dnf"].fillna(False).astype(bool)
    out["target_finish_position"] = raw["finish_position"]
    out["target_quali_to_race_delta"] = raw["grid_position"] - raw["finish_position"]
    out["target_race_gap_pct"] = raw["race_gap_pct"]
    # gaps past 107% (no time, a crash) stay in training: dropping them was
    # tried and did slightly worse even on representative laps. They are
    # scored separately (train.py) and their chance is the odds' job.
    out["target_qualifying_gap_pct"] = raw["quali_gap_pct"]
    out["quali_gap_to_pole"] = raw["quali_gap_to_pole"]  # seconds, for display only

    for layer in (driver, team, relative, strategy):
        new_cols = [c for c in layer.columns if c not in KEY]
        out = out.merge(layer[KEY + new_cols], on=KEY, how="left")

    return out


def main():
    matrix = build()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_parquet(OUT_PATH, index=False)
    matrix.to_csv(OUT_PATH.with_suffix(".csv"), index=False)
    print(f"wrote {len(matrix)} rows, {matrix.shape[1]} columns to {OUT_PATH}")


if __name__ == "__main__":
    main()
