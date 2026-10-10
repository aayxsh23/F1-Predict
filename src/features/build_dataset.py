"""Merge the circuit -> driver -> team -> relative -> weather/strategy layers
into the final model matrix (data/processed/model_matrix.parquet)."""
from pathlib import Path

import pandas as pd

from src.features import circuit_reference, driver_features, relative_features, strategy_features, team_features

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "races"
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


def load_raw() -> pd.DataFrame:
    files = sorted(RAW_DIR.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No raw race data in {RAW_DIR} — run `python -m src.data.ingest` first.")
    return pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)


def build(raw: pd.DataFrame | None = None) -> pd.DataFrame:
    """raw: pre-loaded raw table to use instead of reading data/raw/races/ --
    live prediction passes historical rows plus one appended not-yet-happened
    race here, so the exact same feature layers below (including every
    leakage-safe rolling stat) run identically for training and live rows."""
    raw = circuit_reference.resolve(raw if raw is not None else load_raw())

    driver = driver_features.build(raw)
    team = team_features.build(raw)
    relative = relative_features.build(raw.merge(driver[KEY + ["driver_recent_form"]], on=KEY))
    strategy = strategy_features.build(raw)

    out = raw.reindex(columns=KEY + ["team", "location", "race_date"] + CIRCUIT_COLS + WEATHER_COLS + RACE_INFO_COLS + SPRINT_COLS).copy()
    out["is_sprint_weekend"] = raw.reindex(columns=["sprint_weekend"])["sprint_weekend"].fillna(False).astype(float)
    # coarse steps: hourly forecast wobble shouldn't count as new information
    for col, step in zip(WEATHER_COLS, (1.0, 0.5, 5.0)):
        out[col] = (out[col] / step).round() * step
    out["starting_tire_compound"] = raw["starting_compound"]
    out["target_finish_position"] = raw["finish_position"]
    out["target_quali_to_race_delta"] = raw["grid_position"] - raw["finish_position"]
    out["target_race_gap_pct"] = raw["race_gap_pct"]
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
