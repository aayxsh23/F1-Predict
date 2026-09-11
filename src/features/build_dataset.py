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
WEATHER_COLS = ["air_temp", "track_temp", "rain_probability", "wind_speed", "wet_track_probability"]
KEY = ["season", "round", "driver"]


def load_raw() -> pd.DataFrame:
    files = sorted(RAW_DIR.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No raw race data in {RAW_DIR} — run `python -m src.data.ingest` first.")
    return pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)


def build(raw: pd.DataFrame | None = None) -> pd.DataFrame:
    """raw: pre-loaded raw table to use instead of reading data/raw/races/ —
    live prediction passes historical rows plus one appended not-yet-happened
    race here, so the exact same feature layers below (including every
    leakage-safe rolling stat) run identically for training and live rows."""
    raw = circuit_reference.resolve(raw if raw is not None else load_raw())

    driver = driver_features.build(raw)
    team = team_features.build(raw)
    relative = relative_features.build(raw.merge(driver[KEY + ["driver_recent_form"]], on=KEY))
    strategy = strategy_features.build(raw)

    out = raw[KEY + ["team", "location", "race_date"] + CIRCUIT_COLS + WEATHER_COLS].copy()
    out["starting_tire_compound"] = raw["starting_compound"]
    out["target_finish_position"] = raw["finish_position"]
    out["target_quali_to_race_delta"] = raw["grid_position"] - raw["finish_position"]

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
