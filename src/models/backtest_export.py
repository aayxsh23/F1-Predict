"""Historical actual-vs-predicted export for the /history view. Reads
data/processed/model_matrix.parquet directly -- that file already IS the
built feature table (build_dataset.build()'s own persisted output), so there's
no reason to rebuild it from raw data a second time just to read it back.
Run periodically (e.g. after a retrain), not on every refresh_job.py tick
like live predictions -- historical results don't change."""
import json
from pathlib import Path

import pandas as pd

from src.models.predict import CANONICAL_PRED_COLS, predict_all

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "model_matrix.parquet"
OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "predictions" / "backtest"

TARGET_COLS = {
    "qualifying": "target_qualifying_gap",
    "finish_position": "target_finish_position",
    "quali_delta": "target_quali_to_race_delta",
    "race_time": "target_race_time_gap",
}


def _clean(v):
    # round(float(v), 4) rather than a bare float(v) -- float32 -> float64
    # widening otherwise reintroduces noise into an already-rounded value
    # (e.g. XGBoost's float32 0.20 prints as 0.20000000298023224 once cast)
    return None if pd.isna(v) else round(float(v), 4)


def build_race_payload(predicted_df: pd.DataFrame, season: int, round_number: int) -> dict:
    """predicted_df: the FULL dataset, already run through predict_all() once
    -- reloading each of the 4 models per race (105+ times) instead of once
    for the whole table would be pure waste, since predict_all() doesn't care
    about grouping."""
    race = predicted_df[(predicted_df["season"] == season) & (predicted_df["round"] == round_number)]

    drivers = []
    for _, row in race.iterrows():
        entry = {"driver": row["driver"], "team": row["team"]}
        for target, pred_col in CANONICAL_PRED_COLS.items():
            entry[target] = {"actual": _clean(row[TARGET_COLS[target]]), "predicted": _clean(row[pred_col])}
        drivers.append(entry)

    return {"season": season, "round": round_number, "location": race.iloc[0]["location"], "drivers": drivers}


def export_all() -> list[dict]:
    df = pd.read_parquet(DATA_PATH)
    predicted_df = predict_all(df)
    races = df[["season", "round", "location", "race_date"]].drop_duplicates().sort_values("race_date")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    index = []
    for _, r in races.iterrows():
        season, round_number = int(r["season"]), int(r["round"])
        payload = build_race_payload(predicted_df, season, round_number)
        (OUT_DIR / f"{season}_{round_number}.json").write_text(json.dumps(payload, indent=2))
        index.append({"season": season, "round": round_number, "location": r["location"]})

    (OUT_DIR / "index.json").write_text(json.dumps(index, indent=2))
    return index


if __name__ == "__main__":
    written = export_all()
    print(f"exported backtest data for {len(written)} races -> {OUT_DIR}")
