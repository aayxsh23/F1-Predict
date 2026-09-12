"""GitHub Actions glue: builds the live prediction for the next/current race
and writes it to data/predictions/ as the JSON the FastAPI backend serves.
This is the ONLY place that calls the slow FastF1-touching live_predict path
-- it runs on a schedule inside the workflow, never inside a user-facing
request (see phase7-ui-backend-plan.md's "freshness lives in GitHub, not
backend uptime" decision). The per-driver feature_row is stored raw (not a
precomputed SHAP explanation) so the API's /explain endpoint can run
shap_explanation() on demand, cheaply, without touching FastF1 at request time."""
import json
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from src.data.fastf1_client import event_schedule
from src.models.features import FEATURE_COLS
from src.models.live_predict import build_live_rows, predict_upcoming_race

OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "predictions"


def next_or_current_round(season: int) -> int:
    """The race to predict: the first round whose date hasn't passed yet, or
    the season's last round if every race this season is already done."""
    sched = event_schedule(season)
    upcoming = sched[sched["EventDate"] >= pd.Timestamp.now()]
    return int(upcoming.iloc[0]["RoundNumber"]) if not upcoming.empty else int(sched.iloc[-1]["RoundNumber"])


def _clean(v):
    # round(float(v), 4) rather than a bare float(v) -- float32 -> float64
    # widening otherwise reintroduces noise into an already-rounded value
    # (e.g. XGBoost's float32 0.20 prints as 0.20000000298023224 once cast)
    return None if pd.isna(v) else round(float(v), 4)


def build_prediction_payload(season: int, round_number: int) -> dict:
    live_rows = build_live_rows(season, round_number)
    predicted = predict_upcoming_race(season, round_number)
    feature_rows = live_rows.set_index("driver")[FEATURE_COLS]

    known_sessions = {
        "practice": bool(predicted["practice_pace"].notna().any()),
        "qualifying": bool(predicted["quali_gap_to_pole"].notna().any()),
        "grid": bool(predicted["grid_position"].notna().any()),
        "compound": bool(live_rows["starting_tire_compound"].notna().any()),
    }

    drivers = []
    for _, row in predicted.iterrows():
        driver = row["driver"]
        drivers.append({
            "driver": driver, "team": row["team"],
            "grid_position": _clean(row["grid_position"]),
            "quali_gap_to_pole": _clean(row["quali_gap_to_pole"]),
            "practice_pace": _clean(row["practice_pace"]),
            "predicted_qualifying_gap": _clean(row["predicted_qualifying_gap"]),
            "predicted_finish_position": _clean(row["predicted_finish_position"]),
            "predicted_quali_to_race_delta": _clean(row["predicted_quali_to_race_delta"]),
            "predicted_race_time_gap": _clean(row["predicted_race_time_gap"]),
            "feature_row": {c: _clean(v) if not isinstance(v, str) else v for c, v in feature_rows.loc[driver].items()},
        })

    return {
        "season": season, "round": round_number, "location": live_rows.iloc[0]["location"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "known_sessions": known_sessions,
        "drivers": drivers,
    }


def write_payload(payload: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, indent=2)
    (OUT_DIR / f"{payload['season']}_{payload['round']}.json").write_text(body)
    (OUT_DIR / "latest.json").write_text(body)


def main():
    season = date.today().year
    round_number = next_or_current_round(season)
    payload = build_prediction_payload(season, round_number)
    write_payload(payload)
    print(f"wrote predictions for season {season} round {round_number} ({payload['location']})")


if __name__ == "__main__":
    main()
