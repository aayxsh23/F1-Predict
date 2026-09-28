"""Names and paths the serving side needs, with no heavy imports: the API runs
as a serverless function that doesn't ship pandas or XGBoost (everything that
needs them is precomputed by the scheduled refresh job)."""
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent / "saved"

# one output column per target, shared by every caller
CANONICAL_PRED_COLS = {
    "qualifying": "predicted_qualifying_gap_pct",
    "finish_position": "predicted_finish_position",
    "quali_delta": "predicted_quali_to_race_delta",
    "race_time": "predicted_race_gap_pct",
}
TARGETS = tuple(CANONICAL_PRED_COLS)

# FastF1 renames some venues between seasons (Monaco -> "Monte Carlo"); map
# them to one canonical circuit name instead of duplicating circuit rows.
LOCATION_ALIASES = {
    "Monte Carlo": "Monaco",
    "Miami Gardens": "Miami",
    "Yas Marina": "Yas Island",
}

# every model input in a fan's words; the API sends these, the app never shows a column name
FEATURE_LABELS = {
    "overtaking_difficulty": "How hard it is to pass here",
    "is_street_circuit": "Street circuit",
    "pit_lane_loss_time": "Time lost in the pit lane",
    "safety_car_frequency": "Safety cars at this track",
    "dnf_rate": "Retirements at this track",
    "longest_straight_m": "Longest straight",
    "braking_zone_count": "Heavy braking zones",
    "tyre_degradation_level": "Tyre wear at this track",
    "rain_race_frequency": "How often it rains here",
    "track_length_km": "Lap length",
    "air_temp_forecast": "Forecast air temperature",
    "rain_mm_forecast": "Forecast rain",
    "wind_kph_forecast": "Forecast wind",
    "grid_position": "Starting grid slot",
    "quali_gap_pct": "Qualifying gap to pole",
    "practice_pace": "Best practice lap",
    "practice_long_run_pace": "Practice long-run pace",
    "driver_recent_form": "Recent results",
    "driver_track_form": "Past results here",
    "driver_positions_gained_form": "Usual places gained",
    "driver_dnf_rate": "Recent retirements",
    "team_recent_form": "Team's recent results",
    "team_quali_pace": "Team's qualifying pace",
    "team_race_pace": "Team's race pace",
    "team_reliability": "Team's reliability",
    "team_track_type_form": "Team's form on tracks like this",
    "teammate_quali_gap": "Qualifying vs teammate",
    "teammate_race_pace_gap": "Race pace vs teammate",
    "grid_vs_expected_position": "Grid slot vs usual form",
    "starting_tire_compound": "Starting tyre",
    "expected_stops": "Usual pit stops here",
    "historical_compound_performance": "Past results on this tyre here",
    "grid_x_overtaking_difficulty": "Grid slot on a hard-to-pass track",
}
