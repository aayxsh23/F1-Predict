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
    "Singapore": "Marina Bay",
    "Miami Gardens": "Miami",
    "Yas Marina": "Yas Island",
}
# The other way round: one name, two tracks. The 2020 Sakhir Grand Prix ran
# Bahrain's 3.5 km outer loop, a week after the Bahrain Grand Prix on the
# full circuit; FastF1 calls both "Sakhir". build_dataset.load_raw renames it,
# so its 55-second laps and results stay out of Bahrain's history.
EVENT_VENUES = {(2020, 16): "Sakhir Outer"}

# Constructors that changed name (or owner) but kept the same factory, staff
# and car lineage: form carries across the rename. The app still shows the
# name each team raced under.
TEAM_LINEAGE = {
    "Toro Rosso": "Racing Bulls", "Scuderia Toro Rosso": "Racing Bulls", "AlphaTauri": "Racing Bulls",
    "RB": "Racing Bulls",
    "Sauber": "Audi", "Alfa Romeo Racing": "Audi", "Alfa Romeo": "Audi", "Kick Sauber": "Audi",
    "Force India": "Aston Martin", "Racing Point": "Aston Martin",
    "Renault": "Alpine",
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
    "expected_stops": "Usual pit stops here",
    "grid_x_overtaking_difficulty": "Grid slot on a hard-to-pass track",
    "driver_quali_rating": "Qualifying rating against teammates",
    "teammate_quali_rating_gap": "Rating gap to teammate",
    "circuit_sc_share": "Safety cars in past races here",
    "circuit_dnf_share": "Retirements in past races here",
    "circuit_grid_order": "How closely results here follow the grid",
    "grid_x_circuit_grid_order": "Grid slot where results follow the grid",
    "is_sprint_weekend": "Sprint weekend (one practice session)",
    "sprint_quali_gap_pct": "Sprint qualifying gap to pole",
    "sprint_finish_position": "Sprint result",
    "sprint_race_pace_pct": "Sprint race pace",
    "team_performance_upgrades": "Car upgrades this weekend",
    "team_upgrades_recent": "Car upgrades in the last three races",
    "driver_quali_form": "Driver's recent qualifying pace",
    "driver_quali_vs_teammate_form": "Recent qualifying vs teammate",
}
