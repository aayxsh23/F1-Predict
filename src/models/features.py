"""Shared feature-column list for finish_position/quali_delta/race_time (all
three predict a pre-race-stage or later outcome, so all pre-race info is fair
game). The qualifying model is different -- see QUALI_SAFE_FEATURE_COLS below
-- and gets its own prepare function so the two never get mixed up."""
import pandas as pd

from src.features.build_dataset import CIRCUIT_COLS, WEATHER_COLS

DRIVER_COLS = [
    "grid_position", "quali_gap_to_pole", "practice_pace", "driver_recent_form",
    "driver_track_form", "driver_positions_gained_form", "driver_dnf_rate",
]
TEAM_COLS = ["team_recent_form", "team_quali_pace", "team_race_pace", "team_reliability", "team_track_type_form"]
RELATIVE_COLS = ["teammate_quali_gap", "teammate_race_pace_gap", "grid_vs_expected_position"]
STRATEGY_COLS = ["starting_tire_compound", "expected_stops", "historical_compound_performance"]

FEATURE_COLS = CIRCUIT_COLS + WEATHER_COLS + DRIVER_COLS + TEAM_COLS + RELATIVE_COLS + STRATEGY_COLS
# fixed category list (not inferred per-batch) so a single-row prediction with
# starting_tire_compound unknown yet doesn't end up with zero known categories
COMPOUND_CATEGORIES = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]

# grid_position matters more when overtaking is hard — an interpretable interaction
# for SHAP, not required for model performance (XGBoost learns interactions on its own)
INTERACTION_COLS = ["grid_x_overtaking_difficulty"]


def row_from_dict(feature_row: dict) -> pd.DataFrame:
    """Rebuild a single-row DataFrame from a feature dict that's been through
    a JSON round-trip (refresh_job.py writes it, the API reads it back for
    /explain). JSON's null becomes Python None, and a dict with a None value
    makes pandas infer `object` dtype for that column instead of float64/NaN
    -- which XGBoost's categorical-dtype check then rejects outright, even
    for columns that were never meant to be categorical. Coerce every
    FEATURE_COLS member except the one genuinely categorical column back to
    numeric; anything outside FEATURE_COLS (e.g. an added 'location' column)
    is left alone."""
    row = pd.DataFrame([feature_row])
    numeric_cols = [c for c in FEATURE_COLS if c != "starting_tire_compound" and c in row.columns]
    row[numeric_cols] = row[numeric_cols].apply(pd.to_numeric, errors="coerce")
    return row


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURE_COLS].copy()
    X["grid_x_overtaking_difficulty"] = X["grid_position"] * X["overtaking_difficulty"]
    X["starting_tire_compound"] = pd.Categorical(X["starting_tire_compound"], categories=COMPOUND_CATEGORIES)
    return X


# --- qualifying predictor: a genuinely different information stage ---
#
# Every other model here predicts something that resolves at pre-race or
# later, so "everything known before the race" is a fair feature set. The
# qualifying model predicts qualifying itself, from only what's known before
# qualifying happens (practice sessions + historical form). Reusing
# FEATURE_COLS wholesale would leak the target into its own inputs. Excluded,
# and why:
#   - grid_position, quali_gap_to_pole:      this race's actual qualifying
#                                             result (or the target itself)
#   - teammate_quali_gap:                    derived from this race's actual
#                                             quali_gap_to_pole
#   - grid_vs_expected_position:             derived from this race's actual
#                                             grid_position
#   - starting_tire_compound:                a race-day-only decision, not
#                                             known until Sunday
#   - historical_compound_performance:       computable only once this race's
#                                             starting_tire_compound is known
#                                             (it's part of the lookup key),
#                                             which qualifying-time doesn't have
#   - WEATHER_COLS (all 5):                  ingest.py captures these from the
#                                             RACE session, not qualifying or
#                                             practice — Sunday's weather isn't
#                                             known on Saturday. No separate
#                                             qualifying-weather column exists
#                                             yet, so weather is dropped
#                                             entirely for this model rather
#                                             than fed a future value.
QUALI_SAFE_DRIVER_COLS = ["practice_pace", "driver_recent_form", "driver_track_form", "driver_positions_gained_form", "driver_dnf_rate"]
QUALI_SAFE_TEAM_COLS = TEAM_COLS  # all historical (recent_form-based), none of this race's actuals
QUALI_SAFE_RELATIVE_COLS = ["teammate_race_pace_gap"]  # historical-form comparison only
QUALI_SAFE_STRATEGY_COLS = ["expected_stops"]  # historical; excludes historical_compound_performance (see above)

QUALI_SAFE_FEATURE_COLS = CIRCUIT_COLS + QUALI_SAFE_DRIVER_COLS + QUALI_SAFE_TEAM_COLS + QUALI_SAFE_RELATIVE_COLS + QUALI_SAFE_STRATEGY_COLS


def prepare_features_quali(df: pd.DataFrame) -> pd.DataFrame:
    return df[QUALI_SAFE_FEATURE_COLS].copy()


# single shared mapping of target -> its feature-prep function, so every
# caller (predict.py, the SHAP/RAG explainer) dispatches identically instead
# of each guessing which prepare fn goes with which target
PREPARE_FN = {
    "finish_position": prepare_features, "quali_delta": prepare_features,
    "qualifying": prepare_features_quali, "race_time": prepare_features,
}
