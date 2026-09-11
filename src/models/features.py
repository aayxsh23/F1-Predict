"""Shared feature-column list for the finishing-position model, so training
and prediction never drift apart."""
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


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURE_COLS].copy()
    X["grid_x_overtaking_difficulty"] = X["grid_position"] * X["overtaking_difficulty"]
    X["starting_tire_compound"] = pd.Categorical(X["starting_tire_compound"], categories=COMPOUND_CATEGORIES)
    return X
