"""Rolling re-prediction: call the SAME trained model again as more of a race
weekend's real session data becomes available (FP1-3 -> Qualifying -> pre-race).
Not-yet-known columns are simply left as NaN — XGBoost's native missing-value
handling does the "different stage" work, so there's no separate per-stage model."""
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from src.models.features import prepare_features

MODEL_PATH = Path(__file__).resolve().parent / "saved" / "finish_position_xgb.json"

# columns not yet known before qualifying happens
POST_QUALI_ONLY_COLS = ["grid_position", "quali_gap_to_pole", "grid_vs_expected_position", "teammate_quali_gap"]
# columns only settled once the starting strategy is locked in pre-race
PRE_RACE_ONLY_COLS = ["starting_tire_compound"]


def load_model() -> xgb.XGBRegressor:
    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)
    return model


def predict(model: xgb.XGBRegressor, rows: pd.DataFrame) -> pd.Series:
    """rows must have the raw Phase 1 feature-table columns (see
    src.models.features.FEATURE_COLS); missing columns/values are fine."""
    X = prepare_features(rows)
    return pd.Series(model.predict(X), index=rows.index)


def mask_for_stage(rows: pd.DataFrame, stage: str) -> pd.DataFrame:
    """stage: 'post_practice' | 'post_quali' | 'pre_race'."""
    rows = rows.copy()
    if stage == "post_practice":
        for col in POST_QUALI_ONLY_COLS + PRE_RACE_ONLY_COLS:
            rows[col] = np.nan
    elif stage == "post_quali":
        for col in PRE_RACE_ONLY_COLS:
            rows[col] = np.nan
    elif stage != "pre_race":
        raise ValueError(f"unknown stage: {stage}")
    return rows
