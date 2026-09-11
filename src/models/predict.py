"""Rolling re-prediction: call the SAME trained model again as more of a race
weekend's real session data becomes available (FP1-3 -> Qualifying -> pre-race).
Not-yet-known columns are simply left as NaN — XGBoost's native missing-value
handling does the "different stage" work, so there's no separate per-stage model."""
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from src.models.features import PREPARE_FN

MODEL_DIR = Path(__file__).resolve().parent / "saved"
MODEL_FILES = {
    "finish_position": MODEL_DIR / "finish_position_xgb.json",
    "quali_delta": MODEL_DIR / "quali_delta_xgb.json",
    "qualifying": MODEL_DIR / "qualifying_xgb.json",
    "race_time": MODEL_DIR / "race_time_xgb.json",
}

# columns not yet known before qualifying happens
POST_QUALI_ONLY_COLS = ["grid_position", "quali_gap_to_pole", "grid_vs_expected_position", "teammate_quali_gap"]
# columns only settled once the starting strategy is locked in pre-race
PRE_RACE_ONLY_COLS = ["starting_tire_compound"]


def load_model(target: str = "finish_position") -> xgb.XGBRegressor:
    """target: 'finish_position' | 'quali_delta' | 'qualifying' | 'race_time'
    — all four predictors, sharing this same predict/mask_for_stage machinery
    (qualifying uses its own restricted feature prep, see _PREPARE_FN)."""
    model = xgb.XGBRegressor()
    model.load_model(MODEL_FILES[target])
    return model


def predict(model: xgb.XGBRegressor, rows: pd.DataFrame, target: str = "finish_position") -> pd.Series:
    """rows must have the raw Phase 1 feature-table columns (see
    src.models.features.FEATURE_COLS); missing columns/values are fine."""
    X = PREPARE_FN[target](rows)
    return pd.Series(model.predict(X), index=rows.index)


# one canonical output-column name per target, shared by every caller that
# wants "all 4 predictions on this row set" (live_predict.py, rag/chat.py) so
# they can't quietly diverge into two different naming schemes for the same computation
CANONICAL_PRED_COLS = {
    "qualifying": "predicted_qualifying_gap",
    "finish_position": "predicted_finish_position",
    "quali_delta": "predicted_quali_to_race_delta",
    "race_time": "predicted_race_time_gap",
}


def predict_all(rows: pd.DataFrame) -> pd.DataFrame:
    """Add all 4 predictors' columns to rows (see CANONICAL_PRED_COLS for the
    exact column names) -- the shared implementation behind both
    live_predict.predict_upcoming_race() and rag/chat.py's race table."""
    rows = rows.copy()
    for target, col in CANONICAL_PRED_COLS.items():
        model = load_model(target)
        rows[col] = predict(model, rows, target=target).round(2)
    return rows


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
