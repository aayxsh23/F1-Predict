"""Load the four saved predictors and run them. The same model serves every
point of a race weekend: columns not known yet are NaN (see
features.mask_for_stage for how training prepares for that)."""
from functools import lru_cache
from pathlib import Path

import pandas as pd
import xgboost as xgb

from src.models.features import PREPARE_FN

MODEL_DIR = Path(__file__).resolve().parent / "saved"
TARGETS = ("qualifying", "finish_position", "quali_delta", "race_time")

# one output column per target, shared by every caller
CANONICAL_PRED_COLS = {
    "qualifying": "predicted_qualifying_gap_pct",
    "finish_position": "predicted_finish_position",
    "quali_delta": "predicted_quali_to_race_delta",
    "race_time": "predicted_race_gap_pct",
}


@lru_cache(maxsize=len(TARGETS))
def load_model(target: str = "finish_position") -> xgb.XGBRegressor:
    model = xgb.XGBRegressor()
    model.load_model(MODEL_DIR / f"{target}_xgb.json")
    return model


def predict(model: xgb.XGBRegressor, rows: pd.DataFrame, target: str = "finish_position") -> pd.Series:
    return pd.Series(model.predict(PREPARE_FN[target](rows)), index=rows.index)


def predict_all(rows: pd.DataFrame) -> pd.DataFrame:
    """rows plus one prediction column per target (CANONICAL_PRED_COLS)."""
    rows = rows.copy()
    for target, col in CANONICAL_PRED_COLS.items():
        rows[col] = predict(load_model(target), rows, target=target).round(4)
    return rows


def contributions(model: xgb.XGBRegressor, X: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Exact TreeSHAP values from XGBoost itself: how much each feature moved
    each prediction away from the average one. Returns (per-feature
    contributions, base value); contributions + base = the prediction."""
    raw = model.get_booster().predict(xgb.DMatrix(X, enable_categorical=True), pred_contribs=True)
    return pd.DataFrame(raw[:, :-1], columns=X.columns, index=X.index), pd.Series(raw[:, -1], index=X.index)
