"""Load the four saved predictors and run them. The same model serves every
point of a race weekend: columns not known yet are NaN (see
features.mask_for_stage for how training prepares for that)."""
from functools import lru_cache

import pandas as pd
import xgboost as xgb

from src.models.catalog import CANONICAL_PRED_COLS, MODEL_DIR, TARGETS
from src.models.features import PREPARE_FN


@lru_cache(maxsize=len(TARGETS))
def load_model(target: str = "finish_position") -> xgb.XGBRegressor:
    model = xgb.XGBRegressor()
    model.load_model(MODEL_DIR / f"{target}_xgb.json")
    return model


def predict(model: xgb.XGBRegressor, rows: pd.DataFrame, target: str = "finish_position") -> pd.Series:
    return pd.Series(model.predict(PREPARE_FN[target](rows)), index=rows.index)


def predict_all(rows: pd.DataFrame, blend: bool = True) -> pd.DataFrame:
    """rows (one race) plus one prediction column per target
    (CANONICAL_PRED_COLS), with the fitted blends applied (src/models/blend.py)
    unless blend=False."""
    from src.models import blend as blends
    from src.models.features import stage_of

    rows = rows.copy()
    for target, col in CANONICAL_PRED_COLS.items():
        rows[col] = predict(load_model(target), rows, target=target).round(4)
    return blends.apply_rows(rows, stage_of(rows)) if blend else rows


def contributions(model: xgb.XGBRegressor, X: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Exact TreeSHAP values from XGBoost itself: how much each feature moved
    each prediction away from the average one. Returns (per-feature
    contributions, base value); contributions + base = the prediction."""
    raw = model.get_booster().predict(xgb.DMatrix(X, enable_categorical=True), pred_contribs=True)
    return pd.DataFrame(raw[:, :-1], columns=X.columns, index=X.index), pd.Series(raw[:, -1], index=X.index)
