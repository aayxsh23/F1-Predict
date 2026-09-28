"""Per-prediction breakdown: which inputs pushed one driver's prediction up or
down, and by how much (signed SHAP contributions, in the target's units)."""
import numpy as np
import pandas as pd
import xgboost as xgb

from src.models.features import PREPARE_FN
from src.models.predict import contributions


def _native(v):
    """JSON-safe form of one feature value."""
    if pd.isna(v):
        return None
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    return v


def shap_explanation(model: xgb.XGBRegressor, row: pd.DataFrame, target: str = "finish_position", top_n: int = 8) -> dict:
    """row: single-row frame with the feature-table columns."""
    X = PREPARE_FN[target](row)
    contrib, base = contributions(model, X)
    c = contrib.iloc[0]
    top = c.reindex(c.abs().sort_values(ascending=False).index)[:top_n]
    return {
        "predicted_value": round(float(c.sum() + base.iloc[0]), 3),
        "base_value": round(float(base.iloc[0]), 3),
        "top_contributions": [
            {"feature": f, "value": _native(X.iloc[0][f]), "shap": round(float(s), 4)} for f, s in top.items()
        ],
    }
