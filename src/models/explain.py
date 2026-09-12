"""Per-prediction SHAP explanation -- the numeric feature-attribution
breakdown behind one driver's one prediction (signed contributions, not
prose -- that's rag/explain.py's job). Extends training_common's
shap_circuit_check TreeExplainer pattern from a circuit-level mean aggregate
down to a single row, which is what the Phase 7 API's
/predictions/{season}/{round}/explain endpoint needs: cheap enough to run
per-request on a cold-started free-tier backend, no FastF1/feature-matrix
rebuild required."""
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from src.models.features import PREPARE_FN


def _native(v):
    """JSON-serializable form of one feature value -- numpy scalar types and
    pandas NA don't survive json.dumps as-is."""
    if pd.isna(v):
        return None
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    return v


def shap_explanation(model: xgb.XGBRegressor, row: pd.DataFrame, target: str = "finish_position", top_n: int = 8) -> dict:
    """row: single-row DataFrame with the raw Phase 1 feature-table columns,
    same shape src.models.predict.predict() takes."""
    X = PREPARE_FN[target](row)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)[0]
    predicted_value = float(model.predict(X)[0])

    contributions = sorted(
        zip(X.columns, X.iloc[0].tolist(), shap_values.tolist()),
        key=lambda c: abs(c[2]), reverse=True,
    )[:top_n]

    return {
        "predicted_value": round(predicted_value, 3),
        "base_value": round(float(explainer.expected_value), 3),
        "top_contributions": [
            {"feature": f, "value": _native(v), "shap": round(s, 4)} for f, v, s in contributions
        ],
    }
