"""Phase 3: tune, train, and evaluate the quali-to-race delta XGBoost model —
target #2, reusing the exact same feature table, time-based CV, tuning search,
and SHAP check as Phase 2's finish-position model (training_common.py).

target_quali_to_race_delta = grid_position - finish_position (positive =
gained places, negative = lost places). grid_position is legitimately a
*feature* here too, not leakage — it's known before the race, and it's the
whole reason this target is interesting: a driver starting P1 has far less
room to gain than one starting P18, so the model needs grid_position to learn
that ceiling/floor effect.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from src.models.features import prepare_features
from src.models.training_common import calibration_table, fold_mae, save_model_and_metrics, shap_circuit_check, time_based_splits, tune

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "model_matrix.parquet"
MODEL_DIR = Path(__file__).resolve().parent / "saved"


def load_data() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)
    df = df.dropna(subset=["target_quali_to_race_delta"]).reset_index(drop=True)
    return df


def baseline_mae(df: pd.DataFrame, folds: list) -> float:
    """Naive baseline: predict zero change — you finish where you qualified."""
    errs = []
    for _, test_idx in folds:
        test = df.iloc[test_idx]
        errs.append(mean_absolute_error(test["target_quali_to_race_delta"], np.zeros(len(test))))
    return float(np.mean(errs))


def main():
    df = load_data()
    X = prepare_features(df)
    y = df["target_quali_to_race_delta"]
    folds = list(time_based_splits(df))

    print(f"{len(df)} rows, {len(folds)} time-based folds")
    base_mae = baseline_mae(df, folds)
    print(f"baseline MAE (predicted delta = 0, i.e. finish = grid): {base_mae:.3f}")

    search = tune(X, y, folds)
    print(f"best params: {search.best_params_}")
    tuned_mae = fold_mae(search.best_estimator_, X, y, folds)
    print(f"tuned model MAE (time-based CV): {tuned_mae:.3f}")
    print(f"improvement over baseline: {base_mae - tuned_mae:.3f} positions")

    final_model = search.best_estimator_  # refit=True already retrained on all of X, y
    last_test_idx = folds[-1][1]
    pred = final_model.predict(X.iloc[last_test_idx])
    calib = calibration_table(y.iloc[last_test_idx], pred)
    print("calibration (last fold, predicted vs actual by quintile):")
    print(calib)

    shap_result = shap_circuit_check(final_model, X, df)
    print("SHAP circuit sanity check:", json.dumps(shap_result, indent=2))

    metrics = {
        "n_rows": len(df), "n_folds": len(folds), "baseline_mae": base_mae, "tuned_mae": tuned_mae,
        "best_params": search.best_params_, "shap_circuit_check": shap_result,
    }
    save_model_and_metrics(final_model, metrics, MODEL_DIR, "quali_delta")
    print(f"saved model + metrics to {MODEL_DIR}")


if __name__ == "__main__":
    main()
