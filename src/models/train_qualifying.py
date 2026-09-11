"""Scope-expansion model #3: predict qualifying result *before qualifying
happens*, from practice pace + historical form only (see AGENTS.md's hard
scope constraints and src/models/features.py's QUALI_SAFE_FEATURE_COLS for
exactly which columns this is/isn't allowed to see, and why).

target_qualifying_gap = quali_gap_to_pole (seconds behind pole). Same
time-based CV, tuning search, and SHAP check as the other three models
(training_common.py) — only the feature set and prepare function differ.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from src.models.features import prepare_features_quali
from src.models.training_common import calibration_table, fold_mae, save_model_and_metrics, shap_circuit_check, time_based_splits, tune

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "model_matrix.parquet"
MODEL_DIR = Path(__file__).resolve().parent / "saved"


def load_data() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)
    df = df.dropna(subset=["target_qualifying_gap"]).reset_index(drop=True)
    return df


def baseline_mae(df: pd.DataFrame, folds: list) -> float:
    """Naive baseline: predict the team's own historical qualifying pace
    (team_quali_pace, already in seconds-behind-pole units, same as the
    target) — "you'll qualify about as well as your car normally does" is a
    strong, realistic naive guess, not "you'll be on pole," which target=0
    would imply for everyone."""
    errs = []
    for _, test_idx in folds:
        test = df.iloc[test_idx]
        naive = test["team_quali_pace"].fillna(test["target_qualifying_gap"].mean())
        errs.append(mean_absolute_error(test["target_qualifying_gap"], naive))
    return float(np.mean(errs))


def main():
    df = load_data()
    X = prepare_features_quali(df)
    y = df["target_qualifying_gap"]
    folds = list(time_based_splits(df))

    print(f"{len(df)} rows, {len(folds)} time-based folds")
    base_mae = baseline_mae(df, folds)
    print(f"baseline MAE (predict driver's own recent form as a proxy): {base_mae:.3f}")

    search = tune(X, y, folds)
    print(f"best params: {search.best_params_}")
    tuned_mae = fold_mae(search.best_estimator_, X, y, folds)
    print(f"tuned model MAE (time-based CV): {tuned_mae:.3f}")
    print(f"improvement over baseline: {base_mae - tuned_mae:.3f} seconds")

    final_model = search.best_estimator_
    last_test_idx = folds[-1][1]
    pred = final_model.predict(X.iloc[last_test_idx])
    calib = calibration_table(y.iloc[last_test_idx], pred)
    print("calibration (last fold, predicted vs actual by quintile):")
    print(calib)

    shap_result = shap_circuit_check(final_model, X, df)
    print("SHAP check (no grid_x_overtaking_difficulty split -- grid isn't a feature here):", json.dumps(shap_result, indent=2))

    metrics = {
        "n_rows": len(df), "n_folds": len(folds), "baseline_mae": base_mae, "tuned_mae": tuned_mae,
        "best_params": search.best_params_, "shap_circuit_check": shap_result,
    }
    save_model_and_metrics(final_model, metrics, MODEL_DIR, "qualifying")
    print(f"saved model + metrics to {MODEL_DIR}")


if __name__ == "__main__":
    main()
