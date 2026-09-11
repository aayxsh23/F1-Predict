"""Scope-expansion model #4: predict each driver's race time as a gap to the
race winner, in seconds (0 for the winner). Same pre-race-stage feature set
as finish_position/quali_delta (grid_position, starting_tire_compound etc.
are all fair game here — unlike the qualifying model, this predicts an
outcome that resolves at the race itself, not before qualifying), same
time-based CV/tuning/SHAP machinery (training_common.py).

DNF/not-classified rows have no defined finish time and are dropped, same as
every other target's null-handling — not imputed.
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
    df = df.dropna(subset=["target_race_time_gap"]).reset_index(drop=True)
    return df


def baseline_mae(df: pd.DataFrame, folds: list) -> float:
    """Naive baseline: predict the field's mean gap-to-winner for that same
    race (i.e. "you'll finish a typical amount behind the winner"), computed
    per-race so it isn't just one global constant across very different
    circuit lengths/lap counts."""
    errs = []
    for _, test_idx in folds:
        test = df.iloc[test_idx]
        naive = test.groupby(["season", "round"])["target_race_time_gap"].transform("mean")
        errs.append(mean_absolute_error(test["target_race_time_gap"], naive))
    return float(np.mean(errs))


def main():
    df = load_data()
    X = prepare_features(df)
    y = df["target_race_time_gap"]
    folds = list(time_based_splits(df))

    print(f"{len(df)} rows, {len(folds)} time-based folds")
    base_mae = baseline_mae(df, folds)
    print(f"baseline MAE (predict this race's mean gap-to-winner): {base_mae:.3f}")

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
    print("SHAP circuit sanity check:", json.dumps(shap_result, indent=2))

    metrics = {
        "n_rows": len(df), "n_folds": len(folds), "baseline_mae": base_mae, "tuned_mae": tuned_mae,
        "best_params": search.best_params_, "shap_circuit_check": shap_result,
    }
    save_model_and_metrics(final_model, metrics, MODEL_DIR, "race_time")
    print(f"saved model + metrics to {MODEL_DIR}")


if __name__ == "__main__":
    main()
