"""Phase 2: tune, train, and evaluate the race-finishing-position XGBoost model.

Time-based CV throughout — folds are built from chronologically ordered races
(never randomly), and every fold's test set only ever follows its train set,
matching the leakage discipline already enforced at the feature-engineering
stage in Phase 1.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import randint, uniform
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

from src.models.features import prepare_features

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "model_matrix.parquet"
MODEL_DIR = Path(__file__).resolve().parent / "saved"
N_SPLITS = 5


def time_based_splits(df: pd.DataFrame, n_splits: int = N_SPLITS):
    """Yield (train_row_idx, test_row_idx) folds where every fold's test races
    are strictly later than its train races. df must have a clean 0..n-1
    RangeIndex. Splits are over unique races, not rows, so all drivers in one
    race always land on the same side of a fold."""
    races = df[["season", "round", "race_date"]].drop_duplicates().sort_values("race_date").reset_index(drop=True)
    races["race_key"] = races["season"].astype(str) + "_" + races["round"].astype(str)
    df_key = df["season"].astype(str) + "_" + df["round"].astype(str)

    for train_races, test_races in TimeSeriesSplit(n_splits=n_splits).split(races):
        train_keys = set(races.loc[train_races, "race_key"])
        test_keys = set(races.loc[test_races, "race_key"])
        train_idx = np.flatnonzero(df_key.isin(train_keys).to_numpy())
        test_idx = np.flatnonzero(df_key.isin(test_keys).to_numpy())
        yield train_idx, test_idx


def load_data() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)
    df = df.dropna(subset=["target_finish_position"]).reset_index(drop=True)
    return df


def tune(X: pd.DataFrame, y: pd.Series, folds: list) -> RandomizedSearchCV:
    base = xgb.XGBRegressor(
        objective="reg:absoluteerror", enable_categorical=True, random_state=42, n_jobs=-1,
    )
    param_dist = {
        "max_depth": randint(2, 7),
        "learning_rate": uniform(0.01, 0.2),
        "n_estimators": randint(100, 600),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
        "min_child_weight": randint(1, 10),
        "reg_alpha": uniform(0, 2),
        "reg_lambda": uniform(0.5, 3),
    }
    search = RandomizedSearchCV(
        base, param_dist, n_iter=40, scoring="neg_mean_absolute_error",
        cv=folds, random_state=42, refit=True, n_jobs=-1,
    )
    search.fit(X, y)
    return search


def baseline_mae(df: pd.DataFrame, folds: list) -> float:
    """Naive baseline: predicted finish position = grid position."""
    errs = []
    for _, test_idx in folds:
        test = df.iloc[test_idx]
        errs.append(mean_absolute_error(test["target_finish_position"], test["grid_position"].fillna(10)))
    return float(np.mean(errs))


def fold_mae(model, X: pd.DataFrame, y: pd.Series, folds: list) -> float:
    errs = []
    for train_idx, test_idx in folds:
        m = xgb.XGBRegressor(**model.get_params())
        m.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = m.predict(X.iloc[test_idx])
        errs.append(mean_absolute_error(y.iloc[test_idx], pred))
    return float(np.mean(errs))


def calibration_table(y_true: pd.Series, y_pred: np.ndarray, n_bins: int = 5) -> pd.DataFrame:
    bins = pd.qcut(y_pred, n_bins, duplicates="drop")
    return pd.DataFrame({"predicted": y_pred, "actual": y_true.to_numpy()}).groupby(bins, observed=True).mean()


def shap_circuit_check(model, X: pd.DataFrame, df: pd.DataFrame) -> dict:
    """The plan's explicit sanity check: circuit features (grid x overtaking
    difficulty, in particular) should matter more at Monaco-like (hard to
    overtake) rows than Monza-like (easy to overtake) rows."""
    import shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap_df = pd.DataFrame(shap_values, columns=X.columns, index=X.index)

    hard = df["overtaking_difficulty"] >= 0.7
    easy = df["overtaking_difficulty"] <= 0.3
    result = {
        "mean_abs_shap_grid_x_overtaking__hard_tracks": float(shap_df.loc[hard, "grid_x_overtaking_difficulty"].abs().mean()),
        "mean_abs_shap_grid_x_overtaking__easy_tracks": float(shap_df.loc[easy, "grid_x_overtaking_difficulty"].abs().mean()),
        "top_10_mean_abs_shap": shap_df.abs().mean().sort_values(ascending=False).head(10).to_dict(),
    }
    return result


def main():
    df = load_data()
    X = prepare_features(df)
    y = df["target_finish_position"]
    folds = list(time_based_splits(df))

    print(f"{len(df)} rows, {len(folds)} time-based folds")
    base_mae = baseline_mae(df, folds)
    print(f"baseline MAE (grid position = finish position): {base_mae:.3f}")

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

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    final_model.save_model(MODEL_DIR / "finish_position_xgb.json")
    metrics = {
        "n_rows": len(df), "n_folds": len(folds), "baseline_mae": base_mae, "tuned_mae": tuned_mae,
        "best_params": search.best_params_, "shap_circuit_check": shap_result,
    }
    (MODEL_DIR / "finish_position_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"saved model + metrics to {MODEL_DIR}")


if __name__ == "__main__":
    main()
