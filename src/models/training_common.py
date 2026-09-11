"""Shared training/evaluation machinery for both predictors (finish position,
quali-to-race delta) — same time-based CV, same tuning search, same SHAP
sanity check, so the two targets stay evaluated the exact same way rather
than drifting into two subtly different methodologies."""
import json

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from scipy.stats import randint, uniform

N_SPLITS = 5

PARAM_DIST = {
    "max_depth": randint(2, 7),
    "learning_rate": uniform(0.01, 0.2),
    "n_estimators": randint(100, 600),
    "subsample": uniform(0.6, 0.4),
    "colsample_bytree": uniform(0.6, 0.4),
    "min_child_weight": randint(1, 10),
    "reg_alpha": uniform(0, 2),
    "reg_lambda": uniform(0.5, 3),
}


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


def tune(X: pd.DataFrame, y: pd.Series, folds: list, n_iter: int = 40) -> RandomizedSearchCV:
    base = xgb.XGBRegressor(
        objective="reg:absoluteerror", enable_categorical=True, random_state=42, n_jobs=-1,
    )
    search = RandomizedSearchCV(
        base, PARAM_DIST, n_iter=n_iter, scoring="neg_mean_absolute_error",
        cv=folds, random_state=42, refit=True, n_jobs=-1,
    )
    search.fit(X, y)
    return search


def fold_mae(model: xgb.XGBRegressor, X: pd.DataFrame, y: pd.Series, folds: list) -> float:
    """Re-trains fresh per fold (unlike the refit=True model, which is trained
    on all of X, y) so this reflects genuine held-out performance."""
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


def shap_circuit_check(model: xgb.XGBRegressor, X: pd.DataFrame, df: pd.DataFrame) -> dict:
    """The plan's explicit sanity check: circuit features (grid x overtaking
    difficulty, in particular) should matter more at Monaco-like (hard to
    overtake) rows than Monza-like (easy to overtake) rows — for either target,
    since a hard-to-overtake circuit constrains both how far you finish from
    your grid slot and how much your finishing position can differ from it.

    The grid_x_overtaking_difficulty half only runs when that column is
    actually in X — the qualifying model deliberately excludes grid_position
    (and therefore this interaction) since it isn't known before qualifying,
    so it still gets the top-10 SHAP ranking without the hard/easy split."""
    import shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap_df = pd.DataFrame(shap_values, columns=X.columns, index=X.index)

    result = {"top_10_mean_abs_shap": shap_df.abs().mean().sort_values(ascending=False).head(10).to_dict()}
    if "grid_x_overtaking_difficulty" in X.columns:
        hard = df["overtaking_difficulty"] >= 0.7
        easy = df["overtaking_difficulty"] <= 0.3
        result["mean_abs_shap_grid_x_overtaking__hard_tracks"] = float(shap_df.loc[hard, "grid_x_overtaking_difficulty"].abs().mean())
        result["mean_abs_shap_grid_x_overtaking__easy_tracks"] = float(shap_df.loc[easy, "grid_x_overtaking_difficulty"].abs().mean())
    return result


def save_model_and_metrics(model: xgb.XGBRegressor, metrics: dict, model_dir, name: str) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(model_dir / f"{name}_xgb.json")
    (model_dir / f"{name}_metrics.json").write_text(json.dumps(metrics, indent=2))
