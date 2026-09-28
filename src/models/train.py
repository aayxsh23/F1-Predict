"""Train, evaluate and save all four predictors:
`python -m src.models.train [--targets finish_position ...] [--n-iter 40]`.

For each target:
1. Tune XGBoost hyperparameters on the earliest 60% of races only, with
   time-ordered folds (never validated on a race older than its training).
2. Walk forward through every race after the first MIN_HISTORY: refit on the
   races before it and predict it at every weekend stage. Nothing about a race
   is seen before it is predicted, so these are the honest numbers. The
   headline metrics use only races after the tuning window, since those
   influenced neither the weights nor the hyperparameters.
3. Fit the shipped model on every race.

Every model trains on one copy of each row per weekend stage, with the
columns unknown at that stage blanked (see features.mask_for_stage).
Outputs: saved/<target>_xgb.json, saved/<target>_metrics.json and
data/processed/walkforward.parquet (per race, driver, stage: prediction,
naive baseline, actual), which the History view and the odds calibration use.
"""
import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import randint, spearmanr, uniform
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

from src.models.features import PREPARE_FN, QUALI_STAGES, RACE_STAGES, mask_for_stage
from src.models.predict import MODEL_DIR, contributions

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "model_matrix.parquet"
WALKFORWARD_PATH = ROOT / "data" / "processed" / "walkforward.parquet"
TUNE_SHARE = 0.6
MIN_HISTORY = 10
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
BASE_PARAMS = {"objective": "reg:absoluteerror", "enable_categorical": True, "random_state": 42, "n_jobs": -1}


@dataclass(frozen=True)
class Target:
    column: str
    stages: list[str]
    unit: str
    baseline_name: str
    baseline: Callable[[pd.DataFrame, pd.DataFrame, str], pd.Series]  # (train, rows, stage) -> naive guess


def _finish_baseline(train, rows, stage):
    form = rows["driver_recent_form"].fillna(train["target_finish_position"].median())
    return form if stage in ("pre_weekend", "post_practice") else rows["grid_position"].fillna(form)


TARGETS = {
    "finish_position": Target(
        "target_finish_position", RACE_STAGES, "places",
        "the driver's recent average finish before qualifying, their grid slot after it", _finish_baseline),
    "quali_delta": Target(
        "target_quali_to_race_delta", RACE_STAGES, "places", "no change from the grid",
        lambda train, rows, stage: pd.Series(0.0, index=rows.index)),
    "qualifying": Target(
        "target_qualifying_gap_pct", QUALI_STAGES, "% of the pole lap", "the team's recent qualifying gap",
        lambda train, rows, stage: rows["team_quali_pace"].fillna(train["target_qualifying_gap_pct"].median())),
    "race_time": Target(
        "target_race_gap_pct", RACE_STAGES, "% of the winner's race time", "the typical gap in past races",
        lambda train, rows, stage: pd.Series(train["target_race_gap_pct"].median(), index=rows.index)),
}


def augment(df: pd.DataFrame, stages: list[str]) -> pd.DataFrame:
    return pd.concat([mask_for_stage(df, s).assign(stage=s) for s in stages], ignore_index=True)


def race_order(df: pd.DataFrame) -> pd.DataFrame:
    return df[["season", "round", "race_date"]].drop_duplicates().sort_values("race_date").reset_index(drop=True)


def _race_key(df: pd.DataFrame) -> pd.Series:
    return df["season"] * 100 + df["round"]


def race_folds(df: pd.DataFrame, n_splits: int = N_SPLITS):
    """(train_idx, test_idx) folds over whole races, each test block later than its train block."""
    races = race_order(df)
    rkey, key = _race_key(races), _race_key(df)
    for tr, te in TimeSeriesSplit(n_splits=n_splits).split(races):
        yield np.flatnonzero(key.isin(rkey[tr])), np.flatnonzero(key.isin(rkey[te]))


def fit(name: str, rows: pd.DataFrame, params: dict) -> xgb.XGBRegressor:
    t = TARGETS[name]
    aug = augment(rows, t.stages)
    return xgb.XGBRegressor(**BASE_PARAMS, **params).fit(PREPARE_FN[name](aug), aug[t.column])


def tune(name: str, rows: pd.DataFrame, n_iter: int) -> dict:
    t = TARGETS[name]
    aug = augment(rows, t.stages)
    search = RandomizedSearchCV(
        xgb.XGBRegressor(**BASE_PARAMS), PARAM_DIST, n_iter=n_iter, scoring="neg_mean_absolute_error",
        cv=list(race_folds(aug)), random_state=42, refit=False, n_jobs=-1,
    )
    search.fit(PREPARE_FN[name](aug), aug[t.column])
    return {k: (int(v) if isinstance(v, (np.integer, int)) else float(v)) for k, v in search.best_params_.items()}


def walk_forward(name: str, df: pd.DataFrame, params: dict) -> pd.DataFrame:
    t = TARGETS[name]
    races = race_order(df)
    key, rkey = _race_key(df), _race_key(races)
    out = []
    for i in range(MIN_HISTORY, len(races)):
        train, test = df[key.isin(rkey[:i])], df[key == rkey[i]]
        model = fit(name, train, params)
        for stage in t.stages:
            rows = mask_for_stage(test, stage)
            out.append(pd.DataFrame({
                "season": test["season"], "round": test["round"], "race_index": i, "driver": test["driver"],
                "target": name, "stage": stage, "pred": model.predict(PREPARE_FN[name](rows)),
                "baseline": t.baseline(train, rows, stage).to_numpy(), "actual": test[t.column],
            }))
    return pd.concat(out, ignore_index=True)


def _stage_metrics(wf: pd.DataFrame, position_target: bool) -> dict:
    out = {}
    for stage, g in wf.groupby("stage", sort=False):
        m = {"mae": float((g["pred"] - g["actual"]).abs().mean()),
             "baseline_mae": float((g["baseline"] - g["actual"]).abs().mean()), "n": int(len(g))}
        if position_target:
            rho = [spearmanr(r["pred"], r["actual"])[0] for _, r in g.groupby(["season", "round"]) if len(r) > 2]
            m["mean_race_spearman"] = float(np.nanmean(rho))
        out[stage] = {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}
    return out


def train_target(name: str, df: pd.DataFrame, n_iter: int) -> tuple[dict, pd.DataFrame]:
    t = TARGETS[name]
    df = df.dropna(subset=[t.column]).sort_values("race_date").reset_index(drop=True)
    races = race_order(df)
    cut = int(len(races) * TUNE_SHARE)
    tune_rows = df[_race_key(df).isin(_race_key(races)[:cut])]
    print(f"[{name}] {len(df)} rows, {len(races)} races; tuning on the first {cut}")

    params = tune(name, tune_rows, n_iter)
    wf = walk_forward(name, df, params)
    wf["after_tuning_window"] = wf["race_index"] >= cut
    model = fit(name, df, params)

    live_like = mask_for_stage(df, "post_quali" if "post_quali" in t.stages else "post_practice")
    contrib, _ = contributions(model, PREPARE_FN[name](live_like))
    held_out = wf[wf["after_tuning_window"]]
    first, last = races.iloc[cut], races.iloc[-1]
    metrics = {
        "target": name, "column": t.column, "unit": t.unit, "baseline": t.baseline_name,
        "n_rows": int(len(df)), "n_races": int(len(races)),
        "data_through": str(last["race_date"].date()),
        "evaluation": (f"{held_out[['season', 'round']].drop_duplicates().shape[0]} races from "
                       f"{first['race_date'].date()} to {last['race_date'].date()}, each predicted by a model "
                       f"trained only on earlier races; none of them was used to pick hyperparameters"),
        "stages": _stage_metrics(held_out, name in ("finish_position", "quali_delta")),
        "all_walk_forward_stages": _stage_metrics(wf, name in ("finish_position", "quali_delta")),
        "best_params": params,
        "top_features": contrib.abs().mean().sort_values(ascending=False).head(10).round(4).to_dict(),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(MODEL_DIR / f"{name}_xgb.json")
    (MODEL_DIR / f"{name}_metrics.json").write_text(json.dumps(metrics, indent=2))
    for stage, m in metrics["stages"].items():
        print(f"  {stage:14s} MAE {m['mae']:.3f}  vs baseline {m['baseline_mae']:.3f}"
              + (f"  spearman {m['mean_race_spearman']:.2f}" if "mean_race_spearman" in m else ""))
    return metrics, wf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", nargs="+", default=list(TARGETS), choices=list(TARGETS))
    parser.add_argument("--n-iter", type=int, default=40)
    args = parser.parse_args()

    df = pd.read_parquet(DATA_PATH)
    frames = [train_target(name, df, args.n_iter)[1] for name in args.targets]
    if WALKFORWARD_PATH.exists():  # keep other targets' rows when retraining a subset
        old = pd.read_parquet(WALKFORWARD_PATH)
        frames.append(old[~old["target"].isin(args.targets)])
    pd.concat(frames, ignore_index=True).to_parquet(WALKFORWARD_PATH, index=False)
    print(f"wrote {WALKFORWARD_PATH}")


if __name__ == "__main__":
    main()
