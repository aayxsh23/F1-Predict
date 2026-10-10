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
columns unknown at that stage blanked (see features.mask_for_stage). The two
sprint stages exist only on sprint weekends, so only those rows are copied
for them, and only those races are evaluated at them; metrics also report
every stage on sprint weekends alone (`sprint_weekends`).

The finishing-position and places-gained models learn the order of the cars
that finish: a retirement isn't a 20th-place pace, and the odds sampler
(probabilities.py) already models retirements on its own. They are still
scored on every car, retirements included, next to a `finishers` block.
Every MAE comes with a 95% interval for (model - baseline) from resampling
whole races, because 44 races leave differences under ~0.05 within noise.
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

from src.features.driver_features import QUALI_MAX_GAP
from src.models.features import PREPARE_FN, QUALI_STAGES, RACE_STAGES, SPRINT_STAGES, STAGES, mask_for_stage
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
    finishers_only: bool = False  # train on cars that finished (scored on all)
    first_season: int | None = None  # learn only from this season on (None: everything loaded)


def _finish_baseline(train, rows, stage):
    form = rows["driver_recent_form"].fillna(train["target_finish_position"].median())
    if stage in ("pre_weekend", "post_practice", "post_sprint_quali"):
        return form
    if stage == "post_sprint":  # the naive guess a fan would make: Sunday repeats Saturday's Sprint
        return rows["sprint_finish_position"].fillna(form)
    return rows["grid_position"].fillna(form)


def _quali_baseline(train, rows, stage):
    team = rows["team_quali_pace"].fillna(train["target_qualifying_gap_pct"].median())
    return rows["sprint_quali_gap_pct"].fillna(team)  # NaN (masked) before Sprint Qualifying has run


TARGETS = {
    "finish_position": Target(
        "target_finish_position", RACE_STAGES, "places",
        "the driver's recent average finish before qualifying (their Sprint result once the Sprint has run), "
        "their grid slot after it", _finish_baseline, finishers_only=True,
        first_season=2021),  # 2018-2020 rows made tuning-window error worse (+0.05)
    "quali_delta": Target(
        "target_quali_to_race_delta", RACE_STAGES, "places", "no change from the grid",
        lambda train, rows, stage: pd.Series(0.0, index=rows.index), finishers_only=True, first_season=2021),
    "qualifying": Target(
        "target_qualifying_gap_pct", QUALI_STAGES, "% of the pole lap",
        "the team's recent qualifying gap (the driver's own Sprint Qualifying gap once it has run)", _quali_baseline),
    "race_time": Target(
        "target_race_gap_pct", RACE_STAGES, "% of the winner's race time", "the typical gap in past races",
        lambda train, rows, stage: pd.Series(train["target_race_gap_pct"].median(), index=rows.index),
        # 2021's gaps (a different car generation) made tuning-window error 0.536% -> 0.646%
        first_season=2022),
}


def _sprint(df: pd.DataFrame) -> pd.Series:
    return df["is_sprint_weekend"] == 1


def augment(df: pd.DataFrame, stages: list[str]) -> pd.DataFrame:
    """One masked copy of df per stage; a sprint stage copies sprint weekends only
    (anywhere else it would just repeat post_practice and over-weight it)."""
    return pd.concat([mask_for_stage(df[_sprint(df)] if s in SPRINT_STAGES else df, s).assign(stage=s) for s in stages],
                     ignore_index=True)


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


def _train_rows(name: str, rows: pd.DataFrame) -> pd.DataFrame:
    return rows[~rows["dnf"].astype(bool)] if TARGETS[name].finishers_only else rows


def fit(name: str, rows: pd.DataFrame, params: dict) -> xgb.XGBRegressor:
    t = TARGETS[name]
    aug = augment(_train_rows(name, rows), t.stages)
    return xgb.XGBRegressor(**BASE_PARAMS, **params).fit(PREPARE_FN[name](aug), aug[t.column])


def tune(name: str, rows: pd.DataFrame, n_iter: int) -> dict:
    t = TARGETS[name]
    aug = augment(_train_rows(name, rows), t.stages)
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
        sprint = bool(_sprint(test).any())
        for stage in t.stages:
            if stage in SPRINT_STAGES and not sprint:
                continue
            rows = mask_for_stage(test, stage)
            out.append(pd.DataFrame({
                "season": test["season"], "round": test["round"], "race_index": i, "driver": test["driver"],
                "dnf": test["dnf"].astype(bool), "sprint_weekend": sprint, "target": name, "stage": stage, "pred": model.predict(PREPARE_FN[name](rows)),
                "baseline": t.baseline(train, rows, stage).to_numpy(), "actual": test[t.column],
            }))
    return pd.concat(out, ignore_index=True)


def diff_ci95(g: pd.DataFrame, a: str = "pred", b: str = "baseline", n_boot: int = 2000, seed: int = 0) -> list[float]:
    """95% interval of MAE(a) - MAE(b), resampling whole races (cars in one
    race aren't independent)."""
    per_race = g.assign(_d=(g[a] - g["actual"]).abs() - (g[b] - g["actual"]).abs()).groupby(["season", "round"])["_d"].agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(per_race), size=(n_boot, len(per_race)))
    boot = per_race["sum"].to_numpy()[idx].sum(axis=1) / per_race["count"].to_numpy()[idx].sum(axis=1)
    return [round(float(x), 4) for x in np.percentile(boot, [2.5, 97.5])]


def _stage_metrics(wf: pd.DataFrame, position_target: bool) -> dict:
    out = {}
    for stage in [s for s in STAGES if s in set(wf["stage"])]:
        g = wf[wf["stage"] == stage]
        m = {"mae": float((g["pred"] - g["actual"]).abs().mean()),
             "baseline_mae": float((g["baseline"] - g["actual"]).abs().mean()), "n": int(len(g)),
             "vs_baseline_ci95": diff_ci95(g)}
        if position_target:
            rho = [spearmanr(r["pred"], r["actual"])[0] for _, r in g.groupby(["season", "round"]) if len(r) > 2]
            m["mean_race_spearman"] = float(np.nanmean(rho))
        out[stage] = {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}
    return out


def train_target(name: str, df: pd.DataFrame, n_iter: int) -> tuple[dict, pd.DataFrame]:
    t = TARGETS[name]
    if t.first_season:
        df = df[df["season"] >= t.first_season]
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
        # the same held-out races, sprint weekends only: what each sprint session adds
        "sprint_weekends": _stage_metrics(held_out[held_out["sprint_weekend"]], name in ("finish_position", "quali_delta")),
        **({"finishers": _stage_metrics(held_out[~held_out["dnf"]], True)} if t.finishers_only else {}),
        # qualifying on laps that were real attempts (within 107% of pole)
        **({"representative_laps": _stage_metrics(held_out[held_out["actual"] <= QUALI_MAX_GAP], False)}
           if name == "qualifying" else {}),
        "best_params": params,
        "top_features": contrib.abs().mean().sort_values(ascending=False).head(10).round(4).to_dict(),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(MODEL_DIR / f"{name}_xgb.json")
    (MODEL_DIR / f"{name}_metrics.json").write_text(json.dumps(metrics, indent=2))
    for label, block in (("", metrics["stages"]), ("sprint weekends ", metrics["sprint_weekends"])):
        for stage, m in block.items():
            print(f"  {label}{stage:18s} MAE {m['mae']:.3f}  vs baseline {m['baseline_mae']:.3f}  (n={m['n']})"
                  + (f"  spearman {m['mean_race_spearman']:.2f}" if "mean_race_spearman" in m else ""))
    return metrics, wf


def gate(metrics: list[dict]) -> list[str]:
    """Reasons to refuse a retrained model: at the most informed stage every
    weekend reaches (race day; after practice for qualifying), it must beat
    its naive baseline on unseen races."""
    failures = []
    for m in metrics:
        stage = "race_day" if "race_day" in m["stages"] else "post_practice"
        s = m["stages"][stage]
        if s["mae"] >= s["baseline_mae"]:
            failures.append(f"{m['target']} at {stage}: MAE {s['mae']:.3f} doesn't beat the baseline {s['baseline_mae']:.3f}")
    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", nargs="+", default=list(TARGETS), choices=list(TARGETS))
    parser.add_argument("--n-iter", type=int, default=40)
    parser.add_argument("--gate", action="store_true", help="exit non-zero if any model fails to beat its baseline")
    args = parser.parse_args()

    df = pd.read_parquet(DATA_PATH)
    results = [train_target(name, df, args.n_iter) for name in args.targets]
    frames = [wf for _, wf in results]
    if WALKFORWARD_PATH.exists():  # keep other targets' rows when retraining a subset
        old = pd.read_parquet(WALKFORWARD_PATH)
        frames.append(old[~old["target"].isin(args.targets)])
    pd.concat(frames, ignore_index=True).to_parquet(WALKFORWARD_PATH, index=False)
    print(f"wrote {WALKFORWARD_PATH}")
    if args.gate and (failures := gate([m for m, _ in results])):
        raise SystemExit("gate failed:\n  " + "\n  ".join(failures))


if __name__ == "__main__":
    main()
