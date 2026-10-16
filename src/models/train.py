"""Train, evaluate and save all four predictors:
`python -m src.models.train [--targets finish_position ...] [--n-iter 40]`.

For each target:
1. Tune XGBoost hyperparameters on the earliest 60% of races only, with
   time-ordered folds (never validated on a race older than its training).
   Only with --retune (or when none are stored): otherwise the settings the
   shipped model was trained with are reused, so weekly retrains don't drift.
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
from scipy.stats import randint, spearmanr, uniform
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

from src.features.driver_features import QUALI_MAX_GAP
from src.models.features import PREPARE_FN, QUALI_STAGES, RACE_STAGES, SPRINT_STAGES, STAGES, mask_for_stage
from src.models import families
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
BASE_PARAMS = families.XGB_BASE
LGBM_PARAM_DIST = {
    "num_leaves": randint(4, 48),
    "learning_rate": uniform(0.01, 0.2),
    "n_estimators": randint(100, 600),
    "subsample": uniform(0.6, 0.4),
    "subsample_freq": randint(1, 2),
    "colsample_bytree": uniform(0.6, 0.4),
    "min_child_samples": randint(5, 60),
    "reg_alpha": uniform(0, 2),
    "reg_lambda": uniform(0.5, 3),
}


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


MODEL_SPECS_PATH = MODEL_DIR / "model_specs.json"


def chosen_spec(name: str) -> dict | None:
    """The model family an experiment chose for a target on the tuning window
    (src/models/saved/model_specs.json, written by experiments/); None means
    plain XGBoost."""
    if not MODEL_SPECS_PATH.exists():
        return None
    return json.loads(MODEL_SPECS_PATH.read_text()).get(name)


def spec_of(params: dict) -> dict:
    """A model spec from a target's stored params: plain XGBoost settings, or
    a whole spec ({"family": ...}) for any other family."""
    return params if "family" in params else {"family": "xgb", "params": params}


def fit(name: str, rows: pd.DataFrame, params: dict):
    t = TARGETS[name]
    aug = augment(_train_rows(name, rows), t.stages)
    spec = spec_of(params)
    model = families.build(spec)
    if isinstance(model, families.TwoPart):
        return model.fit(PREPARE_FN[name](aug), aug[t.column], y_split=aug[spec["split_column"]].fillna(False).astype(bool))
    return model.fit(PREPARE_FN[name](aug), aug[t.column])


def _search(estimator, dist: dict, X, y, folds, n_iter: int) -> dict:
    search = RandomizedSearchCV(estimator, dist, n_iter=n_iter, scoring="neg_mean_absolute_error",
                                cv=folds, random_state=42, refit=False, n_jobs=-1)
    search.fit(X, y)
    return {k: (int(v) if isinstance(v, (np.integer, int)) else float(v)) for k, v in search.best_params_.items()}


def tune(name: str, rows: pd.DataFrame, n_iter: int, family: str = "xgb", subset=None) -> dict:
    """Hyperparameters for one tree family on the tuning rows, by MAE over
    time-ordered race folds. `subset` (a boolean array over the training rows)
    tunes a two-part model's regression on its own rows."""
    import xgboost as xgb

    t = TARGETS[name]
    aug = augment(_train_rows(name, rows if subset is None else rows[subset]), t.stages)
    X, y, folds = PREPARE_FN[name](aug), aug[t.column], list(race_folds(aug))
    if family == "lgbm":
        import lightgbm as lgb

        return _search(lgb.LGBMRegressor(**families.LGBM_BASE), LGBM_PARAM_DIST, X, y, folds, n_iter)
    return _search(xgb.XGBRegressor(**BASE_PARAMS), PARAM_DIST, X, y, folds, n_iter)


def retune_spec(name: str, rows: pd.DataFrame, n_iter: int, spec: dict | None) -> dict:
    """Fresh settings for a target's family. A composite keeps its structure
    (weights, threshold) and re-tunes each tree member."""
    if spec is None or spec_of(spec)["family"] == "xgb":
        return tune(name, rows, n_iter)
    spec = json.loads(json.dumps(spec))
    if spec["family"] == "ensemble":
        for m in spec["members"]:
            m["params"] = tune(name, rows, n_iter, family=m["family"])
    elif spec["family"] == "two_part":
        split = rows[spec["split_column"]].fillna(False).astype(bool).to_numpy()
        spec["first"]["params"] = tune(name, rows, n_iter, subset=~split)
        spec["second"]["params"] = tune(name, rows, n_iter, subset=split)
    return spec


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


def saved_params(name: str) -> tuple[dict | None, str | None]:
    """The hyperparameters the shipped model was trained with, and when they
    were last searched for."""
    path = MODEL_DIR / f"{name}_metrics.json"
    if not path.exists():
        return None, None
    m = json.loads(path.read_text())
    return m.get("best_params"), m.get("params_tuned_at") or m.get("trained_at")


def train_target(name: str, df: pd.DataFrame, n_iter: int, retune: bool = True) -> tuple[dict, pd.DataFrame]:
    t = TARGETS[name]
    if t.first_season:
        df = df[df["season"] >= t.first_season]
    df = df.dropna(subset=[t.column]).sort_values("race_date").reset_index(drop=True)
    races = race_order(df)
    cut = int(len(races) * TUNE_SHARE)
    tune_rows = df[_race_key(df).isin(_race_key(races)[:cut])]
    print(f"[{name}] {len(df)} rows, {len(races)} races; tuning on the first {cut}")

    # Searching again every week let a tiny data change pick different
    # settings: gap-to-winner's held-out error drifted 0.627 -> 0.641 -> 0.657
    # in three retrains. Weekly retrains reuse the stored settings; a search
    # runs monthly, on request, or when none are stored.
    stored, stored_at = saved_params(name)
    chosen = chosen_spec(name)
    if chosen and spec_of(stored or {})["family"] != chosen["family"]:  # an experiment switched the family
        stored, stored_at = chosen, chosen.get("chosen_at")
    params, tuned_at = (None, None) if retune else (stored, stored_at)
    if params is None:
        params = retune_spec(name, tune_rows, n_iter, chosen or (stored if stored and "family" in stored else None))
        tuned_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        print(f"[{name}] searched hyperparameters ({n_iter} candidates)")
    else:
        print(f"[{name}] reusing hyperparameters searched {tuned_at}")
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
        "segments": segments(name, wf, df),
        "family": spec_of(params)["family"],
        "best_params": params,
        "params_tuned_at": tuned_at,
        "top_features": contrib.abs().mean().sort_values(ascending=False).head(10).round(4).to_dict(),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    families.save(model, MODEL_DIR, name)
    (MODEL_DIR / f"{name}_metrics.json").write_text(json.dumps(metrics, indent=2))
    for label, block in (("", metrics["stages"]), ("sprint weekends ", metrics["sprint_weekends"])):
        for stage, m in block.items():
            print(f"  {label}{stage:18s} MAE {m['mae']:.3f}  vs baseline {m['baseline_mae']:.3f}  (n={m['n']})"
                  + (f"  spearman {m['mean_race_spearman']:.2f}" if "mean_race_spearman" in m else ""))
    return metrics, wf


SHOWN = {"qualifying": "post_practice"}  # the stage each target is judged at in segments (default post_quali)
ERA_START_SEASONS = (2022, 2026)  # rules resets inside the data (rolling.ERA_STARTS, minus 2017 which precedes it)


def segments(name: str, wf: pd.DataFrame, rows: pd.DataFrame) -> dict:
    """Held-out error at the shown stage for the slices where models usually
    struggle, each against the same naive baseline: era-start rounds, wet
    forecasts, street circuits, sprint weekends, rookies, drivers new to their
    team. A slice the model loses is a to-do, not a hidden number."""
    stage = SHOWN.get(name, "post_quali")
    g = wf[(wf["stage"] == stage) & wf["after_tuning_window"]]
    if g.empty:
        return {}
    key = ["season", "round", "driver"]
    info = rows[key + ["team", "race_date", "rain_mm_forecast", "is_street_circuit", "is_sprint_weekend"]].copy()
    info = info.sort_values("race_date")
    info["starts_before"] = info.groupby("driver").cumcount()
    prev_team = info.groupby("driver")["team"].shift(1)
    first_of_season = info.groupby(["driver", "season"]).cumcount() == 0
    moved = (first_of_season & prev_team.notna() & (prev_team != info["team"]))
    info["new_team_season"] = moved.groupby([info["driver"], info["season"]]).transform("max")
    g = g.merge(info, on=key, how="left")
    slices = {
        "era_start_rounds_1_4": g["season"].isin(ERA_START_SEASONS) & (g["round"] <= 4),
        "wet_forecast_1mm": g["rain_mm_forecast"] >= 1,
        "street_circuit": g["is_street_circuit"] == 1,
        "sprint_weekend": g["is_sprint_weekend"] == 1,
        "rookie_first_10_starts": g["starts_before"] < 10,
        "new_to_team_this_season": g["new_team_season"].fillna(False).astype(bool),
    }
    out = {"stage": stage}
    for label, mask in slices.items():
        s = g[mask]
        if s.empty:
            continue
        out[label] = {"mae": round(float((s["pred"] - s["actual"]).abs().mean()), 4),
                      "baseline_mae": round(float((s["baseline"] - s["actual"]).abs().mean()), 4),
                      "rows": int(len(s)), "races": int(s[["season", "round"]].drop_duplicates().shape[0])}
    return out


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
    parser.add_argument("--retune", action="store_true",
                        help="search hyperparameters again instead of reusing the shipped models' (monthly, or after a rules reset)")
    args = parser.parse_args()

    df = pd.read_parquet(DATA_PATH)
    results = [train_target(name, df, args.n_iter, retune=args.retune) for name in args.targets]
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
