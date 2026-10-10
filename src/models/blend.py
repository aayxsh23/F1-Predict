"""Derived corrections on top of the four models, each a single weight w:

  finish_grid    finishing position once a grid exists:
                 (1 - w) * finish model + w * (grid - places-gained model)
  finish_sprint  finishing position after the Sprint:
                 (1 - w) * finish model + w * Sprint result
  quali_sprint   qualifying gap after Sprint Qualifying:
                 (1 - w) * qualifying model + w * Sprint Qualifying gap
  quali_team     qualifying gap before practice:
                 (1 - w) * qualifying model + w * team's recent qualifying gap

Why: the finish model leans on form and, after qualifying, named the winner
in 45% of held-out races against 66% for "the pole-sitter wins"; the
places-gained model starts from the grid. On sprint weekends the shallow
models barely use the sprint columns (one weekend in five), while the raw
Sprint Qualifying gap is a strong guide to qualifying. And before any
running, the team's own recent qualifying gap (with pre-season testing in it)
is a guess the qualifying model didn't beat on its own.

Each w is fitted on the walk-forward predictions of the tuning-window races
only (the races that also chose the hyperparameters) and then reported on the
held-out races, like the odds' temperature. Not a fifth model: a weighted
average of predictions that already exist, so explanations stay exact sums
(`contributions`). `python -m src.models.blend` (after train.py) writes
saved/blend.json and rewrites the shown accuracy in the targets' metrics.
"""
import json
from functools import lru_cache

import numpy as np
import pandas as pd

from src.models.catalog import CANONICAL_PRED_COLS, MODEL_DIR
from src.models.features import GRID_STAGES, SPRINT_STAGES

BLEND_PATH = MODEL_DIR / "blend.json"
WEIGHT_GRID = np.round(np.arange(0, 1.0001, 0.05), 2)
BLENDS = {  # name: (target, stages, anchor column)
    "finish_grid": ("finish_position", GRID_STAGES, "grid_position"),
    "finish_sprint": ("finish_position", ["post_sprint"], "sprint_finish_position"),
    "quali_sprint": ("qualifying", SPRINT_STAGES, "sprint_quali_gap_pct"),
    "quali_team": ("qualifying", ["pre_weekend"], "team_quali_pace"),
}


@lru_cache(maxsize=1)
def load_weights() -> dict:
    return json.loads(BLEND_PATH.read_text())["weights"] if BLEND_PATH.exists() else {}


def anchor(name: str, rows: pd.DataFrame, delta_pred: pd.Series | None = None) -> pd.Series:
    """What blend `name` pulls toward, NaN where it doesn't exist (no grid yet,
    no Sprint Qualifying lap)."""
    col = BLENDS[name][2]
    a = rows[col].astype(float) if col in rows else pd.Series(np.nan, index=rows.index)
    return a - delta_pred if name == "finish_grid" else a


def _mix(pred: pd.Series, a: pd.Series, w: float) -> pd.Series:
    return pred.where(a.isna(), (1 - w) * pred + w * a)


def active(target: str, stage: str, weights: dict | None = None) -> list[tuple[str, float]]:
    weights = load_weights() if weights is None else weights
    return [(n, weights[n]) for n, (t, stages, _) in BLENDS.items() if t == target and stage in stages and weights.get(n)]


def apply_rows(rows: pd.DataFrame, stage: str, weights: dict | None = None) -> pd.DataFrame:
    """rows with CANONICAL_PRED_COLS (predict.predict_all); blended in place of the raw predictions."""
    rows = rows.copy()
    for target in ("finish_position", "qualifying"):
        col = CANONICAL_PRED_COLS[target]
        for name, w in active(target, stage, weights):
            delta = rows[CANONICAL_PRED_COLS["quali_delta"]] if name == "finish_grid" else None
            rows[col] = _mix(rows[col], anchor(name, rows, delta), w).round(4)
    return rows


def _stage_inputs(matrix: pd.DataFrame, stage: str) -> pd.DataFrame:
    from src.models.features import mask_for_stage

    cols = ["season", "round", "driver", "grid_position", "quali_position", "driver_recent_form",
            "sprint_finish_position", "sprint_quali_gap_pct", "team_quali_pace"]
    return mask_for_stage(matrix[cols], stage).drop(columns=["quali_position", "driver_recent_form", "grid_official"], errors="ignore")


def apply_walkforward(wf: pd.DataFrame, matrix: pd.DataFrame, weights: dict | None = None) -> pd.DataFrame:
    """Walk-forward predictions with the blends applied (raw kept as `pred_model`)."""
    weights = load_weights() if weights is None else weights
    key = ["season", "round", "driver", "stage"]
    wf = wf.copy()
    wf["pred_model"] = wf["pred"]
    for stage in set(wf["stage"]):
        inputs = _stage_inputs(matrix, stage).assign(stage=stage)
        delta = wf[(wf["target"] == "quali_delta") & (wf["stage"] == stage)][key + ["pred"]].rename(columns={"pred": "_delta"})
        for target in ("finish_position", "qualifying"):
            sel = (wf["target"] == target) & (wf["stage"] == stage)
            if not sel.any():
                continue
            rows = wf.loc[sel, key + ["pred"]].merge(inputs, on=key, how="left").merge(delta, on=key, how="left")
            for name, w in active(target, stage, weights):
                rows["pred"] = _mix(rows["pred"], anchor(name, rows, rows["_delta"]), w)
            wf.loc[sel, "pred"] = rows["pred"].to_numpy()
    return wf


def contributions(target: str, stage: str, row: pd.DataFrame, field: pd.DataFrame,
                  weights: dict | None = None) -> tuple[pd.Series, float]:
    """Exact per-feature contributions (and base) of the prediction the app
    shows for one row: the model's own TreeSHAP values, scaled by (1 - w), plus
    w times the anchor's, the anchor's own value entering as its deviation
    from the field's average (so the base stays "a typical car here")."""
    from src.models import predict
    from src.models.features import PREPARE_FN

    c, b = predict.contributions(predict.load_model(target), PREPARE_FN[target](row))
    c, b = c.iloc[0], float(b.iloc[0])
    for name, w in active(target, stage, weights):
        col = BLENDS[name][2]
        value = float(row[col].iloc[0]) if col in row else np.nan
        if np.isnan(value):
            continue
        mean = float(field[col].mean())
        c, b = c * (1 - w), b * (1 - w)
        if name == "finish_grid":
            cd, bd = predict.contributions(predict.load_model("quali_delta"), PREPARE_FN["quali_delta"](row))
            c, b = c.sub(w * cd.iloc[0], fill_value=0.0), b - w * float(bd.iloc[0])
        c[col] = c.get(col, 0.0) + w * (value - mean)
        b += w * mean
    return c, b


def _report(wf: pd.DataFrame, name: str) -> dict:
    """Held-out effect of one blend at its stages."""
    from src.models.train import diff_ci95

    target, stages, _ = BLENDS[name]
    out = {}
    for stage in stages:
        g = wf[(wf["target"] == target) & (wf["stage"] == stage) & wf["after_tuning_window"]]
        if g.empty:
            continue
        m = {"n": int(len(g)), "mae_model": round(float((g["pred_model"] - g["actual"]).abs().mean()), 4),
             "mae_blend": round(float((g["pred"] - g["actual"]).abs().mean()), 4),
             "blend_minus_model_ci95": diff_ci95(g, "pred", "pred_model")}
        if target == "finish_position":
            races = list(g.groupby(["season", "round"]))
            for col in ("pred_model", "pred"):
                m[f"winner_{col}"] = round(float(np.mean([r.loc[r[col].idxmin(), "actual"] == 1 for _, r in races])), 3)
                m[f"podium_{col}"] = round(float(np.mean([len(set(r.nsmallest(3, col)["driver"]) & set(r[r["actual"] <= 3]["driver"]))
                                                          for _, r in races])), 3)
        out[stage] = m
    return out


def fit() -> dict:
    """Choose each weight on tuning-window races, report on held-out races,
    and rewrite the targets' shown metrics from the blended predictions."""
    from datetime import datetime, timezone

    from src.features.build_dataset import OUT_PATH
    from src.features.driver_features import QUALI_MAX_GAP
    from src.models.train import TARGETS, WALKFORWARD_PATH, _stage_metrics

    wf = pd.read_parquet(WALKFORWARD_PATH)
    matrix = pd.read_parquet(OUT_PATH)
    weights = {}
    for name, (target, stages, _) in BLENDS.items():
        # the places-gained predictions ride along: finish_grid's anchor is built from them
        pool = wf[wf["target"].isin([target, "quali_delta"]) & wf["stage"].isin(stages) & ~wf["after_tuning_window"]]
        errs = {}
        for w in WEIGHT_GRID:
            b = apply_walkforward(pool, matrix, {name: float(w)})
            b = b[b["target"] == target]
            if target == "finish_position":
                b = b[~b["dnf"].astype(bool)]  # retirements are the odds sampler's job
            errs[float(w)] = float((b["pred"] - b["actual"]).abs().mean()) if len(b) else np.inf
        weights[name] = min(errs, key=lambda w: (round(errs[w], 4), w))  # ties: the smaller correction
        print(f"{name}: w={weights[name]} (tuning-window MAE {errs[0.0]:.3f} -> {errs[weights[name]]:.3f})")

    blended = apply_walkforward(wf, matrix, weights)
    report = {n: _report(blended, n) for n in BLENDS}
    BLEND_PATH.write_text(json.dumps({
        "weights": weights, "held_out": report,
        "method": "one weight per blend, chosen by MAE on tuning-window walk-forward predictions (finishers for "
                  "finishing position); held_out = races after the tuning window",
        "fitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2))
    load_weights.cache_clear()

    held = blended[blended["after_tuning_window"]]
    for target in {t for t, _, _ in BLENDS.values()}:
        path = MODEL_DIR / f"{target}_metrics.json"
        m = json.loads(path.read_text())
        g = held[held["target"] == target]
        position = target == "finish_position"
        m["model_only_stages"] = m.get("model_only_stages", m["stages"])
        m["stages"] = _stage_metrics(g, position)
        m["sprint_weekends"] = _stage_metrics(g[g["sprint_weekend"]], position)
        if TARGETS[target].finishers_only:
            m["finishers"] = _stage_metrics(g[~g["dnf"]], True)
        if target == "qualifying":
            m["representative_laps"] = _stage_metrics(g[g["actual"] <= QUALI_MAX_GAP], False)
        m["blends"] = {n: {"weight": weights[n], "held_out": report[n]} for n, (t, _, _) in BLENDS.items() if t == target}
        path.write_text(json.dumps(m, indent=2))
    return {"weights": weights, "held_out": report}


if __name__ == "__main__":
    print(json.dumps(fit(), indent=2))
