"""The model-family contest: candidates for one target run through the SAME
walk-forward as the shipped model (train.walk_forward: each race predicted by
a model trained only on earlier races, every weekend stage), and the
leaderboard reports, per stage:

  tuning_mae      on the tuning-window races (the ones allowed to choose)
  held_out_mae    on the later races (the honest number)
  vs_incumbent    95% race-level interval of held-out MAE(candidate) - MAE(incumbent)

Two-member ensembles are scored without refitting anything: an ensemble's
prediction is the weighted mean of its members', so their walk-forward
predictions are combined with the weight that is best on the tuning window.

Results go to src/models/saved/leaderboard.json (one entry per target and
candidate) and each candidate's walk-forward predictions to
data/processed/candidates/. The experiments/ scripts call this; a winner is
switched on in src/models/saved/model_specs.json, which train.py reads.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.train import DATA_PATH, MODEL_DIR, TARGETS, TUNE_SHARE, _race_key, diff_ci95, race_order, saved_params, walk_forward

_race_keys = _race_key

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE_DIR = ROOT / "data" / "processed" / "candidates"
LEADERBOARD_PATH = MODEL_DIR / "leaderboard.json"
WEIGHTS = np.round(np.arange(0, 1.0001, 0.05), 2)


def target_rows(name: str, df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, int]:
    """The rows a target trains on and the number of tuning-window races
    (same filtering as train.train_target)."""
    t = TARGETS[name]
    df = pd.read_parquet(DATA_PATH) if df is None else df
    if t.first_season:
        df = df[df["season"] >= t.first_season]
    df = df.dropna(subset=[t.column]).sort_values("race_date").reset_index(drop=True)
    return df, int(len(race_order(df)) * TUNE_SHARE)


def incumbent_params(name: str) -> dict:
    params, _ = saved_params(name)
    if params is None:
        raise FileNotFoundError(f"no stored settings for {name}: train it first")
    return params


def run(name: str, label: str, params: dict, df: pd.DataFrame | None = None, use_cache: bool = True) -> pd.DataFrame:
    """Walk-forward predictions of one candidate (cached by label)."""
    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    path = CANDIDATE_DIR / f"{name}__{label}.parquet"
    if use_cache and path.exists():
        return pd.read_parquet(path)
    rows, cut = target_rows(name, df)
    wf = walk_forward(name, rows, params)
    wf["after_tuning_window"] = wf["race_index"] >= cut
    wf.to_parquet(path, index=False)
    return wf


def finishers(name: str, wf: pd.DataFrame) -> pd.DataFrame:
    """Finishing position and places gained are judged on finishers when
    choosing (retirements are the odds sampler's job), as the blends are."""
    return wf[~wf["dnf"].astype(bool)] if TARGETS[name].finishers_only else wf


def combine(a: pd.DataFrame, b: pd.DataFrame, w: float) -> pd.DataFrame:
    """(1 - w) * a + w * b, aligned on race, driver and stage."""
    key = ["season", "round", "driver", "stage"]
    j = a.merge(b[key + ["pred"]].rename(columns={"pred": "_b"}), on=key)
    return j.assign(pred=(1 - w) * j["pred"] + w * j["_b"]).drop(columns="_b")


def best_weight(name: str, a: pd.DataFrame, b: pd.DataFrame) -> float:
    errs = {}
    for w in WEIGHTS:
        c = finishers(name, combine(a, b, float(w)))
        tune = c[~c["after_tuning_window"]]
        errs[float(w)] = round(float((tune["pred"] - tune["actual"]).abs().mean()), 4)
    return min(errs, key=lambda w: (errs[w], w))


def score(name: str, cand: pd.DataFrame, incumbent: pd.DataFrame) -> dict:
    key = ["season", "round", "driver", "stage"]
    out = {}
    for stage in [s for s in TARGETS[name].stages if s in set(cand["stage"])]:
        c = cand[cand["stage"] == stage]
        i = incumbent[incumbent["stage"] == stage][key + ["pred"]].rename(columns={"pred": "incumbent"})
        j = c.merge(i, on=key)
        tune, held = j[~j["after_tuning_window"]], j[j["after_tuning_window"]]
        tf, hf = finishers(name, tune), finishers(name, held)
        out[stage] = {
            "tuning_mae": round(float((tf["pred"] - tf["actual"]).abs().mean()), 4),
            "tuning_mae_incumbent": round(float((tf["incumbent"] - tf["actual"]).abs().mean()), 4),
            "held_out_mae": round(float((held["pred"] - held["actual"]).abs().mean()), 4),
            "held_out_mae_incumbent": round(float((held["incumbent"] - held["actual"]).abs().mean()), 4),
            "vs_incumbent_ci95": diff_ci95(held, "pred", "incumbent"),
            "held_out_races": int(held[["season", "round"]].drop_duplicates().shape[0]),
        }
        if TARGETS[name].finishers_only:
            out[stage]["held_out_mae_finishers"] = round(float((hf["pred"] - hf["actual"]).abs().mean()), 4)
            out[stage]["vs_incumbent_ci95_finishers"] = diff_ci95(hf, "pred", "incumbent")
    return out


def record(name: str, label: str, spec: dict, result: dict, note: str = "") -> None:
    board = json.loads(LEADERBOARD_PATH.read_text()) if LEADERBOARD_PATH.exists() else {}
    board.setdefault(name, {})[label] = {"spec": spec, "stages": result, "note": note,
                                          "scored_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    LEADERBOARD_PATH.write_text(json.dumps(board, indent=2))


def summary(name: str, result: dict, stages: list[str]) -> str:
    lines = []
    for s in stages:
        if s in result:
            r = result[s]
            lines.append(f"  {s:18s} tuning {r['tuning_mae']:.4f} vs {r['tuning_mae_incumbent']:.4f} | held-out "
                         f"{r['held_out_mae']:.4f} vs {r['held_out_mae_incumbent']:.4f} CI {r['vs_incumbent_ci95']}")
    return "\n".join(lines)


def choose(name: str, label: str, why: str) -> dict:
    """Switch a target to a leaderboard candidate: written to
    saved/model_specs.json, which train.py reads on the next retrain."""
    from src.models.train import MODEL_SPECS_PATH

    board = json.loads(LEADERBOARD_PATH.read_text())
    spec = {**board[name][label]["spec"], "chosen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "chosen_from": f"leaderboard.json: {name} / {label}", "why": why}
    specs = json.loads(MODEL_SPECS_PATH.read_text()) if MODEL_SPECS_PATH.exists() else {}
    specs[name] = spec
    MODEL_SPECS_PATH.write_text(json.dumps(specs, indent=2))
    return spec
