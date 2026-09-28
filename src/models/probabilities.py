"""Win / podium / top-10 probabilities for one race, sampled from the
finish-position model's scores.

Each finisher order is a Plackett-Luce draw (Gumbel-max trick: sort
score/tau + Gumbel noise), and each car retires independently with a
probability blended from the driver's and team's recency-weighted DNF rates,
so a retirement always classifies last.

Two temperatures (how noisy results are, in finishing-position units), both fitted
once by maximum likelihood on walk-forward OUT-OF-FOLD predictions -- the same
time-based folds training uses -- over finishers only, since retirements are
sampled separately above: tau_top from the first three finishing places (win,
podium), tau_field from the whole order (top-10, position band). One temperature
for both was tried first and was under-confident at the front (cars it gave
20-40% actually won 50% of the time) because the pack shuffles far more than
the leaders do. `python -m src.models.probabilities --calibrate` rewrites
saved/pl_calibration.json; the API reads it, never refits at runtime.

The model that produces the scores is a plain L1 regressor, so these are
probabilities *of that model's ordering under measured noise*, not a second
model. The Brier check printed by --calibrate is the honest test of how far to
trust them."""
import argparse
import json
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

CALIBRATION_PATH = Path(__file__).resolve().parent / "saved" / "pl_calibration.json"
N_SAMPLES = 10_000
_DNF_FLOOR, _DNF_CEIL = 0.01, 0.5


@lru_cache(maxsize=1)
def load_calibration() -> dict:
    if not CALIBRATION_PATH.exists():
        raise FileNotFoundError(f"{CALIBRATION_PATH} missing -- run `python -m src.models.probabilities --calibrate`")
    return json.loads(CALIBRATION_PATH.read_text())


def dnf_probability(driver_dnf_rate, team_reliability, prior: float) -> np.ndarray:
    """Average of the driver's DNF rate and the team's (1 - reliability); either
    may be NaN (rookies, new teams) and `prior` covers both missing."""
    d = np.asarray(driver_dnf_rate, dtype=float)
    t = 1.0 - np.asarray(team_reliability, dtype=float)
    p = np.where(np.isnan(d), t, np.where(np.isnan(t), d, (d + t) / 2))
    return np.clip(np.where(np.isnan(p), prior, p), _DNF_FLOOR, _DNF_CEIL)


def sample_positions(score, tau: float, p_dnf, n: int = N_SAMPLES, seed: int = 0) -> np.ndarray:
    """score: higher = better, one per car. Returns (n, cars) sampled finishing
    positions, 1 = win."""
    rng = np.random.default_rng(seed)
    score = np.asarray(score, dtype=float)
    perf = score / tau + rng.gumbel(size=(n, len(score)))
    retired = rng.random(perf.shape) < np.asarray(p_dnf, dtype=float)
    perf = np.where(retired, -1e9 + rng.random(perf.shape), perf)  # jitter: DNFs don't finish in car order
    return (-perf).argsort(axis=1).argsort(axis=1) + 1


def race_probabilities(predicted_finish, driver_dnf_rate, team_reliability, seed: int = 0, cal: dict | None = None) -> dict:
    """predicted_finish: the model's raw predicted_finish_position per car (lower
    = better). Returns per-car arrays: win, podium, top10, expected_position,
    band (P10, P90 finishing position). `cal` overrides the saved calibration
    (only --calibrate needs that, to score a tau it hasn't saved yet)."""
    cal = cal or load_calibration()
    p_dnf = dnf_probability(driver_dnf_rate, team_reliability, cal["dnf_prior"])
    score = -np.asarray(predicted_finish, dtype=float)
    top = sample_positions(score, cal["tau_top"], p_dnf, seed=seed)
    field = sample_positions(score, cal["tau_field"], p_dnf, seed=seed + 1)
    return {
        "win": (top == 1).mean(axis=0),
        "podium": (top <= 3).mean(axis=0),
        "top10": (field <= 10).mean(axis=0),
        "expected_position": field.mean(axis=0),
        "band": np.percentile(field, [10, 90], axis=0).T,
    }


def _pl_nll(tau: float, races: list[np.ndarray], k: int) -> float:
    """Negative log-likelihood of the first k finishing places of each observed
    order. Each array holds one race's scores, best actual finisher first."""
    total = 0.0
    for s in races:
        z = s / tau
        suffix_lse = np.logaddexp.accumulate(z[::-1])[::-1]  # log-sum-exp of z[j:]
        m = min(k, len(z) - 1)
        total -= float(np.sum(z[:m] - suffix_lse[:m]))
    return total


def fit_tau(races: list[np.ndarray], k: int = 10**6) -> float:
    return float(minimize_scalar(lambda t: _pl_nll(t, races, k), bounds=(0.05, 20.0), method="bounded").x)


def _oof_frame() -> tuple[pd.DataFrame, float]:
    """Walk-forward out-of-fold finish-position predictions for every race the
    time-based folds hold out, using the shipped tuned hyperparameters on the
    current feature matrix, plus the raw `dnf` flag."""
    import xgboost as xgb

    from src.features.build_dataset import load_raw
    from src.models.features import prepare_features
    from src.models.train_finish_position import load_data
    from src.models.training_common import time_based_splits

    metrics = json.loads((Path(__file__).resolve().parent / "saved" / "finish_position_metrics.json").read_text())
    df = load_data()
    X = prepare_features(df)
    y = df["target_finish_position"]
    raw = load_raw()

    parts = []
    for train_idx, test_idx in time_based_splits(df):
        m = xgb.XGBRegressor(objective="reg:absoluteerror", enable_categorical=True, random_state=42, n_jobs=-1, **metrics["best_params"])
        m.fit(X.iloc[train_idx], y.iloc[train_idx])
        part = df.iloc[test_idx][["season", "round", "driver", "target_finish_position", "driver_dnf_rate", "team_reliability"]].copy()
        part["pred"] = m.predict(X.iloc[test_idx])
        parts.append(part)
    oof = pd.concat(parts, ignore_index=True)
    return oof.merge(raw[["season", "round", "driver", "dnf"]], on=["season", "round", "driver"], how="left"), float(raw["dnf"].mean())


def calibrate() -> dict:
    oof, dnf_prior = _oof_frame()
    finishers = oof[~oof["dnf"].fillna(False).astype(bool)]
    races = [
        -g.sort_values("target_finish_position")["pred"].to_numpy()  # score = -pred, best actual finisher first
        for _, g in finishers.groupby(["season", "round"]) if len(g) >= 3
    ]
    cal = {"tau_top": fit_tau(races, k=3), "tau_field": fit_tau(races), "dnf_prior": dnf_prior}

    # honest check: Brier score of each sampled probability against what happened,
    # and how often the actual finishing position landed inside the P10-P90 band
    P, Y, in_band = [], [], []
    for i, (_, g) in enumerate(oof.groupby(["season", "round"])):
        pr = race_probabilities(g["pred"], g["driver_dnf_rate"], g["team_reliability"], seed=i, cal=cal)
        actual = g["target_finish_position"].to_numpy()
        P.append(np.c_[pr["win"], pr["podium"], pr["top10"]])
        Y.append(np.c_[actual == 1, actual <= 3, actual <= 10])
        in_band += list((actual >= pr["band"][:, 0]) & (actual <= pr["band"][:, 1]))
    P, Y = np.vstack(P), np.vstack(Y).astype(float)
    n_cars = len(oof) / oof.groupby(["season", "round"]).ngroups
    brier = ((P - Y) ** 2).mean(axis=0)
    uniform = ((np.array([1, 3, 10]) / n_cars - Y) ** 2).mean(axis=0)

    cal.update({
        "n_races": int(oof.groupby(["season", "round"]).ngroups),
        "brier": {"win": float(brier[0]), "podium": float(brier[1]), "top10": float(brier[2])},
        "brier_uniform_field": {"win": float(uniform[0]), "podium": float(uniform[1]), "top10": float(uniform[2])},
        "band_p10_p90_coverage": float(np.mean(in_band)),
        "mean_p_dnf": float(np.mean(dnf_probability(oof["driver_dnf_rate"], oof["team_reliability"], dnf_prior))),
        "method": "walk-forward OOF (time_based_splits), Plackett-Luce MLE over finishers, DNFs sampled separately",
        "fitted_at": datetime.now(timezone.utc).isoformat(),
    })
    CALIBRATION_PATH.write_text(json.dumps(cal, indent=2))
    load_calibration.cache_clear()
    return cal


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrate", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(calibrate(), indent=2))
