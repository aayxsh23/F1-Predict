"""Chances, not just point predictions: win / podium / top-10 odds, a likely
finishing range, retirement risk and "beats teammate", sampled from the
finish-position model; pole / Q3 / out-in-Q1 odds sampled from the qualifying
model.

Each simulated result is a Plackett-Luce draw (Gumbel-max trick: sort
score/tau + Gumbel noise). In races each car also retires independently with a
probability blended from the driver's and team's recent DNF rates, and a
retired car classifies last.

tau is how noisy results are relative to the model's ordering. It is fitted
by maximum likelihood on the walk-forward predictions train.py saves (every
race predicted by a model that never saw it), separately per weekend stage,
because a Thursday forecast is less certain than a Saturday-night one. Two
taus per race stage: tau_top (fitted on the first three places) drives
win/podium, tau_field (whole order) drives top-10 and the range, since the
midfield shuffles far more than the front. A third, tau_band, sets only the
likely-finishing range: fitted on the tuning-window races so that the
10th-90th percentile range holds the real result 80% of the time (tau_field
alone gave 87%: honest but needlessly wide), and checked on held-out races.

In qualifying a car can also fail to set a representative lap (a crash, a
failure, track limits); it then classifies at the back, with a probability
from the driver's recent record (`driver_quali_nolap_rate`, shrunk toward
the field's rate).

Predictions are the ones the app shows: the blends (blend.py) applied.
`python -m src.models.probabilities` refits and prints the honest check
(Brier scores vs guessing evenly).
"""
import json
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np

CALIBRATION_PATH = Path(__file__).resolve().parent / "saved" / "pl_calibration.json"
N_SAMPLES = 10_000
BAND_TARGET = 0.80  # the P10-P90 range should hold the result this often
NOLAP_SHRINK = 5  # qualifying sessions of a driver's own record worth the field's rate
_DNF_FLOOR, _DNF_CEIL = 0.01, 0.5
Q3_SIZE = 10


@lru_cache(maxsize=1)
def load_calibration() -> dict:
    if not CALIBRATION_PATH.exists():
        raise FileNotFoundError(f"{CALIBRATION_PATH} missing -- run `python -m src.models.probabilities`")
    return json.loads(CALIBRATION_PATH.read_text())


def q1_cut(n_cars: int) -> int:
    """Cars knocked out in Q1: 6 with the 22-car grid from 2026, 5 before."""
    return 6 if n_cars >= 22 else 5


def dnf_probability(driver_dnf_rate, team_reliability, prior: float) -> np.ndarray:
    """Average of the driver's DNF rate and the team's (1 - reliability); either
    may be NaN (rookies, new teams) and `prior` covers both missing."""
    d = np.asarray(driver_dnf_rate, dtype=float)
    t = 1.0 - np.asarray(team_reliability, dtype=float)
    p = np.where(np.isnan(d), t, np.where(np.isnan(t), d, (d + t) / 2))
    return np.clip(np.where(np.isnan(p), prior, p), _DNF_FLOOR, _DNF_CEIL)


def sample_positions(score, tau: float, p_dnf=0.0, n: int = N_SAMPLES, seed: int = 0) -> np.ndarray:
    """score: higher = better, one per car. Returns (n, cars) sampled finishing
    positions, 1 = first."""
    rng = np.random.default_rng(seed)
    score = np.asarray(score, dtype=float)
    perf = score / tau + rng.gumbel(size=(n, len(score)))
    retired = rng.random(perf.shape) < np.broadcast_to(np.asarray(p_dnf, dtype=float), perf.shape)
    perf = np.where(retired, -1e9 + rng.random(perf.shape), perf)  # jitter: retirements don't classify in car order
    return (-perf).argsort(axis=1).argsort(axis=1) + 1


def _stage_cal(cal: dict, kind: str, stage: str) -> dict:
    by_stage = cal[kind]
    return by_stage.get(stage) or by_stage[next(iter(by_stage))]


def race_probabilities(predicted_finish, driver_dnf_rate, team_reliability, stage: str = "post_quali",
                       teams=None, seed: int = 0, cal: dict | None = None) -> dict:
    """predicted_finish: the model's predicted finishing position per car
    (lower = better). Per-car arrays: win, podium, top10, expected_position,
    band (P10, P90), retire, beats_teammate (NaN without a teammate)."""
    cal = cal or load_calibration()
    t = _stage_cal(cal, "race", stage)
    p_dnf = dnf_probability(driver_dnf_rate, team_reliability, cal["dnf_prior"])
    score = -np.asarray(predicted_finish, dtype=float)
    top = sample_positions(score, t["tau_top"], p_dnf, seed=seed)
    field = sample_positions(score, t["tau_field"], p_dnf, seed=seed + 1)
    band = sample_positions(score, t.get("tau_band", t["tau_field"]), p_dnf, n=4000, seed=seed + 2)
    beats = np.full(len(score), np.nan)
    if teams is not None:
        teams = list(teams)
        for i, team in enumerate(teams):
            mates = [j for j, other in enumerate(teams) if other == team and j != i]
            if mates:
                beats[i] = (field[:, [i]] < field[:, mates]).all(axis=1).mean()
    return {
        "win": (top == 1).mean(axis=0),
        "podium": (top <= 3).mean(axis=0),
        "top10": (field <= 10).mean(axis=0),
        "expected_position": field.mean(axis=0),
        "band": np.percentile(band, [10, 90], axis=0).T,
        "retire": p_dnf,
        "beats_teammate": beats,
        "samples": field,
    }


def nolap_probability(driver_nolap_rate, prior: float) -> np.ndarray:
    r = np.asarray(driver_nolap_rate, dtype=float)
    return np.clip(np.where(np.isnan(r), prior, r), 0.0, 0.5)


def quali_probabilities(predicted_gap_pct, stage: str = "post_practice", seed: int = 0, cal: dict | None = None,
                        nolap_rate=None) -> dict:
    """predicted_gap_pct: predicted qualifying gap to pole per car (lower =
    better). Per-car arrays: pole, q3, q1_out, expected_position."""
    cal = cal or load_calibration()
    tau = _stage_cal(cal, "qualifying", stage)["tau"]
    gaps = np.asarray(predicted_gap_pct, dtype=float)
    p_nolap = 0.0 if nolap_rate is None else nolap_probability(nolap_rate, cal.get("nolap_prior", 0.0))
    pos = sample_positions(-gaps, tau, p_nolap, seed=seed)
    n = pos.shape[1]
    return {
        "pole": (pos == 1).mean(axis=0),
        "q3": (pos <= Q3_SIZE).mean(axis=0),
        "q1_out": (pos > n - q1_cut(n)).mean(axis=0),
        "expected_position": pos.mean(axis=0),
    }


# --- calibration ---

def _pl_nll(tau: float, races: list[np.ndarray], k: int) -> float:
    """Negative log-likelihood of the first k places of each observed order;
    each array holds one race's scores, best actual finisher first."""
    total = 0.0
    for s in races:
        z = s / tau
        suffix_lse = np.logaddexp.accumulate(z[::-1])[::-1]
        m = min(k, len(z) - 1)
        total -= float(np.sum(z[:m] - suffix_lse[:m]))
    return total


def fit_tau(races: list[np.ndarray], k: int = 10**6) -> float:
    from scipy.optimize import minimize_scalar

    return float(minimize_scalar(lambda t: _pl_nll(t, races, k), bounds=(0.01, 30.0), method="bounded").x)


def _ordered_scores(g) -> list[np.ndarray]:
    return [-r.sort_values("actual")["pred"].to_numpy() for _, r in g.groupby(["season", "round"]) if len(r) >= 3]


def _coverage(g, tau: float, dnf_prior: float, n: int = 1500) -> float:
    inside = []
    for i, (_, r) in enumerate(g.groupby(["season", "round"])):
        p_dnf = dnf_probability(r["driver_dnf_rate"], r["team_reliability"], dnf_prior)
        pos = sample_positions(-r["pred"].to_numpy(), tau, p_dnf, n=n, seed=i)
        lo, hi = np.percentile(pos, [10, 90], axis=0)
        inside += list((r["actual"].to_numpy() >= lo) & (r["actual"].to_numpy() <= hi))
    return float(np.mean(inside))


def fit_tau_band(g, dnf_prior: float, start: float) -> float:
    """tau whose P10-P90 range holds BAND_TARGET of results (coverage falls as tau shrinks)."""
    lo, hi = 0.05 * start, start
    if _coverage(g, hi, dnf_prior) <= BAND_TARGET:
        return start
    for _ in range(12):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if _coverage(g, mid, dnf_prior) > BAND_TARGET else (mid, hi)
    return hi


def calibrate() -> dict:
    import pandas as pd

    from src.features.build_dataset import OUT_PATH, load_raw
    from src.models.blend import apply_walkforward
    from src.models.train import WALKFORWARD_PATH

    full_matrix = pd.read_parquet(OUT_PATH)
    wf = apply_walkforward(pd.read_parquet(WALKFORWARD_PATH).drop(columns=["dnf"], errors="ignore"), full_matrix)
    raw = load_raw()[["season", "round", "driver", "dnf"]]
    matrix = full_matrix[["season", "round", "driver", "driver_dnf_rate", "team_reliability", "driver_quali_nolap_rate"]]
    dnf_prior = float(raw["dnf"].mean())
    q = full_matrix["quali_position"].notna()
    nolap_prior = float((full_matrix.loc[q, "quali_gap_pct"].isna() | (full_matrix.loc[q, "quali_gap_pct"] > 7)).mean())
    cal: dict = {"dnf_prior": dnf_prior, "nolap_prior": nolap_prior, "race": {}, "qualifying": {}, "checks": {}}

    race = wf[wf["target"] == "finish_position"].merge(raw, on=["season", "round", "driver"]).merge(
        matrix, on=["season", "round", "driver"], how="left")
    for stage, g in race.groupby("stage"):
        finishers = g[~g["dnf"].astype(bool)]
        taus = {"tau_top": fit_tau(_ordered_scores(finishers), k=3), "tau_field": fit_tau(_ordered_scores(finishers))}
        taus["tau_band"] = fit_tau_band(g[~g["after_tuning_window"]], dnf_prior, taus["tau_field"])
        cal["race"][stage] = taus
        P, Y, inside = [], [], []
        for i, (_, r) in enumerate(g.groupby(["season", "round"])):
            pr = race_probabilities(r["pred"], r["driver_dnf_rate"], r["team_reliability"], seed=i,
                                    cal={"dnf_prior": dnf_prior, "race": {stage: taus}})
            a = r["actual"].to_numpy()
            P.append(np.c_[pr["win"], pr["podium"], pr["top10"]])
            Y.append(np.c_[a == 1, a <= 3, a <= 10])
            if r["after_tuning_window"].all():  # range coverage: held-out races only
                inside += list((a >= pr["band"][:, 0]) & (a <= pr["band"][:, 1]))
        P, Y = np.vstack(P), np.vstack(Y).astype(float)
        n_cars = len(g) / g.groupby(["season", "round"]).ngroups
        cal["checks"][f"race_{stage}"] = {
            "brier": dict(zip(["win", "podium", "top10"], ((P - Y) ** 2).mean(axis=0).round(4).tolist())),
            "brier_if_guessing_evenly": dict(zip(["win", "podium", "top10"], ((np.array([1, 3, 10]) / n_cars - Y) ** 2).mean(axis=0).round(4).tolist())),
            "range_p10_p90_coverage": round(float(np.mean(inside)), 3),
        }

    # ranked by the official qualifying order, with each car's no-lap chance in the draw
    quali = wf[wf["target"] == "qualifying"]
    quali_all = full_matrix[q][["season", "round", "driver", "quali_position"]].merge(
        quali.drop(columns=["actual"]), on=["season", "round", "driver"]).merge(
        matrix[["season", "round", "driver", "driver_quali_nolap_rate"]], on=["season", "round", "driver"], how="left")
    for stage, g in quali.groupby("stage"):
        cal["qualifying"][stage] = {"tau": fit_tau(_ordered_scores(g))}
        P, Y = [], []
        for i, (_, r) in enumerate(quali_all[quali_all["stage"] == stage].groupby(["season", "round"])):
            pr = quali_probabilities(r["pred"].fillna(r["pred"].max()), stage, seed=i, cal=cal,
                                     nolap_rate=r["driver_quali_nolap_rate"])
            order = r["quali_position"].rank(method="first").to_numpy()
            P.append(np.c_[pr["pole"], pr["q3"]])
            Y.append(np.c_[order == 1, order <= Q3_SIZE])
        P, Y = np.vstack(P), np.vstack(Y).astype(float)
        cal["checks"][f"qualifying_{stage}"] = {"brier": dict(zip(["pole", "q3"], ((P - Y) ** 2).mean(axis=0).round(4).tolist()))}

    cal["method"] = "Plackett-Luce MLE on walk-forward predictions (src/models/train.py), per weekend stage"
    cal["fitted_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    CALIBRATION_PATH.write_text(json.dumps(cal, indent=2))
    load_calibration.cache_clear()
    return cal


if __name__ == "__main__":
    print(json.dumps(calibrate(), indent=2))
