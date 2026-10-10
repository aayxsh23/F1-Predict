"""Chances, not just point predictions: win / podium / top-10 odds, a likely
finishing range, retirement risk and "beats teammate", sampled from the
finish-position model; pole / Q3 / out-in-Q1 odds sampled from the qualifying
model.

Each simulated result is a Plackett-Luce draw (Gumbel-max trick: sort
score/tau + Gumbel noise). In races each car also retires with the field's
recent retirement rate (the share of starters who retired over the last
DNF_WINDOW races) and a retired car classifies last. Per-driver and per-team
rates were used until 2026-10-11 and scored worse than that single rate
(Brier 0.1321 vs 0.1297 on held-out races): ten or twenty races of one car's
history are mostly noise.

tau is how noisy results are relative to the model's ordering. Each one is
fitted, separately per weekend stage and on the tuning-window races only, for
the probabilities it produces, then checked on the held-out races:
  tau_top    win and podium (Brier score)
  tau_field  top ten (Brier score); also expected position and beats-teammate
  tau_band   the likely range if the car finishes: its P10-P90 should hold
             a finisher's result 80% of the time. (With every car's
             retirement chance above 10%, a range that counted retirements
             would end at last place for everyone.)
Until 2026-10-11 tau_top was a Plackett-Luce likelihood fit to the whole top
three, which left favourites under-stated: drivers given 40-60% to win won 87%
of the time. In qualifying, tau is fitted for pole and Q3 the same way, and a
car can also fail to set a representative lap (a crash, a failure, track
limits): it then classifies at the back, with a probability from the driver's
recent record (`driver_quali_nolap_rate`, field rate when missing).

Predictions are the ones the app shows: the blends (blend.py) applied.
`python -m src.models.probabilities` refits and prints the honest check
(held-out Brier scores vs guessing evenly, and the win reliability table).
"""
import json
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np

CALIBRATION_PATH = Path(__file__).resolve().parent / "saved" / "pl_calibration.json"
N_SAMPLES = 10_000
FIT_SAMPLES = 2_000  # per race while searching temperatures (same draws for every tau)
TAU_GRID = np.geomspace(0.3, 8.0, 36)  # candidate temperatures, on the 1-22 position scale
BAND_TARGET = 0.80  # the P10-P90 range should hold the result this often
DNF_WINDOW = 48  # races in the field's retirement rate (~two seasons; 12-96 all scored alike on the tuning races)
Q3_SIZE = 10


@lru_cache(maxsize=1)
def load_calibration() -> dict:
    if not CALIBRATION_PATH.exists():
        raise FileNotFoundError(f"{CALIBRATION_PATH} missing -- run `python -m src.models.probabilities`")
    return json.loads(CALIBRATION_PATH.read_text())


def q1_cut(n_cars: int) -> int:
    """Cars knocked out in Q1: 6 with the 22-car grid from 2026, 5 before."""
    return 6 if n_cars >= 22 else 5


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


def race_probabilities(predicted_finish, stage: str = "post_quali", teams=None, seed: int = 0,
                       cal: dict | None = None, p_dnf: float | None = None) -> dict:
    """predicted_finish: the model's predicted finishing position per car
    (lower = better). Per-car arrays: win, podium, top10, expected_position,
    band (P10, P90), retire, beats_teammate (NaN without a teammate). p_dnf:
    every car's retirement chance (default: the calibrated field rate)."""
    cal = cal or load_calibration()
    t = _stage_cal(cal, "race", stage)
    score = -np.asarray(predicted_finish, dtype=float)
    p = np.full(len(score), cal["dnf_prior"] if p_dnf is None else p_dnf)
    top = sample_positions(score, t["tau_top"], p, seed=seed)
    field = sample_positions(score, t["tau_field"], p, seed=seed + 1)
    band = sample_positions(score, t.get("tau_band", t["tau_field"]), 0.0, n=4000, seed=seed + 2)  # if it finishes
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
        "retire": p,
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

def field_retirement_rates(raw) -> "pd.DataFrame":
    """Per race: the share of starters who retired over the DNF_WINDOW races
    before it (season, round, p_dnf). Strictly earlier races only; the very
    first race, with nothing before it, gets the overall rate."""
    races = raw.groupby(["season", "round"]).agg(date=("race_date", "first"), n=("dnf", "size"), d=("dnf", "sum"))
    races = races.reset_index().sort_values("date").reset_index(drop=True)
    d, n = races["d"].rolling(DNF_WINDOW, min_periods=1).sum().shift(1), races["n"].rolling(DNF_WINDOW, min_periods=1).sum().shift(1)
    return races.assign(p_dnf=(d / n).fillna(races["d"].sum() / races["n"].sum()))[["season", "round", "p_dnf"]]


def _race_brier(g, tau: float, n: int = FIT_SAMPLES) -> np.ndarray:
    """Mean squared error of win / podium / top-10 chances over the races in g
    (columns pred, actual, p_dnf), with temperature tau."""
    err = np.zeros(3)
    cars = 0
    for i, (_, r) in enumerate(g.groupby(["season", "round"])):
        pos = sample_positions(-r["pred"].to_numpy(), tau, float(r["p_dnf"].iloc[0]), n=n, seed=i)
        a = r["actual"].to_numpy()
        for j, k in enumerate((1, 3, 10)):
            err[j] += (((pos <= k).mean(axis=0) - (a <= k)) ** 2).sum()
        cars += len(r)
    return err / max(cars, 1)


def _coverage(g, tau: float, n: int = 1500) -> float:
    """Share of finishers whose result fell inside their P10-P90 range (drawn without retirements)."""
    inside = []
    for i, (_, r) in enumerate(g.groupby(["season", "round"])):
        pos = sample_positions(-r["pred"].to_numpy(), tau, 0.0, n=n, seed=i)
        lo, hi = np.percentile(pos, [10, 90], axis=0)
        fin = ~r["dnf"].astype(bool).to_numpy()
        inside += list(((r["actual"].to_numpy() >= lo) & (r["actual"].to_numpy() <= hi))[fin])
    return float(np.mean(inside))


def fit_tau_band(g, start: float) -> float:
    """tau whose P10-P90 range holds BAND_TARGET of results (coverage falls as tau shrinks)."""
    lo, hi = 0.05 * start, start
    if _coverage(g, hi) <= BAND_TARGET:
        return start
    for _ in range(12):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if _coverage(g, mid) > BAND_TARGET else (mid, hi)
    return hi


def _quali_brier(g, tau: float, nolap_prior: float, n: int = FIT_SAMPLES) -> float:
    """Mean squared error of pole and Q3 chances (columns pred, quali_position,
    driver_quali_nolap_rate)."""
    err, cars = 0.0, 0
    for i, (_, r) in enumerate(g.groupby(["season", "round"])):
        pos = sample_positions(-r["pred"].to_numpy(), tau, nolap_probability(r["driver_quali_nolap_rate"], nolap_prior), n=n, seed=i)
        order = r["quali_position"].rank(method="first").to_numpy()
        err += (((pos == 1).mean(axis=0) - (order == 1)) ** 2).sum() + (((pos <= Q3_SIZE).mean(axis=0) - (order <= Q3_SIZE)) ** 2).sum()
        cars += len(r)
    return err / max(cars, 1)


def _reliability(p: np.ndarray, y: np.ndarray, edges=(0, .05, .1, .2, .4, 1.0)) -> list[dict]:
    """How often things given each range of chances actually happened."""
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if m.any():
            out.append({"stated": f"{lo:.0%}-{hi:.0%}", "n": int(m.sum()), "mean_stated": round(float(p[m].mean()), 3),
                        "happened": round(float(y[m].mean()), 3)})
    return out


def calibrate() -> dict:
    import pandas as pd

    from src.features.build_dataset import OUT_PATH, load_raw
    from src.models.blend import apply_walkforward
    from src.models.train import WALKFORWARD_PATH

    key = ["season", "round", "driver"]
    full_matrix = pd.read_parquet(OUT_PATH)
    wf = apply_walkforward(pd.read_parquet(WALKFORWARD_PATH).drop(columns=["dnf"], errors="ignore"), full_matrix)
    raw = load_raw()[["season", "round", "race_date", "driver", "dnf"]]
    rates = field_retirement_rates(raw)
    last = raw.groupby(["season", "round"]).agg(date=("race_date", "first"), d=("dnf", "sum"), n=("dnf", "size")).sort_values("date").tail(DNF_WINDOW)
    dnf_prior = float(last["d"].sum() / last["n"].sum())
    q = full_matrix["quali_position"].notna()
    nolap_prior = float((full_matrix.loc[q, "quali_gap_pct"].isna() | (full_matrix.loc[q, "quali_gap_pct"] > 7)).mean())
    cal: dict = {"dnf_prior": round(dnf_prior, 4), "dnf_window_races": DNF_WINDOW, "nolap_prior": nolap_prior,
                 "race": {}, "qualifying": {}, "checks": {}}

    race = wf[wf["target"] == "finish_position"].merge(raw[key + ["dnf"]], on=key).merge(rates, on=["season", "round"])
    for stage, g in race.groupby("stage"):
        tune, held = g[~g["after_tuning_window"]], g[g["after_tuning_window"]]
        scores = {float(t): _race_brier(tune, t) for t in TAU_GRID}
        tau_top = min(scores, key=lambda t: scores[t][0] + scores[t][1])
        tau_field = min(scores, key=lambda t: scores[t][2])
        taus = {"tau_top": tau_top, "tau_field": tau_field, "tau_band": fit_tau_band(tune, tau_field)}
        cal["race"][stage] = taus
        P, Y, inside = [], [], []
        for i, (_, r) in enumerate(held.groupby(["season", "round"])):
            pr = race_probabilities(r["pred"], stage=stage, seed=i, p_dnf=float(r["p_dnf"].iloc[0]),
                                    cal={"dnf_prior": dnf_prior, "race": {stage: taus}})
            a = r["actual"].to_numpy()
            P.append(np.c_[pr["win"], pr["podium"], pr["top10"]])
            Y.append(np.c_[a == 1, a <= 3, a <= 10])
            fin = ~r["dnf"].astype(bool).to_numpy()
            inside += list(((a >= pr["band"][:, 0]) & (a <= pr["band"][:, 1]))[fin])
        if not P:
            continue
        P, Y = np.vstack(P), np.vstack(Y).astype(float)
        n_cars = len(held) / held.groupby(["season", "round"]).ngroups
        held_cars = held.drop_duplicates(key)
        cal["checks"][f"race_{stage}"] = {
            "held_out_races": int(held.groupby(["season", "round"]).ngroups),
            "brier": dict(zip(["win", "podium", "top10"], ((P - Y) ** 2).mean(axis=0).round(4).tolist())),
            "brier_if_guessing_evenly": dict(zip(["win", "podium", "top10"], ((np.array([1, 3, 10]) / n_cars - Y) ** 2).mean(axis=0).round(4).tolist())),
            "range_p10_p90_coverage_finishers": round(float(np.mean(inside)), 3),
            "retirement_brier": round(float(((held_cars["p_dnf"] - held_cars["dnf"].astype(float)) ** 2).mean()), 4),
            "win_reliability": _reliability(P[:, 0], Y[:, 0]),
        }

    # qualifying, ranked by the official order, each car's no-lap chance in the draw
    quali = wf[wf["target"] == "qualifying"]
    quali_all = full_matrix[q][key + ["quali_position", "driver_quali_nolap_rate"]].merge(quali.drop(columns=["actual"]), on=key)
    for stage, g in quali_all.groupby("stage"):
        tune, held = g[~g["after_tuning_window"]], g[g["after_tuning_window"]]
        scores = {float(t): _quali_brier(tune, t, nolap_prior) for t in TAU_GRID}
        cal["qualifying"][stage] = {"tau": min(scores, key=scores.get)}
        P, Y = [], []
        for i, (_, r) in enumerate(held.groupby(["season", "round"])):
            pr = quali_probabilities(r["pred"], stage, seed=i, cal=cal, nolap_rate=r["driver_quali_nolap_rate"])
            order = r["quali_position"].rank(method="first").to_numpy()
            P.append(np.c_[pr["pole"], pr["q3"]])
            Y.append(np.c_[order == 1, order <= Q3_SIZE])
        if P:
            P, Y = np.vstack(P), np.vstack(Y).astype(float)
            cal["checks"][f"qualifying_{stage}"] = {"brier": dict(zip(["pole", "q3"], ((P - Y) ** 2).mean(axis=0).round(4).tolist())),
                                                    "pole_reliability": _reliability(P[:, 0], Y[:, 0])}

    cal["method"] = ("temperatures per weekend stage chosen on tuning-window walk-forward predictions for the Brier score of "
                     "what they produce (tau_top: win + podium; tau_field: top 10; qualifying: pole + Q3), tau_band for 80% "
                     "range coverage; retirements: the field rate over the last DNF_WINDOW races; checks on held-out races")
    cal["fitted_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    CALIBRATION_PATH.write_text(json.dumps(cal, indent=2))
    load_calibration.cache_clear()
    return cal


if __name__ == "__main__":
    print(json.dumps(calibrate(), indent=2))
