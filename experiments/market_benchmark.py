"""Our win odds against the betting market's. Phase 10 item 2.8.

The gold standard in sports forecasting is the market's price just before the
start. This compares it with our win odds after qualifying (the closest stage
we publish) on the held-out races, and tests whether ours adds anything to the
market's price.

Input: data/external/market_win_odds.csv, one row per car: season, round,
driver (three-letter code), odds (decimal, the last price before the start).
It is gitignored: the source's terms decide what may be shared, so publish only
the aggregate comparison. An exchange's historical data (e.g. Betfair's) suits
it, since its margin is small. A race counts only if every car we scored has a
price.

  margin       prices -> probabilities: 1/odds, scaled to sum to 1 per race
  scores       per car, win log loss and Brier (as the model card scores
               them); per race, the winner's log loss (-log of the
               probability given to the car that won), ours minus the
               market's with a race-level 95% interval
  adds info?   p proportional to market^a * ours^b, fitted over the races (a
               conditional logit): a likelihood-ratio test of b = 0, and the
               leave-one-race-out winner log loss of that blend against the
               market alone (refitted, so both get the same recalibration)

    python -m experiments.market_benchmark
    python -m experiments.market_benchmark --self-check   # synthetic races, no data needed
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import chi2

ODDS_PATH = Path("data/external/market_win_odds.csv")
STAGE = "post_quali"
FLOOR = 1e-4  # a car the sampler never saw win still gets this much


def our_win_odds() -> pd.DataFrame:
    """Held-out races: our post-qualifying win probability per car, drawn as the model card draws it."""
    from src.features.build_dataset import OUT_PATH, load_raw
    from src.models.blend import apply_walkforward
    from src.models.probabilities import _stage_grid, field_retirement_rates, load_calibration, race_probabilities
    from src.models.train import WALKFORWARD_PATH

    matrix = pd.read_parquet(OUT_PATH)
    wf = apply_walkforward(pd.read_parquet(WALKFORWARD_PATH).drop(columns=["dnf"], errors="ignore"), matrix)
    raw = load_raw()[["season", "round", "race_date", "driver", "dnf"]]
    race = wf[(wf["target"] == "finish_position") & (wf["stage"] == STAGE) & wf["after_tuning_window"]]
    race = race.merge(field_retirement_rates(raw), on=["season", "round"])
    race["grid_at_stage"] = _stage_grid(race, matrix)[0]
    cal = load_calibration()
    out = []
    for i, ((season, rnd), r) in enumerate(race.groupby(["season", "round"])):
        pr = race_probabilities(r["pred"], stage=STAGE, seed=i, p_dnf=float(r["p_dnf"].iloc[0]), grid=r["grid_at_stage"], cal=cal)
        out.append(pd.DataFrame({"season": season, "round": rnd, "driver": r["driver"].to_numpy(), "ours": pr["win"],
                                 "won": r["actual"].to_numpy() == 1}))
    return pd.concat(out, ignore_index=True)


def _arrays(races):
    """Races of different sizes as padded (race, car) arrays: log market, log ours, real-car mask, winner index."""
    n = max(len(m) for m, _, _ in races)
    M, O, mask = np.zeros((len(races), n)), np.zeros((len(races), n)), np.zeros((len(races), n), dtype=bool)
    for i, (m, o, _) in enumerate(races):
        M[i, :len(m)], O[i, :len(o)], mask[i, :len(m)] = m, o, True
    return M, O, mask, np.array([w for _, _, w in races])


def _nll(a, b, M, O, mask, w) -> np.ndarray:
    """Per race, -log of the winner's probability under p proportional to market^a * ours^b."""
    z = np.where(mask, a * M + b * O, -np.inf)
    return -(z[np.arange(len(w)), w] - logsumexp(z, axis=1))


def _fit(M, O, mask, w, with_ours=True) -> tuple[float, float]:
    if with_ours:
        a, b = minimize(lambda p: _nll(p[0], p[1], M, O, mask, w).sum(), [1.0, 0.0], method="Nelder-Mead").x
        return float(a), float(b)
    return float(minimize(lambda p: _nll(p[0], 0.0, M, O, mask, w).sum(), [1.0], method="Nelder-Mead").x[0]), 0.0


def _interval(d) -> list[float]:
    d = np.asarray(d)
    half = 1.96 * d.std(ddof=1) / np.sqrt(len(d))
    return [round(float(d.mean() - half), 4), round(float(d.mean() + half), 4)]


def adds_information(races) -> dict:
    """races: (log market, log ours, winner index) per race."""
    M, O, mask, w = _arrays(races)
    a, b = _fit(M, O, mask, w)
    a0, _ = _fit(M, O, mask, w, with_ours=False)
    lr = 2 * (_nll(a0, 0.0, M, O, mask, w).sum() - _nll(a, b, M, O, mask, w).sum())
    market, blend = [], []
    for k in range(len(w)):
        keep, one = np.arange(len(w)) != k, [k]
        rest = (M[keep], O[keep], mask[keep], w[keep])
        market.append(_nll(*_fit(*rest, with_ours=False), M[one], O[one], mask[one], w[one])[0])
        blend.append(_nll(*_fit(*rest), M[one], O[one], mask[one], w[one])[0])
    return {"market_weight": round(a, 3), "our_weight": round(b, 3),
            "p_value_ours_adds_nothing": round(float(chi2.sf(max(lr, 0.0), 1)), 4),
            "leave_one_race_out_winner_log_loss": {"market": round(float(np.mean(market)), 4),
                                                   "market_plus_ours": round(float(np.mean(blend)), 4),
                                                   "difference_95": _interval(np.array(blend) - np.array(market))}}


def compare(df: pd.DataFrame) -> dict:
    """df: one row per car with season, round, ours, market, won."""
    from src.models.probabilities import log_loss

    y = df["won"].to_numpy(dtype=float)
    out = {}
    for col in ("ours", "market"):
        p = df[col].to_numpy()
        out[col] = {"win_log_loss": round(log_loss(p, y), 4), "win_brier": round(float(((p - y) ** 2).mean()), 4)}
    winners = df[df["won"]].set_index(["season", "round"])
    ours_w, market_w = -np.log(np.clip(winners["ours"], FLOOR, 1)), -np.log(np.clip(winners["market"], FLOOR, 1))
    out["winner_log_loss"] = {"ours": round(float(ours_w.mean()), 4), "market": round(float(market_w.mean()), 4),
                              "ours_minus_market_95": _interval(ours_w - market_w)}
    races = [(np.log(np.clip(g["market"].to_numpy(), FLOOR, 1)), np.log(np.clip(g["ours"].to_numpy(), FLOOR, 1)),
              int(np.argmax(g["won"].to_numpy()))) for _, g in df.groupby(["season", "round"])]
    out["does_ours_add_to_the_market"] = adds_information(races)
    return out


def self_check():
    """Synthetic races with known answers: a forecast that is noise adds nothing; a sharper one adds a lot."""
    rng = np.random.default_rng(0)

    def races(n, market_noise, ours):
        out = []
        for _ in range(n):
            s = rng.normal(0, 1.5, 20)
            w = int(rng.choice(20, p=np.exp(s - logsumexp(s))))
            m, o = s + rng.normal(0, market_noise, 20), ours(s)
            out.append((m - logsumexp(m), o - logsumexp(o), w))
        return out

    noise = adds_information(races(150, 0.3, lambda s: rng.normal(0, 1.5, 20)))
    assert abs(noise["our_weight"]) < 0.2 and noise["p_value_ours_adds_nothing"] > 0.01, noise
    sharper = adds_information(races(150, 1.5, lambda s: s + rng.normal(0, 0.3, 20)))
    assert sharper["our_weight"] > sharper["market_weight"] and sharper["p_value_ours_adds_nothing"] < 0.001, sharper
    assert sharper["leave_one_race_out_winner_log_loss"]["difference_95"][1] < 0, sharper
    print("self-check ok", json.dumps({"noise": noise, "sharper": sharper}, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-check", action="store_true")
    if ap.parse_args().self_check:
        return self_check()
    if not ODDS_PATH.exists():
        raise SystemExit(f"{ODDS_PATH} missing: one row per car with season, round, driver, odds (see this file's docstring)")
    odds = pd.read_csv(ODDS_PATH)[["season", "round", "driver", "odds"]]
    ours = our_win_odds()
    df = ours.merge(odds, on=["season", "round", "driver"], how="left")
    priced = df.groupby(["season", "round"])["odds"].transform(lambda s: s.notna().all())
    df = df[priced].copy()
    inv = 1 / df["odds"]
    booksum = inv.groupby([df["season"], df["round"]]).sum()
    df["market"] = inv / inv.groupby([df["season"], df["round"]]).transform("sum")
    result = {"races_compared": int(booksum.size), "held_out_races": int(ours.groupby(["season", "round"]).ngroups),
              "average_margin_removed": round(float(booksum.mean() - 1), 4), **compare(df)}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
