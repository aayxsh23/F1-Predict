"""Title odds: one strength for the rest of the season, or each driver's
strength at each remaining venue, and should team form drift? Backtested.
Phase 10 item 2.5.

After every round of 2023-2025 (2022 lacks sprint results in the data), the
rest of the season is simulated from that round's real standings, and the
odds are scored against the real champions (drivers and constructors).
Nothing after the round is used: the finishing model is refitted on races up
to that round (shipped family and settings), each remaining venue's inputs
are built from data up to that round (line-up as at that round, car upgrades
published so far), and its pre-weekend prediction is the strength.

  single          today's method: the next venue's strength for every round
  venue           each remaining round's own venue strength
  + drift s       a random walk in each team's form, s places per round

Drift is chosen on 2023-2024 by the champions' log loss; 2025 only checks.

    python -m experiments.title_odds_backtest      # ~15 minutes
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.agent.scenarios import COUNTBACK, SPRINT_POINTS, title_odds
from src.features.build_dataset import build, load_raw, load_upgrades
from src.models.catalog import TEAM_LINEAGE
from src.models.features import PREPARE_FN, mask_for_stage
from src.models.probabilities import load_calibration
from src.models.train import MODEL_SPECS_PATH, fit

SEASONS = (2023, 2024, 2025)
CHOOSE = (2023, 2024)
DRIFTS = (0.25, 0.5, 1.0, 1.5)
N = 3000
SPRINT = {i + 1: p for i, p in enumerate(SPRINT_POINTS)}
OUT = Path(__file__).resolve().parents[1] / "data" / "processed" / "title_odds_backtest.json"


def table(season_raw: pd.DataFrame, upto: int) -> pd.DataFrame:
    """Points and Grand Prix places per driver after round `upto` (race + sprint points)."""
    r = season_raw[season_raw["round"] <= upto]
    pts = r.groupby("driver")["points"].sum() + r["sprint_finish_position"].map(SPRINT).fillna(0).groupby(r["driver"]).sum()
    places = pd.DataFrame({k: (r["finish_position"] == k + 1).groupby(r["driver"]).sum() for k in range(COUNTBACK)})
    team = r.sort_values("round").groupby("driver")["team"].last().replace(TEAM_LINEAGE)
    return pd.DataFrame({"points": pts}).join(places).join(team).fillna(0)


def champion(t: pd.DataFrame, by: str | None = None) -> str:
    g = t.groupby(by).sum(numeric_only=True) if by else t
    key = g["points"] + sum(g[k] * 1e-3 ** (k + 1) for k in range(COUNTBACK))
    return key.idxmax()


def venue_strengths(raw: pd.DataFrame, season: int, upto: int, spec: dict) -> dict[int, pd.Series]:
    """Pre-weekend predicted finish per driver at every round after `upto`, from data up to `upto` only."""
    before = raw[(raw["season"] < season) | ((raw["season"] == season) & (raw["round"] <= upto))]
    season_raw = raw[raw["season"] == season]
    lineup = before[(before["season"] == season) & (before["round"] == upto)][["driver", "driver_number", "team"]]
    events = season_raw[season_raw["round"] > upto].drop_duplicates("round")
    future = pd.concat([lineup.assign(season=season, round=int(e["round"]), location=e["location"], race_date=e["race_date"],
                                      race_start_utc=e["race_start_utc"], sprint_weekend=e["sprint_weekend"], dnf=False)
                        for _, e in events.iterrows()], ignore_index=True)
    up = load_upgrades()
    if {"season", "round"} <= set(up.columns):
        up = up[(up["season"] < season) | ((up["season"] == season) & (up["round"] <= upto))]
    full = build(raw=pd.concat([before, future], ignore_index=True), upgrades=up)
    seen = full[(full["season"] < season) | ((full["season"] == season) & (full["round"] <= upto))]
    model = fit("finish_position", seen.dropna(subset=["target_finish_position"]), spec)
    rows = mask_for_stage(full[(full["season"] == season) & (full["round"] > upto)], "pre_weekend")
    rows = rows.assign(pred=model.predict(PREPARE_FN["finish_position"](rows)))
    return {int(r): g.set_index("driver")["pred"] for r, g in rows.groupby("round")}


def score(p: dict | np.ndarray, names: list[str], actual: str) -> tuple[float, float]:
    probs = np.array([p[n] for n in names]) if isinstance(p, dict) else np.asarray(p)
    y = np.array([n == actual for n in names], dtype=float)
    return float(-np.log(max(probs[names.index(actual)], 1e-3))), float(((probs - y) ** 2).sum())


def main() -> None:
    raw = load_raw()
    spec = json.loads(MODEL_SPECS_PATH.read_text())["finish_position"]
    cal = load_calibration()
    tau, p_dnf = cal["race"]["pre_weekend"]["tau_field"], cal["dnf_prior"]
    variants = ["single", "venue"] + [f"venue + drift {d}" for d in DRIFTS] + [f"single + drift {d}" for d in DRIFTS]
    rows = []
    for season in SEASONS:
        season_raw = raw[raw["season"] == season]
        last = int(season_raw["round"].max())
        final = table(season_raw, last)
        champ, team_champ = champion(final), champion(final, "team")
        sprint_rounds = set(season_raw.loc[season_raw["sprint_weekend"].astype(bool), "round"])
        for upto in range(1, last):
            now = table(season_raw, upto)
            strength = venue_strengths(raw, season, upto, spec)
            remaining = [{"round": r, "is_sprint": r in sprint_rounds} for r in sorted(strength)]
            names = list(now.index)
            worst = max(s.max() for s in strength.values()) + 2
            per_round = np.array([[strength[r["round"]].get(d, worst) for d in names] for r in remaining])
            places = now[list(range(COUNTBACK))].to_numpy()
            for v in variants:
                drift = float(v.split("drift ")[1]) if "drift" in v else 0.0
                s = per_round if v.startswith("venue") else per_round[0]
                o = title_odds(now["points"].to_numpy(), s, remaining, tau=tau, p_dnf=p_dnf, groups=list(now["team"]),
                               n=N, seed=upto, drift=drift, places=places)
                dl, db = score(o["champion"], names, champ)
                tl, tb = score(o["team_champion"], sorted(set(now["team"])), team_champ) if team_champ in set(now["team"]) else (np.nan, np.nan)
                rows.append({"season": season, "after_round": upto, "variant": v, "driver_log_loss": dl, "driver_brier": db,
                             "team_log_loss": tl, "team_brier": tb, "p_champion": float(np.exp(-dl))})
            print(season, upto, "done", flush=True)
    df = pd.DataFrame(rows)
    df.to_json(OUT, orient="records")
    agg = lambda d: d.groupby("variant")[["driver_log_loss", "team_log_loss", "driver_brier", "team_brier"]].mean().round(4)  # noqa: E731
    choose = agg(df[df["season"].isin(CHOOSE)])
    check = agg(df[~df["season"].isin(CHOOSE)])
    pd.set_option("display.width", 200)
    print("\nchoosing seasons", CHOOSE, "\n", choose.sort_values("driver_log_loss"))
    print("\ncheck season(s)", sorted(set(SEASONS) - set(CHOOSE)), "\n", check.loc[choose.index].sort_values("driver_log_loss"))


if __name__ == "__main__":
    main()
