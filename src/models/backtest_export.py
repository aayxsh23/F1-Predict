"""The History view's data: for every past race, what the model predicted
before it and what actually happened. Predictions come from train.py's walk-
forward run, so each race was predicted by a model trained only on earlier
races (the first MIN_HISTORY races have no such model and are left out).
Race targets are shown as of the evening after qualifying (grid known,
tyres not), qualifying as of after practice: what a live forecast has at
those points.

Also writes summary.json: accuracy per season against the naive baselines,
for the "track record" view. Run after train.py.
"""
import json
from pathlib import Path

import pandas as pd

from src.features.build_dataset import load_raw
from src.models.train import TARGETS, WALKFORWARD_PATH

OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "predictions" / "backtest"
SHOWN_STAGE = {"qualifying": "post_practice", "finish_position": "post_quali", "quali_delta": "post_quali", "race_time": "post_quali"}


def _clean(v):
    return None if pd.isna(v) else round(float(v), 4)


def export_all() -> list[dict]:
    from src.features.build_dataset import OUT_PATH
    from src.models.blend import apply_walkforward

    wf = apply_walkforward(pd.read_parquet(WALKFORWARD_PATH), pd.read_parquet(OUT_PATH))  # what the app would have shown
    wf = pd.concat([wf[(wf["target"] == t) & (wf["stage"] == s)] for t, s in SHOWN_STAGE.items()])
    raw = load_raw()
    info = raw.drop_duplicates(["season", "round", "driver"]).set_index(["season", "round", "driver"])
    races = raw[["season", "round", "location", "race_date", "quali_pole_s", "race_winner_time_s"]].drop_duplicates(["season", "round"]).sort_values("race_date")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for stale in OUT_DIR.glob("*_*.json"):  # races that no longer have a walk-forward prediction
        stale.unlink()
    index = []
    for _, r in races.iterrows():
        season, rnd = int(r["season"]), int(r["round"])
        race = wf[(wf["season"] == season) & (wf["round"] == rnd)]
        if race[race["target"] == "finish_position"].empty:  # e.g. 2018-2020: only qualifying learns from them
            continue
        drivers = []
        for driver, g in race.groupby("driver"):
            d = info.loc[(season, rnd, driver)]
            entry = {"driver": driver, "team": d["team"],
                     "driver_number": int(d["driver_number"]) if str(d["driver_number"]).isdigit() else None}
            for target in TARGETS:
                row = g[g["target"] == target]
                entry[target] = {"actual": _clean(row["actual"].iloc[0]) if len(row) else None,
                                 "predicted": _clean(row["pred"].iloc[0]) if len(row) else None}
            entry["finish_position"]["actual"] = _clean(d["finish_position"])  # retirements too, for the result column
            drivers.append(entry)
        payload = {"season": season, "round": rnd, "location": r["location"],
                   "method": "walk-forward: predicted by a model trained only on earlier races",
                   # the real pole lap and race length, to turn % gaps into seconds
                   "pole_time_s": _clean(r["quali_pole_s"]), "race_duration_s": _clean(r["race_winner_time_s"]),
                   "drivers": drivers}
        (OUT_DIR / f"{season}_{rnd}.json").write_text(json.dumps(payload, indent=2))
        index.append({"season": season, "round": rnd, "location": r["location"]})

    (OUT_DIR / "index.json").write_text(json.dumps(index, indent=2))
    (OUT_DIR / "summary.json").write_text(json.dumps(summary(wf), indent=2))
    results = raw.sort_values(["race_date", "finish_position"])[
        ["season", "round", "location", "driver", "team", "grid_position", "quali_position", "finish_position", "status", "points"]]
    (OUT_DIR / "results.json").write_text(json.dumps(
        [{k: (_clean(v) if isinstance(v, float) else v) for k, v in r.items()} for r in results.to_dict("records")]))
    return index


def summary(wf: pd.DataFrame) -> dict:
    """Per target and season: model error vs naive baseline, and how often
    the predicted winner / podium were right."""
    out = {}
    for target, g in wf.groupby("target"):
        seasons = {}
        for season, s in g.groupby("season"):
            row = {"races": int(s[["season", "round"]].drop_duplicates().shape[0]),
                   "mae": round(float((s["pred"] - s["actual"]).abs().mean()), 3),
                   "baseline_mae": round(float((s["baseline"] - s["actual"]).abs().mean()), 3)}
            if target in ("finish_position", "qualifying"):
                hits = []
                for _, r in s.groupby("round"):
                    top_pred = r.nsmallest(1, "pred")["driver"].iloc[0]
                    top_true = r.nsmallest(1, "actual")["driver"].iloc[0]
                    hits.append(top_pred == top_true)
                row["called_the_winner" if target == "finish_position" else "called_pole"] = round(float(sum(hits) / len(hits)), 3)
            seasons[int(season)] = row
        out[target] = {"unit": TARGETS[target].unit, "baseline": TARGETS[target].baseline_name,
                       "stage_shown": SHOWN_STAGE[target], "by_season": seasons}
    return out


if __name__ == "__main__":
    written = export_all()
    print(f"exported walk-forward backtest for {len(written)} races -> {OUT_DIR}")
