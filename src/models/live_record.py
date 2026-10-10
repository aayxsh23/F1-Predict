"""The forecasts the app actually published, scored once the race is in.

train.py's walk-forward simulates what each forecast would have been; this
scores what was really shown: every snapshot in data/predictions/timeline/
(one per weekend stage, the last version of each) against the race result in
data/raw/races/. If the live pipeline ever diverges from what training
assumed (a weather forecast, the FIA grid, a session that didn't load), it
shows up here and nowhere else.

Writes data/predictions/live_record.json:
  races      per race, per published snapshot: how far off the finishing order
             was (all cars, and cars that finished), order agreement, whether
             the winner and how many podium finishers were called, the same
             error for "finish where you start" once a grid existed, and the
             qualifying-gap error for snapshots made before qualifying
  by_stage   the same averaged per snapshot label over every scored race
`python -m src.models.live_record`; the Monday ingest and the weekly retrain run it.
"""
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
TIMELINE_DIR = ROOT / "data" / "predictions" / "timeline"
RAW_DIR = ROOT / "data" / "raw" / "races"
OUT_PATH = ROOT / "data" / "predictions" / "live_record.json"
QUALI_MAX_GAP = 7.0  # same "representative lap" line as the model cards
STAGE_ORDER = ["pre_weekend", "post_practice", "post_sprint_quali", "post_sprint", "post_quali", "race_day"]


def _result(season: int, rnd: int) -> pd.DataFrame | None:
    files = list(RAW_DIR.glob(f"{season}_{rnd:02d}_*.parquet"))
    return pd.read_parquet(files[0]) if files else None


def _r(v) -> float | None:
    return None if v is None or pd.isna(v) else round(float(v), 3)


def score_snapshot(snap: dict, res: pd.DataFrame) -> dict:
    pred = pd.DataFrame(snap["drivers"])
    g = pred.merge(res[["driver", "finish_position", "dnf", "quali_gap_pct", "quali_position", "grid_position"]], on="driver")
    g = g.dropna(subset=["predicted_finish_position", "finish_position"])
    fin = g[~g["dnf"].astype(bool)]
    out = {"label": snap["label"], "stage": snap.get("stage"), "generated_at": snap.get("generated_at"), "cars": int(len(g)),
           "finish_mae": _r((g["predicted_finish_position"] - g["finish_position"]).abs().mean()),
           "finish_mae_finishers": _r((fin["predicted_finish_position"] - fin["finish_position"]).abs().mean()),
           "order_agreement": _r(spearmanr(fin["predicted_finish_position"], fin["finish_position"])[0]) if len(fin) > 2 else None}
    if len(g):
        out["winner_called"] = bool(g.loc[g["predicted_finish_position"].idxmin(), "finish_position"] == 1)
        out["podium_called"] = int(len(set(g.nsmallest(3, "predicted_finish_position")["driver"]) & set(g[g["finish_position"] <= 3]["driver"])))
    grid = {"post_quali": "quali_position", "race_day": "grid_position"}.get(snap.get("stage"))
    if grid:
        out["grid_baseline_mae"] = _r((g[grid] - g["finish_position"]).abs().mean())
    if snap.get("stage") in STAGE_ORDER[:4] and "predicted_qualifying_gap_pct" in g:
        q = g[g["quali_gap_pct"] <= QUALI_MAX_GAP].dropna(subset=["predicted_qualifying_gap_pct"])
        if len(q):
            out["quali_mae"] = _r((q["predicted_qualifying_gap_pct"] - q["quali_gap_pct"]).abs().mean())
            out["pole_called"] = bool(q.loc[q["predicted_qualifying_gap_pct"].idxmin(), "quali_position"] == q["quali_position"].min())
    return out


def build() -> dict:
    races = []
    for path in sorted(TIMELINE_DIR.glob("*.json")):
        m = re.fullmatch(r"(\d{4})_(\d+)", path.stem)
        res = _result(int(m[1]), int(m[2])) if m else None
        if res is None:
            continue  # not raced (or not ingested) yet
        snaps = sorted(json.loads(path.read_text()), key=lambda s: STAGE_ORDER.index(s["stage"]) if s.get("stage") in STAGE_ORDER else 0)
        races.append({"season": int(m[1]), "round": int(m[2]), "location": str(res["location"].iloc[0]),
                      "snapshots": [score_snapshot(s, res) for s in snaps]})
    rows = pd.DataFrame([{**s, "race": (r["season"], r["round"])} for r in races for s in r["snapshots"]])
    by_stage = []
    if len(rows):
        for label, g in sorted(rows.groupby("label"), key=lambda x: STAGE_ORDER.index(x[1]["stage"].iloc[0]) if x[1]["stage"].iloc[0] in STAGE_ORDER else 0):
            entry = {"label": label, "races": int(g["race"].nunique()), "finish_mae": _r(g["finish_mae"].mean()),
                     "finish_mae_finishers": _r(g["finish_mae_finishers"].mean()), "winner_called_share": _r(g["winner_called"].mean())}
            for c in ("grid_baseline_mae", "quali_mae"):
                if c in g and g[c].notna().any():
                    entry[c] = _r(g[c].mean())
            by_stage.append(entry)
    return {"method": "every forecast snapshot the app published (data/predictions/timeline), scored against the race result",
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "races": sorted(races, key=lambda r: (r["season"], r["round"]), reverse=True), "by_stage": by_stage}


def main() -> None:
    record = build()
    old = json.loads(OUT_PATH.read_text()) if OUT_PATH.exists() else {}
    if {k: v for k, v in old.items() if k != "generated_at"} == {k: v for k, v in record.items() if k != "generated_at"}:
        print("live record unchanged")
        return
    OUT_PATH.write_text(json.dumps(record, indent=2))
    print(f"scored {len(record['races'])} races -> {OUT_PATH}")


if __name__ == "__main__":
    main()
