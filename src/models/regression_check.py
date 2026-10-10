"""Champion/challenger: a retrained model must not be clearly worse than the
one it replaces.

The weekly retrain's own gate (train.py --gate) only asks "does it still beat
the naive guess?"; a model can pass that and still be worse than what's live.
This compares the new walk-forward predictions (blends applied) with the
shipped ones still sitting in data/predictions/backtest/ (written by the model
being replaced), race by race, on the most recent RECENT_RACES races both
predicted, at the stage the History view shows each target. Both are honest
walk-forward predictions of the same races, so this is a fair contest.

Fails (exit 1) when the new error is higher with the whole 95% interval above
zero: clearly worse, not just a noisy week. Run after `src.models.blend` and
before `src.models.backtest_export` overwrites the old predictions.
"""
import json
import sys

import pandas as pd

from src.features.build_dataset import OUT_PATH
from src.models.backtest_export import OUT_DIR, SHOWN_STAGE
from src.models.blend import apply_walkforward
from src.models.train import WALKFORWARD_PATH, diff_ci95

RECENT_RACES = 24  # about a season: the races that matter most for what's live next


def shipped_predictions() -> pd.DataFrame:
    rows = []
    for path in OUT_DIR.glob("*_*.json"):
        b = json.loads(path.read_text())
        for d in b["drivers"]:
            for target in SHOWN_STAGE:
                e = d.get(target) or {}
                if e.get("predicted") is not None:
                    rows.append({"season": b["season"], "round": b["round"], "driver": d["driver"], "target": target, "old": e["predicted"]})
    return pd.DataFrame(rows)


def check() -> list[str]:
    old = shipped_predictions()
    if old.empty:
        return []
    wf = apply_walkforward(pd.read_parquet(WALKFORWARD_PATH), pd.read_parquet(OUT_PATH))
    wf = pd.concat([wf[(wf["target"] == t) & (wf["stage"] == s)] for t, s in SHOWN_STAGE.items()])
    j = wf.merge(old, on=["season", "round", "driver", "target"]).dropna(subset=["actual"])
    failures = []
    for target, g in j.groupby("target"):
        races = g[["season", "round"]].drop_duplicates().sort_values(["season", "round"]).tail(RECENT_RACES)
        g = g.merge(races, on=["season", "round"])
        new_mae, old_mae = (g["pred"] - g["actual"]).abs().mean(), (g["old"] - g["actual"]).abs().mean()
        lo, hi = diff_ci95(g, "pred", "old")
        verdict = "CLEARLY WORSE" if lo > 0 else "ok"
        print(f"{target:16s} last {len(races)} races: new {new_mae:.3f} vs shipped {old_mae:.3f}  (new - shipped CI [{lo}, {hi}])  {verdict}")
        if lo > 0:
            failures.append(f"{target}: new MAE {new_mae:.3f} vs shipped {old_mae:.3f} on the last {len(races)} races (CI [{lo}, {hi}])")
    return failures


if __name__ == "__main__":
    if failures := check():
        sys.exit("refusing the retrained models, clearly worse than the shipped ones:\n  " + "\n  ".join(failures))
