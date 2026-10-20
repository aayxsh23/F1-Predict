"""Qualifying: learn from a session-chained gap instead of the official gap
to pole? Phase 10 item 1.6.

The official target is each driver's best lap as % over pole. A driver out in
Q1 set that lap half an hour before pole, on a slower track, so part of the
"gap" is track evolution, not pace. The chained gap compares only within a
session:

  Q3 runners   gap to pole (unchanged)
  out in Q2    gap to the fastest Q2 car + that car's chained gap (its Q3 gap)
  out in Q1    gap to the fastest Q1 car + that car's chained gap

The candidate (the shipped XGBoost + LightGBM ensemble, same settings) is
trained on the chained gap and SCORED ON THE OFFICIAL GAP, which is what the
app shows; the incumbent is the same ensemble trained on the official gap.
Races or drivers without session times keep the official value.

    python -m experiments.quali_chained_target [--fresh]
"""
import json
import sys

import numpy as np
import pandas as pd

from src.models import compare
from src.models.train import MODEL_SPECS_PATH

TARGET = "target_qualifying_gap_pct"


def chained_gap(race: pd.DataFrame) -> pd.Series:
    """% gap per driver of one race (columns q1_time, q2_time, q3_time in seconds), NaN where it can't be chained."""
    g = race[["q1_time", "q2_time", "q3_time"]].astype(float)
    gap3 = g["q3_time"] / g["q3_time"].min() - 1

    def via(session: str, later: pd.Series) -> pd.Series:
        col = g[session]
        if col.notna().sum() == 0:
            return pd.Series(np.nan, index=g.index)
        fastest = col.idxmin()
        return col / col.min() - 1 + later.get(fastest, np.nan)

    gap2 = via("q2_time", gap3)
    gap1 = via("q1_time", gap3.where(gap3.notna(), gap2))
    return gap3.where(gap3.notna(), gap2.where(gap2.notna(), gap1)) * 100


def chained_targets() -> pd.DataFrame:
    from src.features.build_dataset import load_raw

    raw = load_raw()
    rows = []
    for (season, rnd), race in raw.groupby(["season", "round"]):
        c = chained_gap(race.set_index("driver"))
        rows.append(pd.DataFrame({"season": season, "round": rnd, "driver": c.index, "chained": c.to_numpy()}))
    return pd.concat(rows, ignore_index=True)


def main(fresh: bool = False) -> dict:
    from src.features.build_dataset import build

    base = build()
    spec = json.loads(MODEL_SPECS_PATH.read_text())["qualifying"]
    spec = {k: v for k, v in spec.items() if k in ("family", "weights", "members", "params")}
    official = base[["season", "round", "driver", TARGET]]
    ch = chained_targets()
    df = base.merge(ch, on=["season", "round", "driver"], how="left")
    used = df["chained"].notna() & df[TARGET].notna()
    print(f"chained gap available for {used.mean():.1%} of rows with a target; "
          f"mean |chained - official| = {(df.loc[used, 'chained'] - df.loc[used, TARGET]).abs().mean():.3f} points", flush=True)
    df[TARGET] = df["chained"].where(df["chained"].notna(), df[TARGET])
    incumbent = compare.run("qualifying", "ensemble_official_target", spec, df=base, use_cache=not fresh)
    cand = compare.run("qualifying", "ensemble_chained_target", spec, df=df.drop(columns=["chained"]), use_cache=not fresh)
    # judged on the official gap, the number the app shows
    cand = cand.drop(columns=["actual"]).merge(official.rename(columns={TARGET: "actual"}), on=["season", "round", "driver"])
    result = compare.score("qualifying", cand, incumbent)
    compare.record("qualifying", "ensemble_chained_target", {**spec, "target": "session-chained gap"}, result,
                   note="trained on the session-chained gap, scored on the official gap; incumbent = same ensemble on the official gap")
    print(compare.summary("qualifying", result, ["pre_weekend", "post_practice", "post_sprint_quali"]))
    return result


if __name__ == "__main__":
    main(fresh="--fresh" in sys.argv)
