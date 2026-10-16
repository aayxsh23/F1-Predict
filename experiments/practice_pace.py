"""Practice pace: one best lap across all practice sessions, or a per-session
measure?

Today `practice_pace` is each driver's single best practice lap as % over the
field's best, across FP1-FP3 together. The track gets faster through a
weekend and teams run different programmes per session, so a best lap from
FP3 and one from FP1 aren't comparable. Candidates, from the same cached laps
(FastF1, offline; laps FastF1 marks inaccurate left out):

  session_weighted  each session's best lap as % over THAT session's field
                    best, averaged with weights 1/2/3 for FP1/FP2/FP3
  latest_session    the % gap in the latest session the driver set a time in

Deleted (track-limits) laps can't be dropped here: that needs race-control
messages, which the cache doesn't hold (a network fetch for every session).

    python -m experiments.practice_pace compute 2018 2021 <cache dir>   (once per cache)
    python -m experiments.practice_pace compare
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
VARIANTS_PATH = ROOT / "data" / "processed" / "practice_variants.parquet"
WEIGHT = {"FP1": 1.0, "FP2": 2.0, "FP3": 3.0}


def compute(first: int, last: int, cache_dir: str) -> None:
    import fastf1

    fastf1.set_log_level("CRITICAL")
    fastf1.Cache.enable_cache(cache_dir)
    fastf1.Cache.offline_mode(True)
    rows = []
    for season in range(first, last + 1):
        sched = fastf1.get_event_schedule(season)
        for _, e in sched[sched["RoundNumber"] > 0].iterrows():
            per_session = []
            for code in ("FP1", "FP2", "FP3"):
                try:
                    s = fastf1.get_session(season, int(e["RoundNumber"]), code)
                    s.load(laps=True, telemetry=False, weather=False, messages=False)
                    laps = s.laps
                except Exception:
                    continue
                if laps is None or laps.empty:
                    continue
                ok = laps.dropna(subset=["LapTime"])
                if "IsAccurate" in ok:
                    ok = ok[ok["IsAccurate"].fillna(False).astype(bool)] if ok["IsAccurate"].any() else ok
                best = ok.groupby("Driver")["LapTime"].min().dt.total_seconds()
                if best.empty:
                    continue
                per_session.append(pd.DataFrame({"driver": best.index, "session": code, "gap": (best / best.min() - 1).to_numpy() * 100}))
            if not per_session:
                continue
            g = pd.concat(per_session)
            g["w"] = g["session"].map(WEIGHT)
            weighted = g.groupby("driver").apply(lambda x: np.average(x["gap"], weights=x["w"]), include_groups=False)
            latest = g.assign(order=g["session"].map(WEIGHT)).sort_values("order").groupby("driver")["gap"].last()
            for d in weighted.index:
                rows.append({"season": season, "round": int(e["RoundNumber"]), "driver": d,
                             "practice_session_weighted": float(weighted[d]), "practice_latest_session": float(latest[d])})
            print(season, int(e["RoundNumber"]), len(weighted), flush=True)
    new = pd.DataFrame(rows)
    old = pd.read_parquet(VARIANTS_PATH) if VARIANTS_PATH.exists() else pd.DataFrame(columns=new.columns)
    old = old[~old["season"].between(first, last)]
    pd.concat([old, new], ignore_index=True).to_parquet(VARIANTS_PATH, index=False)


def compare_variants(fresh: bool = False) -> dict:
    from src.features.build_dataset import build
    from src.models import compare

    base = build()
    variants = pd.read_parquet(VARIANTS_PATH)
    out = {}
    for target, stages in (("qualifying", ["post_practice"]), ("finish_position", ["post_practice"])):
        params = compare.incumbent_params(target)
        spec = {"family": "xgb", "params": params["params"] if "family" in params else params}
        before = compare.run(target, "xgb_practice_best_lap", spec, df=base, use_cache=not fresh)
        for col in ("practice_session_weighted", "practice_latest_session"):
            df = base.merge(variants[["season", "round", "driver", col]], on=["season", "round", "driver"], how="left")
            # a race the cache doesn't cover keeps today's value rather than a hole
            df["practice_pace"] = df[col].where(df[col].notna() | df["practice_pace"].isna(), df["practice_pace"])
            df["practice_pace"] = df["practice_pace"].where(df["practice_pace"].isna() | df[col].isna(), df[col])
            after = compare.run(target, f"xgb_{col}", spec, df=df.drop(columns=[col]), use_cache=not fresh)
            result = compare.score(target, after, before)
            compare.record(target, f"xgb_{col}", {**spec, "features": col}, result, note="incumbent here = best practice lap across sessions")
            print(f"\n{target}: {col} vs best lap across sessions")
            print(compare.summary(target, result, stages))
            out[(target, col)] = result
    return out


if __name__ == "__main__":
    if sys.argv[1] == "compute":
        compute(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
    else:
        compare_variants(fresh="--fresh" in sys.argv)
