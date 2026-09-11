import pandas as pd

from src.features.rolling import recent_form


def build(df: pd.DataFrame) -> pd.DataFrame:
    """Driver-vs-teammate features. df must already have `driver_recent_form`
    merged in. teammate_quali_gap uses this race's actual quali times (known
    pre-race, not leakage); teammate_race_pace_gap compares each driver's
    historical race-pace form (prior races only) — actual race pace from the
    race being predicted would leak the outcome."""
    df = df.copy()
    df["_hist_race_pace"] = recent_form(df, "driver", "race_date", "race_pace_pct")

    key = ["season", "round", "team"]
    teammates = df[key + ["driver", "quali_gap_to_pole", "_hist_race_pace"]].rename(
        columns={"driver": "_tm_driver", "quali_gap_to_pole": "_tm_quali", "_hist_race_pace": "_tm_pace"}
    )
    merged = df.merge(teammates, on=key, how="left")
    merged = merged[merged["driver"] != merged["_tm_driver"]]
    tm_avg = merged.groupby(["season", "round", "driver"], as_index=False).agg(
        _tm_quali=("_tm_quali", "mean"), _tm_pace=("_tm_pace", "mean"),
    )

    out = df.merge(tm_avg, on=["season", "round", "driver"], how="left")
    out["teammate_quali_gap"] = out["quali_gap_to_pole"] - out["_tm_quali"]
    out["teammate_race_pace_gap"] = out["_hist_race_pace"] - out["_tm_pace"]
    out["grid_vs_expected_position"] = out["grid_position"] - out["driver_recent_form"]

    return out[["season", "round", "driver", "teammate_quali_gap", "teammate_race_pace_gap", "grid_vs_expected_position"]]
