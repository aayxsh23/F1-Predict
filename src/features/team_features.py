import pandas as pd

from src.features.rolling import recent_form


def build(df: pd.DataFrame) -> pd.DataFrame:
    """Per-constructor-per-race form, all recency-weighted over strictly
    prior races. df must already have `track_type` merged in (circuit layer)."""
    df = df.copy()
    df["team_recent_form"] = recent_form(df, "team", "race_date", "finish_position")
    df["team_quali_pace"] = recent_form(df, "team", "race_date", "quali_gap_to_pole")
    df["team_race_pace"] = recent_form(df, "team", "race_date", "race_pace_pct")
    df["_dnf_f"] = df["dnf"].astype(float)
    df["team_reliability"] = 1 - recent_form(df, "team", "race_date", "_dnf_f", decay=0.9)

    df["_team_type_key"] = df["team"] + "|" + df["track_type"]
    df["team_track_type_form"] = recent_form(df, "_team_type_key", "race_date", "finish_position")

    return df[[
        "season", "round", "driver", "team_recent_form", "team_quali_pace",
        "team_race_pace", "team_reliability", "team_track_type_form",
    ]]
