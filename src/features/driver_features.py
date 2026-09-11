import pandas as pd

from src.features.rolling import recent_form, track_form


def build(df: pd.DataFrame) -> pd.DataFrame:
    """Per-driver-per-race features. grid_position/quali_gap_to_pole/practice_pace
    are pre-race-session actuals (known before the race, not leakage); the
    *_form columns are recency-weighted over strictly prior races."""
    df = df.copy()
    df["driver_recent_form"] = recent_form(df, "driver", "race_date", "finish_position")
    df["_gain"] = df["grid_position"] - df["finish_position"]
    df["driver_positions_gained_form"] = recent_form(df, "driver", "race_date", "_gain")
    df["_dnf_f"] = df["dnf"].astype(float)
    df["driver_dnf_rate"] = recent_form(df, "driver", "race_date", "_dnf_f", decay=0.9)
    df["driver_track_form"] = track_form(df, "driver", "location", "season", "race_date", "finish_position")

    return df[[
        "season", "round", "driver", "grid_position", "quali_gap_to_pole", "practice_pace",
        "driver_recent_form", "driver_track_form", "driver_positions_gained_form", "driver_dnf_rate",
    ]]
