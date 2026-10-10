import pandas as pd

from src.features.rolling import recent_form, track_form

# outside 107% of pole a qualifying "gap" is a crash, a failure or no lap at all, not pace
QUALI_MAX_GAP = 7.0


def build(df: pd.DataFrame) -> pd.DataFrame:
    """Per-driver-per-race features. grid_position/quali_gap_pct/practice pace
    are pre-race-session actuals (known before the race, not leakage); the
    *_form columns are recency-weighted over strictly prior races."""
    df = df.copy()
    df["driver_recent_form"] = recent_form(df, "driver", "race_date", "finish_position")
    df["_gain"] = df["grid_position"] - df["finish_position"]
    df["driver_positions_gained_form"] = recent_form(df, "driver", "race_date", "_gain")
    df["_dnf_f"] = df["dnf"].astype(float)
    df["driver_dnf_rate"] = recent_form(df, "driver", "race_date", "_dnf_f", decay=0.9)
    df["driver_track_form"] = track_form(df, "driver", "location", "season", "race_date", "finish_position")

    # the driver's own recent qualifying pace (representative laps only); the
    # team's is team_quali_pace, and pole is usually decided between teammates
    df["_q"] = df["quali_gap_pct"].where(df["quali_gap_pct"] <= QUALI_MAX_GAP)
    df["driver_quali_form"] = recent_form(df, "driver", "race_date", "_q")

    # how often a driver ends qualifying without a representative lap; the
    # qualifying odds use it, the models don't
    nolap = df["quali_gap_pct"].isna() | (df["quali_gap_pct"] > QUALI_MAX_GAP)
    df["_nolap"] = nolap.astype(float).where(df.reindex(columns=["quali_position"])["quali_position"].notna())
    df["driver_quali_nolap_rate"] = recent_form(df, "driver", "race_date", "_nolap", decay=0.9)

    return df.reindex(columns=[
        "season", "round", "driver", "grid_position", "quali_gap_pct", "practice_pace", "practice_long_run_pace",
        "driver_recent_form", "driver_track_form", "driver_positions_gained_form", "driver_dnf_rate",
        "driver_quali_form", "driver_quali_nolap_rate",
    ])
