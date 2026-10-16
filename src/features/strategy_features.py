import pandas as pd

from src.features.rolling import recent_form


def build(df: pd.DataFrame) -> pd.DataFrame:
    """expected_stops: the usual number of pit stops at this circuit, over
    prior races only (this race's actual stops would leak the outcome).
    Starting tyres and "past results on this tyre here" were inputs once;
    a live forecast never knows the starting tyre, so both were dropped."""
    df = df.copy()
    df["expected_stops"] = recent_form(df, "location", "race_date", "num_pit_stops", decay=0.9)
    return df[["season", "round", "driver", "expected_stops"]]
