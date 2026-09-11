import pandas as pd

from src.features.rolling import recent_form


def build(df: pd.DataFrame) -> pd.DataFrame:
    """expected_stops and historical_compound_performance are both historical
    (prior races only) — this race's actual pit stops / finish position would
    leak the outcome. starting_tire_compound itself is a pre-race actual and
    is passed through directly in build_dataset, not here."""
    df = df.copy()
    df["expected_stops"] = recent_form(df, "location", "race_date", "num_pit_stops", decay=0.9)

    df["_loc_compound_key"] = df["location"] + "|" + df["starting_compound"].astype(str)
    df["historical_compound_performance"] = recent_form(
        df, "_loc_compound_key", "race_date", "finish_position", decay=0.9
    )

    return df[["season", "round", "driver", "expected_stops", "historical_compound_performance"]]
