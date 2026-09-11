"""Static circuit reference table, keyed by circuit + configuration era so a
pre/post-reconfig circuit (e.g. Singapore 2023) doesn't inherit one row's
values across a layout change. Hand-curated proxy values — see
circuit_reference.csv. `track_type` is derived here (not a v1 plan column)
purely to drive team_track_type_form downstream.

New calendar additions (e.g. Madrid, 2026) need a row added here before any
race there can be predicted — `resolve()` raises if a location has no match
rather than silently guessing. Madrid's row is a rougher estimate than the
other 25: there's no real race history to inform it (inaugural circuit),
just publicly announced layout facts plus reasonable proxies for the rest.
"""
from pathlib import Path

import numpy as np
import pandas as pd

_CSV = Path(__file__).with_name("circuit_reference.csv")

# FastF1's schedule metadata occasionally renames a location between seasons
# for the same physical circuit (e.g. Monaco -> "Monte Carlo", Miami ->
# "Miami Gardens" starting 2026) — map those back to the canonical CSV key
# rather than duplicating a row for the same track under a new label.
LOCATION_ALIASES = {
    "Monte Carlo": "Monaco",
    "Miami Gardens": "Miami",
    "Yas Marina": "Yas Island",
}


def load() -> pd.DataFrame:
    return pd.read_csv(_CSV)


def resolve(df: pd.DataFrame, circuits: pd.DataFrame | None = None) -> pd.DataFrame:
    """Merge circuit reference columns onto df (needs `location`, `season`),
    picking the era row whose [valid_from_year, valid_to_year] window
    contains the race's season."""
    circuits = circuits if circuits is not None else load()
    lookup_location = df["location"].replace(LOCATION_ALIASES)
    matched = []
    for _, era in circuits.iterrows():
        lo = era["valid_from_year"]
        hi = era["valid_to_year"] if not pd.isna(era["valid_to_year"]) else 9999
        mask = (lookup_location == era["location"]) & (df["season"] >= lo) & (df["season"] <= hi)
        if mask.any():
            rows = df.loc[mask].copy()
            for col in circuits.columns:
                if col not in ("location", "valid_from_year", "valid_to_year"):
                    rows[col] = era[col]
            matched.append(rows)

    result = pd.concat(matched).sort_index() if matched else df.iloc[0:0]
    missing = set(df.index) - set(result.index)
    if missing:
        bad_locations = df.loc[sorted(missing), "location"].unique()
        raise ValueError(f"No circuit reference row for locations: {list(bad_locations)}")

    result["track_type"] = np.where(
        result["is_street_circuit"] == 1, "street",
        np.where(result["longest_straight_m"] > 1500, "low_downforce", "high_downforce"),
    )
    return result
