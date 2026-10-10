"""Which columns each predictor sees, and what is unknown at each point of a
race weekend.

One model per target serves every stage of the weekend. It is trained on
copies of each historical row with the not-yet-known columns blanked out
(`mask_for_stage`), so a Thursday forecast with no grid and a Saturday-night
forecast with the grid are both situations the model has actually seen. The
earlier version trained only on complete rows (grid known 99% of the time) and
its pre-qualifying forecasts were worse than guessing each driver's recent
average finish (4.09 vs 3.62 places MAE, measured 2026-09-28).

Sprint weekends (2023 on) have two extra stages between practice and
qualifying: after Sprint Qualifying and after the Sprint. Their columns are
NaN on every other weekend, so those two stages are only trained and
evaluated on sprint-weekend rows (SPRINT_STAGES).
"""
import numpy as np
import pandas as pd

from src.features.build_dataset import CIRCUIT_COLS, SPRINT_COLS, WEATHER_COLS
from src.models.catalog import FEATURE_LABELS  # noqa: F401  (re-exported for older imports)

# FP long-run pace was tried as a second practice input and added nothing
# (held-out MAE 3.749 vs 3.744 without it); it stays in the payload for fans.
PRACTICE_COLS = ["practice_pace"]
QUALI_COLS = ["grid_position", "quali_gap_pct", "teammate_quali_gap", "grid_vs_expected_position"]
RACE_DAY_COLS = ["starting_tire_compound", "historical_compound_performance"]
SPRINT_QUALI_COLS = ["sprint_quali_gap_pct"]
SPRINT_RACE_COLS = ["sprint_finish_position", "sprint_race_pace_pct"]
assert SPRINT_QUALI_COLS + SPRINT_RACE_COLS == SPRINT_COLS

DRIVER_COLS = [
    "grid_position", "quali_gap_pct", *PRACTICE_COLS, "driver_recent_form",
    "driver_track_form", "driver_positions_gained_form", "driver_dnf_rate",
]
TEAM_COLS = ["team_recent_form", "team_quali_pace", "team_race_pace", "team_reliability", "team_track_type_form"]
RELATIVE_COLS = ["teammate_quali_gap", "teammate_race_pace_gap", "grid_vs_expected_position"]
STRATEGY_COLS = ["starting_tire_compound", "expected_stops", "historical_compound_performance"]

# is_sprint_weekend is known from the calendar: it tells the model a weekend
# has one practice session, not three
SPRINT_FEATURE_COLS = ["is_sprint_weekend", *SPRINT_COLS]

FEATURE_COLS = CIRCUIT_COLS + WEATHER_COLS + DRIVER_COLS + TEAM_COLS + RELATIVE_COLS + STRATEGY_COLS + SPRINT_FEATURE_COLS
# fixed category list (not inferred per batch) so a row with no compound yet still has the full category set
COMPOUND_CATEGORIES = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]

# The qualifying model predicts qualifying itself, so it may only see what is
# known before qualifying: no grid, no quali times, nothing decided on race
# day, and no weather (the forecast is for Sunday's race, not Saturday). The
# sprint sessions run before Grand Prix qualifying from 2024, and in 2023
# (when they ran after it) neither depended on it: the Sprint grid came from
# the Shootout. So they are fair inputs.
QUALI_SAFE_FEATURE_COLS = (
    CIRCUIT_COLS + PRACTICE_COLS
    + ["driver_recent_form", "driver_track_form", "driver_positions_gained_form", "driver_dnf_rate"]
    + TEAM_COLS + ["teammate_race_pace_gap", "expected_stops"] + SPRINT_FEATURE_COLS
)

# Weekend stages, in order, and what is still unknown at each.
UNKNOWN_AT = {
    "pre_weekend": PRACTICE_COLS + SPRINT_QUALI_COLS + SPRINT_RACE_COLS + QUALI_COLS + RACE_DAY_COLS,
    "post_practice": SPRINT_QUALI_COLS + SPRINT_RACE_COLS + QUALI_COLS + RACE_DAY_COLS,
    "post_sprint_quali": SPRINT_RACE_COLS + QUALI_COLS + RACE_DAY_COLS,
    "post_sprint": QUALI_COLS + RACE_DAY_COLS,
    "post_quali": RACE_DAY_COLS,
    "race_day": [],
}
STAGES = list(UNKNOWN_AT)
SPRINT_STAGES = ["post_sprint_quali", "post_sprint"]
RACE_STAGES = STAGES
QUALI_STAGES = ["pre_weekend", "post_practice", *SPRINT_STAGES]


def mask_for_stage(rows: pd.DataFrame, stage: str) -> pd.DataFrame:
    if stage not in UNKNOWN_AT:
        raise ValueError(f"unknown stage: {stage}")
    rows = rows.copy()
    for col in UNKNOWN_AT[stage]:
        if col in rows.columns:
            rows[col] = np.nan
    return rows


def stage_of(rows: pd.DataFrame) -> str:
    """The stage a live row set is actually at, read off what's filled in."""
    if rows["starting_tire_compound"].notna().any():
        return "race_day"
    if rows["grid_position"].notna().any() or rows["quali_gap_pct"].notna().any():
        return "post_quali"
    if rows[SPRINT_RACE_COLS].notna().any().any():
        return "post_sprint"
    if rows[SPRINT_QUALI_COLS].notna().any().any():
        return "post_sprint_quali"
    if rows[PRACTICE_COLS].notna().any().any():
        return "post_practice"
    return "pre_weekend"


def row_from_dict(feature_row: dict) -> pd.DataFrame:
    """Single-row frame from a JSON feature dict. JSON null becomes None, which
    makes pandas type the column `object`, which XGBoost rejects -- so coerce
    every numeric feature back to float."""
    row = pd.DataFrame([feature_row])
    numeric_cols = [c for c in FEATURE_COLS if c != "starting_tire_compound" and c in row.columns]
    row[numeric_cols] = row[numeric_cols].apply(pd.to_numeric, errors="coerce")
    return row


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df.reindex(columns=FEATURE_COLS).copy()
    # grid slot matters more where passing is hard; spelled out so explanations can name it
    X["grid_x_overtaking_difficulty"] = X["grid_position"] * X["overtaking_difficulty"]
    X["starting_tire_compound"] = pd.Categorical(X["starting_tire_compound"], categories=COMPOUND_CATEGORIES)
    return X


def prepare_features_quali(df: pd.DataFrame) -> pd.DataFrame:
    return df.reindex(columns=QUALI_SAFE_FEATURE_COLS).astype(float)


PREPARE_FN = {
    "finish_position": prepare_features, "quali_delta": prepare_features,
    "qualifying": prepare_features_quali, "race_time": prepare_features,
}
