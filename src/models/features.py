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

After qualifying the grid is the qualifying order: that is all a Saturday-
night forecast has. Penalties move a fifth of the field, so on race day, once
the FIA publishes the official grid, the forecast switches to it
(`race_day`). Training sees both, exactly as a live forecast would.
"""
import numpy as np
import pandas as pd

from src.features.build_dataset import CIRCUIT_COLS, MEASURED_CIRCUIT_COLS, SPRINT_COLS, WEATHER_COLS
from src.models.catalog import FEATURE_LABELS  # noqa: F401  (re-exported for older imports)

# FP long-run pace was tried as a second practice input and added nothing
# (held-out MAE 3.749 vs 3.744 without it); it stays in the payload for fans.
PRACTICE_COLS = ["practice_pace"]
QUALI_COLS = ["grid_position", "quali_gap_pct", "teammate_quali_gap", "grid_vs_expected_position"]
# this weekend's car upgrades: the FIA publishes them on the Friday, before practice
EVENT_COLS = ["team_performance_upgrades"]
SPRINT_QUALI_COLS = ["sprint_quali_gap_pct"]
SPRINT_RACE_COLS = ["sprint_finish_position", "sprint_race_pace_pct"]
assert SPRINT_QUALI_COLS + SPRINT_RACE_COLS == SPRINT_COLS

DRIVER_COLS = [
    "grid_position", "quali_gap_pct", *PRACTICE_COLS, "driver_recent_form",
    "driver_track_form", "driver_positions_gained_form", "driver_dnf_rate",
]
# A wet-weather skill input (each driver's finish vs form in past wet races)
# was tried on 2026-10-10 and made held-out finishing error worse (+0.007
# places, 95% CI +0.002 to +0.013): too few wet races to learn from.
TEAM_COLS = ["team_recent_form", "team_quali_pace", "team_race_pace", "team_reliability", "team_track_type_form",
             "team_upgrades_recent", *EVENT_COLS]
RELATIVE_COLS = ["teammate_quali_gap", "teammate_race_pace_gap", "grid_vs_expected_position"]
# Starting tyres were inputs once; they are never known before a live race
# starts, so they only ever filled the race_day copies (2026-10-10: identical
# accuracy with and without them) and were dropped.
STRATEGY_COLS = ["expected_stops"]

# is_sprint_weekend is known from the calendar: it tells the model a weekend
# has one practice session, not three
SPRINT_FEATURE_COLS = ["is_sprint_weekend", *SPRINT_COLS]

# Circuit facts: finishing position and places gained read the ones measured
# from earlier races (features/circuit_history.py: better on the tuning races,
# held-out no worse); qualifying and gap to the winner keep the hand-typed
# ones, which the measured ones made worse (2026-10-11,
# experiments/circuit_history.py).
TYPED_OUTCOME_COLS = ["safety_car_frequency", "dnf_rate", "overtaking_difficulty"]
FINISH_CIRCUIT_COLS = [c for c in CIRCUIT_COLS if c not in TYPED_OUTCOME_COLS] + MEASURED_CIRCUIT_COLS


def _race_inputs(circuit: list[str]) -> list[str]:
    return list(dict.fromkeys(circuit + WEATHER_COLS + DRIVER_COLS + TEAM_COLS + RELATIVE_COLS + STRATEGY_COLS + SPRINT_FEATURE_COLS))


FINISH_FEATURE_COLS = _race_inputs(FINISH_CIRCUIT_COLS)  # finishing position, places gained
RACE_TIME_FEATURE_COLS = _race_inputs(CIRCUIT_COLS)  # gap to the winner
FEATURE_COLS = list(dict.fromkeys(FINISH_FEATURE_COLS + RACE_TIME_FEATURE_COLS))  # every input a race model reads

# The qualifying model predicts qualifying itself, so it may only see what is
# known before qualifying: no grid, no quali times, nothing decided on race
# day, and no weather (the forecast is for Sunday's race, not Saturday). The
# sprint sessions run before Grand Prix qualifying from 2024, and in 2023
# (when they ran after it) neither depended on it: the Sprint grid came from
# the Shootout. So they are fair inputs.
# The driver's own qualifying form (pace, and against the teammate) is for
# this model only: on 2026-10-11 it helped qualifying on the tuning races but
# made the finishing-position model worse on the held-out ones.
QUALI_SAFE_FEATURE_COLS = (
    CIRCUIT_COLS + PRACTICE_COLS
    + ["driver_recent_form", "driver_track_form", "driver_positions_gained_form", "driver_dnf_rate"]
    + TEAM_COLS + ["teammate_race_pace_gap", "expected_stops"] + SPRINT_FEATURE_COLS
    + ["driver_quali_form", "driver_quali_vs_teammate_form"]
)

# Weekend stages, in order, and what is still unknown at each.
UNKNOWN_AT = {
    "pre_weekend": EVENT_COLS + PRACTICE_COLS + SPRINT_QUALI_COLS + SPRINT_RACE_COLS + QUALI_COLS,
    "post_practice": SPRINT_QUALI_COLS + SPRINT_RACE_COLS + QUALI_COLS,
    "post_sprint_quali": SPRINT_RACE_COLS + QUALI_COLS,
    "post_sprint": QUALI_COLS,
    "post_quali": [],  # the grid is the qualifying order (mask_for_stage)
    "race_day": [],  # the official grid, penalties applied
}
STAGES = list(UNKNOWN_AT)
SPRINT_STAGES = ["post_sprint_quali", "post_sprint"]
GRID_STAGES = ["post_quali", "race_day"]
RACE_STAGES = STAGES
QUALI_STAGES = ["pre_weekend", "post_practice", *SPRINT_STAGES]


# Which forecast a stage would really have had: on Thursday/Friday the race
# forecast is about three days out, on Saturday about one. Past races carry
# those (build_dataset.WEATHER_LEAD_COLS, from 2022/2024) besides the nearly
# same-day archive forecast; a live race only has the forecast of the moment,
# which is already the right one, so the swap does nothing there.
WEATHER_LEAD_FOR = {"pre_weekend": "_d3", "post_practice": "_d3", "post_sprint_quali": "_d1", "post_sprint": "_d1", "post_quali": "_d1"}


def mask_for_stage(rows: pd.DataFrame, stage: str) -> pd.DataFrame:
    if stage not in UNKNOWN_AT:
        raise ValueError(f"unknown stage: {stage}")
    rows = rows.copy()
    lead = WEATHER_LEAD_FOR.get(stage)
    if lead:
        for col in WEATHER_COLS:
            if f"{col}{lead}" in rows:
                rows[col] = rows[f"{col}{lead}"].where(rows[f"{col}{lead}"].notna(), rows[col])
    for col in UNKNOWN_AT[stage]:
        if col in rows.columns:
            rows[col] = np.nan
    if stage != "race_day":
        rows["grid_official"] = 0.0
    if stage == "post_quali" and "quali_position" in rows:
        rows["grid_position"] = rows["quali_position"]
        if "driver_recent_form" in rows:
            rows["grid_vs_expected_position"] = rows["grid_position"] - rows["driver_recent_form"]
    return rows


def stage_of(rows: pd.DataFrame) -> str:
    """The stage a live row set is actually at, read off what's filled in."""
    if rows.reindex(columns=["grid_official"])["grid_official"].eq(1).any():
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
    numeric_cols = [c for c in FEATURE_COLS if c in row.columns]
    row[numeric_cols] = row[numeric_cols].apply(pd.to_numeric, errors="coerce")
    return row


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Finishing position and places gained."""
    X = df.reindex(columns=FINISH_FEATURE_COLS).astype(float)
    # grid slot matters more where results follow the grid; spelled out so explanations can name it
    X["grid_x_circuit_grid_order"] = X["grid_position"] * X["circuit_grid_order"]
    return X


def prepare_features_race_time(df: pd.DataFrame) -> pd.DataFrame:
    """Gap to the winner."""
    X = df.reindex(columns=RACE_TIME_FEATURE_COLS).astype(float)
    X["grid_x_overtaking_difficulty"] = X["grid_position"] * X["overtaking_difficulty"]
    return X


def prepare_features_quali(df: pd.DataFrame) -> pd.DataFrame:
    return df.reindex(columns=QUALI_SAFE_FEATURE_COLS).astype(float)


PREPARE_FN = {
    "finish_position": prepare_features, "quali_delta": prepare_features,
    "qualifying": prepare_features_quali, "race_time": prepare_features_race_time,
}
