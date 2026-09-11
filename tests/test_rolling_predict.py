"""Self-check for rolling re-prediction on the three pre-race-stage predictors
(finish position, quali-to-race delta, race time -- all resolve at or after
the race, so all three respond to the same post_practice/post_quali/pre_race
staging): the same saved model, called with progressively more real session
data, should actually respond to the new information (not return an identical
number regardless of stage), and stay in a plausible range for its target.

The qualifying predictor is checked separately below -- it's a genuinely
different information stage (predicts qualifying itself, before qualifying
happens), so rolling re-prediction through post_quali/pre_race doesn't apply
to it, and its feature set must never include this-race's actual quali/grid/
compound columns (that's the whole point of the model).

Run with `python tests/test_rolling_predict.py` (requires all four models to
be trained first).
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.features import QUALI_SAFE_FEATURE_COLS
from src.models.predict import load_model, mask_for_stage, predict

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "model_matrix.parquet"

# (model target, target column, plausible output range)
TARGETS = [
    ("finish_position", "target_finish_position", (-5, 25)),
    ("quali_delta", "target_quali_to_race_delta", (-25, 25)),
    ("race_time", "target_race_time_gap", (-5, 180)),
]

# columns that would leak this race's actual qualifying/grid/compound result
# into a model meant to predict qualifying BEFORE it happens
QUALI_LEAKAGE_COLS = {
    "grid_position", "quali_gap_to_pole", "teammate_quali_gap", "grid_vs_expected_position",
    "starting_tire_compound", "historical_compound_performance",
    "air_temp", "track_temp", "rain_probability", "wind_speed", "wet_track_probability",
}


def _check_target(target: str, target_col: str, plausible_range: tuple):
    model = load_model(target)
    df = pd.read_parquet(DATA_PATH)
    row = df[df[target_col].notna()].sample(1, random_state=7)

    preds = {}
    for stage in ("post_practice", "post_quali", "pre_race"):
        staged = mask_for_stage(row, stage)
        preds[stage] = predict(model, staged, target=target).iloc[0]

    print(f"  [{target}] predictions by stage: {preds}")
    assert len({round(v, 4) for v in preds.values()}) > 1, f"[{target}] prediction never changed across stages"
    lo, hi = plausible_range
    for v in preds.values():
        assert lo <= v <= hi, f"[{target}] prediction {v} outside a plausible range"


def test_rolling_prediction_responds_to_new_information():
    for target, target_col, plausible_range in TARGETS:
        _check_target(target, target_col, plausible_range)


def test_qualifying_predictor_is_leakage_safe_and_plausible():
    assert not QUALI_LEAKAGE_COLS & set(QUALI_SAFE_FEATURE_COLS), "qualifying feature set includes a this-race actual it shouldn't see"

    model = load_model("qualifying")
    df = pd.read_parquet(DATA_PATH)
    row = df[df["target_qualifying_gap"].notna()].sample(1, random_state=7)
    pred = predict(model, row, target="qualifying").iloc[0]

    print(f"  [qualifying] predicted gap to pole: {pred:.2f}s")
    assert -1 <= pred <= 15, f"[qualifying] prediction {pred} outside a plausible gap-to-pole range"


if __name__ == "__main__":
    test_rolling_prediction_responds_to_new_information()
    test_qualifying_predictor_is_leakage_safe_and_plausible()
    print("ok  test_rolling_predict")
