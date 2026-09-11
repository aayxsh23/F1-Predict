"""Self-check for rolling re-prediction on BOTH predictors (finish position,
Phase 2; quali-to-race delta, Phase 3): the same saved model, called with
progressively more real session data, should actually respond to the new
information (not return an identical number regardless of stage), and stay
in a plausible range for its target. Run with
`python tests/test_rolling_predict.py` (requires both models to be trained
first: `python -m src.models.train_finish_position` and
`python -m src.models.train_quali_delta`).
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.predict import load_model, mask_for_stage, predict

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "model_matrix.parquet"

# (model target, target column, plausible output range)
TARGETS = [
    ("finish_position", "target_finish_position", (-5, 25)),
    ("quali_delta", "target_quali_to_race_delta", (-25, 25)),
]


def _check_target(target: str, target_col: str, plausible_range: tuple):
    model = load_model(target)
    df = pd.read_parquet(DATA_PATH)
    row = df[df[target_col].notna()].sample(1, random_state=7)

    preds = {}
    for stage in ("post_practice", "post_quali", "pre_race"):
        staged = mask_for_stage(row, stage)
        preds[stage] = predict(model, staged).iloc[0]

    print(f"  [{target}] predictions by stage: {preds}")
    assert len({round(v, 4) for v in preds.values()}) > 1, f"[{target}] prediction never changed across stages"
    lo, hi = plausible_range
    for v in preds.values():
        assert lo <= v <= hi, f"[{target}] prediction {v} outside a plausible range"


def test_rolling_prediction_responds_to_new_information():
    for target, target_col, plausible_range in TARGETS:
        _check_target(target, target_col, plausible_range)


if __name__ == "__main__":
    test_rolling_prediction_responds_to_new_information()
    print("ok  test_rolling_prediction_responds_to_new_information")
