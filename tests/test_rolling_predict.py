"""Self-check for Phase 2 rolling re-prediction: the same saved model, called
with progressively more real session data, should actually respond to the
new information (not return an identical number regardless of stage), and
stay in a plausible finishing-position range. Run with
`python tests/test_rolling_predict.py` (requires the model to be trained first
via `python -m src.models.train_finish_position`).
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.predict import load_model, mask_for_stage, predict

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "model_matrix.parquet"


def test_rolling_prediction_responds_to_new_information():
    model = load_model()
    df = pd.read_parquet(DATA_PATH)
    row = df[df["target_finish_position"].notna()].sample(1, random_state=7)

    preds = {}
    for stage in ("post_practice", "post_quali", "pre_race"):
        staged = mask_for_stage(row, stage)
        preds[stage] = predict(model, staged).iloc[0]

    print(f"  predictions by stage: {preds}")
    assert len({round(v, 4) for v in preds.values()}) > 1, "prediction never changed across stages"
    for v in preds.values():
        assert -5 <= v <= 25, f"prediction {v} outside a plausible finishing-position range"


if __name__ == "__main__":
    test_rolling_prediction_responds_to_new_information()
    print("ok  test_rolling_prediction_responds_to_new_information")
