"""The saved models answer every point of a race weekend: the same model, fed
progressively more real session data (features.mask_for_stage), should change
its answer and stay plausible. The qualifying model must never see anything
decided in qualifying or on race day. Run with `python tests/test_rolling_predict.py`."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.features import QUALI_COLS, QUALI_SAFE_FEATURE_COLS, RACE_DAY_COLS, RACE_STAGES, mask_for_stage, stage_of
from src.models.predict import load_model, predict

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "model_matrix.parquet"

TARGETS = [  # (target, column, plausible range)
    ("finish_position", "target_finish_position", (-5, 25)),
    ("quali_delta", "target_quali_to_race_delta", (-25, 25)),
    ("race_time", "target_race_gap_pct", (-1, 30)),
]


def _row(col):
    df = pd.read_parquet(DATA_PATH)
    return df[df[col].notna() & df["practice_pace"].notna() & df["starting_tire_compound"].notna()].sample(1, random_state=7)


def test_predictions_respond_to_each_stage():
    for target, col, (lo, hi) in TARGETS:
        row, model = _row(col), load_model(target)
        preds = {s: float(predict(model, mask_for_stage(row, s), target=target).iloc[0]) for s in RACE_STAGES}
        print(f"  [{target}] {preds}")
        assert len({round(v, 4) for v in preds.values()}) > 1, f"[{target}] never changed across stages"
        assert all(lo <= v <= hi for v in preds.values()), f"[{target}] implausible: {preds}"


def test_stage_is_read_off_what_is_filled_in():
    row = _row("target_finish_position")
    assert [stage_of(mask_for_stage(row, s)) for s in RACE_STAGES] == RACE_STAGES


def test_qualifying_model_is_leakage_safe_and_plausible():
    assert not (set(QUALI_COLS) | set(RACE_DAY_COLS)) & set(QUALI_SAFE_FEATURE_COLS)
    assert not {"air_temp_forecast", "rain_mm_forecast", "wind_kph_forecast"} & set(QUALI_SAFE_FEATURE_COLS), \
        "Sunday's forecast isn't Saturday's weather"
    pred = float(predict(load_model("qualifying"), _row("target_qualifying_gap_pct"), target="qualifying").iloc[0])
    assert -0.5 <= pred <= 8, f"gap to pole {pred}% of a lap is implausible"


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} checks passed")
